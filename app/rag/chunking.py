from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel, Field

# Structure-aware section splitting (docs/22 P1.3, opt-in). ATX heading levels 1-6 at line
# start, one space, then the heading text -- deliberately minimal (not a full markdown
# parser: no fenced-code-block awareness, no closing '#' stripping), matching the spec's
# own heading regex.
_ATX_HEADING_RE = re.compile(r"^(#{1,6}) (.+)$", re.MULTILINE)

# Sentence-boundary packing for oversized blocks (flag on only): a terminator followed by
# whitespace-or-end. Mirrors the boundary regex proven in
# app/orchestration/context_budget.py's ``_truncate_text`` (trims backward from a cap;
# this packs forward instead), reimplemented locally rather than imported -- app/rag sits
# below app/orchestration in the dependency direction (orchestration depends on rag, never
# the reverse), so importing across that boundary would invert it.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?:\s|$)")

# Separator for both the heading-hierarchy section path (``A › B › C``) and the final
# ``<doc title> › <section path>`` contextual header line.
_PATH_SEPARATOR = " › "


class ChunkingConfig(BaseModel):
    chunk_size_chars: int = Field(default=1000, ge=1)
    chunk_overlap_chars: int = Field(default=120, ge=0)
    # Opt-in structure-aware chunking (docs/22 P1.3). False (default) reproduces the
    # original blind paragraph-accumulation chunker byte-for-byte, regardless of
    # doc_title. True splits on markdown ATX headings first (chunks never straddle a
    # section boundary), packs oversized blocks at sentence/word boundaries instead of a
    # fixed character offset, and prepends a "<doc title> › <section path>" header to
    # every emitted chunk's embedded text. This changes chunk text -> chunk ids (sha256 of
    # source:index:text, app/rag/ingestion.py) -> flipping this setting triggers a natural
    # full re-ingest per source on next contact (the #86 unchanged-document skip sees a
    # different id set and falls through to a full re-embed). Flip the default only after
    # the owner-side semantic benchmark (docs/24) shows flag-on helps recall/nDCG.
    structure_aware: bool = False


def chunk_text(
    text: str,
    *,
    config: ChunkingConfig | None = None,
    doc_title: str | None = None,
) -> list[str]:
    """Split ``text`` into chunks per ``config``.

    ``doc_title`` feeds the contextual header line and is used only when
    ``config.structure_aware`` is True (docs/22 P1.3); the legacy (default) path ignores
    it entirely and is byte-identical to the pre-P1.3 chunker, so a caller may pass it
    unconditionally without needing to branch on the flag itself.
    """
    active_config = config or ChunkingConfig()
    if not active_config.structure_aware:
        return _chunk_text_legacy(text, active_config)
    return _chunk_text_structure_aware(text, active_config, doc_title=doc_title)


def _chunk_text_legacy(text: str, config: ChunkingConfig) -> list[str]:
    """The original (pre-P1.3) blind paragraph-accumulation chunker.

    Untouched logic, pinned byte-for-byte by tests/unit/test_chunking.py's golden-baseline
    test (captured from HEAD before P1.3 landed).
    """
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    return _accumulate_blocks(blocks, config, _split_oversized_block)


def _accumulate_blocks(
    blocks: list[str],
    config: ChunkingConfig,
    split_oversized: Callable[[str, ChunkingConfig], list[str]],
) -> list[str]:
    """Shared paragraph-accumulation loop: pack ``blocks`` up to ``chunk_size_chars``,
    seeding overlap between chunks, deferring any single oversized block to
    ``split_oversized``.

    Used both by the legacy chunker (``_split_oversized_block``, unchanged) and, once per
    section, by the structure-aware chunker (``_split_oversized_block_structured``) -- the
    accumulation policy is identical either way; only the oversized-block strategy
    differs. Calling this once per section (rather than once for the whole document) is
    what keeps chunks from ever straddling a section boundary and what scopes overlap
    seeding to within a section.
    """
    chunks: list[str] = []
    current = ""

    for block in blocks:
        if len(block) > config.chunk_size_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_oversized(block, config))
            continue

        candidate = block if not current else f"{current}\n\n{block}"
        if current and len(candidate) > config.chunk_size_chars:
            chunks.append(current.strip())
            current = _seed_overlap(chunks[-1], config, next_block=block)
            continue
        current = candidate

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk.strip()]


def _split_oversized_block(block: str, config: ChunkingConfig) -> list[str]:
    """Legacy fixed-window hard split (may cut mid-word). Unchanged since pre-P1.3; also
    reused as the structure-aware cascade's last-resort tier for a single unbroken token
    longer than the cap (``_split_oversized_block_structured``)."""
    step = max(config.chunk_size_chars - config.chunk_overlap_chars, 1)
    return [
        block[start : start + config.chunk_size_chars].strip()
        for start in range(0, len(block), step)
        if block[start : start + config.chunk_size_chars].strip()
    ]


def _seed_overlap(previous_chunk: str, config: ChunkingConfig, *, next_block: str) -> str:
    if config.chunk_overlap_chars <= 0:
        return next_block

    overlap = previous_chunk[-config.chunk_overlap_chars :].lstrip()
    if overlap:
        return f"{overlap}\n\n{next_block}"
    return next_block


# --- Structure-aware chunking (docs/22 P1.3, flag on) ------------------------------------


def _chunk_text_structure_aware(
    text: str, config: ChunkingConfig, *, doc_title: str | None
) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    for section_path, body in _split_into_sections(normalized):
        blocks = [block.strip() for block in body.split("\n\n") if block.strip()]
        if not blocks:
            continue
        header = _section_header(doc_title, section_path)
        for body_chunk in _accumulate_blocks(blocks, config, _split_oversized_block_structured):
            chunks.append(f"{header}\n\n{body_chunk}" if header else body_chunk)
    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split ``text`` on markdown ATX headings (levels 1-6) into ``(section_path, body)``
    pairs in document order.

    ``section_path`` joins the active heading hierarchy with ``_PATH_SEPARATOR``: entering
    ``## B`` under ``# A`` gives ``"A › B"``; a heading at level N first pops every stack
    entry at level >= N (so a later ``## D`` after ``### C`` pops back to ``"A › D"``, not
    ``"A › B › C › D"``). Text before the first heading is a root section with an empty
    path. A section's body is exactly the text between its heading line and the next
    heading of ANY level (or end of document) -- the heading line itself is never part of
    any body, so its text is never duplicated between the header and the chunk content.
    """
    headings = list(_ATX_HEADING_RE.finditer(text))
    if not headings:
        return [("", text)]

    sections: list[tuple[str, str]] = [("", text[: headings[0].start()])]
    stack: list[tuple[int, str]] = []

    for index, match in enumerate(headings):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = _PATH_SEPARATOR.join(item_title for _, item_title in stack)

        body_start = match.end() + 1  # skip the single '\n' ending the heading line
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections.append((path, text[body_start:body_end]))

    return sections


def _section_header(doc_title: str | None, section_path: str) -> str:
    """Join whichever of ``doc_title``/``section_path`` are present with
    ``_PATH_SEPARATOR``; an absent part (``None`` or empty) is skipped, and if both are
    absent the header is empty (the caller then emits no header line at all). When the
    section path's root segment already IS the doc title (the common lore shape whose
    only H1 is the document's own title, which ``ingestion._derive_doc_title`` also picks
    as ``doc_title``), the title is not repeated -- ``Title › Section``, never
    ``Title › Title › Section``."""
    if doc_title and (
        section_path == doc_title or section_path.startswith(doc_title + _PATH_SEPARATOR)
    ):
        return section_path
    parts = [part for part in (doc_title, section_path) if part]
    return _PATH_SEPARATOR.join(parts)


def _split_oversized_block_structured(block: str, config: ChunkingConfig) -> list[str]:
    """Oversized-block cascade for the structure-aware path: sentence boundaries, then
    word boundaries, then (only for a single unbroken token) the legacy fixed-window hard
    split -- never a mid-word cut while any boundary exists at some tier."""
    max_chars = config.chunk_size_chars

    def _word_tier(oversized_unit: str) -> list[str]:
        return _pack_by_boundary(
            oversized_unit,
            max_chars=max_chars,
            split_units=_split_into_words,
            next_tier=lambda oversized_token: _split_oversized_block(oversized_token, config),
        )

    return _pack_by_boundary(
        block,
        max_chars=max_chars,
        split_units=_split_into_sentences,
        next_tier=_word_tier,
    )


def _pack_by_boundary(
    text: str,
    *,
    max_chars: int,
    split_units: Callable[[str], list[str]],
    next_tier: Callable[[str], list[str]],
) -> list[str]:
    """Greedily pack ``text`` into pieces <= ``max_chars`` at the boundaries
    ``split_units`` finds, joining consecutive units with a single space.

    A unit still longer than ``max_chars`` on its own (no boundary of this kind exists
    inside it) is handed to ``next_tier`` instead of being packed directly, so an
    oversized unit cascades to a finer boundary rather than ever being cut arbitrarily at
    this tier.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    units = split_units(text)
    if len(units) <= 1:
        return next_tier(text)

    packed: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if len(unit) > max_chars:
            if current:
                packed.append(" ".join(current))
                current, current_len = [], 0
            packed.extend(next_tier(unit))
            continue
        addition = len(unit) if not current else 1 + len(unit)
        if current and current_len + addition > max_chars:
            packed.append(" ".join(current))
            current, current_len = [unit], len(unit)
        else:
            current.append(unit)
            current_len += addition
    if current:
        packed.append(" ".join(current))
    return packed


def _split_into_sentences(text: str) -> list[str]:
    """Split on a terminator (``.``/``!``/``?``) followed by whitespace-or-end, keeping the
    terminator with the sentence it closes. Same boundary regex as
    app/orchestration/context_budget.py's ``_truncate_text`` (reimplemented locally --
    see ``_SENTENCE_BOUNDARY_RE``'s comment for why it isn't imported), applied here to
    pack forward instead of trimming backward from a cap."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        piece = text[start : match.end()].strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_into_words(text: str) -> list[str]:
    return text.split()
