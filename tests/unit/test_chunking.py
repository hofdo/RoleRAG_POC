from __future__ import annotations

from app.rag.chunking import ChunkingConfig, chunk_text


def test_chunk_text_preserves_headings_and_is_deterministic() -> None:
    text = """# Winter Palace

The palace was built on older ruins and keeps its records beneath the east wing.

## Archivists

The archivists maintain coded ledgers and sealed witness statements.
"""

    config = ChunkingConfig(chunk_size_chars=120, chunk_overlap_chars=20)

    first = chunk_text(text, config=config)
    second = chunk_text(text, config=config)

    assert first == second
    assert len(first) == 2
    assert first[0].startswith("# Winter Palace")
    assert "## Archivists" in first[1]


def test_chunk_text_skips_empty_chunks_and_applies_overlap() -> None:
    text = "\n\n".join(
        [
            "First note about the west gallery.",
            "Second note about the hidden stair.",
            "Third note about the regent's courier.",
        ]
    )

    chunks = chunk_text(text, config=ChunkingConfig(chunk_size_chars=60, chunk_overlap_chars=10))

    assert len(chunks) == 3
    assert all(chunk.strip() for chunk in chunks)
    assert "hidden" in chunks[1]
    assert "stair." in chunks[2]


def test_chunking_config_defaults_structure_aware_to_false() -> None:
    assert ChunkingConfig().structure_aware is False


# ---------------------------------------------------------------------------
# Golden flag-off byte-identity baseline (docs/22 P1.3).
#
# These three fixtures' outputs were captured by running the CURRENT (pre-P1.3)
# `chunk_text` on HEAD before chunking.py was touched -- not hand-written -- so this is a
# genuine byte-for-byte pin of the legacy behavior (including its existing quirks, e.g. the
# overlap-seed window landing mid-heading-marker in the first fixture below). Any future
# change to the legacy path that alters so much as one character here is a behavior change
# to the byte-identical default and must not land silently.
# ---------------------------------------------------------------------------

_GOLDEN_HEADINGS_TEXT = """# Winter Palace

The palace was built on older ruins and keeps its records beneath the east wing.

## Archivists

The archivists maintain coded ledgers and sealed witness statements.

### Sealed Vault

Only the seneschal holds the vault key, passed hand to hand for three generations.

## Gardens

The gardens bloom in spring with roses smuggled from the southern coast.
"""

_GOLDEN_HEADINGS_CONFIG = ChunkingConfig(chunk_size_chars=90, chunk_overlap_chars=15)

_GOLDEN_HEADINGS_OUTPUT = [
    "# Winter Palace",
    "# Winter Palace\n\nThe palace was built on older ruins and keeps its records beneath "
    "the east wing.",
    "the east wing.\n\n## Archivists",
    "## Archivists\n\nThe archivists maintain coded ledgers and sealed witness statements.",
    "ess statements.\n\n### Sealed Vault",
    "## Sealed Vault\n\nOnly the seneschal holds the vault key, passed hand to hand for "
    "three generations.",
    "ee generations.\n\n## Gardens",
    "ns.\n\n## Gardens\n\nThe gardens bloom in spring with roses smuggled from the southern coast.",
]

_GOLDEN_OVERSIZED_TEXT = (
    "The regent's courier rode through the night carrying sealed dispatches meant only for "
    "the seneschal's own hands and nobody else was permitted to break the wax before the "
    "appointed hour arrived at the eastern gate where the guards had been waiting since dusk "
    "without any word from the capital."
)

_GOLDEN_OVERSIZED_CONFIG = ChunkingConfig(chunk_size_chars=100, chunk_overlap_chars=20)

_GOLDEN_OVERSIZED_OUTPUT = [
    "The regent's courier rode through the night carrying sealed dispatches meant only for "
    "the seneschal'",
    "y for the seneschal's own hands and nobody else was permitted to break the wax before "
    "the appointed",
    "efore the appointed hour arrived at the eastern gate where the guards had been waiting "
    "since dusk wi",
    "aiting since dusk without any word from the capital.",
]

_GOLDEN_PLAIN_TEXT = "\n\n".join(
    [
        "First note about the west gallery and its collection of old maps.",
        "Second note about the hidden stair behind the tapestry.",
        "Third note about the regent's courier and the sealed letter.",
        "Fourth note about the kitchens and the missing silver.",
    ]
)

_GOLDEN_PLAIN_CONFIG = ChunkingConfig(chunk_size_chars=70, chunk_overlap_chars=10)

_GOLDEN_PLAIN_OUTPUT = [
    "First note about the west gallery and its collection of old maps.",
    "old maps.\n\nSecond note about the hidden stair behind the tapestry.",
    "tapestry.\n\nThird note about the regent's courier and the sealed letter.",
    "ed letter.\n\nFourth note about the kitchens and the missing silver.",
]


def test_chunk_text_golden_baseline_nested_headings_is_byte_identical() -> None:
    assert chunk_text(_GOLDEN_HEADINGS_TEXT, config=_GOLDEN_HEADINGS_CONFIG) == (
        _GOLDEN_HEADINGS_OUTPUT
    )
    # doc_title is ignored entirely on the legacy (structure_aware=False) path.
    assert (
        chunk_text(_GOLDEN_HEADINGS_TEXT, config=_GOLDEN_HEADINGS_CONFIG, doc_title="Anything")
        == _GOLDEN_HEADINGS_OUTPUT
    )


def test_chunk_text_golden_baseline_oversized_paragraph_is_byte_identical() -> None:
    assert chunk_text(_GOLDEN_OVERSIZED_TEXT, config=_GOLDEN_OVERSIZED_CONFIG) == (
        _GOLDEN_OVERSIZED_OUTPUT
    )
    assert (
        chunk_text(_GOLDEN_OVERSIZED_TEXT, config=_GOLDEN_OVERSIZED_CONFIG, doc_title="Anything")
        == _GOLDEN_OVERSIZED_OUTPUT
    )


def test_chunk_text_golden_baseline_plain_multi_paragraph_is_byte_identical() -> None:
    assert chunk_text(_GOLDEN_PLAIN_TEXT, config=_GOLDEN_PLAIN_CONFIG) == _GOLDEN_PLAIN_OUTPUT
    assert (
        chunk_text(_GOLDEN_PLAIN_TEXT, config=_GOLDEN_PLAIN_CONFIG, doc_title="Anything")
        == _GOLDEN_PLAIN_OUTPUT
    )


# ---------------------------------------------------------------------------
# Structure-aware chunking (docs/22 P1.3, structure_aware=True).
# ---------------------------------------------------------------------------


def test_structure_aware_nested_section_paths_and_headers() -> None:
    text = """# Winter Palace

Intro text about the palace grounds and its long history under three dynasties.

## Archivists

The archivists maintain coded ledgers and sealed witness statements in the east wing.

### Sealed Vault

Only the seneschal holds the vault key.

## Gardens

The gardens bloom in spring.
"""
    config = ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=20, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title="Winter Palace Archive")

    assert chunks == [
        "Winter Palace Archive › Winter Palace\n\nIntro text about the palace grounds and its "
        "long history under three dynasties.",
        "Winter Palace Archive › Winter Palace › Archivists\n\nThe archivists maintain coded "
        "ledgers and sealed witness statements in the east wing.",
        "Winter Palace Archive › Winter Palace › Archivists › Sealed Vault\n\nOnly the "
        "seneschal holds the vault key.",
        # A level-2 heading ("## Gardens") pops BOTH the level-3 ("Sealed Vault") and the
        # level-2 ("Archivists") stack entries -- the path is "... > Gardens", not
        # "... > Archivists > Sealed Vault > Gardens".
        "Winter Palace Archive › Winter Palace › Gardens\n\nThe gardens bloom in spring.",
    ]


def test_structure_aware_header_has_blank_line_separator() -> None:
    text = "# Doc\n\nBody text right here."
    config = ChunkingConfig(structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title="Title")

    assert chunks == ["Title › Doc\n\nBody text right here."]
    header, _, body = chunks[0].partition("\n\n")
    assert header == "Title › Doc"
    assert body == "Body text right here."


def test_structure_aware_no_title_no_path_omits_header_entirely() -> None:
    text = "Just a plain paragraph with no headings at all in this document."
    config = ChunkingConfig(chunk_size_chars=200, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title=None)

    assert chunks == [text]


def test_structure_aware_title_only_header_when_no_heading_path() -> None:
    text = "Just a plain paragraph with no headings at all in this document."
    config = ChunkingConfig(chunk_size_chars=200, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title="My Doc")

    assert chunks == [f"My Doc\n\n{text}"]


def test_structure_aware_path_only_header_when_no_doc_title() -> None:
    text = "# Heading\n\nBody content."
    config = ChunkingConfig(structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title=None)

    assert chunks == ["Heading\n\nBody content."]


def test_structure_aware_header_deduplicates_title_matching_root_segment() -> None:
    text = "# Title\n\nRoot body.\n\n## Sub\n\nSub body."
    config = ChunkingConfig(chunk_size_chars=200, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title="Title")

    # The doc title already IS the section path's root segment (the common lore shape
    # whose only H1 is the document title), so it is not repeated: "Title" and
    # "Title › Sub", never "Title › Title" / "Title › Title › Sub".
    assert chunks == ["Title\n\nRoot body.", "Title › Sub\n\nSub body."]


def test_structure_aware_header_keeps_distinct_title_prefix() -> None:
    text = "# Chapter\n\nBody."
    config = ChunkingConfig(structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title="Book")

    assert chunks == ["Book › Chapter\n\nBody."]


def test_structure_aware_root_section_before_first_heading() -> None:
    text = "Root intro paragraph here.\n\n# First Heading\n\nBody under first heading."
    config = ChunkingConfig(chunk_size_chars=200, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title="Doc")

    assert chunks == [
        "Doc\n\nRoot intro paragraph here.",
        "Doc › First Heading\n\nBody under first heading.",
    ]


def test_structure_aware_sections_never_straddle_heading_boundaries() -> None:
    text = (
        "# Sec1\n\nAlpha content lives here only in section one, nowhere else at all.\n\n"
        "## Sec2\n\nBeta content lives here only in section two, nowhere else at all."
    )
    config = ChunkingConfig(chunk_size_chars=60, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title=None)

    assert chunks == [
        "Sec1\n\nAlpha content lives here only in section one, nowhere else",
        "Sec1\n\nat all.",
        "Sec1 › Sec2\n\nBeta content lives here only in section two, nowhere else at",
        "Sec1 › Sec2\n\nall.",
    ]
    assert all(not ("Alpha" in chunk and "Beta" in chunk) for chunk in chunks)


def test_structure_aware_overlap_seeding_resets_at_section_boundary() -> None:
    text = (
        "# Notes\n\n"
        "First note about the west gallery and its collection of old maps.\n\n"
        "Second note about the hidden stair behind the tapestry.\n\n"
        "## Next\n\n"
        "Third note about the regent's courier and the sealed letter."
    )
    config = ChunkingConfig(chunk_size_chars=70, chunk_overlap_chars=10, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title=None)

    assert chunks == [
        "Notes\n\nFirst note about the west gallery and its collection of old maps.",
        "Notes\n\nold maps.\n\nSecond note about the hidden stair behind the tapestry.",
        # No "tapestry."-derived overlap fragment leaks into the "Next" section: the body
        # here starts exactly at "Third note", proving _accumulate_blocks resets its
        # ``current`` buffer per section rather than seeding overlap from the prior section.
        "Notes › Next\n\nThird note about the regent's courier and the sealed letter.",
    ]


def test_structure_aware_empty_intermediate_section_emits_no_chunk() -> None:
    # "# A" has no body of its own (immediately followed by "## B") -- it contributes zero
    # chunks, but its title still lives in "B"'s section path.
    text = "# A\n\n## B\n\nReal content under B."
    config = ChunkingConfig(structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title="Doc")

    assert chunks == ["Doc › A › B\n\nReal content under B."]


def test_structure_aware_heading_only_document_emits_no_chunks() -> None:
    text = "# A\n\n## B\n"
    config = ChunkingConfig(structure_aware=True)

    assert chunk_text(text, config=config, doc_title="Doc") == []


def test_structure_aware_sentence_boundary_packing_never_cuts_mid_word() -> None:
    text = "## Chapter\n\n" + " ".join(f"Sentence number {i} is here." for i in range(1, 8))
    config = ChunkingConfig(chunk_size_chars=60, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title=None)

    assert chunks == [
        "Chapter\n\nSentence number 1 is here. Sentence number 2 is here.",
        "Chapter\n\nSentence number 3 is here. Sentence number 4 is here.",
        "Chapter\n\nSentence number 5 is here. Sentence number 6 is here.",
        "Chapter\n\nSentence number 7 is here.",
    ]
    # The header pushes chunk 0 a few chars past chunk_size_chars=60 -- documented,
    # budget-wise the header is free (only the BODY is packed to the cap).
    assert len(chunks[0]) > 60
    # Every packed body is made of whole words from the original text -- no word fragments.
    all_words = set(text.split())
    for chunk in chunks:
        _, _, body = chunk.partition("\n\n")
        assert all(word in all_words for word in body.split())


def test_structure_aware_single_long_sentence_falls_back_to_word_boundary() -> None:
    long_sentence = (
        "this is one extremely long sentence without any terminal punctuation that just "
        "keeps going and going far past the cap"
    )
    text = "## Chapter\n\n" + long_sentence
    config = ChunkingConfig(chunk_size_chars=50, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title=None)

    assert chunks == [
        "Chapter\n\nthis is one extremely long sentence without any",
        "Chapter\n\nterminal punctuation that just keeps going and",
        "Chapter\n\ngoing far past the cap",
    ]
    # Every packed body breaks on a real word boundary (no fragment ends mid-word).
    words = set(long_sentence.split())
    for chunk in chunks:
        _, _, body = chunk.partition("\n\n")
        assert all(word in words for word in body.split())


def test_structure_aware_unbroken_token_falls_back_to_hard_split() -> None:
    token = "x" * 80
    text = "## Chapter\n\n" + token
    config = ChunkingConfig(chunk_size_chars=30, chunk_overlap_chars=5, structure_aware=True)

    chunks = chunk_text(text, config=config, doc_title=None)

    assert chunks == [
        "Chapter\n\n" + "x" * 30,
        "Chapter\n\n" + "x" * 30,
        "Chapter\n\n" + "x" * 30,
        "Chapter\n\n" + "x" * 5,
    ]


def test_structure_aware_empty_document_returns_empty_list() -> None:
    config = ChunkingConfig(structure_aware=True)

    assert chunk_text("", config=config, doc_title="Doc") == []
    assert chunk_text("   \n\n  ", config=config, doc_title="Doc") == []
