from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    chunk_size_chars: int = Field(default=1000, ge=1)
    chunk_overlap_chars: int = Field(default=120, ge=0)


def chunk_text(text: str, *, config: ChunkingConfig | None = None) -> list[str]:
    active_config = config or ChunkingConfig()
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    chunks: list[str] = []
    current = ""

    for block in blocks:
        if len(block) > active_config.chunk_size_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_oversized_block(block, active_config))
            continue

        candidate = block if not current else f"{current}\n\n{block}"
        if current and len(candidate) > active_config.chunk_size_chars:
            chunks.append(current.strip())
            current = _seed_overlap(chunks[-1], active_config, next_block=block)
            continue
        current = candidate

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk.strip()]


def _split_oversized_block(block: str, config: ChunkingConfig) -> list[str]:
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
