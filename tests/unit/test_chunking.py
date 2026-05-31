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
