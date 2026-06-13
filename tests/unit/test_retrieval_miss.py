from __future__ import annotations

from typing import Any

from app.diagnostics.retrieval_miss import summarize_retrieval_miss


def _item(
    chunk_id: str,
    *,
    selected_rank: int | None = None,
    adjusted_score: float = 0.5,
) -> dict[str, Any]:
    return {"id": chunk_id, "selected_rank": selected_rank, "adjusted_score": adjusted_score}


def test_selected_chunk_is_not_a_miss() -> None:
    report = summarize_retrieval_miss(
        expected_ids=["want"],
        selected=[_item("want", selected_rank=1)],
        rejected=[_item("other")],
    )

    assert report.missed_ids == ()
    assert report.absent_ids == ()
    assert report.ranks[0].selected is True
    assert report.ranks[0].overall_rank == 1


def test_rejected_chunk_reports_overall_rank() -> None:
    report = summarize_retrieval_miss(
        expected_ids=["want"],
        selected=[_item("a", selected_rank=1), _item("b", selected_rank=2)],
        rejected=[_item("c"), _item("want"), _item("d")],
    )

    assert report.missed_ids == ("want",)
    assert report.absent_ids == ()
    # 2 selected + second position in the rejected tail.
    assert report.ranks[0].overall_rank == 4
    assert report.ranks[0].selected is False


def test_absent_chunk_is_missed_with_no_rank() -> None:
    report = summarize_retrieval_miss(
        expected_ids=["want"],
        selected=[_item("a", selected_rank=1)],
        rejected=[_item("b")],
    )

    assert report.missed_ids == ("want",)
    assert report.absent_ids == ("want",)
    assert report.ranks[0].overall_rank is None


def test_orders_selected_by_rank_then_rejected() -> None:
    report = summarize_retrieval_miss(
        expected_ids=["x", "y"],
        selected=[_item("p", selected_rank=2), _item("q", selected_rank=1)],
        rejected=[_item("x"), _item("y")],
    )

    by_id = {rank.chunk_id: rank.overall_rank for rank in report.ranks}
    assert by_id["x"] == 3
    assert by_id["y"] == 4
