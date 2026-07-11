"""Hand-computed unit tests for the recall@k / nDCG@k / MRR metric layer
(docs/22 P0.4). Pure math, no embedding provider involved -- see
app/evals/retrieval_metrics.py for the formulas being pinned here.
"""

from __future__ import annotations

import math

import pytest

from app.evals.retrieval_metrics import mrr, ndcg_at_k, recall_at_k


class TestRecallAtK:
    def test_binary_recall_counts_judgment_one_and_two(self) -> None:
        judgments = {"a": 2, "b": 1}
        ranked = ["x", "a", "y", "b"]

        assert recall_at_k(ranked, judgments, k=2) == pytest.approx(0.5)  # top2: x, a -> 1/2
        assert recall_at_k(ranked, judgments, k=4) == pytest.approx(1.0)  # both found

    def test_strict_recall_only_counts_judgment_two(self) -> None:
        judgments = {"a": 2, "b": 1}
        ranked = ["x", "a", "y", "b"]

        # relevant set shrinks to {a}; top2 already contains it.
        assert recall_at_k(ranked, judgments, k=2, relevance_floor=2) == pytest.approx(1.0)
        # b (judgment 1) never counts toward the strict floor.
        assert recall_at_k(["x", "b"], judgments, k=2, relevance_floor=2) == pytest.approx(0.0)

    def test_recall_ignores_unjudged_chunks_in_ranking(self) -> None:
        judgments = {"a": 1}
        ranked = ["z", "y", "x", "a"]

        assert recall_at_k(ranked, judgments, k=3) == pytest.approx(0.0)
        assert recall_at_k(ranked, judgments, k=4) == pytest.approx(1.0)

    def test_recall_is_vacuously_satisfied_with_no_relevant_chunks(self) -> None:
        assert recall_at_k(["a", "b"], {}, k=5) == pytest.approx(1.0)
        assert recall_at_k([], {}, k=5) == pytest.approx(1.0)

    def test_recall_handles_empty_ranking_with_relevant_chunks(self) -> None:
        assert recall_at_k([], {"a": 2}, k=5) == pytest.approx(0.0)


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self) -> None:
        judgments = {"a": 2, "b": 1}
        ranked = ["a", "b", "c"]

        assert ndcg_at_k(ranked, judgments, k=3) == pytest.approx(1.0)

    def test_worst_ranking_scores_zero(self) -> None:
        judgments = {"a": 2, "b": 1}
        ranked = ["c", "d", "e"]  # neither relevant chunk retrieved at all

        assert ndcg_at_k(ranked, judgments, k=3) == pytest.approx(0.0)

    def test_reordered_ranking_hand_computed(self) -> None:
        """judgments={a:2, b:1}, ranked=[b, a, x], k=3.

        gains = [1, 2, 0] (b=1, a=2, x unjudged=0)
        dcg = 1/log2(2) + 2/log2(3) + 0/log2(4) = 1 + 1.26186... = 2.26186...
        ideal gains (sorted desc) = [2, 1]
        idcg = 2/log2(2) + 1/log2(3) = 2 + 0.63093... = 2.63093...
        ndcg = dcg / idcg = 0.85972... (computed independently below, not by
        calling the module under test, to actually catch a formula bug).
        """
        judgments = {"a": 2, "b": 1}
        ranked = ["b", "a", "x"]

        dcg = 1 / math.log2(2) + 2 / math.log2(3) + 0 / math.log2(4)
        idcg = 2 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg

        assert ndcg_at_k(ranked, judgments, k=3) == pytest.approx(expected)
        assert ndcg_at_k(ranked, judgments, k=3) == pytest.approx(0.8597187, rel=1e-6)

    def test_k_truncates_the_ranking(self) -> None:
        # The relevant chunk sits outside the top-1 window: k=1 sees nothing
        # relevant (ndcg=0), k=2 finds it but discounted at rank 2 rather than
        # rank 1, so it still falls short of the ideal (a hand-computed 0.63093,
        # not 1.0 -- 1.0 would require "a" ranked first, matching the ideal order).
        judgments = {"a": 2}
        ranked = ["x", "a"]

        assert ndcg_at_k(ranked, judgments, k=1) == pytest.approx(0.0)
        assert ndcg_at_k(ranked, judgments, k=2) == pytest.approx(
            (2 / math.log2(3)) / (2 / math.log2(2)), rel=1e-6
        )

        # Swapping the order so the relevant chunk actually leads matches the
        # ideal ranking exactly, at both window sizes.
        assert ndcg_at_k(["a", "x"], judgments, k=1) == pytest.approx(1.0)
        assert ndcg_at_k(["a", "x"], judgments, k=2) == pytest.approx(1.0)

    def test_no_judged_chunks_scores_zero(self) -> None:
        assert ndcg_at_k(["a", "b"], {}, k=5) == pytest.approx(0.0)
        assert ndcg_at_k([], {}, k=5) == pytest.approx(0.0)


class TestMrr:
    def test_reciprocal_rank_of_first_relevant_hit(self) -> None:
        judgments = {"a": 1, "b": 2}

        assert mrr(["x", "y", "a"], judgments) == pytest.approx(1 / 3)
        assert mrr(["a", "y"], judgments) == pytest.approx(1.0)

    def test_strict_floor_skips_judgment_one(self) -> None:
        judgments = {"a": 1, "b": 2}

        assert mrr(["a", "b"], judgments, relevance_floor=2) == pytest.approx(0.5)

    def test_zero_when_nothing_relevant_found(self) -> None:
        assert mrr(["x", "y"], {"a": 2}) == pytest.approx(0.0)
        assert mrr([], {"a": 2}) == pytest.approx(0.0)
