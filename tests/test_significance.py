"""
Tests for the significance checks: bootstrap ranges and the paired test.
Small hand-made score lists, so no retriever or AWS call is needed.
"""
import pytest

from src.vector_store.evaluate import evaluate
from src.vector_store.significance import bootstrap_ci, paired_test


def test_bootstrap_range_contains_the_average():
    ci = bootstrap_ci([1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0, 1.0])
    assert ci.mean == pytest.approx(0.6875)
    assert ci.low <= ci.mean <= ci.high


def test_same_seed_gives_the_same_result():
    scores = [0.2, 0.9, 0.4, 1.0, 0.0, 0.7]
    assert bootstrap_ci(scores, seed=7) == bootstrap_ci(scores, seed=7)
    assert paired_test(scores, scores[::-1], seed=7) == paired_test(scores, scores[::-1], seed=7)


def test_identical_setups_show_no_difference():
    scores = [0.5, 1.0, 0.0, 0.8]
    result = paired_test(scores, scores)
    assert (result.mean_diff, result.low, result.high) == (0.0, 0.0, 0.0)
    assert result.p_value == 1.0
    assert not result.significant


def test_clear_winner_is_significant():
    better = [1.0] * 20
    worse = [0.0] * 20
    result = paired_test(better, worse)
    assert result.mean_diff == 1.0
    assert result.low > 0
    assert result.p_value < 0.05 and result.significant


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        paired_test([1.0, 0.0], [1.0])


class FixedSetup:
    def __init__(self, answers):
        self.answers = answers

    def ranked_ids(self, query, k):
        return self.answers[query][:k]


def test_evaluate_keeps_per_query_scores_in_query_order():
    queries = [
        {"id": "Q1", "kind": "paraphrase", "query": "q1", "relevant": ["A"]},
        {"id": "Q2", "kind": "exact_id", "query": "q2", "relevant": ["B"]},
    ]
    report = evaluate({"s": FixedSetup({"q1": ["A"], "q2": ["X"]})}, queries)
    rows = report["s"]["per_query"]
    assert [r["id"] for r in rows] == ["Q1", "Q2"]
    assert [r["ndcg"] for r in rows] == [1.0, 0.0]
