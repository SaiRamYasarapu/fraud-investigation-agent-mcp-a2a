"""
Tests for the retrieval metrics. Hand-worked examples, so the maths is checked
without any retriever or AWS call.
"""
import json
import math

import pytest

from src.vector_store.evaluate import (
    EVAL_PATH,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_relevant_items_in_top_k():
    assert recall_at_k(["A", "B", "C"], {"B", "Z"}, k=5) == 0.5
    assert recall_at_k(["A", "B", "C"], {"C"}, k=2) == 0.0


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["A", "B", "C"], {"C", "B"}) == 0.5
    assert reciprocal_rank(["A", "B"], {"Z"}) == 0.0


def test_ndcg_is_one_for_perfect_order_and_lower_when_pushed_down():
    assert ndcg_at_k(["A", "B"], {"A"}, k=5) == 1.0
    # relevant item at position 2: 1/log2(3) out of an ideal 1/log2(2) = 1
    assert ndcg_at_k(["X", "A"], {"A"}, k=5) == pytest.approx(1 / math.log2(3))


class FixedSetup:
    def __init__(self, answers):
        self.answers = answers

    def ranked_ids(self, query, k):
        return self.answers[query][:k]


def test_evaluate_averages_per_setup_and_lists_misses():
    queries = [
        {"id": "Q1", "kind": "paraphrase", "query": "q1", "relevant": ["A"]},
        {"id": "Q2", "kind": "exact_id", "query": "q2", "relevant": ["B"]},
    ]
    report = evaluate({"s": FixedSetup({"q1": ["A"], "q2": ["X"]})}, queries)
    assert report["s"]["all"]["recall"] == 0.5
    assert report["s"]["exact_id"]["mrr"] == 0.0
    assert report["s"]["misses"] == ["Q2"]


def test_eval_queries_reference_real_cases():
    from src.vector_store.retriever import DATA_PATH

    with open(DATA_PATH) as f:
        case_ids = {c["case_id"] for c in json.load(f)}
    with open(EVAL_PATH) as f:
        queries = json.load(f)
    assert len({q["id"] for q in queries}) == len(queries)
    for q in queries:
        assert q["relevant"] and set(q["relevant"]) <= case_ids, q["id"]
