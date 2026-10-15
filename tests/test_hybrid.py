"""
Tests for hybrid search. They run offline: the vector lane is either the local
TF-IDF retriever or a fake that returns a fixed ranking.
"""
import json

from src.vector_store.embeddings import LocalTfidfEmbedder
from src.vector_store.hybrid import (
    BM25,
    HybridCaseRetriever,
    HybridPolicyRetriever,
    reciprocal_rank_fusion,
)
from src.vector_store.retriever import (
    DATA_PATH,
    FraudCaseRetriever,
    PolicyRetriever,
    RetrievedCase,
)


class FakeVectorRetriever:
    """Returns the given case IDs in order, whatever the query."""

    def __init__(self, case_ids):
        with open(DATA_PATH) as f:
            cases = {c["case_id"]: c for c in json.load(f)}
        self.results = [
            RetrievedCase(cid, cases[cid]["typology"], cases[cid]["pattern"],
                          cases[cid]["risk_notes"], cases[cid]["resolution"], 0.5)
            for cid in case_ids
        ]

    def retrieve(self, query, top_k=3):
        return self.results[:top_k]


def test_bm25_ranks_the_document_with_the_rare_word_first():
    bm25 = BM25(["wire to a new payee", "card used at an emulator", "wire to an old payee"])
    assert bm25.top("emulator login", k=3)[0][0] == 1


def test_bm25_leaves_out_documents_with_no_matching_word():
    bm25 = BM25(["wire transfer", "cash deposit"])
    assert [i for i, _ in bm25.top("wire", k=2)] == [0]


def test_rrf_rewards_items_found_by_both_lanes():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    assert [item for item, _ in fused] == ["a", "c", "b", "d"]


def test_exact_case_id_is_found_by_the_keyword_lane():
    # The vector lane never returns FC-1043, so only BM25 can find it.
    hybrid = HybridCaseRetriever(vector_retriever=FakeVectorRetriever(["FC-1001", "FC-1002", "FC-1003"]))
    assert "FC-1043" in [r.case_id for r in hybrid.retrieve("FC-1043", top_k=3)]


def test_vector_results_with_no_similarity_are_dropped():
    # Offline TF-IDF gives an ID-only query a score of 0 for every case, so the
    # vector lane adds nothing and the exact match comes first.
    hybrid = HybridCaseRetriever(vector_retriever=FraudCaseRetriever(embedder=LocalTfidfEmbedder()))
    assert hybrid.retrieve("FC-1043", top_k=1)[0].case_id == "FC-1043"


def test_paraphrase_is_found_by_the_vector_lane():
    # No shared words with FC-1038, so only the vector lane (here a fake) can find it.
    hybrid = HybridCaseRetriever(vector_retriever=FakeVectorRetriever(["FC-1038", "FC-1040"]))
    results = hybrid.retrieve("grandmother tricked by a caller impersonating her lender", top_k=3)
    assert "FC-1038" in [r.case_id for r in results]


def test_hybrid_policy_search_finds_the_sanctions_procedure():
    hybrid = HybridPolicyRetriever(vector_retriever=PolicyRetriever(embedder=LocalTfidfEmbedder()))
    top = hybrid.retrieve("sanctions watchlist name match escalation", top_k=3)
    assert any("sanctions" in p.source for p in top)
