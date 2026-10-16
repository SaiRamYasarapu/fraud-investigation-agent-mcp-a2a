"""
Tests for reranking. They run offline: the Bedrock client is a fake that
returns a fixed response, so no AWS call is made.
"""
from src.vector_store.embeddings import LocalTfidfEmbedder
from src.vector_store.hybrid import HybridCaseRetriever
from src.vector_store.rerank import (
    BedrockReranker,
    KeepOrderReranker,
    RerankedCaseRetriever,
    get_reranker,
)
from src.vector_store.retriever import FraudCaseRetriever, RetrievedCase


def case(case_id):
    return RetrievedCase(case_id, "typology", f"pattern {case_id}", "notes", "resolution", 0.0)


class FakeBase:
    """A retriever that always returns cases A, B, C, D in that order."""

    def retrieve(self, query, top_k=3):
        return [case(c) for c in "ABCD"][:top_k]


class ReverseReranker:
    """Puts the last candidate first."""

    def rerank(self, query, texts, top_n):
        order = list(reversed(range(len(texts))))[:top_n]
        return [(i, 1.0 - rank / 10) for rank, i in enumerate(order)]


class FakeBedrockClient:
    def __init__(self, results=None, error=None):
        self.results, self.error, self.calls = results, error, []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"results": self.results}


def test_keep_order_reranker_keeps_the_hybrid_order():
    assert [i for i, _ in KeepOrderReranker().rerank("q", ["a", "b", "c"], 2)] == [0, 1]


def test_reranked_retriever_uses_the_reranker_order_and_top_k():
    results = RerankedCaseRetriever(base=FakeBase(), reranker=ReverseReranker()).retrieve("q", top_k=2)
    assert [r.case_id for r in results] == ["D", "C"]
    assert results[0].score > results[1].score


def test_bedrock_reranker_sorts_by_relevance_score():
    client = FakeBedrockClient(results=[{"index": 0, "relevanceScore": 0.1}, {"index": 2, "relevanceScore": 0.9},
                                        {"index": 1, "relevanceScore": 0.5}])
    ranked = BedrockReranker(region="us-west-2", client=client).rerank("q", ["a", "b", "c"], 2)
    assert ranked == [(2, 0.9), (1, 0.5)]
    sent = client.calls[0]
    assert sent["queries"][0]["textQuery"]["text"] == "q"
    assert len(sent["sources"]) == 3
    assert "amazon.rerank-v1:0" in sent["rerankingConfiguration"]["bedrockRerankingConfiguration"][
        "modelConfiguration"]["modelArn"]


def test_bedrock_failure_falls_back_to_the_hybrid_order():
    client = FakeBedrockClient(error=RuntimeError("AccessDenied"))
    ranked = BedrockReranker(region="us-west-2", client=client).rerank("q", ["a", "b", "c"], 2)
    assert [i for i, _ in ranked] == [0, 1]


def test_reranker_is_off_by_default(monkeypatch):
    monkeypatch.delenv("RERANK_MODE", raising=False)
    monkeypatch.delenv("USE_LIVE_BEDROCK", raising=False)
    assert isinstance(get_reranker(), KeepOrderReranker)


def test_full_offline_pipeline_finds_an_exact_case_id():
    hybrid = HybridCaseRetriever(vector_retriever=FraudCaseRetriever(embedder=LocalTfidfEmbedder()))
    top = RerankedCaseRetriever(base=hybrid, reranker=KeepOrderReranker()).retrieve("FC-1043", top_k=1)[0]
    assert top.case_id == "FC-1043"
