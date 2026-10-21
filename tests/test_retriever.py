"""
Tests for the retrieval layer. All of these run offline: the local embedder
needs no AWS, and the Titan cache test uses a fake embedding function instead
of calling Bedrock.
"""
import hashlib
import os

import numpy as np
import pytest

from src.vector_store.embeddings import LocalTfidfEmbedder, TitanEmbedder
from src.vector_store.retriever import (
    FraudCaseRetriever,
    PolicyRetriever,
    get_case_retriever,
    get_policy_retriever,
)


def local_cases():
    return FraudCaseRetriever(embedder=LocalTfidfEmbedder())


def test_returns_top_k_cases_with_expected_fields():
    results = local_cases().retrieve("password reset then wire to new payee", top_k=3)
    assert len(results) == 3
    assert {"case_id", "typology", "pattern", "risk_notes", "resolution", "score"} <= set(
        results[0].__dict__
    )
    assert results[0].score >= results[1].score >= results[2].score


def test_account_takeover_query_finds_takeover_case_first():
    top = local_cases().retrieve(
        "password reset followed minutes later by a wire to a newly added beneficiary"
    )[0]
    assert top.typology == "account_takeover"


def test_sanctions_query_finds_sanctions_case_first():
    top = local_cases().retrieve("possible watchlist name match during sanctions screening")[0]
    assert top.typology == "sanctions_hit"


def test_policies_are_chunked_by_section():
    retriever = PolicyRetriever(embedder=LocalTfidfEmbedder())
    assert len(retriever.chunks) > 8  # more chunks than files
    assert all(c.section for c in retriever.chunks)


def test_policy_search_finds_sanctions_procedure():
    top = PolicyRetriever(embedder=LocalTfidfEmbedder()).retrieve(
        "partial name match on sanctions list with different date of birth"
    )[0]
    assert top.source == "sanctions_screening_procedure.md"


def fake_embedding(text: str) -> list[float]:
    """Deterministic stand-in for Titan, so tests never call AWS."""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return list(np.random.default_rng(seed).random(16))


def test_titan_cache_embeds_each_text_only_once(tmp_path):
    first = TitanEmbedder(cache_dir=tmp_path, embed_fn=fake_embedding)
    first.embed(["alpha", "beta"])
    assert first.calls_made == 2

    # On a new run the embedder reads the cache from disk: no new calls for known text,
    # one call for the new text.
    second = TitanEmbedder(cache_dir=tmp_path, embed_fn=fake_embedding)
    second.embed(["alpha", "beta", "gamma"])
    assert second.calls_made == 1


def test_titan_vectors_are_normalized(tmp_path):
    vectors = TitanEmbedder(cache_dir=tmp_path, embed_fn=fake_embedding).embed(["x", "y"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


@pytest.mark.skipif(
    os.environ.get("EMBEDDINGS_MODE") != "titan",
    reason="live Titan test: set EMBEDDINGS_MODE=titan and log in to AWS to run it",
)
def test_live_titan_meaning_match():
    retriever = FraudCaseRetriever(embedder=TitanEmbedder())
    top = retriever.retrieve("credentials changed, then funds sent abroad to someone new")[0]
    assert top.typology == "account_takeover"


def test_backend_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_BACKEND", raising=False)
    monkeypatch.setenv("EMBEDDINGS_MODE", "local")
    assert isinstance(get_case_retriever(), FraudCaseRetriever)
    assert isinstance(get_policy_retriever(), PolicyRetriever)


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "elastic")
    with pytest.raises(ValueError):
        get_case_retriever()


def test_pgvector_retriever_wraps_the_store():
    from src.vector_store.pg_store import PgCaseRetriever

    class FakeStore:
        def search_cases(self, query, top_k):
            return [("case", query, top_k)]

    assert PgCaseRetriever(store=FakeStore()).retrieve("wire", top_k=2) == [("case", "wire", 2)]
