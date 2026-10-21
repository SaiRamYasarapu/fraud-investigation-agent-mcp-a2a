"""
Reranking: a second, more careful pass over the top hybrid results.

Hybrid search (hybrid.py) is fast but shallow: it scores words and vectors
separately, so a harmless look-alike such as FC-1054 ("customer reset their
password after forgetting it, then paid regular bills") can rank as high as a
real account takeover. A reranker reads the query and each candidate together
and scores how well that candidate actually answers the query, so it can tell
the two apart.

The flow is: hybrid search returns about 20 candidates, the reranker re-scores
them, and the best top_k are returned in the same RetrievedCase / PolicyChunk
shape as every other retriever, so the agents do not change.

Rerankers:
- BedrockReranker calls the Amazon Bedrock Rerank API (model amazon.rerank-v1:0
  by default). If the call fails, it keeps the hybrid order instead of
  crashing the investigation.
- KeepOrderReranker keeps the hybrid order. It is used offline, so tests and
  demos run without AWS.

RERANK_MODE (bedrock / off) picks one. If it is not set, it follows
USE_LIVE_BEDROCK, the same way embeddings.py does.

Usage:
    python -m src.vector_store.rerank
"""
from __future__ import annotations

import logging
import os
from dataclasses import replace

from src.vector_store.hybrid import (
    CANDIDATES,
    HybridCaseRetriever,
    HybridPolicyRetriever,
)
from src.vector_store.retriever import PolicyChunk, RetrievedCase

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "amazon.rerank-v1:0"


class KeepOrderReranker:
    """Offline stand-in: keeps the incoming order and gives a falling score."""

    def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        n = min(top_n, len(texts))
        return [(i, round(1.0 - i / max(len(texts), 1), 4)) for i in range(n)]


class BedrockReranker:
    """Scores (query, text) pairs with a Bedrock rerank model.

    `client` can be passed in (tests use a fake); otherwise a
    bedrock-agent-runtime client is created on first use."""

    def __init__(self, model_id: str | None = None, region: str | None = None, client=None):
        self.model_id = model_id or os.environ.get("RERANK_MODEL_ID", DEFAULT_RERANK_MODEL)
        self.region = region or os.environ.get("AWS_REGION", "us-west-2")
        self._client = client
        self.fallback = KeepOrderReranker()

    @property
    def client(self):
        if self._client is None:
            import boto3  # imported lazily so offline runs never need it

            self._client = boto3.client("bedrock-agent-runtime", region_name=self.region)
        return self._client

    def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        if not texts:
            return []
        top_n = min(top_n, len(texts))
        try:
            response = self.client.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": query}}],
                sources=[
                    {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": t}}}
                    for t in texts
                ],
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "numberOfResults": top_n,
                        "modelConfiguration": {
                            "modelArn": f"arn:aws:bedrock:{self.region}::foundation-model/{self.model_id}"
                        },
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001 - any AWS failure (network, permissions, throttling) keeps the hybrid order
            logger.warning("Bedrock rerank failed, keeping hybrid order: %s", exc)
            return self.fallback.rerank(query, texts, top_n)
        ranked = sorted(response["results"], key=lambda r: r["relevanceScore"], reverse=True)
        return [(r["index"], float(r["relevanceScore"])) for r in ranked[:top_n]]


def rerank_mode() -> str:
    mode = os.environ.get("RERANK_MODE", "").strip().lower()
    if mode in ("bedrock", "off"):
        return mode
    live = os.environ.get("USE_LIVE_BEDROCK", "false").lower() == "true"
    return "bedrock" if live else "off"


def get_reranker() -> BedrockReranker | KeepOrderReranker:
    return BedrockReranker() if rerank_mode() == "bedrock" else KeepOrderReranker()


class RerankedCaseRetriever:
    """Same retrieve() as the other case retrievers: hybrid candidates, then rerank."""

    def __init__(self, base=None, reranker=None, candidates: int = CANDIDATES):
        self.base = base or HybridCaseRetriever()
        self.reranker = reranker or get_reranker()
        self.candidates = candidates

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedCase]:
        pool = self.base.retrieve(query, top_k=self.candidates)
        texts = [f"{c.typology}: {c.pattern} {c.risk_notes}" for c in pool]
        return [replace(pool[i], score=round(s, 4)) for i, s in self.reranker.rerank(query, texts, top_k)]


class RerankedPolicyRetriever:
    """Same retrieve() as the other policy retrievers: hybrid candidates, then rerank."""

    def __init__(self, base=None, reranker=None, candidates: int = CANDIDATES):
        self.base = base or HybridPolicyRetriever()
        self.reranker = reranker or get_reranker()
        self.candidates = candidates

    def retrieve(self, query: str, top_k: int = 3) -> list[PolicyChunk]:
        pool = self.base.retrieve(query, top_k=self.candidates)
        return [replace(pool[i], score=round(s, 4))
                for i, s in self.reranker.rerank(query, [p.text for p in pool], top_k)]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    question = "customer reset their password then immediately wired money overseas"
    hybrid = HybridCaseRetriever()
    print(f"Query: {question}  (rerank mode: {rerank_mode()})")
    print("  hybrid only :", [r.case_id for r in hybrid.retrieve(question, top_k=5)])
    reranked = RerankedCaseRetriever(base=hybrid)
    for r in reranked.retrieve(question, top_k=5):
        print(f"  reranked [{r.score}] {r.case_id} ({r.typology})")
