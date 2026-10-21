"""
Hybrid search: keyword search (BM25) run next to vector search, with the two
ranked lists merged by Reciprocal Rank Fusion (RRF).

Vector search matches meaning, so it finds a case even when it is worded
differently, but it blurs exact words such as case IDs, names and rare terms.
BM25 scores exact words, so it catches those. RRF merges the two lists using
only each item's position, so the two kinds of score never have to be on the
same scale.

The vector lane is whatever get_case_retriever() / get_policy_retriever()
return, so hybrid search works with both RETRIEVAL_BACKEND=memory and
RETRIEVAL_BACKEND=pgvector.

Usage:
    python -m src.vector_store.hybrid
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from src.vector_store.retriever import (
    DATA_PATH,
    POLICY_DIR,
    PolicyChunk,
    RetrievedCase,
    chunk_policy,
    get_case_retriever,
    get_policy_retriever,
)

# Lowercase words and numbers; a hyphen inside a token keeps IDs like fc-1002 whole.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
CANDIDATES = 20  # how many results each lane hands to the fusion step
RRF_K = 60  # standard RRF constant; larger values flatten the rank differences


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25:
    """BM25 ranks documents by keyword match, weighing three things: rare
    words count more than common ones, repeated words help but flatten fast
    (k1 controls that), and it adjusts for document length so long docs don't
    win on size alone (b controls that). Written from scratch, no imported library used."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_counts = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(tc.values()) for tc in self.term_counts]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        doc_freq = Counter(term for tc in self.term_counts for term in tc)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        results = []
        for counts, length in zip(self.term_counts, self.lengths):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            results.append(score)
        return results

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """(document index, score) for the k best documents. Documents that
        share no word with the query are left out."""
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order[:k] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Merge ranked lists of IDs. An item at position r (starting at 1) in a list
    earns 1 / (k + r) from that list; an item found by both lanes adds both."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


class HybridCaseRetriever:
    """Same retrieve() as the other case retrievers, ranked by vector + BM25."""

    def __init__(self, vector_retriever=None, data_path: Path = DATA_PATH):
        with open(data_path) as f:
            self.cases = json.load(f)
        self.by_id = {c["case_id"]: c for c in self.cases}
        # The case ID is part of the keyword text so an analyst can search by it.
        self.bm25 = BM25(
            [f"{c['case_id']} {c['typology']} {c['pattern']} {c['risk_notes']}" for c in self.cases]
        )
        self.vector = vector_retriever or get_case_retriever()

    def rankings(self, query: str) -> tuple[list[str], list[str]]:
        """Case IDs from each lane, best first."""
        # A score of 0 means no similarity at all (possible with offline TF-IDF), so
        # those results are not counted as votes.
        vector_ids = [r.case_id for r in self.vector.retrieve(query, top_k=CANDIDATES) if r.score > 0]
        keyword_ids = [self.cases[i]["case_id"] for i, _ in self.bm25.top(query, CANDIDATES)]
        return vector_ids, keyword_ids

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedCase]:
        results = []
        for case_id, score in reciprocal_rank_fusion(list(self.rankings(query)))[:top_k]:
            c = self.by_id[case_id]
            results.append(
                RetrievedCase(
                    case_id=c["case_id"],
                    typology=c["typology"],
                    pattern=c["pattern"],
                    risk_notes=c["risk_notes"],
                    resolution=c["resolution"],
                    score=round(score, 4),
                )
            )
        return results


class HybridPolicyRetriever:
    """Same retrieve() as the other policy retrievers, ranked by vector + BM25."""

    def __init__(self, vector_retriever=None, policy_dir: Path = POLICY_DIR):
        self.chunks: list[PolicyChunk] = []
        for path in sorted(Path(policy_dir).glob("*.md")):
            self.chunks.extend(chunk_policy(path))
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25([c.text for c in self.chunks])
        self.vector = vector_retriever or get_policy_retriever()

    def rankings(self, query: str) -> tuple[list[str], list[str]]:
        vector_ids = [p.chunk_id for p in self.vector.retrieve(query, top_k=CANDIDATES) if p.score > 0]
        keyword_ids = [self.chunks[i].chunk_id for i, _ in self.bm25.top(query, CANDIDATES)]
        return vector_ids, keyword_ids

    def retrieve(self, query: str, top_k: int = 3) -> list[PolicyChunk]:
        results = []
        for chunk_id, score in reciprocal_rank_fusion(list(self.rankings(query)))[:top_k]:
            c = self.by_id[chunk_id]
            results.append(PolicyChunk(c.chunk_id, c.source, c.section, c.text, round(score, 4)))
        return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    cases = HybridCaseRetriever()
    for question in (
        "customer reset their password then immediately wired money overseas",
        "FC-1043",
    ):
        vector_ids, keyword_ids = cases.rankings(question)
        print(f"Query: {question}")
        print(f"  vector : {vector_ids[:5]}")
        print(f"  keyword: {keyword_ids[:5]}")
        for r in cases.retrieve(question, top_k=5):
            print(f"  hybrid [{r.score}] {r.case_id} ({r.typology})")
        print()
