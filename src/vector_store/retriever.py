"""
Vector store / RAG layer.

Two retrievers share the same embedding layer (src/vector_store/embeddings.py):

- FraudCaseRetriever: one chunk per historical fraud case in data/fraud_cases.json.
- PolicyRetriever: the policy documents in data/policies, split into chunks at
  their "##" section headings so each chunk covers one topic.

Which embedder runs depends on the environment. Offline (the default), a local
TF-IDF embedder is used, so the project runs with no AWS access. With 
EMBEDDINGS_MODE=titan, or USE_LIVE_BEDROCK=true, Amazon Titan Text Embeddings V2 
on Bedrock is used, so matches are based on meaning rather than shared words. 
Titan vectors are cached on disk under .cache/embeddings.

Similarity is cosine similarity.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.vector_store.embeddings import get_embedder

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_PATH = DATA_DIR / "fraud_cases.json"
POLICY_DIR = DATA_DIR / "policies"
MAX_CHUNK_CHARS = 1200


@dataclass
class RetrievedCase:
    case_id: str
    typology: str
    pattern: str
    risk_notes: str
    resolution: str
    score: float


@dataclass
class PolicyChunk:
    chunk_id: str
    source: str
    section: str
    text: str
    score: float = 0.0


def _rank(query_vec: np.ndarray, matrix: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Cosine similarity of one normalized query vector against normalized rows."""
    scores = matrix @ query_vec
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(i, float(scores[i])) for i in order[:top_k]]


class FraudCaseRetriever:
    """Embeds every historical fraud case once, then embeds the incoming
    description and returns the top_k most similar cases."""

    def __init__(self, data_path: Path = DATA_PATH, embedder=None):
        with open(data_path) as f:
            self.cases = json.load(f)

        # One chunk per case. The typology is included so the case's label
        # contributes to what it is "about".
        self._corpus = [
            f"{c['typology']} {c['pattern']} {c['risk_notes']}" for c in self.cases
        ]
        self.embedder = embedder or get_embedder()
        self.embedder.fit(self._corpus)
        self._matrix = self.embedder.embed(self._corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedCase]:
        """Return the top_k most similar historical fraud cases to `query`."""
        query_vec = self.embedder.embed([query])[0]
        return [
            RetrievedCase(
                case_id=self.cases[i]["case_id"],
                typology=self.cases[i]["typology"],
                pattern=self.cases[i]["pattern"],
                risk_notes=self.cases[i]["risk_notes"],
                resolution=self.cases[i]["resolution"],
                score=round(score, 4),
            )
            for i, score in _rank(query_vec, self._matrix, top_k)
        ]


def chunk_policy(path: Path) -> list[PolicyChunk]:
    """Split one markdown policy into chunks at its "## " headings. Sections
    longer than MAX_CHUNK_CHARS are split further at paragraph breaks."""
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem

    chunks: list[PolicyChunk] = []
    for block in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = block.partition("\n")
        pieces, current = [], ""
        for para in [p.strip() for p in body.split("\n\n") if p.strip()]:
            if current and len(current) + len(para) > MAX_CHUNK_CHARS:
                pieces.append(current)
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            pieces.append(current)
        for piece in pieces:
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{path.stem}#{len(chunks) + 1}",
                    source=path.name,
                    section=heading.strip(),
                    text=f"{title} - {heading.strip()}\n{piece}",
                )
            )
    return chunks


class PolicyRetriever:
    """Searches the policy documents in data/policies."""

    def __init__(self, policy_dir: Path = POLICY_DIR, embedder=None):
        self.chunks: list[PolicyChunk] = []
        for path in sorted(Path(policy_dir).glob("*.md")):
            self.chunks.extend(chunk_policy(path))
        corpus = [c.text for c in self.chunks]
        self.embedder = embedder or get_embedder()
        self.embedder.fit(corpus)
        self._matrix = self.embedder.embed(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[PolicyChunk]:
        query_vec = self.embedder.embed([query])[0]
        results = []
        for i, score in _rank(query_vec, self._matrix, top_k):
            c = self.chunks[i]
            results.append(PolicyChunk(c.chunk_id, c.source, c.section, c.text, round(score, 4)))
        return results


if __name__ == "__main__":
    question = "customer reset their password then immediately wired money overseas"
    for r in FraudCaseRetriever().retrieve(question):
        print(f"[{r.score}] {r.case_id} ({r.typology}) -> {r.pattern}")
    print()
    for p in PolicyRetriever().retrieve(question):
        print(f"[{p.score}] {p.chunk_id} ({p.section})")
