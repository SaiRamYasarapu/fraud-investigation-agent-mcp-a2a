"""
Vector store / RAG layer.

Uses TF-IDF + cosine similarity instead of a neural embedding model so it can
run offline with no API keys or network calls. In production, I would be using
Amazon Titan Text Embeddings via Bedrock, and would store the vectors in Pinecone
or OpenSearch - see PRODUCTION_SWAP.md. The retrieve() interface is the integration
point. Agents depend only on this, not on the internals, so swapping the retriever
wouldn't require any agent code to be updated.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "fraud_cases.json"


@dataclass
class RetrievedCase:
    case_id: str
    typology: str
    pattern: str
    risk_notes: str
    resolution: str
    score: float


class FraudCaseRetriever:
    """This class converts the existing fraud case patterns into vectors, and
    also converts the user-given description into a vector. It compares the
    user-given description with the already-embedded vectors, and returns the
    top three similar fraud cases."""

    def __init__(self, data_path: Path = DATA_PATH):
        with open(data_path) as f:
            self.cases = json.load(f)

        # Build the "index": one document string per case, embedded together
        # so the vocabulary is shared across all cases.
        self._corpus = [
            f"{c['typology']} {c['pattern']} {c['risk_notes']}" for c in self.cases
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self.vectorizer.fit_transform(self._corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedCase]:
        """Return the top_k most similar historical fraud cases to `query`."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(
            zip(self.cases, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            RetrievedCase(
                case_id=c["case_id"],
                typology=c["typology"],
                pattern=c["pattern"],
                risk_notes=c["risk_notes"],
                resolution=c["resolution"],
                score=round(float(score), 4),
            )
            for c, score in ranked[:top_k]
        ]


if __name__ == "__main__":
    retriever = FraudCaseRetriever()
    results = retriever.retrieve(
        "customer reset their password then immediately wired money overseas"
    )
    for r in results:
        print(f"[{r.score}] {r.case_id} ({r.typology}) -> {r.pattern}")
