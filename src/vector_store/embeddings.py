"""
Embedding layer for retrieval.

Two embedders share one small interface (fit + embed), so the retrievers never
need to know which one is running:

- LocalTfidfEmbedder: runs fully offline with no AWS calls. It turns text into
  TF-IDF vectors fitted on the corpus being searched. Deterministic, free.
- TitanEmbedder: calls Amazon Titan Text Embeddings V2 on Bedrock, which
  produces meaning-based vectors. Results are cached on disk, keyed by a
  hash of the model and the text, so each piece of text is only embedded once
  and is re-embedded automatically when the text changes.

Which one runs is decided by environment variables:
- EMBEDDINGS_MODE=titan or EMBEDDINGS_MODE=local picks one explicitly.
- If EMBEDDINGS_MODE is not set, it follows USE_LIVE_BEDROCK
  (true -> Titan, false -> local), matching src/llm/bedrock_client.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "embeddings"
TITAN_MODEL_ID = os.environ.get("TITAN_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
TITAN_DIMENSIONS = 1024


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LocalTfidfEmbedder:
    """Offline embedder. Must be fitted on the corpus before use."""

    name = "local-tfidf"

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer.fit(corpus)

    def embed(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(self._vectorizer.transform(texts).toarray())


class TitanEmbedder:
    """Amazon Titan Text Embeddings V2 on Bedrock, with an on-disk cache."""

    name = "titan-v2"

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        # embed_fn lets tests swap in a fake so they never call AWS.
        self._embed_fn = embed_fn or self._call_bedrock
        self._client = None
        self._cache_path = Path(cache_dir) / f"{TITAN_MODEL_ID.replace(':', '_')}.json"
        self._cache: dict[str, list[float]] = self._load_cache()
        self.calls_made = 0

    def fit(self, corpus: list[str]) -> None:
        """Titan is pre-trained, so there is nothing to fit."""

    def embed(self, texts: list[str]) -> np.ndarray:
        changed = False
        vectors = []
        for text in texts:
            key = self._key(text)
            if key not in self._cache:
                self._cache[key] = self._embed_fn(text)
                self.calls_made += 1
                changed = True
            vectors.append(self._cache[key])
        if changed:
            self._save_cache()
        return l2_normalize(np.asarray(vectors, dtype=float))

    # -- internals -------------------------------------------------------
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(f"{TITAN_MODEL_ID}\n{text}".encode()).hexdigest()

    def _call_bedrock(self, text: str) -> list[float]:
        if self._client is None:
            import boto3  # imported lazily so offline runs never need AWS

            session = boto3.Session(
                profile_name=os.environ.get("AWS_PROFILE"),
                region_name=os.environ.get("AWS_REGION", "us-west-2"),
            )
            self._client = session.client("bedrock-runtime")
        response = self._client.invoke_model(
            modelId=TITAN_MODEL_ID,
            body=json.dumps(
                {"inputText": text, "dimensions": TITAN_DIMENSIONS, "normalize": True}
            ),
        )
        return json.loads(response["body"].read())["embedding"]

    def _load_cache(self) -> dict[str, list[float]]:
        if self._cache_path.exists():
            with open(self._cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        tmp.replace(self._cache_path)


def embeddings_mode() -> str:
    mode = os.environ.get("EMBEDDINGS_MODE", "").strip().lower()
    if mode in ("titan", "local"):
        return mode
    live = os.environ.get("USE_LIVE_BEDROCK", "false").lower() == "true"
    return "titan" if live else "local"


def get_embedder() -> LocalTfidfEmbedder | TitanEmbedder:
    return TitanEmbedder() if embeddings_mode() == "titan" else LocalTfidfEmbedder()
