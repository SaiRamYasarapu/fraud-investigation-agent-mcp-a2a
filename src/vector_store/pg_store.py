"""
Search over the pgvector `documents` table.

ingest.py fills the table. This module embeds a query with Titan and asks
Postgres for the nearest chunks by cosine distance. It returns the same
RetrievedCase and PolicyChunk objects as the in-memory retrievers, so the
agents can use either one.

Usage:
    python -m src.vector_store.pg_store
"""
from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

from src.vector_store.embeddings import TitanEmbedder
from src.vector_store.ingest import DEFAULT_DATABASE_URL
from src.vector_store.retriever import PolicyChunk, RetrievedCase

# <=> is pgvector's cosine distance operator. 1 - distance gives the cosine
# similarity, the same score the in-memory retriever reports.
SEARCH_SQL = """
    SELECT chunk_id, source, content, metadata, 1 - (embedding <=> %s) AS score
    FROM documents
    WHERE metadata->>'kind' = %s
    ORDER BY embedding <=> %s
    LIMIT %s
"""


class PgVectorStore:
    """Nearest-neighbour search over the documents table."""

    def __init__(self, database_url: str | None = None, embedder=None):
        self.database_url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
        self.embedder = embedder or TitanEmbedder()

    def _search(self, query: str, kind: str, top_k: int) -> list[tuple]:
        query_vec = self.embedder.embed([query])[0]
        with psycopg.connect(self.database_url) as conn:
            register_vector(conn)
            return conn.execute(SEARCH_SQL, (query_vec, kind, query_vec, top_k)).fetchall()

    def search_cases(self, query: str, top_k: int = 3) -> list[RetrievedCase]:
        return [
            RetrievedCase(
                case_id=meta["case_id"],
                typology=meta["typology"],
                pattern=meta["pattern"],
                risk_notes=meta["risk_notes"],
                resolution=meta["resolution"],
                score=round(float(score), 4),
            )
            for _chunk_id, _source, _content, meta, score in self._search(query, "case", top_k)
        ]

    def search_policies(self, query: str, top_k: int = 3) -> list[PolicyChunk]:
        return [
            PolicyChunk(chunk_id, source, meta["section"], content, round(float(score), 4))
            for chunk_id, source, content, meta, score in self._search(query, "policy", top_k)
        ]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    store = PgVectorStore()
    question = "customer reset their password then immediately wired money overseas"
    for r in store.search_cases(question):
        print(f"[{r.score}] {r.case_id} ({r.typology}) -> {r.pattern}")
    print()
    for p in store.search_policies(question):
        print(f"[{p.score}] {p.chunk_id} ({p.section})")