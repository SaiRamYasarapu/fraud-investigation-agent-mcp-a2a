"""
Ingestion pipeline: loads the fraud cases and policy documents into the
pgvector `documents` table.

For every chunk it stores the text, a few metadata fields, a hash of the text,
and the Titan embedding. Running it again is safe:
- unchanged chunks are skipped (their hash matches what is stored),
- changed or new chunks are embedded and upserted by chunk_id,
- chunks that no longer exist in the data are deleted.

The table has a 1024-dimension vector column, so this always uses Titan.

Usage:
    python -m src.vector_store.ingest
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from src.vector_store.embeddings import TitanEmbedder
from src.vector_store.retriever import DATA_PATH, POLICY_DIR, chunk_policy

DEFAULT_DATABASE_URL = "postgresql://fraud:fraud@localhost:5432/fraud_db"


@dataclass
class Chunk:
    chunk_id: str
    source: str
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def case_chunks(data_path: Path = DATA_PATH) -> list[Chunk]:
    """One chunk per historical fraud case."""
    with open(data_path, encoding="utf-8") as f:
        cases = json.load(f)
    return [
        Chunk(
            chunk_id=c["case_id"],
            source=Path(data_path).name,
            content=f"{c['typology']} {c['pattern']} {c['risk_notes']}",
            metadata={
                "kind": "case",
                "case_id": c["case_id"],
                "typology": c["typology"],
                "pattern": c["pattern"],
                "risk_notes": c["risk_notes"],
                "resolution": c["resolution"],
            },
        )
        for c in cases
    ]


def policy_chunks(policy_dir: Path = POLICY_DIR) -> list[Chunk]:
    """Policy documents split at their "##" headings."""
    chunks = []
    for path in sorted(Path(policy_dir).glob("*.md")):
        for p in chunk_policy(path):
            chunks.append(
                Chunk(
                    chunk_id=p.chunk_id,
                    source=p.source,
                    content=p.text,
                    metadata={"kind": "policy", "section": p.section},
                )
            )
    return chunks


def all_chunks() -> list[Chunk]:
    chunks = case_chunks() + policy_chunks()
    ids = [c.chunk_id for c in chunks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate chunk ids: {sorted(duplicates)}")
    return chunks


def ingest(conn: psycopg.Connection, chunks: list[Chunk], embedder) -> dict[str, int]:
    """Bring the documents table in line with `chunks`. Returns counts."""
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id, content_hash FROM documents")
        stored = dict(cur.fetchall())

        to_embed = [c for c in chunks if stored.get(c.chunk_id) != c.content_hash]
        if to_embed:
            vectors = embedder.embed([c.content for c in to_embed])
            for chunk, vector in zip(to_embed, vectors):
                cur.execute(
                    """
                    INSERT INTO documents (chunk_id, source, content, metadata, content_hash, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        source = EXCLUDED.source,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        content_hash = EXCLUDED.content_hash,
                        embedding = EXCLUDED.embedding
                    """,
                    (chunk.chunk_id, chunk.source, chunk.content, Jsonb(chunk.metadata),
                     chunk.content_hash, vector),
                )

        current_ids = {c.chunk_id for c in chunks}
        removed = [cid for cid in stored if cid not in current_ids]
        if removed:
            cur.execute("DELETE FROM documents WHERE chunk_id = ANY(%s)", (removed,))

    conn.commit()
    return {
        "total": len(chunks),
        "embedded": len(to_embed),
        "unchanged": len(chunks) - len(to_embed),
        "removed": len(removed),
    }


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    chunks = all_chunks()
    embedder = TitanEmbedder()
    with psycopg.connect(url) as conn:
        counts = ingest(conn, chunks, embedder)
    print(
        f"{counts['total']} chunks: {counts['embedded']} embedded, "
        f"{counts['unchanged']} unchanged, {counts['removed']} removed "
        f"({embedder.calls_made} Titan calls, the rest came from the cache)"
    )


if __name__ == "__main__":
    main()
