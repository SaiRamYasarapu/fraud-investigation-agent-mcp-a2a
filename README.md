# Fraud & Risk Investigation Agent (MCP + A2A + RAG)

I built this to learn how production agent systems fit together end to end:
retrieval, tool calling, agent hand-offs and evaluation. A Detection Agent
retrieves similar historical cases and policy sections through RAG, then hands
off over the A2A protocol to an Investigation Agent that gathers account
evidence through MCP tools. A LangGraph graph routes each case to clear, flag,
escalate or compliance hold. It runs fully offline by default, and on Amazon
Bedrock when switched to live mode.

All data is synthetic, written by me for this project (see data/README.md).

## Architecture

    transaction ──> LangGraph orchestrator
                       │
                       ▼
                 Detection Agent ──── RAG: hybrid search (vector + BM25, RRF)
                       │                    └─ optional Bedrock reranking
                       │ A2A task
                       ▼
               Investigation Agent ── MCP client ──> MCP server (stdio)
                                                     ├─ account_lookup
                                                     ├─ kyc_check
                                                     └─ sanctions_screen
                       │
                       ▼
            verdict + rationale (live Claude on Bedrock, or rule-based offline)
                       │
                       ▼
           clear / flag / escalate / compliance hold

## Retrieval stack

| Layer | Offline (default) | Live |
|---|---|---|
| Embeddings | Local TF-IDF | Amazon Titan Text Embeddings V2 (cached on disk) |
| Vector store | In-memory | PostgreSQL + pgvector (HNSW), idempotent ingestion |
| Keyword search | BM25 | BM25 |
| Fusion | Reciprocal Rank Fusion (k=60, 20 candidates per lane) | same |
| Reranking | Keeps hybrid order | Amazon Bedrock Rerank (`amazon.rerank-v1:0`) |

## Retrieval evaluation

I didn't want to judge retrieval by eye, so I wrote 63 labelled queries
(data/eval_queries.json) in four kinds and scored every setup on the top 5
results with Recall@5, MRR and nDCG@5:

- paraphrase (37): a case described in different words
- exact case ID (5): for example "FC-1043"
- legitimate look-alike (9): harmless activity that resembles fraud
- hard paraphrase (12): almost no words in common with the case text

Offline results (the vector lane here is the TF-IDF stand-in, not Titan):

| Setup | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| Vector | 0.812 | 0.836 | 0.808 |
| BM25 | 0.902 | 0.919 | 0.891 |
| Hybrid (RRF) | 0.902 | 0.910 | 0.886 |

What I found:
- Vector search found 0 of 5 exact case IDs, while BM25 and hybrid found all
  5. That is the reason I added hybrid search.
- Hard paraphrases are the weak spot for every setup (Recall@5 of 0.43 to
  0.48), because TF-IDF only matches shared words. Closing that gap is what
  Titan embeddings are for, and the live run is my next step.
- Offline, BM25 alone scores as well as hybrid. I kept hybrid because the
  vector lane is expected to matter once Titan replaces TF-IDF. I plan to test
  that, not assume it.
- Every setup misses Q58 (elder exploitation) and Q60 (refund abuse), which
  tells me those two typologies need better case text or more examples.
- Offline, reranking keeps the hybrid order, so its scores equal hybrid. Set
  RERANK_MODE=bedrock to score the live reranker.

Run it:

    python -m src.vector_store.evaluate            # results table
    python -m src.vector_store.evaluate --misses   # plus missed queries per setup

## How to run

    py -V:3.12 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python main.py

`python main.py` runs four scenarios (an account takeover, two sanctions
matches and a routine purchase) fully offline, with no AWS account needed.

### Live mode (Amazon Bedrock)

Create a `.env` file (git-ignored, never committed):

    AWS_PROFILE=fraud-agent
    AWS_REGION=us-west-2
    USE_LIVE_BEDROCK=true
    BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0

Each piece can also be switched on its own:

| Variable | Values | Controls |
|---|---|---|
| USE_LIVE_BEDROCK | true / false | Claude reasoning; default for the two below |
| EMBEDDINGS_MODE | titan / local | Embeddings |
| RERANK_MODE | bedrock / off | Reranking |
| RETRIEVAL_BACKEND | memory / pgvector | Vector store |

### pgvector (optional)

    docker compose up -d
    Get-Content db/schema.sql | docker exec -i fraud_pgvector psql -U fraud -d fraud_db
    python -m src.vector_store.ingest

Then set RETRIEVAL_BACKEND=pgvector. Ingestion is safe to re-run: unchanged
chunks are skipped, changed ones are re-embedded, and deleted ones are removed.

## Tests

    pip install pytest ruff
    pytest
    ruff check .

34 tests cover the retriever, hybrid search, reranking, policy retrieval in the
Detection Agent, live-reply parsing, evaluation metrics and the MCP protocol.

## Project layout

    src/agents/        Detection and Investigation agents, A2A protocol, LangGraph orchestrator, MCP client
    src/mcp_server/    MCP server exposing account_lookup, kyc_check, sanctions_screen
    src/vector_store/  embeddings, retriever, pgvector store, ingestion, hybrid search, reranking, evaluation
    src/llm/           Bedrock client with offline fallback
    data/              synthetic cases, accounts, sanctions list, policies, evaluation queries
    deploy/            Dockerfile and AgentCore Runtime reference config

My design choices are in DECISIONS.md, and the path to production is in
PRODUCTION_SWAP.md.

## What's next

- Test whether the hybrid improvement is statistically real (bootstrap
  confidence intervals and a paired test on per-query scores), not just a
  higher average.
- Run the evaluation live with Titan embeddings and Bedrock reranking.
- Add LLM-as-judge scoring for answer groundedness, checked against my own
  labels on a sample.
