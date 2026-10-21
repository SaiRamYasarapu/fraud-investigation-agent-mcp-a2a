# Going from offline demo to live production

I built this repo so each "demo shortcut" lives behind one interface, and
swapping it doesn't touch agent logic. This doc lists what I've already
swapped, and exactly what to change and where for the rest.

## 1. Vector store: done locally, next step is managed Postgres

Already built: Titan Text Embeddings V2 with a disk cache, PostgreSQL +
pgvector (HNSW index), and an idempotent ingestion pipeline
(src/vector_store/ingest.py). Switch it on with RETRIEVAL_BACKEND=pgvector.

For production, I would move the database from local Docker to Amazon Aurora
PostgreSQL (pgvector is supported there) and point DATABASE_URL at it. No code
changes: the retriever interface stays the same, so detection_agent.py does
not change.

## 2. LLM reasoning: rule-based -> live Bedrock

Already built in. Set in .env:

```bash
USE_LIVE_BEDROCK=true
AWS_REGION=us-west-2
BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0   # development
# I switch to Sonnet for demos and final testing
```

`src/llm/bedrock_client.py` picks this up automatically; no code changes.

## 3. Reranking: off -> Bedrock Rerank

Already built in (src/vector_store/rerank.py). Set RERANK_MODE=bedrock to
rerank the top 20 hybrid results with amazon.rerank-v1:0. If the call fails,
the hybrid order is kept, so an investigation never stops on a reranking
error.

## 4. MCP: local stdio subprocess -> AgentCore Gateway

`src/agents/mcp_client.py` currently spawns `src/mcp_server/server.py` as a
subprocess. In production, register the same tool functions with AgentCore
Gateway (which converts them into a managed, authenticated MCP endpoint)
and change the transport in `MCPToolClient`:

```python
from mcp.client.streamable_http import streamablehttp_client

class MCPToolClient:
    def __init__(self):
        self._url = os.environ["AGENTCORE_GATEWAY_MCP_URL"]

    async def _call(self, tool_name, arguments):
        async with streamablehttp_client(self._url, headers=self._auth_headers()) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                ...
```

`MCPToolClient` also starts a new MCP server process for every tool call, so one
investigation starts it two or three times. In production, I would keep one
session open and reuse it for all the tool calls in an investigation.

## 5. A2A: in-process registry -> real agent services

`src/agents/a2a_protocol.py`'s `A2AServer`/`A2AClient` currently resolve
in-process. To run Detection and Investigation as genuinely separate
services, replace `A2AServer.resolve` + direct handler call with an HTTP
POST to the other agent's AgentCore Runtime endpoint using the A2A
JSON-RPC task schema, and run each agent as its own deployable (own
Dockerfile / own AgentCore Runtime entry).

## 6. Deploy

```bash
agentcore configure --entrypoint main.py --name fraud-risk-investigation
agentcore launch
```

See `deploy/agentcore_runtime_config.yaml` for the full mapping of Runtime,
Memory, Gateway, and Identity to this codebase.
