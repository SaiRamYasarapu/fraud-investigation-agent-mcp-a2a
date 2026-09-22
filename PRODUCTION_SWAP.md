# Going from offline demo to live production

This repo is built so each "demo shortcut" lives behind one interface, and
swapping it doesn't touch agent logic. This doc lists exactly what to
change and where.

## 1. Vector store: TF-IDF -> Bedrock Titan Embeddings + Pinecone

Replace `src/vector_store/retriever.py` internals. The `retrieve()` method
signature stays the same, so `detection_agent.py` needs zero changes.

```python
import boto3, json
import pinecone

bedrock = boto3.client("bedrock-runtime")
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("fraud-cases")

def embed(text: str) -> list[float]:
    resp = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text}),
    )
    return json.loads(resp["body"].read())["embedding"]

def retrieve(query: str, top_k: int = 3):
    vec = embed(query)
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return [match["metadata"] for match in results["matches"]]
```

## 2. LLM reasoning: rule-based -> live Bedrock

Already built in - set env vars:

```bash
export USE_LIVE_BEDROCK=true
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

`src/llm/bedrock_client.py` picks this up automatically; no code changes.

## 3. MCP: local stdio subprocess -> AgentCore Gateway

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

## 4. A2A: in-process registry -> real agent services

`src/agents/a2a_protocol.py`'s `A2AServer`/`A2AClient` currently resolve
in-process. To run Detection and Investigation as genuinely separate
services, replace `A2AServer.resolve` + direct handler call with an HTTP
POST to the other agent's AgentCore Runtime endpoint using the A2A
JSON-RPC task schema, and run each agent as its own deployable (own
Dockerfile / own AgentCore Runtime entry).

## 5. Deploy

```bash
agentcore configure --entrypoint main.py --name fraud-risk-investigation
agentcore launch
```

See `deploy/agentcore_runtime_config.yaml` for the full mapping of Runtime,
Memory, Gateway, and Identity to this codebase.
