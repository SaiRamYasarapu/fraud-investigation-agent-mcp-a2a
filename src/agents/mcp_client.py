"""
MCP client wrapper used by the Investigation Agent.

This literally creates the connection between our Investigation Agent and the tools present in server.py,
and lets our Investigation Agent trigger those tools. In this file, we created a method, call_tool, and by
using this method, our Investigation Agent can connect to server.py and trigger the tools, without our agent
code needing to manage asyncio.

In production, AgentCore Gateway would sit here instead of a local stdio subprocess - swapping this class's
transport is the only change we need to go from a local MCP server to MCP tools behind AgentCore Gateway.
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "server.py"


class MCPToolClient:
    def __init__(self):
        # sys.executable, not the string "python3" - Windows only has
        # "python", not "python3", so hardcoding "python3" makes the
        # subprocess fail to launch (which MCP reports as "Connection
        # closed" since the pipe closes before a handshake happens).
        self._params = StdioServerParameters(
            command=sys.executable, args=[str(SERVER_SCRIPT)]
        )

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        return asyncio.run(self._call(tool_name, arguments))

    async def _call(self, tool_name: str, arguments: dict) -> dict:
        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                text = result.content[0].text
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw": text}
