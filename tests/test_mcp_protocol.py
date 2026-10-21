"""
Sanity check that the MCP server speaks the actual protocol: spins it up
as a subprocess over stdio, connects an MCP ClientSession, lists the
tools, and calls one.
"""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PARAMS = StdioServerParameters(
    # sys.executable rather than the literal string "python3": Windows only
    # has "python" on PATH, not "python3", so this must resolve to whatever
    # interpreter is actually running this script.
    command=sys.executable,
    args=["src/mcp_server/server.py"],
)

EXPECTED_TOOLS = {"account_lookup", "kyc_check", "sanctions_screen"}


async def run_session():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool(
                "account_lookup", arguments={"account_id": "ACC-3003"}
            )
            return {t.name for t in tools.tools}, json.loads(result.content[0].text)


def test_mcp_server_lists_tools_and_answers_a_call():
    tool_names, account = asyncio.run(run_session())
    assert tool_names == EXPECTED_TOOLS
    assert account["customer_name"] == "R. Okafor"


if __name__ == "__main__":
    names, account = asyncio.run(run_session())
    print("Tools exposed over MCP:", ", ".join(sorted(names)))
    print("account_lookup(ACC-3003):", account)
