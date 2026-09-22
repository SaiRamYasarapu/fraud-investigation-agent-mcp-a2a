"""
Sanity check that the MCP server speaks the actual protocol: spins it up
as a subprocess over stdio, connects an MCP ClientSession, lists the
tools, and calls one.
"""
import asyncio
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


async def main():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools exposed over MCP:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description.splitlines()[0]}")

            result = await session.call_tool(
                "account_lookup", arguments={"account_id": "ACC-3003"}
            )
            print("\nCalled account_lookup via MCP:")
            print(" ", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
