"""
MCP server for the Investigation Agent.

This is where we build the connection between the Investigation Agent and our
three tools: account lookup, KYC check, and sanctions screen. It runs locally
over stdio, so it can be exercised with any MCP client, including the MCP CLI
inspector, without any cloud dependency. But in production, Amazon Bedrock
AgentCore Gateway would front a server like this, converting it into a managed,
authenticated MCP endpoint.
"""
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

with open(DATA_DIR / "accounts.json") as f:
    ACCOUNTS = json.load(f)

with open(DATA_DIR / "sanctions_list.json") as f:
    SANCTIONS = json.load(f)

mcp = FastMCP("fraud-investigation-tools")


@mcp.tool()
def account_lookup(account_id: str) -> dict:
    """Look up an account's profile: open date, home country, average
    monthly transaction volume, and whether a password reset happened
    recently. Use this to establish baseline behavior for an account."""
    account = ACCOUNTS.get(account_id)
    if not account:
        return {"error": f"No account found for id {account_id}"}
    return account


@mcp.tool()
def kyc_check(account_id: str) -> dict:
    """Return the KYC (Know Your Customer) verification status and risk
    tier for an account. Use this before recommending any account action."""
    account = ACCOUNTS.get(account_id)
    if not account:
        return {"error": f"No account found for id {account_id}"}
    return {
        "account_id": account_id,
        "kyc_status": account["kyc_status"],
        "risk_tier": account["risk_tier"],
    }


@mcp.tool()
def sanctions_screen(full_name: str) -> dict:
    """Screen a person or entity name against the sanctions watchlist.
    Returns whether there is a match and which list it came from. Any
    positive match requires an immediate compliance hold."""
    matches = [
        entry
        for entry in SANCTIONS
        if entry["name"].lower() == full_name.strip().lower()
    ]
    return {
        "queried_name": full_name,
        "match_found": len(matches) > 0,
        "matches": matches,
    }


if __name__ == "__main__":
    # Runs the server over stdio, the transport an MCP client connects to
    # as a subprocess.
    mcp.run(transport="stdio")
