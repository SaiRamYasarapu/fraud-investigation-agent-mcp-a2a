"""
End-to-end demo runner.

Simulates three transactions being investigated by the two-agent system:
  1. A clear account-takeover pattern (should escalate to human review)
  2. A sanctions list match (should trigger a compliance hold)
  3. A benign transaction (should clear)

Run: python3 main.py
"""
import json
from src.agents.orchestrator import build_graph

SCENARIOS = [
    {
        "label": "Scenario 1: Possible account takeover",
        "transaction_description": (
            "Customer ACC-3001 had a password reset, and minutes later a wire "
            "transfer was sent to a newly added beneficiary in a high-risk "
            "jurisdiction."
        ),
        "account_id": "ACC-3001",
    },
    {
        "label": "Scenario 2: Sanctions list match",
        "transaction_description": (
            "A transaction from account ACC-3003 was flagged during real-time "
            "sanctions screening for a possible watchlist name match."
        ),
        "account_id": "ACC-3003",
    },
    {
        "label": "Scenario 3: Confirmed sanctions list match",
        "transaction_description": (
            "A transaction from account ACC-3004 was flagged during real-time "
            "sanctions screening for a possible watchlist name match."
        ),
        "account_id": "ACC-3004",
    },
    {
        "label": "Scenario 4: Routine transaction",
        "transaction_description": (
            "Customer ACC-3002 made a standard in-network purchase consistent "
            "with their normal spending pattern, no new device or location."
        ),
        "account_id": "ACC-3002",
    },
]


def main():
    graph = build_graph()

    for scenario in SCENARIOS:
        print("=" * 80)
        print(scenario["label"])
        print("=" * 80)

        result = graph.invoke(
            {
                "transaction_description": scenario["transaction_description"],
                "account_id": scenario["account_id"],
                "detection_output": None,
                "verdict": None,
                "rationale": None,
                "final_action": None,
            }
        )

        print(f"\nVerdict:      {result['verdict']}")
        print(f"Final action: {result['final_action']}")
        print(f"Rationale:    {result['rationale']}")

        top_cases = result["detection_output"]["retrieved_cases"][:2]
        print("\nTop retrieved historical cases (RAG over vector store):")
        for c in top_cases:
            print(f"  [{c['score']}] {c['case_id']} - {c['typology']}")

        evidence = result["detection_output"]["account_evidence"]
        print("\nAccount evidence gathered via A2A -> MCP tool calls:")
        print(" ", json.dumps(evidence, indent=2).replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
