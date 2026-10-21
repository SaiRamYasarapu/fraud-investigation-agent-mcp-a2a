from src.agents.detection_agent import DetectionAgent
from src.agents.orchestrator import build_graph
from src.llm.bedrock_client import POLICY_HEADER, RuleBasedFraudReasoner
from src.vector_store.retriever import PolicyChunk


def test_offline_reasoner_ignores_policy_text():
    prompt = (
        "Transaction under review:\nRoutine grocery purchase.\n\n"
        f"{POLICY_HEADER}\n- [account_takeover_playbook.md > Signs] account takeover, money mule\n"
    )
    assert '"CLEAR"' in RuleBasedFraudReasoner().reason("", prompt)


def test_prompt_includes_policy_sections():
    policy = PolicyChunk(chunk_id="p1", source="wire_transfer_review_policy.md",
                         section="When a wire must be held", text="Hold new-beneficiary wires.")
    prompt = DetectionAgent._build_prompt(None, "a wire", [], {}, [policy])
    assert POLICY_HEADER in prompt
    assert "[wire_transfer_review_policy.md > When a wire must be held]" in prompt


def test_pipeline_returns_policies_and_keeps_routine_verdict():
    result = build_graph().invoke({
        "transaction_description": (
            "Customer ACC-3002 made a standard in-network purchase consistent "
            "with their normal spending pattern, no new device or location."
        ),
        "account_id": "ACC-3002",
        "detection_output": None, "verdict": None, "rationale": None, "final_action": None,
    })
    assert len(result["detection_output"]["retrieved_policies"]) == 2
    assert result["verdict"] == "CLEAR"


def test_parse_verdict_reads_json_wrapped_in_text():
    from src.agents.orchestrator import parse_verdict
    raw = 'Here is my assessment:\n```json\n{"verdict": "ESCALATE_HUMAN_REVIEW", "rationale": "Reset then wire."}\n```'
    assert parse_verdict(raw) == {"verdict": "ESCALATE_HUMAN_REVIEW", "rationale": "Reset then wire."}


def test_parse_verdict_routes_unreadable_reply_for_review():
    from src.agents.orchestrator import parse_verdict
    assert parse_verdict("I think this looks fine.")["verdict"] == "FLAG_FOR_REVIEW"
    assert parse_verdict('{"verdict": "MAYBE"}')["verdict"] == "FLAG_FOR_REVIEW"
