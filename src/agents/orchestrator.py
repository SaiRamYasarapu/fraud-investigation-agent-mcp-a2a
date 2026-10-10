"""
LangGraph supervisor for the Fraud & Risk Investigation Agent system.

This is the orchestration layer: it decides the control flow (detect ->
decide route -> possibly escalate) as an explicit state graph rather than
hard-coded if/else in main.py. In production this graph is what runs inside
Amazon Bedrock AgentCore Runtime, with AgentCore Memory persisting `state`
across turns of a multi-turn investigation.
"""
import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.detection_agent import DetectionAgent
from src.agents.investigation_agent import InvestigationAgent


class InvestigationState(TypedDict):
    transaction_description: str
    account_id: str
    detection_output: dict | None
    verdict: str | None
    rationale: str | None
    final_action: str | None


def build_graph():
    # Instantiating both agents registers them with the A2A registry, so by
    # the time the graph runs, DetectionAgent can already reach
    # InvestigationAgent by id.
    detection_agent = DetectionAgent()
    investigation_agent = InvestigationAgent()  # noqa: F841 (registers itself via A2A)

    def run_detection(state: InvestigationState) -> InvestigationState:
        result = detection_agent.analyze(
            state["transaction_description"], state["account_id"]
        )
        parsed = json.loads(result["verdict_raw"])
        return {
            **state,
            "detection_output": result,
            "verdict": parsed["verdict"],
            "rationale": parsed["rationale"],
        }

    def route_decision(state: InvestigationState) -> InvestigationState:
        verdict = state["verdict"]
        if verdict == "ESCALATE_COMPLIANCE_HOLD":
            action = "HOLD_TRANSACTION_AND_NOTIFY_COMPLIANCE"
        elif verdict == "ESCALATE_HUMAN_REVIEW":
            action = "ROUTE_TO_HUMAN_FRAUD_ANALYST"
        elif verdict == "FLAG_FOR_REVIEW":
            action = "QUEUE_FOR_ANALYST_REVIEW"
        else:
            action = "CLEAR_WITH_STANDARD_MONITORING"
        return {**state, "final_action": action}

    graph = StateGraph(InvestigationState)
    graph.add_node("detect", run_detection)
    graph.add_node("route", route_decision)
    graph.set_entry_point("detect")
    graph.add_edge("detect", "route")
    graph.add_edge("route", END)

    return graph.compile()
