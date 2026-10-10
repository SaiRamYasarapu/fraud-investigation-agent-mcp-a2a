"""
LLM client for the agents.

Here, I built two methods just to get the result. One is live mode — in this mode,
we use Amazon Bedrock to call a model like Claude or Nova, where Amazon Bedrock is
the platform, via boto3's Bedrock runtime client. In production, this is the method
we would use.

The other method is offline mode, which doesn't need any AWS credentials. In this mode,
we use a predetermined class, RuleBasedFraudReasoner, which is a small deterministic reasoner
that produces the same shape of output (verdict & rationale), from the same inputs an LLM prompt
would receive. This is our default mode when running offline. To go live, we just need to set AWS
credentials, BEDROCK_MODEL_ID, and should flip USE_LIVE_BEDROCK to true.

"""
import json
import os
from typing import ClassVar

USE_LIVE_BEDROCK = os.environ.get("USE_LIVE_BEDROCK", "false").lower() == "true"
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
)


class BedrockLLM:
    """Thin wrapper agents call instead of touching boto3 directly, so the
    live/offline switch lives in exactly one place."""

    def __init__(self):
        self._client = None
        if USE_LIVE_BEDROCK:
            import boto3  # imported lazily so offline runs never need it

            self._client = boto3.client("bedrock-runtime")

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        if USE_LIVE_BEDROCK:
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                }
            )
            response = self._client.invoke_model(
                modelId=BEDROCK_MODEL_ID, body=body
            )
            payload = json.loads(response["body"].read())
            return payload["content"][0]["text"]

        return RuleBasedFraudReasoner().reason(system_prompt, user_prompt)


class RuleBasedFraudReasoner:
    """used only when USE_LIVE_BEDROCK is False. Produces a verdict using simple keyword/score
    heuristics over the same retrieved-case and tool-result data."""

    HIGH_RISK_TYPOLOGIES: ClassVar[set[str]] = {
        "account_takeover",
        "sanctions_hit",
        "structuring_smurfing",
        "money_mule_network",
    }

    def reason(self, system_prompt: str, user_prompt: str) -> str:
        text = user_prompt.lower()
        matched_typologies = [
            t for t in self.HIGH_RISK_TYPOLOGIES
            if t in text or t.replace("_", " ") in text
        ]
        compact = "".join(text.split()).replace('"', "").replace("'", "")
        sanctions_hit = "match_found:true" in compact
        recent_reset = "recent_password_reset:true" in compact

        if sanctions_hit:
            verdict = "ESCALATE_COMPLIANCE_HOLD"
            rationale = (
                "Positive sanctions list match detected. Per policy this requires "
                "an immediate compliance hold regardless of transaction size or "
                "other risk signals."
            )
        elif matched_typologies and recent_reset:
            verdict = "ESCALATE_HUMAN_REVIEW"
            rationale = (
                f"Transaction pattern matches known high-risk typology "
                f"({', '.join(matched_typologies)}) and coincides with a recent "
                f"password reset, consistent with account takeover precursors. "
                f"Recommend escalation to a human fraud analyst before any funds move."
            )
        elif matched_typologies:
            verdict = "FLAG_FOR_REVIEW"
            rationale = (
                f"Transaction pattern resembles a known high-risk typology "
                f"({', '.join(matched_typologies)}) but lacks a strong corroborating "
                f"signal. Recommend flagging for review rather than auto-blocking."
            )
        else:
            verdict = "CLEAR"
            rationale = (
                "No strong match to known high-risk fraud typologies and no "
                "corroborating account anomaly found. Recommend clearing with "
                "standard monitoring."
            )

        return json.dumps({"verdict": verdict, "rationale": rationale})
