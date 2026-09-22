"""
Investigation Agent.

Investigation Agent fetches account profile and KYC status using two different tools over
the MCP protocol. It also uses a third tool, sanctions screen, but that is only executed
when the typology given by the Detection Agent via A2A matches "sanctions_hit," and only
if an account was actually found while using the account lookup tool.

"""
from src.agents.mcp_client import MCPToolClient
from src.agents.a2a_protocol import AgentCard, A2AServer, A2ATask, A2ATaskResult


class InvestigationAgent:
    AGENT_ID = "investigation-agent"

    def __init__(self):
        self.mcp_client = MCPToolClient()
        A2AServer(
            card=AgentCard(
                agent_id=self.AGENT_ID,
                name="Fraud Investigation Agent",
                skills=["gather_account_evidence"],
            ),
            handler=self._handle_a2a_task,
        )

    def _handle_a2a_task(self, task: A2ATask) -> A2ATaskResult:
        if task.intent != "gather_account_evidence":
            return A2ATaskResult(
                task_id=task.task_id,
                from_agent=self.AGENT_ID,
                status="failed",
                output={"error": f"Unsupported intent: {task.intent}"},
            )

        account_id = task.payload["account_id"]
        suspected_typology = task.payload.get("suspected_typology")

        # MCP tool calls
        account_profile = self.mcp_client.call_tool(
            "account_lookup", {"account_id": account_id}
        )
        kyc = self.mcp_client.call_tool("kyc_check", {"account_id": account_id})

        sanctions_result = None
        if suspected_typology == "sanctions_hit" and "customer_name" in account_profile:
            sanctions_result = self.mcp_client.call_tool(
                "sanctions_screen", {"full_name": account_profile["customer_name"]}
            )

        output = {
            "account_id": account_id,
            "account_profile": account_profile,
            "kyc": kyc,
            "sanctions_screen": sanctions_result,
        }
        return A2ATaskResult(
            task_id=task.task_id, from_agent=self.AGENT_ID, status="completed", output=output
        )
