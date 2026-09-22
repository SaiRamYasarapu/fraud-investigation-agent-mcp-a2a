"""
Detection Agent.

Detection Agent uses a RAG system to retrieve similar fraud cases from the vector store, and it
also calls the Investigation Agent via A2A client to get the account info, the KYC status, and
the sanctions screen status as well. After gathering all this information, it uses a live LLM,
or a predetermined class, RuleBasedFraudReasoner, to return a verdict and rationale.
"""
from src.vector_store.retriever import FraudCaseRetriever
from src.llm.bedrock_client import BedrockLLM
from src.agents.a2a_protocol import AgentCard, A2AServer, A2AClient, A2ATask, A2ATaskResult

SYSTEM_PROMPT = (
    "You are a fraud detection analyst. Given a transaction description and "
    "similar historical fraud cases, identify the likely typology and note "
    "what account-level evidence would confirm or rule it out."
)


class DetectionAgent:
    AGENT_ID = "detection-agent"

    def __init__(self):
        self.retriever = FraudCaseRetriever()
        self.llm = BedrockLLM()
        self.a2a = A2AClient(from_agent_id=self.AGENT_ID)
        # Registers this agent so InvestigationAgent can reach it via A2A.
        A2AServer(
            card=AgentCard(
                agent_id=self.AGENT_ID,
                name="Fraud Detection Agent",
                skills=["retrieve_similar_fraud_cases"],
            ),
            handler=self._handle_a2a_task,
        )

    def analyze(self, transaction_description: str, account_id: str) -> dict:
        """analyze() is the function that runs the entire Detection Agent pipeline from
        start to finish. This is the single method orchestrator.py calls to do all of the Detection Agent's work."""
        retrieved = self.retriever.retrieve(transaction_description, top_k=3)
        top_match = retrieved[0] if retrieved else None

        # Ask the Investigation Agent for account evidence over A2A.
        evidence_result: A2ATaskResult = self.a2a.send_task(
            to_agent="investigation-agent",
            intent="gather_account_evidence",
            payload={"account_id": account_id, "suspected_typology": top_match.typology if top_match else None},
        )

        prompt = self._build_prompt(transaction_description, retrieved, evidence_result.output)
        raw_verdict = self.llm.invoke(SYSTEM_PROMPT, prompt)

        return {
            "retrieved_cases": [r.__dict__ for r in retrieved],
            "account_evidence": evidence_result.output,
            "llm_prompt": prompt,
            "verdict_raw": raw_verdict,
        }

    def _build_prompt(self, txn_desc, retrieved, evidence: dict) -> str:
        cases_block = "\n".join(
            f"- [{r.typology}] {r.pattern} (resolution: {r.resolution})"
            for r in retrieved
        )
        return (
            f"Transaction under review:\n{txn_desc}\n\n"
            f"Similar historical fraud cases (retrieved via RAG):\n{cases_block}\n\n"
            f"Account evidence gathered by Investigation Agent:\n{evidence}\n\n"
            f"Provide a verdict."
        )

    def _handle_a2a_task(self, task: A2ATask) -> A2ATaskResult:
        """Handles inbound A2A tasks. This is a function where our Detection Agent receives
        the information, in case it is called by some other agent via the A2A protocol."""
        if task.intent == "retrieve_similar_fraud_cases":
            query = task.payload.get("query", "")
            retrieved = self.retriever.retrieve(query, top_k=3)
            return A2ATaskResult(
                task_id=task.task_id,
                from_agent=self.AGENT_ID,
                status="completed",
                output={"retrieved_cases": [r.__dict__ for r in retrieved]},
            )
        return A2ATaskResult(
            task_id=task.task_id,
            from_agent=self.AGENT_ID,
            status="failed",
            output={"error": f"Unsupported intent: {task.intent}"},
        )
