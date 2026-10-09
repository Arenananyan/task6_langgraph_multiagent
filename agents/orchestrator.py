from typing import Literal

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from state import AgentState


MAX_DECISIONS = 8


class RouteDecision(BaseModel):
    next_agent: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(
        description="Specific instruction for the selected agent."
    )
    reason: str = Field(
        description="Explanation of why this agent was selected."
    )


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

router_llm = llm.with_structured_output(
    RouteDecision,
    method="json_schema",
)


def orchestrator_node(state: AgentState) -> dict:
    decision_count = state.get("decision_count", 0)

    if decision_count >= MAX_DECISIONS:
        raise RuntimeError(
            "Maximum number of orchestrator decisions reached."
        )

    history = "\n".join(
        f"{message.type}: {message.content}"
        for message in state["messages"][-12:]
    )

    execution_success = state.get("execution_success", False)
    test_rmse = state.get("test_rmse")

    prompt = f"""
You are the Orchestrator of a LangGraph multi-agent system.

Available agents:

1. scientist:
   Develops the scientific plan, equations, neural network approach,
   training procedure, and free-run evaluation strategy.
   Cannot execute code.

2. general:
   Answers conceptual questions and provides general explanations.
   Cannot execute code.

3. executor:
   Writes and executes Python code using a dedicated coding tool.
   Must report execution status and test RMSE.

4. finish:
   Ends the workflow after successful execution and RMSE reporting.

Your task is to solve the Silverbox oscillator modeling problem
using a physics-informed neural network.

Choose the next agent based on the conversation and current state.
Do not follow a fixed sequence.

Rules:
- Make an independent LLM-based routing decision.
- Give the selected agent a concrete instruction.
- Explain the reason for your decision.
- You may select finish ONLY when execution_success is True
  AND test_rmse is not None.
- If execution has not succeeded, continue working toward
  a successful executable solution.
- The maximum number of routing decisions is {MAX_DECISIONS}.

Current decision number: {decision_count + 1}
Execution successful: {execution_success}
Test RMSE: {test_rmse}

Conversation history:
{history}
"""

    decision = router_llm.invoke(prompt)

    if decision.next_agent == "finish":
        if not execution_success or test_rmse is None:
            raise RuntimeError(
                "Orchestrator attempted to finish before successful execution."
            )

    return {
        "next_agent": decision.next_agent,
        "instruction": decision.instruction,
        "reason": decision.reason,
        "decision_count": decision_count + 1,
    }
