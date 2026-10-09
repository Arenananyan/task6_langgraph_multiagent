from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from state import AgentState


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1200,
)


GENERAL_SYSTEM_PROMPT = """
You are the General Agent in a LangGraph multi-agent system.

Your role is to provide clear conceptual explanations and general guidance.

You may:
- Explain LangGraph and multi-agent collaboration.
- Explain scientific and machine-learning terminology.
- Clarify the Silverbox oscillator modeling task.
- Answer high-level questions from the Orchestrator.

You must not:
- Write or execute Python programs.
- Claim to have run experiments or computed new results.
- Invent dataset details or numerical results.
- Perform the Scientist's detailed planning responsibilities.
- Perform the Executor's implementation responsibilities.

Important project context:
- The task is to reproduce a neural acceleration model for the Silverbox oscillator.
- Numerical integration uses RK4.
- The previous Task 5 test free-run RMSE was 0.024516 V.
- That RMSE is a reference from earlier work, not a new Task 6 result.

Respond concisely and accurately.
"""


def general_node(state: AgentState) -> dict:
    instruction = state["instruction"]

    recent_history = "\n".join(
        f"{message.name or message.type}: {message.content}"
        for message in state["messages"][-6:]
    )

    response = llm.invoke(
        [
            SystemMessage(content=GENERAL_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Orchestrator instruction:\n{instruction}\n\n"
                    f"Recent conversation:\n{recent_history}"
                )
            ),
        ]
    )

    return {
        "messages": [
            AIMessage(
                content=response.content,
                name="general",
            )
        ]
    }
