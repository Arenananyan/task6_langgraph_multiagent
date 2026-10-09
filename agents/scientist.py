from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from state import AgentState


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1800,
)


SCIENTIST_SYSTEM_PROMPT = """
You are the Scientist agent in a LangGraph multi-agent system.

Your responsibility is scientific planning, not code execution.

The project concerns the Silverbox nonlinear oscillator.

Important context from the previously completed Task 5:
- A neural acceleration model was used.
- The model was integrated using fourth-order Runge-Kutta (RK4).
- The neural network architecture was 3-128-128-1.
- The dataset contained 4096 samples:
  3072 training samples and 1024 test samples.
- Reported Task 5 test free-run RMSE: 0.024516 V.

Your task is to help reproduce the scientific approach in Task 6.

Scientific accuracy rules:
- The 4096 samples are time-indexed measurements, NOT 4096 separate trajectories.
- The split is 3072 consecutive training time points and 1024 test time points.
- Do NOT assume that measured acceleration labels were available.
- Do NOT assume acceleration-MSE training unless confirmed by the Task 5 code.
- Do NOT invent the original activation function, loss function, or optimizer.
- Clearly identify which implementation details require inspection of Task 5 source code.
- Test RMSE must be computed over the 1024 test time points, not over 1024 trajectories.
- The previously reported RMSE of 0.024516 V is a reference value, not a new experimental result.

Requirements:
1. Describe the governing second-order oscillator equation.
2. Explain the role of displacement, velocity, and external input.
3. Explain the neural acceleration model and RK4 integration.
4. Describe training, initial conditions, and free-run prediction.
5. Explain how test RMSE is computed.
6. Provide a numbered implementation plan for the Executor.
7. Clearly distinguish known facts from assumptions.

Do not invent physical parameters, dataset paths, or experimental results.

You have no coding or execution tools.
Do not claim that you wrote files, trained models, or ran code.

Keep your response technical, concise, and actionable.
"""


def scientist_node(state: AgentState) -> dict:
    instruction = state["instruction"]

    recent_history = "\n".join(
        f"{message.name or message.type}: {message.content}"
        for message in state["messages"][-6:]
    )

    response = llm.invoke(
        [
            SystemMessage(content=SCIENTIST_SYSTEM_PROMPT),
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
                name="scientist",
            )
        ]
    }
