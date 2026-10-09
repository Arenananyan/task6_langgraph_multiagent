"""LangGraph workflow for the Silverbox multi-agent system."""

from langgraph.graph import END, START, StateGraph

from state import AgentState
from agents.orchestrator import MAX_DECISIONS, orchestrator_node
from agents.scientist import scientist_node
from agents.general import general_node
from agents.executor import executor_node


def route_from_orchestrator(state: AgentState) -> str:
    """Follow the LLM decision, enforcing completion and turn limits."""
    next_agent = state["next_agent"]

    if next_agent == "finish":
        if state.get("execution_success") and state.get("test_rmse") is not None:
            return "finish"
        raise RuntimeError("Cannot finish without a successful execution and RMSE.")

    # Do not dispatch another agent when the decision budget is exhausted.
    if state["decision_count"] >= MAX_DECISIONS:
        return "limit_reached"

    return next_agent


def limit_reached_node(state: AgentState) -> dict:
    """Record an unsuccessful termination when the decision limit is reached."""
    from langchain_core.messages import AIMessage

    return {
        "messages": [
            AIMessage(
                content="Decision limit reached before successful completion.",
                name="system_status",
            )
        ]
    }


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("scientist", scientist_node)
    builder.add_node("general", general_node)
    builder.add_node(
        "executor",
        lambda state: executor_node(state, review_mode=True),
    )
    builder.add_node("limit_reached", limit_reached_node)

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "scientist": "scientist",
            "general": "general",
            "executor": "executor",
            "finish": END,
            "limit_reached": "limit_reached",
        },
    )

    def route_after_agent(state: AgentState) -> str:
        if state.get("decision_count", 0) >= MAX_DECISIONS:
            return "limit_reached"
        return "orchestrator"

    for agent in ("scientist", "general"):
        builder.add_conditional_edges(
            agent,
            route_after_agent,
            {
                "orchestrator": "orchestrator",
                "limit_reached": "limit_reached",
            },
        )

    def route_after_executor(state: AgentState) -> str:
        import json

        for message in reversed(state["messages"]):
            if message.name == "executor":
                try:
                    result = json.loads(message.content)
                except (ValueError, TypeError):
                    break

                if result.get("status") == "awaiting_review":
                    return "awaiting_review"
                break

        return route_after_agent(state)

    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "awaiting_review": END,
            "orchestrator": "orchestrator",
            "limit_reached": "limit_reached",
        },
    )
    builder.add_edge("limit_reached", END)

    return builder.compile()
