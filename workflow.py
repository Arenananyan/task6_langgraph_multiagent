"""Helpers for continuing the multi-agent workflow after code review."""

from langgraph.graph.message import add_messages

from state import AgentState


def merge_execution_update(
    previous_state: AgentState,
    execution_update: dict,
) -> AgentState:
    """Merge a reviewed execution result into the existing graph state."""

    updated_state = dict(previous_state)

    updated_state["messages"] = add_messages(
        previous_state["messages"],
        execution_update["messages"],
    )

    updated_state["execution_success"] = execution_update[
        "execution_success"
    ]

    updated_state["test_rmse"] = execution_update["test_rmse"]

    # Preserve the previous routing context and decision count.
    updated_state["decision_count"] = previous_state["decision_count"]

    return updated_state


def continue_after_review(previous_state: AgentState, execution_update: dict):
    """Resume orchestration with the reviewed execution result."""

    from graph import build_graph

    updated_state = merge_execution_update(
        previous_state,
        execution_update,
    )

    graph = build_graph()
    return graph.invoke(updated_state)
