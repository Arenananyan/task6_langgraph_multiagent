from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


AgentName = Literal["scientist", "general", "executor", "finish"]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    next_agent: AgentName
    instruction: str
    reason: str

    decision_count: int

    execution_success: bool
    test_rmse: float | None
