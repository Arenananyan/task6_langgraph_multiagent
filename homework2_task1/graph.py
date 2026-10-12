"""Four-agent Silverbox graph with a real, resumable human turn."""

from __future__ import annotations

from typing import Annotated, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field


MAX_DECISIONS = 8
Role = Literal["scientist", "general", "executor", "human", "finish", "limit_reached"]
DecisionRole = Literal["scientist", "general", "executor", "human", "finish"]
Pending = Literal["", "executor", "finish"]


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    next_role: Role
    instruction: str
    reason: str
    decision_count: int
    execution_success: bool
    test_rmse: float | None
    human_question: str
    human_kind: Literal["clarification", "approval"]
    pending_role: Pending
    pending_instruction: str
    approved_role: Pending
    human_reply: str


class RouteDecision(BaseModel):
    next_role: DecisionRole
    instruction: str = Field(description="Specific instruction for the selected role")
    reason: str = Field(description="One sentence explaining this decision")
    question: str = Field(default="", description="Question to ask when next_role is human")
    human_kind: Literal["clarification", "approval"] = "clarification"
    pending_role: Pending = Field(default="", description="executor or finish for approval; empty for clarification")


def groq_router(state: State) -> RouteDecision:
    """Ask the model to choose the next graph edge from the shared dialogue."""
    from langchain_groq import ChatGroq

    history = "\n".join(
        f"{message.name or message.type}: {message.content}"
        for message in state.get("messages", [])[-14:]
    )
    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    router = model.with_structured_output(RouteDecision, method="json_schema")
    prompt = f"""You are the orchestrator of a four-role Silverbox LangGraph system.
Read the dialogue and choose the next edge by reasoning; do not use a fixed role order.
Roles: scientist plans equations/training/free-run evaluation; general answers conceptual
questions; executor writes and RUNS Python; human clarifies or approves; finish ends.
The user speaks only to you. Specialists see all shared messages.
Choose human with human_kind=clarification when the latest request has two materially
different interpretations; ask a specific question and set pending_role to empty.
Choose human with human_kind=approval BEFORE executor writes/runs code or overwrites
a saved result, and BEFORE finish reports a measurement. Set pending_role to the
proposed executor or finish action and give a concrete yes/no question. Put the
exact pending instruction in instruction. A refusal returns to you without that
action. You may finish only after actual script execution printed test RMSE.
Never fabricate a metric. Give a one-sentence reason. Max {MAX_DECISIONS} decisions.
Current decision: {state.get('decision_count', 0) + 1}.
Successful execution: {state.get('execution_success', False)}.
Measured test RMSE: {state.get('test_rmse')}.
Dialogue:\n{history}"""
    decision = router.invoke(prompt)
    if decision.next_role in ("executor", "finish"):
        # Ask the model to produce the required human routing decision itself.
        decision = router.invoke(
            prompt + "\nIMPORTANT: Your previous answer skipped mandatory human approval. "
            "Choose next_role=human, human_kind=approval, pending_role equal to the "
            "intended executor or finish action, and ask a concrete yes/no question."
        )
    return decision


def _specialist(name: str, state: State) -> dict:
    # Existing Task 6 specialists remain the implementation of the four roles.
    if name == "scientist":
        from agents.scientist import scientist_node
        return scientist_node(state)
    if name == "general":
        from agents.general import general_node
        return general_node(state)
    if name == "executor":
        from agents.executor import executor_node
        return executor_node(state, review_mode=False)
    raise ValueError(name)


def build_graph(checkpointer, router: Callable[[State], RouteDecision] = groq_router,
                specialist: Callable[[str, State], dict] = _specialist):
    """Compile a graph with a durable checkpoint supplied by the caller."""
    builder = StateGraph(State)

    def orchestrator(state: State) -> dict:
        count = state.get("decision_count", 0)
        if count >= MAX_DECISIONS:
            return {"next_role": "limit_reached", "instruction": "", "reason": "Decision limit reached.",
                    "messages": [AIMessage(content="Decision limit reached.", name="system_status")]}

        decision = router(state)
        if not isinstance(decision, RouteDecision):
            decision = RouteDecision.model_validate(decision)
        if decision.next_role in ("executor", "finish"):
            raise ValueError("The model must request human approval before executor or finish.")
        if decision.next_role == "human":
            if not decision.question.strip():
                raise ValueError("Human routing requires a concrete question.")
            if decision.human_kind == "approval" and decision.pending_role not in ("executor", "finish"):
                raise ValueError("Approval requires a pending executor or finish action.")
            if decision.human_kind == "clarification" and decision.pending_role:
                raise ValueError("Clarification cannot contain a pending action.")
            if decision.pending_role == "finish" and not (
                state.get("execution_success") and state.get("test_rmse") is not None
            ):
                raise ValueError("Cannot request approval to finish before measured RMSE.")
        question = decision.question
        if decision.next_role == "human" and decision.human_kind == "approval":
            if decision.pending_role == "executor":
                question += " The executor will write and run a Python file; an existing attempt file may be overwritten. Proceed?"
            elif decision.pending_role == "finish":
                question += f" The measured test RMSE is {state['test_rmse']:.9f} V. Report it and finish?"
        return {
            "next_role": decision.next_role,
            "instruction": decision.instruction,
            "reason": decision.reason,
            "decision_count": count + 1,
            "human_question": question,
            "human_kind": decision.human_kind,
            "pending_role": decision.pending_role,
            "pending_instruction": decision.instruction if decision.pending_role else "",
            "approved_role": "",
            "messages": [AIMessage(content=f"Next: {decision.next_role}. Reason: {decision.reason}", name="orchestrator")],
        }

    def human(state: State) -> dict:
        # interrupt restarts this node on resume. No file writes or model calls precede it.
        reply = interrupt({
            "question": state["human_question"],
            "kind": state["human_kind"],
            "pending_role": state.get("pending_role", ""),
            "reason": state["reason"],
        })
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("A non-empty human reply is required.")
        reply = reply.strip()
        update: dict = {"messages": [HumanMessage(content=reply, name="human")], "human_reply": reply}
        if state["human_kind"] == "clarification":
            return {**update, "approved_role": "", "pending_role": ""}

        # Approval is deliberately narrow. Any ambiguous response counts as refusal.
        accepted = reply.casefold() in {"yes", "y", "approve", "approved", "accept", "այո", "հա", "հաստատում եմ"}
        if accepted:
            return {**update, "approved_role": state["pending_role"],
                    "instruction": state["pending_instruction"], "pending_role": ""}
        return {**update, "approved_role": "", "pending_role": "",
                "messages": [*update["messages"], AIMessage(
                    content=f"Human refused the proposed {state['pending_role']} action; it did not run.",
                    name="system_status")]}

    def node(name: str):
        def run(state: State) -> dict:
            return specialist(name, state)
        return run

    builder.add_node("orchestrator", orchestrator)
    builder.add_node("human", human)
    for name in ("scientist", "general", "executor"):
        builder.add_node(name, node(name))
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", lambda s: s["next_role"], {
        "scientist": "scientist", "general": "general", "human": "human",
        "finish": END, "limit_reached": END})
    builder.add_conditional_edges("human", lambda s: s.get("approved_role") or "orchestrator", {
        "executor": "executor", "finish": END, "orchestrator": "orchestrator"})
    for name in ("scientist", "general", "executor"):
        builder.add_edge(name, "orchestrator")
    return builder.compile(checkpointer=checkpointer)
