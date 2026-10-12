"""Offline integration checks for checkpoint, interruption and refusal semantics."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from homework2_task1.graph import RouteDecision, build_graph


class HumanTurnTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.decisions = []

        def router(state):
            self.calls.append(("orchestrator", [m.name for m in state["messages"]]))
            return self.decisions.pop(0)

        def specialist(name, state):
            self.calls.append((name, [m.name for m in state["messages"]]))
            if name == "executor":
                return {"messages": [AIMessage(content="TEST_RMSE_V: 0.0245", name="executor")],
                        "execution_success": True, "test_rmse": 0.0245}
            return {"messages": [AIMessage(content="Answered", name=name)]}

        self.graph = build_graph(InMemorySaver(), router=router, specialist=specialist)
        self.config = {"configurable": {"thread_id": self.id()}, "recursion_limit": 35}

    def start(self, text):
        return self.graph.invoke({"messages": [HumanMessage(content=text, name="user")],
                                  "decision_count": 0, "execution_success": False,
                                  "test_rmse": None}, self.config)

    def question(self):
        return self.graph.get_state(self.config).interrupts[0].value

    def test_ambiguous_reply_is_saved_and_read_by_next_agent(self):
        self.decisions = [
            RouteDecision(next_role="human", instruction="", reason="Two meanings.",
                          question="Do you mean data or model?", human_kind="clarification"),
            RouteDecision(next_role="general", instruction="Explain data.", reason="Clarified."),
            RouteDecision(next_role="human", instruction="Run evaluation.", reason="Execution needs approval.",
                          question="Run it?", human_kind="approval", pending_role="executor"),
        ]
        self.start("Explain the record")
        self.assertEqual(self.question()["kind"], "clarification")
        self.graph.invoke(Command(resume="The data record"), self.config)
        self.assertIn(("general", ["user", "orchestrator", "human", "orchestrator"]), self.calls)
        self.assertEqual(self.question()["pending_role"], "executor")
        self.assertFalse(any(name == "executor" for name, _ in self.calls))

    def test_refusal_does_not_run_then_acceptance_does(self):
        approve = RouteDecision(next_role="human", instruction="Evaluate Task 5.",
                                reason="Must ask before executing.", question="Run generated code?",
                                human_kind="approval", pending_role="executor")
        finish = RouteDecision(next_role="human", instruction="Report measured RMSE.",
                               reason="Must ask before reporting.", question="Report the result?",
                               human_kind="approval", pending_role="finish")
        self.decisions = [approve, approve, finish]
        self.start("Evaluate the Silverbox model")
        self.assertEqual(self.question()["pending_role"], "executor")
        self.graph.invoke(Command(resume="no"), self.config)
        self.assertEqual(self.question()["pending_role"], "executor")
        self.assertFalse(any(name == "executor" for name, _ in self.calls))
        self.graph.invoke(Command(resume="yes"), self.config)
        self.assertEqual(sum(name == "executor" for name, _ in self.calls), 1)
        self.assertEqual(self.question()["pending_role"], "finish")
        self.graph.invoke(Command(resume="yes"), self.config)
        state = self.graph.get_state(self.config)
        self.assertFalse(state.interrupts)
        self.assertEqual(state.values["test_rmse"], 0.0245)
        self.assertEqual(sum(name == "executor" for name, _ in self.calls), 1)

    def test_sqlite_resume_after_graph_is_recreated(self):
        def router(state):
            names = [message.name for message in state["messages"]]
            if "human" not in names:
                return RouteDecision(next_role="human", instruction="", reason="Ambiguous.",
                                     question="Which record?", human_kind="clarification")
            if "general" in names:
                return RouteDecision(next_role="human", instruction="Evaluate.", reason="Approval needed.",
                                     question="Run evaluation?", human_kind="approval", pending_role="executor")
            return RouteDecision(next_role="general", instruction="Explain it.", reason="Clarified.")

        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(path) as saver:
                graph = build_graph(saver, router, lambda name, state: {
                    "messages": [AIMessage(content="Explained.", name=name)]})
                graph.invoke({"messages": [HumanMessage(content="What is it?", name="user")],
                              "decision_count": 0}, self.config)
                self.assertEqual(graph.get_state(self.config).interrupts[0].value["question"],
                                 "Which record?")
            with SqliteSaver.from_conn_string(path) as saver:
                graph = build_graph(saver, router, lambda name, state: {
                    "messages": [AIMessage(content="Explained.", name=name)]})
                graph.invoke(Command(resume="The dataset record"), self.config)
                messages = graph.get_state(self.config).values["messages"]
                self.assertTrue(any(m.name == "human" and m.content == "The dataset record"
                                    for m in messages))
                self.assertEqual(graph.get_state(self.config).interrupts[0].value["pending_role"],
                                 "executor")

    def test_executor_cannot_be_routed_without_human_approval(self):
        self.decisions = [RouteDecision(next_role="executor", instruction="Run it.",
                                        reason="Needed.")]
        with self.assertRaisesRegex(ValueError, "human approval"):
            self.start("Run the model")
        self.assertFalse(any(name == "executor" for name, _ in self.calls))

    def test_finish_approval_requires_real_result(self):
        self.decisions = [RouteDecision(next_role="human", instruction="Report result.",
                                        reason="Ready.", question="Finish?",
                                        human_kind="approval", pending_role="finish")]
        with self.assertRaisesRegex(ValueError, "measured RMSE"):
            self.start("Report the result")


if __name__ == "__main__":
    unittest.main()
