"""Start, resume and inspect persistent Homework 2 Task 1 conversations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from homework2_task1.graph import build_graph


ROOT = Path(__file__).resolve().parent.parent
DATABASE = ROOT / "outputs" / "homework2_task1" / "checkpoints.sqlite"


def _report(graph, config, previous_count: int = 0) -> None:
    snapshot = graph.get_state(config)
    state = snapshot.values
    for message in state.get("messages", [])[previous_count:]:
        print(f"{message.name or message.type}: {message.content}")
    if snapshot.interrupts:
        payload = snapshot.interrupts[0].value
        print("\nHUMAN QUESTION:", payload["question"])
        print("Type:", payload["kind"])
        print("Pending action:", payload["pending_role"] or "none")
        print("Resume this same thread with: python -m homework2_task1.cli resume --thread-id",
              config["configurable"]["thread_id"], "--reply 'YOUR ANSWER'")
    else:
        print("\nThread stopped.")
        if state.get("execution_success") and state.get("test_rmse") is not None:
            print(f"Actual test RMSE: {state['test_rmse']:.9f} V")
    print("Decisions:", state.get("decision_count", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start")
    start.add_argument("--thread-id", required=True)
    start.add_argument("--request", required=True)
    resume = sub.add_parser("resume")
    resume.add_argument("--thread-id", required=True)
    resume.add_argument("--reply", required=True)
    status = sub.add_parser("status")
    status.add_argument("--thread-id", required=True)
    status.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if args.action != "status" and not os.getenv("GROQ_API_KEY"):
        parser.error("Set GROQ_API_KEY in the repository's .env file before starting or resuming.")
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    config = {"configurable": {"thread_id": args.thread_id}, "recursion_limit": 35}
    with SqliteSaver.from_conn_string(str(DATABASE)) as saver:
        graph = build_graph(saver)
        snapshot = graph.get_state(config)
        if args.action == "start":
            if snapshot.values:
                parser.error("This thread ID already exists; choose another or resume it.")
            if not args.request.strip():
                parser.error("Request cannot be empty.")
            graph.invoke({"messages": [HumanMessage(content=args.request.strip(), name="user")],
                          "decision_count": 0, "execution_success": False, "test_rmse": None}, config)
            _report(graph, config)
        elif args.action == "resume":
            if not snapshot.interrupts:
                parser.error("This thread has no pending human question.")
            if not args.reply.strip():
                parser.error("Reply cannot be empty.")
            count = len(snapshot.values.get("messages", []))
            graph.invoke(Command(resume=args.reply), config)
            _report(graph, config, count)
        else:
            if not snapshot.values:
                parser.error("Unknown thread ID.")
            _report(graph, config)
            if args.json_out:
                data = {
                    "thread_id": args.thread_id,
                    "decisions": snapshot.values.get("decision_count", 0),
                    "execution_success": snapshot.values.get("execution_success", False),
                    "test_rmse": snapshot.values.get("test_rmse"),
                    "pending_question": snapshot.interrupts[0].value if snapshot.interrupts else None,
                    "messages": [{"role": m.name or m.type, "text": m.content}
                                 for m in snapshot.values.get("messages", [])],
                }
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                print("Saved transcript:", args.json_out)


if __name__ == "__main__":
    main()
