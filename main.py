"""Entry point for the Silverbox LangGraph multi-agent workflow."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from graph import build_graph


PROJECT_ROOT = Path(__file__).resolve().parent
STATE_PATH = PROJECT_ROOT / "outputs" / "workflow_state.json"


def main():
    task = (
        "Reproduce Task 5 Silverbox oscillator modeling. "
        "Develop the scientific plan, generate a Python evaluation "
        "script using the existing Model E checkpoint and real dataset, "
        "perform a continuous RK4 free-run, and report test RMSE."
    )

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "next_agent": "scientist",
        "instruction": "",
        "reason": "",
        "decision_count": 0,
        "execution_success": False,
        "test_rmse": None,
    }

    graph = build_graph()
    result = graph.invoke(initial_state)

    print("\n===== WORKFLOW RESULT =====")
    print("Decision count:", result["decision_count"])
    print("Execution success:", result["execution_success"])
    print("Test RMSE:", result["test_rmse"])

    print("\n===== AGENT MESSAGES =====")

    for message in result["messages"]:
        role = message.name or message.type
        print(f"\n[{role}]")
        print(str(message.content)[:1500])

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "messages": [
            {
                "type": message.type,
                "name": message.name,
                "content": message.content,
            }
            for message in result["messages"]
        ],
        "next_agent": result["next_agent"],
        "instruction": result["instruction"],
        "reason": result["reason"],
        "decision_count": result["decision_count"],
        "execution_success": result["execution_success"],
        "test_rmse": result["test_rmse"],
    }

    STATE_PATH.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSTATE_SAVED:", STATE_PATH)

    if result["messages"]:
        last_message = result["messages"][-1]
        if last_message.name == "executor":
            try:
                summary = json.loads(last_message.content)
                if summary.get("status") == "awaiting_review":
                    print("AWAITING_CODE_REVIEW:", summary["script_path"])
            except (ValueError, TypeError, KeyError):
                pass


if __name__ == "__main__":
    main()
