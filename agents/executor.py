"""Executor agent: generate, run, and optionally repair scientific Python code."""

import json
import math
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from state import AgentState
from tools.python_runner import run_python_code


MODEL_NAME = "openai/gpt-oss-120b"
MAX_ATTEMPTS = 2

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=3500,
)

EXECUTOR_PROMPT = """
You are the Executor agent in a LangGraph multi-agent system.

You must generate a complete, executable Python script for the assigned task.

Project facts:
- Working directory: task6_langgraph_multiagent.
- Task 5 reference code: task5_reference/scr/model.py and integrator.py.
- Processed dataset: task5_reference/data/processed/silverbox_excerpt.npz.
- Trained checkpoint:
  task5_reference/results/checkpoints/model_e_w128_s2000.pt.
- The dataset has 4096 time samples; first 3072 train, last 1024 test.
- Use one continuous RK4 free-run over all 4096 samples.
- Do NOT reset the state at the train/test boundary.
- Use the actual checkpoint weights; do not retrain.
- Import AccelerationNetwork from model and rollout from integrator.
- Instantiate AccelerationNetwork(hidden_size=int(checkpoint["hidden_size"])).
- Load weights from checkpoint["model_state_dict"], NOT the entire checkpoint.
- The NPZ dataset contains keys: u, y, fs, train_size.
- There are NO x or v arrays in the NPZ dataset.
- Compute dt as 1.0 / float(data["fs"]).
- Compute initial velocity as (y[1] - y[0]) / dt.
- Use the existing rollout(model, y0, v0, u, dt) function.

- Generated scripts are saved inside the outputs directory.
- Resolve the project root using Path(__file__).resolve().parent.parent.
- Build all Task 5 paths relative to that project root.
- Convert both u and y arrays to torch.float32 tensors.
- Pass tensor initial conditions y0 and v0 to rollout, not Python floats.
- The rollout function returns TWO tensors: y_pred, v_pred.
- Unpack its result as y_pred, _ = rollout(model, y0, v0, u, dt).

- Do NOT rewrite RK4 unless explicitly necessary.
- Calculate test RMSE using y_pred[train_size:] and y[train_size:].
- Do not guess class names, dataset keys, or physical parameters.

- Use torch.load(..., map_location="cpu", weights_only=True).
- Add task5_reference/scr to sys.path before importing model and integrator.
- Calculate test RMSE from predictions at indices 3072:4096.
- Print the calculated metric as TEST_RMSE_V: followed by a number.

Requirements:
- Return ONLY complete Python source code, without Markdown fences.
- Do not hardcode or print the Task 5 reference RMSE as a computed result.
- Do not use network calls, shell commands, subprocess, or file deletion.
- Do not modify files in task5_reference.
- Never claim execution occurred; the Python runner will actually execute it.

Task 2 focusing action:
- If the instruction asks to focus a simulated beam, use only the installed
  bridge `from tools.focusing_tool import FocusSettings, run_focus_experiment`.
- Construct settings only with these valid fields: `iterations`, `seed`, and,
  when needed, `noise_fraction`, `dmd_rows`, `dmd_cols`, `camera_rows`,
  `camera_cols`, `target_rows`, `target_cols`. Do not invent fields such as
  `dmd_amplitude`, `target_center`, `target_radius`, or `background_radius`.
- The selected baseline is a 16x16 amplitude DMD, 64x64 camera, 5x5 target,
  four adjacent 5x5 background regions, 2% Gaussian noise, and 500 iterations.
- Do not append the Task 5 RMSE code to a Task 2 focusing script. Call
  `run_focus_experiment(...)`, print its returned dictionary as JSON, and
  report the actual `final_enhancement` and saved image paths.
"""


def extract_code(text: str) -> str:
    """Accept plain Python or remove a single Markdown code fence."""
    text = text.strip()

    match = re.fullmatch(
        r"```(?:python)?\s*\n(.*?)\n```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else text


def extract_rmse(stdout: str) -> float | None:
    """Read the reported test RMSE from the executed program's stdout."""
    matches = re.findall(
        r"^TEST_RMSE_V:\s*"
        r"([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
        stdout,
        flags=re.MULTILINE,
    )

    if not matches:
        return None

    value = float(matches[-1])
    return value if math.isfinite(value) and value >= 0 else None


def extract_enhancement(stdout: str) -> float | None:
    """Read the measured Task 2 enhancement from executor output."""
    matches = re.findall(
        r"^FINAL_ENHANCEMENT:\s*"
        r"([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
        stdout,
        flags=re.MULTILINE,
    )
    if not matches:
        return None
    value = float(matches[-1])
    return value if math.isfinite(value) and value >= 0 else None



def validate_generated_code(code: str) -> None:
    """Reject obvious dangerous operations before execution."""
    import ast

    tree = ast.parse(code)

    forbidden_modules = {
        "subprocess", "socket", "requests", "httpx",
        "urllib", "shutil", "ctypes", "multiprocessing",
    }

    forbidden_calls = {
        "eval", "exec", "compile", "__import__",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden_modules:
                    raise ValueError(f"Forbidden import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in forbidden_modules:
                raise ValueError(f"Forbidden import: {node.module}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in forbidden_calls:
                    raise ValueError(f"Forbidden call: {node.func.id}")

            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in {
                    "system", "popen", "remove", "unlink", "rmtree",
                }:
                    raise ValueError(
                        f"Forbidden operation: {node.func.attr}"
                    )


def focus_script(instruction: str, history: str) -> str:
    """Let the executor model choose validated settings; run a fixed tool adapter."""
    from pydantic import BaseModel, Field
    from typing import Literal

    class Plan(BaseModel):
        dmd_size: Literal[16, 24, 32]
        target_size: Literal[3, 5, 7]
        iterations: int = Field(ge=1000, le=2000)
        trials: int = Field(ge=3, le=5)
        reason: str

    plan = llm.with_structured_output(Plan).invoke([
        SystemMessage(content="Choose settings for a noisy amplitude-only scattering experiment. "
                      "Camera is 64x64, Gaussian noise 2%. Choose DMD macropixel grid, "
                      "square target size (four adjacent backgrounds of equal size), "
                      "iterations and independent trials. Budget: 300 seconds total. "
                      "Prefer 16x16 for speed; explain the tradeoff. Do not invent measurements."),
        HumanMessage(content=instruction + "\nDialogue:\n" + history),
    ])
    print("Executor selected focusing settings: " + plan.model_dump_json(), flush=True)
    settings = dict(dmd_rows=plan.dmd_size, dmd_cols=plan.dmd_size,
                    target_rows=plan.target_size, target_cols=plan.target_size,
                    iterations=plan.iterations)
    return ("import json\nfrom datetime import datetime\n"
            "from tools.focusing_tool import FocusSettings, run_focus_trials\n"
            f"settings = FocusSettings(**{settings!r})\n"
            "folder = 'outputs/task2_noisy_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')\n"
            f"print('Calling focusing tool: {plan.trials} independent noisy trials', flush=True)\n"
            f"result = run_focus_trials(settings, trials={plan.trials}, output_dir=folder)\n"
            "print('SUMMARY_PATH: ' + folder + '/summary.json')\n"
            "print('FINAL_ENHANCEMENT: ' + str(result['mean_final_enhancement']))\n"
            "print('Reported metric: mean final enhancement across all trials')\n")


def executor_node(state: AgentState, review_mode: bool = False) -> dict:
    history = "\n".join(
        f"{message.name or message.type}: {message.content}"
        for message in state["messages"][-8:]
    )

    instruction = state["instruction"]
    focusing_task = "focus" in instruction.casefold() or "enhancement" in instruction.casefold()
    feedback = ""
    last_result = None
    measured_rmse = None
    measured_enhancement = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if focusing_task:
            # Keep Task 2 isolated from the legacy Task 5 RMSE generator.
            code = focus_script(instruction, history)
        else:
            response = llm.invoke(
                [
                    SystemMessage(content=EXECUTOR_PROMPT),
                    HumanMessage(
                        content=(
                            f"Orchestrator instruction:\n{instruction}\n\n"
                            f"Previous agent messages:\n{history}\n\n"
                            f"Previous execution feedback:\n{feedback}"
                        )
                    ),
                ]
            )
            code = extract_code(response.content)

        try:
            validate_generated_code(code)
        except (ValueError, SyntaxError) as error:
            last_result = {
                "script_path": None,
                "success": False,
                "exit_code": None,
                "stdout": "",
                "stderr": f"Code validation failed: {error}",
                "timed_out": False,
            }
        else:
            last_result = run_python_code(
                code=code,
                filename=f"silverbox_attempt_{attempt}.py",
                timeout=300,
                dry_run=review_mode,
            )

        if last_result.get("dry_run"):
            return {
                "messages": [
                    AIMessage(
                        content=json.dumps(
                            {
                                "status": "awaiting_review",
                                "script_path": last_result["script_path"],
                                "executed": False,
                            },
                            indent=2,
                        ),
                        name="executor",
                    )
                ],
                "execution_success": False,
                "test_rmse": None,
            }

        if last_result["success"] and focusing_task:
            measured_enhancement = extract_enhancement(last_result["stdout"])
        elif last_result["success"]:
            measured_rmse = extract_rmse(last_result["stdout"])

        metric_found = measured_enhancement is not None if focusing_task else measured_rmse is not None
        if last_result["success"] and metric_found:
            break

        feedback = (
            f"Attempt {attempt} failed or did not print the required "
            f"{'FINAL_ENHANCEMENT' if focusing_task else 'TEST_RMSE_V'}.\n"
            f"Exit code: {last_result['exit_code']}\n"
            f"Timed out: {last_result['timed_out']}\n"
            f"STDOUT:\n{last_result['stdout'][-4000:]}\n"
            f"STDERR:\n{last_result['stderr'][-4000:]}\n"
            "Repair the program and return complete corrected Python code."
        )

    success = (
        last_result is not None
        and last_result["success"]
        and (measured_enhancement is not None if focusing_task else measured_rmse is not None)
    )

    summary = {
        "success": success,
        "attempts": attempt,
        "script_path": last_result["script_path"],
        "exit_code": last_result["exit_code"],
        "timed_out": last_result["timed_out"],
        "stdout": last_result["stdout"][-4000:],
        "stderr": last_result["stderr"][-4000:],
        "test_rmse_v": measured_rmse if success and not focusing_task else None,
        "final_enhancement": measured_enhancement if success and focusing_task else None,
    }

    return {
        "messages": [
            AIMessage(
                content=json.dumps(summary, indent=2),
                name="executor",
            )
        ],
        "execution_success": success,
        "test_rmse": measured_rmse if success and not focusing_task else None,
        "final_enhancement": measured_enhancement if success and focusing_task else None,
    }


def execute_reviewed_script(script_path: str) -> dict:
    """Execute a previously reviewed script and return its actual result."""
    from pathlib import Path

    from tools.python_runner import OUTPUT_DIR

    path = Path(script_path).resolve()
    allowed_dir = OUTPUT_DIR.resolve()

    if path.parent != allowed_dir or path.suffix != ".py":
        raise ValueError("Script must be a Python file directly inside outputs/")

    if not path.is_file():
        raise FileNotFoundError(f"Reviewed script not found: {path}")

    code = path.read_text(encoding="utf-8")
    validate_generated_code(code)

    result = run_python_code(
        code=code,
        filename=path.name,
        timeout=300,
    )

    rmse = extract_rmse(result["stdout"]) if result["success"] else None

    return {
        "script_path": result["script_path"],
        "execution_success": result["success"] and rmse is not None,
        "test_rmse": rmse,
        "exit_code": result["exit_code"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "timed_out": result["timed_out"],
    }


def reviewed_execution_state_update(script_path: str) -> dict:
    """Run reviewed code and prepare an AgentState update."""
    result = execute_reviewed_script(script_path)

    summary = {
        "status": (
            "execution_completed"
            if result["execution_success"]
            else "execution_failed"
        ),
        "script_path": result["script_path"],
        "exit_code": result["exit_code"],
        "timed_out": result["timed_out"],
        "stdout": result["stdout"][-4000:],
        "stderr": result["stderr"][-4000:],
        "test_rmse_v": result["test_rmse"],
    }

    return {
        "messages": [
            AIMessage(
                content=json.dumps(summary, indent=2),
                name="executor",
            )
        ],
        "execution_success": result["execution_success"],
        "test_rmse": (
            result["test_rmse"]
            if result["execution_success"]
            else None
        ),
    }
