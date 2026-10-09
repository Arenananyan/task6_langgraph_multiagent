"""Write and execute generated Python scripts with a time limit."""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAX_TIMEOUT_SECONDS = 300


def run_python_code(
    code: str,
    filename: str = "generated_solution.py",
    timeout: int = MAX_TIMEOUT_SECONDS,
    dry_run: bool = False,
) -> dict:
    """Save Python code, execute it, and capture the result."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".py"):
        raise ValueError("Filename must end with .py")

    if Path(filename).name != filename:
        raise ValueError("Filename must not contain directory paths")

    if not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError("Timeout must be between 1 and 300 seconds")

    script_path = OUTPUT_DIR / filename
    script_path.write_text(code, encoding="utf-8")

    if dry_run:
        return {
            "script_path": str(script_path),
            "success": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "dry_run": True,
        }


    # Do not pass the Groq API key to generated programs.
    environment = os.environ.copy()
    environment.pop("GROQ_API_KEY", None)
    environment.pop("GEMINI_API_KEY", None)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )

        return {
            "script_path": str(script_path),
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired as error:
        return {
            "script_path": str(script_path),
            "success": False,
            "exit_code": None,
            "stdout": (
                error.stdout.decode(errors="replace")
                if isinstance(error.stdout, bytes)
                else error.stdout or ""
            ),
            "stderr": (
                error.stderr.decode(errors="replace")
                if isinstance(error.stderr, bytes)
                else error.stderr or ""
            ),
            "timed_out": True,
        }
