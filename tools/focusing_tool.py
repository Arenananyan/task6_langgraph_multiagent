"""Bridge from the Task 6 executor to the standalone Task 2 simulator."""

from __future__ import annotations

import sys
from pathlib import Path


TASK2_ROOT = Path(__file__).resolve().parents[2] / "homework2_task2"
if str(TASK2_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK2_ROOT))

from focus_tool import FocusSettings, run_focus_experiment  # noqa: E402

__all__ = ["FocusSettings", "run_focus_experiment"]
