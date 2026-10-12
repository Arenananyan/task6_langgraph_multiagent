"""Callable tool wrapper for the simulated focusing experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

try:
    from .focusing import focus, save_result
    from .simulator import BenchConfig, SimulatedBench
except ImportError:  # Support direct execution from the task folder.
    from focusing import focus, save_result
    from simulator import BenchConfig, SimulatedBench


@dataclass(frozen=True)
class FocusSettings:
    """All settings chosen for one experiment and recorded in the result."""

    dmd_rows: int = 16
    dmd_cols: int = 16
    camera_rows: int = 64
    camera_cols: int = 64
    target_rows: int = 5
    target_cols: int = 5
    noise_fraction: float = 0.02
    iterations: int = 1000
    seed: int = 42


def run_focus_experiment(settings: FocusSettings = FocusSettings(), output_dir: str | Path = "outputs") -> dict:
    """Run one experiment and return JSON-friendly measured results."""
    config = BenchConfig(
        dmd_shape=(settings.dmd_rows, settings.dmd_cols),
        camera_shape=(settings.camera_rows, settings.camera_cols),
        target_shape=(settings.target_rows, settings.target_cols),
        noise_fraction=settings.noise_fraction,
        seed=settings.seed,
    )
    bench = SimulatedBench(config)
    result = focus(bench, iterations=settings.iterations)
    save_result(result, Path(output_dir))
    return {
        "settings": asdict(settings),
        "target_region": asdict(result.final.target),
        "initial_enhancement": result.initial.enhancement,
        "final_enhancement": result.final.enhancement,
        "improvement_factor": result.final.enhancement / result.initial.enhancement,
        "outputs": [
            str(Path(output_dir) / "focus_result.npz"),
            str(Path(output_dir) / "initial_vs_final_camera.png"),
            str(Path(output_dir) / "enhancement_history.png"),
        ],
    }


def run_focus_trials(settings: FocusSettings = FocusSettings(), trials: int = 5,
                     output_dir: str | Path = "outputs/trials") -> dict:
    """Run independent trials with different random seeds and save a summary."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    root = Path(output_dir)
    seed_rng = __import__("numpy").random.default_rng()
    results = []
    for index in range(1, trials + 1):
        seed = int(seed_rng.integers(0, 2**32 - 1))
        trial_settings = FocusSettings(**{**asdict(settings), "seed": seed})
        result = run_focus_experiment(trial_settings, root / f"trial_{index:02d}")
        results.append({"trial": index, "seed": seed, **result})
    enhancements = [float(item["final_enhancement"]) for item in results]
    summary = {
        "trials": results,
        "mean_final_enhancement": float(__import__("numpy").mean(enhancements)),
        "best_final_enhancement": float(max(enhancements)),
        "worst_final_enhancement": float(min(enhancements)),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
