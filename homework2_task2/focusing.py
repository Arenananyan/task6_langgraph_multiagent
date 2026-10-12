"""Iterative amplitude-only focusing on the simulated optical bench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .simulator import BenchConfig, Measurement, SimulatedBench
except ImportError:
    from simulator import BenchConfig, Measurement, SimulatedBench


@dataclass(frozen=True)
class FocusResult:
    initial_pattern: np.ndarray
    final_pattern: np.ndarray
    initial: Measurement
    final: Measurement
    history: np.ndarray


def focus(bench: SimulatedBench, iterations: int = 500) -> FocusResult:
    """Coordinate search using fresh noisy camera measurements only.

    Remeasure the incumbent each step to avoid keeping a lucky noisy score.
    Independently confirm the winning proposal before accepting it.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    pattern = bench.rng.random(bench.config.dmd_shape)
    initial_pattern = pattern.copy()
    initial = bench.measure(pattern)
    current = bench.measure(pattern)
    amplitude_levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    history = np.empty(iterations + 1, dtype=np.float64)
    history[0] = current.enhancement

    for step in range(1, iterations + 1):
        row = int(bench.rng.integers(bench.config.dmd_shape[0]))
        col = int(bench.rng.integers(bench.config.dmd_shape[1]))
        current = bench.measure(pattern)
        best_pattern = pattern
        best_measurement = current
        for amplitude in amplitude_levels:
            candidate = pattern.copy()
            candidate[row, col] = amplitude
            proposal = bench.measure(candidate)
            if proposal.enhancement > best_measurement.enhancement:
                best_pattern = candidate
                best_measurement = proposal
        # Fresh confirmation reduces selection bias across the five proposals.
        if best_pattern is not pattern:
            confirmation = bench.measure(best_pattern)
            baseline = bench.measure(pattern)
            if confirmation.enhancement > baseline.enhancement:
                pattern = best_pattern
                current = confirmation
            else:
                current = baseline
        history[step] = current.enhancement

    final = bench.measure(pattern)

    return FocusResult(initial_pattern, pattern, initial, final, history)


def save_result(result: FocusResult, output_dir: Path) -> None:
    """Save reproducible numeric results and diagnostic images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "focus_result.npz",
        initial_intensity=result.initial.intensity,
        final_intensity=result.final.intensity,
        initial_pattern=result.initial_pattern,
        final_pattern=result.final_pattern,
        enhancement_history=result.history,
        target=np.array([
            result.final.target.row_start, result.final.target.row_stop,
            result.final.target.col_start, result.final.target.col_stop,
        ]),
    )

    import matplotlib.pyplot as plt

    vmax = max(float(result.initial.intensity.max()), float(result.final.intensity.max()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor="white",
                             constrained_layout=True)
    image = None
    for ax, measurement, title in (
        (axes[0], result.initial, "Initial camera intensity"),
        (axes[1], result.final, "Final camera intensity"),
    ):
        image = ax.imshow(measurement.intensity, cmap="inferno", vmin=0, vmax=vmax,
                          interpolation="nearest", origin="lower")
        ax.add_patch(plt.Rectangle(
            (measurement.target.col_start - 0.5, measurement.target.row_start - 0.5),
            measurement.target.col_stop - measurement.target.col_start,
            measurement.target.row_stop - measurement.target.row_start,
            fill=False, edgecolor="#22d3ee", linewidth=2.0,
        ))
        ax.set_title(f"{title}\nEnhancement  {measurement.enhancement:.3f}",
                     fontsize=13, fontweight="bold", pad=9)
        ax.set_xlabel("Camera x pixel")
        ax.set_ylabel("Camera y pixel")
        ax.set_xlim(-0.5, measurement.intensity.shape[1] - 0.5)
        ax.set_ylim(-0.5, measurement.intensity.shape[0] - 0.5)
        ax.set_xticks(np.arange(0, measurement.intensity.shape[1], 8))
        ax.set_yticks(np.arange(0, measurement.intensity.shape[0], 8))
    cbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Measured intensity (normalized units)")
    fig.suptitle("Simulated scattering bench: initial and focused camera intensity",
                 fontsize=16, fontweight="bold")
    fig.savefig(output_dir / "initial_vs_final_camera.png", dpi=220,
                facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.history, color="tab:blue")
    ax.set(xlabel="Iteration", ylabel="Enhancement", title="Focusing progress")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "enhancement_history.png", dpi=180)
    plt.close(fig)


def main() -> None:
    # Five independent trials make the reported result less dependent on one
    # lucky random transmission matrix and target location.
    from focus_tool import FocusSettings, run_focus_trials

    output_dir = Path(__file__).resolve().parent / "outputs" / "trials"
    summary = run_focus_trials(FocusSettings(iterations=1000), trials=5,
                               output_dir=output_dir)
    for item in summary["trials"]:
        print(f"TRIAL_{item['trial']}: seed={item['seed']} "
              f"initial={item['initial_enhancement']:.6f} "
              f"final={item['final_enhancement']:.6f}")
    print(f"MEAN_FINAL_ENHANCEMENT: {summary['mean_final_enhancement']:.6f}")
    print(f"BEST_FINAL_ENHANCEMENT: {summary['best_final_enhancement']:.6f}")
    print(f"SAVED_OUTPUTS: {output_dir}")


if __name__ == "__main__":
    main()
