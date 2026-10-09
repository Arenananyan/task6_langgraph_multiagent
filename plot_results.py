"""Create scientific plots for the Silverbox Model E evaluation."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "outputs" / "plots"

sys.path.insert(0, str(ROOT / "task5_reference" / "scr"))

from model import AccelerationNetwork
from integrator import rollout


def save_plot(filename):
    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("SAVED:", path)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = (
        ROOT
        / "task5_reference"
        / "data"
        / "processed"
        / "silverbox_excerpt.npz"
    )

    with np.load(data_path) as data:
        u_np = data["u"].copy()
        y_np = data["y"].copy()
        fs = float(data["fs"])
        train_size = int(data["train_size"])

    checkpoint_path = (
        ROOT
        / "task5_reference"
        / "results"
        / "checkpoints"
        / "model_e_w128_s2000.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = AccelerationNetwork(
        hidden_size=int(checkpoint["hidden_size"])
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    u = torch.tensor(u_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    dt = 1.0 / fs
    y0 = y[0]
    v0 = (y[1] - y[0]) / dt

    with torch.no_grad():
        predicted, _ = rollout(model, y0, v0, u, dt)

    measured = y.numpy()
    predicted = predicted.numpy()

    time = np.arange(len(measured)) / fs
    error = predicted - measured
    split_time = train_size / fs

    train_rmse = float(
        np.sqrt(np.mean(error[:train_size] ** 2))
    )
    test_rmse = float(
        np.sqrt(np.mean(error[train_size:] ** 2))
    )

    print(f"TRAIN_RMSE_V: {train_rmse:.9f}")
    print(f"TEST_RMSE_V: {test_rmse:.9f}")

    # 1. Actual and predicted output
    plt.figure(figsize=(12, 5))
    plt.plot(time, measured, label="Measured", linewidth=1)
    plt.plot(
        time,
        predicted,
        label="Model E prediction",
        linewidth=1,
        alpha=0.8,
    )
    plt.axvline(
        split_time,
        linestyle="--",
        color="black",
        label="Train / test boundary",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Output voltage (V)")
    plt.title("Silverbox: Measured vs Predicted Output")
    plt.legend()
    plt.grid(alpha=0.25)
    save_plot("01_measured_vs_predicted.png")

    # 2. Prediction error
    plt.figure(figsize=(12, 4))
    plt.plot(time, error, linewidth=0.9)
    plt.axhline(0, linestyle="--", color="black")
    plt.axvline(split_time, linestyle="--", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Prediction error (V)")
    plt.title("Silverbox: Prediction Error Over Time")
    plt.grid(alpha=0.25)
    save_plot("02_prediction_error.png")

    # 3. Train and test RMSE
    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Training", "Testing"],
        [train_rmse, test_rmse],
    )
    plt.bar_label(
        bars,
        labels=[
            f"{train_rmse:.6f} V",
            f"{test_rmse:.6f} V",
        ],
        padding=4,
    )
    plt.ylabel("RMSE (V)")
    plt.title("Silverbox Model E: Train vs Test RMSE")
    plt.grid(axis="y", alpha=0.25)
    save_plot("03_train_test_rmse.png")

    # 4. Measured vs predicted scatter on the test set
    actual_test = measured[train_size:]
    predicted_test = predicted[train_size:]

    limits = [
        min(actual_test.min(), predicted_test.min()),
        max(actual_test.max(), predicted_test.max()),
    ]

    plt.figure(figsize=(7, 6))
    plt.scatter(
        actual_test,
        predicted_test,
        s=9,
        alpha=0.55,
        label="Test samples",
    )
    plt.plot(
        limits,
        limits,
        linestyle="--",
        color="black",
        label="Perfect prediction",
    )
    plt.xlabel("Measured output (V)")
    plt.ylabel("Predicted output (V)")
    plt.title("Silverbox: Test Set Prediction Agreement")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.axis("equal")
    plt.xlim(limits)
    plt.ylim(limits)
    save_plot("04_test_prediction_scatter.png")

    assert np.isfinite(train_rmse)
    assert np.isfinite(test_rmse)

    expected_rmse = 0.02451591193675995
    assert abs(test_rmse - expected_rmse) < 1e-6, (
        f"RMSE mismatch: {test_rmse} vs {expected_rmse}"
    )

    print("ALL_FOUR_PLOTS_CREATED")
    print("RMSE_MATCH_CONFIRMED")


if __name__ == "__main__":
    main()
