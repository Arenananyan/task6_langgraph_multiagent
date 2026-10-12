# Homework 2 — Task 2: Simulated scattering bench

This folder implements the simulated optical bench from Lecture 2. A complex transmission matrix propagates a DMD amplitude pattern to a camera:

\[
E_{out}=T E_{in}, \qquad I_{out}=|E_{out}|^2.
\]

The experiment uses a `16x16` DMD, a `64x64` camera, a random `5x5` target region, four adjacent background regions, 2% Gaussian camera noise, and 1000 focusing iterations. The optimizer performs discrete coordinate descent over amplitude values `0, 0.25, 0.5, 0.75, 1`.

## Run the tests

```bash
python3 -m unittest homework2_task2.test_simulator -v
```

## Run five independent trials

```bash
python3 homework2_task2/focusing.py
```

Each trial saves `initial_vs_final_camera.png`, `enhancement_history.png`, and `focus_result.npz` under `outputs/trials/trial_XX/`; the aggregate results are saved in `outputs/trials/summary.json`.
