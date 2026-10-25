import csv
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCK = 20      # episodes per point on the curves
SHOW = 600      # plot the first 600 episodes, where learning happens
FIGURES = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")


def read_runs(folder):
    runs = []
    for seed in (0, 1, 2):
        with open(Path("results") / folder / f"training_seed{seed}.csv") as f:
            runs.append([{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)])
    return runs


def plot(groups, title, filename):
    plt.figure(figsize=(7, 4))
    for label, folder in groups:
        runs = read_runs(folder)
        x, y = [], []
        for start in range(0, SHOW, BLOCK):
            block = [r["steps"] for run in runs for r in run[start:start + BLOCK]]
            x.append(start + BLOCK)
            y.append(statistics.mean(block))
        plt.plot(x, y, marker=".", label=label)
    plt.yscale("log")
    plt.xlabel("Training episode")
    plt.ylabel(f"Mean steps per episode ({BLOCK}-episode blocks, 3 seeds)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=200)
    plt.close()
    print("Saved", FIGURES / filename)


def stats(groups):
    print(f"{'run':<14}{'first success':>14}{'truncated':>11}{'total actions':>15}"
          f"{'steps last 500':>16}{'return last 500':>17}{'best episode':>14}")
    for label, folder in groups:
        runs = read_runs(folder)
        first = [next(r["episode"] for r in run if r["success"]) for run in runs]
        trunc = [sum(r["truncated"] for r in run) for run in runs]
        total = [sum(r["steps"] for r in run) for run in runs]
        last_steps = [statistics.mean(r["steps"] for r in run[-500:]) for run in runs]
        last_return = [statistics.mean(r["return"] for r in run[-500:]) for run in runs]
        best = [min(r["steps"] for r in run) for run in runs]
        print(f"{label:<14}{statistics.mean(first):>14.1f}{statistics.mean(trunc):>11.1f}"
              f"{statistics.mean(total):>15.0f}{statistics.mean(last_steps):>16.1f}"
              f"{statistics.mean(last_return):>17.1f}{str(best):>14}")


FIGURES.mkdir(exist_ok=True)
baseline = [("S-W", "baseline_SW"), ("L-D", "baseline_LD"), ("H-P", "baseline_HP")]
custom = [("epsilon 0.05", "custom_e005"), ("epsilon 0.2", "custom_e020"),
          ("epsilon 0.5", "custom_e050")]
plot(baseline, "Baseline Q-learning on emu_campus", "baseline_training.png")
plot(custom, "my_campus S-W: effect of epsilon", "custom_epsilon_training.png")
print("\nBaseline (means over seeds 0, 1, 2):")
stats(baseline)
print("\nCustom map epsilon experiment (means over seeds 0, 1, 2):")
stats(custom)