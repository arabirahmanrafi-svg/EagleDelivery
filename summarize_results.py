import argparse
import csv
from pathlib import Path
import statistics


def read_csv(filename):
    with open(filename, newline="") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description="Summarize saved training and evaluation CSVs")
    parser.add_argument("folder")
    parser.add_argument("--block", type=int, default=100)
    args = parser.parse_args()
    if args.block < 1:
        parser.error("Block size must be positive.")
    folder = Path(args.folder)
    summaries = read_csv(folder / "summary.csv")
    if not summaries:
        parser.error("Summary has no runs.")
    print("Evaluation across", len(summaries), "independent training seeds:")
    for field in ("success_rate", "mean_return"):
        values = [float(row[field]) for row in summaries]
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(field, "mean =", round(statistics.mean(values), 4),
              "sample SD =", round(sd, 4))
    print("Per-seed mean_steps_success:", [row["mean_steps_success"] for row in summaries])

    training = [read_csv(folder / ("training_seed" + row["seed"] + ".csv"))
                for row in summaries]
    lengths = {len(run) for run in training}
    if len(lengths) != 1:
        parser.error("Training runs must have the same episode count.")
    curve = []
    for start in range(0, len(training[0]), args.block):
        end = min(start + args.block, len(training[0]))
        success_rates = [statistics.mean(float(r["success"]) for r in run[start:end])
                         for run in training]
        mean_returns = [statistics.mean(float(r["return"]) for r in run[start:end])
                        for run in training]
        curve.append({"episode_end": end,
                      "mean_training_success_rate": statistics.mean(success_rates),
                      "sd_training_success_rate": (statistics.stdev(success_rates)
                                                   if len(success_rates) > 1 else 0.0),
                      "mean_training_return": statistics.mean(mean_returns)})
    destination = folder / "learning_curve.csv"
    with open(destination, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    print("Wrote", destination)
    print("Plot episode_end against mean_training_success_rate.")


if __name__ == "__main__":
    main()
