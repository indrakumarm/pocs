"""Plot a replica/topology result directory, keeping workloads and arrival modes separate."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(directory):
    paths = sorted(directory.glob("*-closed.csv")) + sorted(directory.glob("*-open.csv"))
    if not paths:
        raise ValueError(f"No benchmark CSVs in {directory}")
    outputs = []
    for path in paths:
        frame = pd.read_csv(path)
        mode = frame["mode"].iloc[0]
        load = "concurrency" if mode == "closed" else "request_rate"
        xlabel = "Total client concurrency" if mode == "closed" else "Offered requests/s"
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for tag, group in frame.groupby("tag", sort=False):
            group = group.sort_values(load)
            axes[0, 0].plot(group[load], group.agg_tok_s, marker="o", label=tag)
            axes[0, 1].plot(group[load], group.ttft_p99_ms / 1000, marker="o", label=tag)
            axes[1, 0].plot(group[load], group.itl_p99_ms, marker="o", label=tag)
            axes[1, 1].plot(group[load], group.succeeded / group.requests * 100,
                            marker="o", label=tag)
        titles = ("Successful output tokens/s", "TTFT p99 (s)",
                  "Stream chunk gap p99 (ms)", "Successful requests (%)")
        for axis, title in zip(axes.flat, titles):
            axis.set(xlabel=xlabel, title=title)
            axis.grid(alpha=.3)
            axis.legend(fontsize=8)
        axes[1, 1].set_ylim(0, 105)
        fig.suptitle(f"{frame.workload.iloc[0]} / {mode} (fixed-count, includes drain time)")
        fig.tight_layout()
        output = path.with_suffix(".png")
        fig.savefig(output, dpi=120)
        plt.close(fig)
        outputs.append(output)
        print(f"Wrote {output}")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    plot_results(args.directory)


if __name__ == "__main__":
    main()
