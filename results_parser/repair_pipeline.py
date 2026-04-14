import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import re

COLORS = [
    "#3266ad", "#d9534f", "#3b6d11", "#ba7517",
    "#7f77dd", "#0f6e56", "#993c1d", "#993556",
]

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [r for r in records]

def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "comparison"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot repair pipeline metrics from one or more log directories."
    )
    parser.add_argument(
        "pipeline_dirs",
        nargs="+",
        type=Path,
        help="One or more directories containing .jsonl log files.",
    )
    parser.add_argument(
        "--title",
        default="Repair pipeline comparison",
        help="Title shown at the top of the plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image filename (for example plots/my_run.png).",
    )
    return parser.parse_args()


def find_jsonl_files(pipeline_dirs):
    jsonl_files = []
    for pipeline_dir in pipeline_dirs:
        if not pipeline_dir.exists() or not pipeline_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {pipeline_dir}")
        jsonl_files.extend(sorted(pipeline_dir.glob("*.jsonl")))
    return sorted(jsonl_files)


def build_methods(jsonl_files, include_parent_prefix):
    methods = {}
    for path in jsonl_files:
        records = load_jsonl(path)
        if not records:
            continue

        method_name = f"{path.parent.name}/{path.stem}" if include_parent_prefix else path.stem

        if method_name in methods:
            suffix = 2
            candidate = f"{method_name}_{suffix}"
            while candidate in methods:
                suffix += 1
                candidate = f"{method_name}_{suffix}"
            method_name = candidate

        methods[method_name] = records
        # methods = {k: v for k, v in methods.items() if not k.startswith("retain_unlearn")}
    return methods


def default_output_path(title):
    return Path(f"repair_pipeline_{slugify(title)}.png")


def main():
    args = parse_args()
    jsonl_files = find_jsonl_files(args.pipeline_dirs)
    if not jsonl_files:
        joined_dirs = ", ".join(str(p) for p in args.pipeline_dirs)
        print(f"No .jsonl files found in: {joined_dirs}")
        exit()

    methods = build_methods(jsonl_files, include_parent_prefix=len(args.pipeline_dirs) > 1)
    n_methods = len(methods)
    if n_methods == 0:
        raise ValueError("No valid repair records found in any .jsonl file.")

    output_path = args.output if args.output is not None else default_output_path(args.title)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 5 + 4 * n_methods))
    fig.suptitle(args.title, fontsize=14, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(n_methods + 1, 2, figure=fig, hspace=0.55, wspace=0.35)
    bar_width = 0.8 / max(n_methods, 1)

    for m_idx, (method_name, records) in enumerate(methods.items()):
        num_traces = [r["num_traces"] * r["iteration"] for r in records]
        train_goal = [r["TrainGoalFrac"] for r in records]
        train_fail = [r["TrainFailFrac"] for r in records]
        train_unsafe = [r["TrainUnsafeFrac"] for r in records]
        eval_goal = [r["EvalGoalFrac"] for r in records]
        eval_fail = [r["EvalFailFrac"] for r in records]
        eval_unsafe = [r["EvalUnsafeFrac"] for r in records]

        ax = fig.add_subplot(gs[m_idx, :])
        ax.plot(num_traces, train_goal, marker="o", label="Train goal", color="#3266ad", linewidth=2)
        ax.plot(num_traces, train_fail, marker="o", label="Train fail", color="#d9534f", linewidth=2)
        # ax.plot(num_traces, train_unsafe, marker="o", label="Train unsafe", color="#0f6e56", linewidth=2)
        ax.plot(num_traces, eval_goal, marker="s", label="Eval goal", color="#7f77dd", linewidth=2, linestyle="--", alpha=0.6)
        ax.plot(num_traces, eval_fail, marker="s", label="Eval fail", color="#f0a0a0", linewidth=2, linestyle="--", alpha=0.6)
        # ax.plot(num_traces, eval_unsafe, marker="s", label="Eval unsafe", color="#3b6d11", linewidth=2, linestyle="--", alpha=0.6)
        ax.set_xlabel("Number of episodes (cumulative)")
        ax.set_ylabel("Fraction")
        ax.set_ylim(-0.05, 1.1)
        # ax.set_xticks(num_traces)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{method_name} - train / eval fractions", fontsize=11)
        ax.grid(True, alpha=0.3)

    ax_faults = fig.add_subplot(gs[n_methods, 0])
    for m_idx, (method_name, records) in enumerate(methods.items()):
        color = COLORS[m_idx % len(COLORS)]
        x_vals = [r["num_traces"] * r["iteration"] for r in records]
        faults_cum = [r["num_faults"] for r in records]
        ax_faults.plot(x_vals, faults_cum, label=method_name, color=color, marker="o", markersize=3)

    longest_records = max(methods.values(), key=len)
    x_max = longest_records[-1]["num_traces"] * longest_records[-1]["iteration"]
    ax_faults.set_xlim(0, x_max)

    ax_faults.set_xlabel("Number of episodes (cumulative)")
    ax_faults.set_ylabel("Faults")
    ax_faults.set_title("Faults fixed", fontsize=11)
    ax_faults.legend(fontsize=8, ncol=1)
    ax_faults.grid(True, alpha=0.3, axis="y")

    if "repair_seconds" in records[0]:
        ax_rt = fig.add_subplot(gs[n_methods, 1])
        for m_idx, (method_name, records) in enumerate(methods.items()):
            color = COLORS[m_idx % len(COLORS)]
            repair_secs = [r["repair_seconds"] for r in records[:-1]]
            x_vals = [r["num_traces"] * r["iteration"] for r in records[:-1]]
            ax_rt.plot(x_vals, repair_secs, label=method_name, color=color, marker="o", markersize=3)

        longest_records = max(methods.values(), key=len)
        x_max = longest_records[-2]["num_traces"] * longest_records[-2]["iteration"]  # -2 because you slice [:-1]
        ax_rt.set_xlim(0, x_max)

        ax_rt.set_xlabel("Number of episodes (cumulative)")
        ax_rt.set_ylabel("Seconds (log scale)")
        ax_rt.set_yscale("log")
        ax_rt.set_title("Repair runtime per method", fontsize=11)
        ax_rt.legend(fontsize=9)
        ax_rt.grid(True, alpha=0.3, axis="y")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()