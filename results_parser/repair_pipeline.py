import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import sys

PIPELINE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

COLORS = [
    "#3266ad", "#d9534f", "#3b6d11", "#ba7517",
    "#7f77dd", "#0f6e56", "#993c1d", "#993556",
]

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [r for r in records if "repair_seconds" in r]

jsonl_files = sorted(PIPELINE_DIR.glob("*.jsonl"))
if not jsonl_files:
    raise FileNotFoundError(f"No .jsonl files found in {PIPELINE_DIR}")

methods = {}
for path in jsonl_files:
    records = load_jsonl(path)
    if records:
        methods[path.stem] = records

n_methods = len(methods)
if n_methods == 0:
    raise ValueError("No valid repair records found in any .jsonl file.")

fig = plt.figure(figsize=(14, 5 + 4 * n_methods))
fig.suptitle("Repair pipeline — 2 way line (20, 10) det", fontsize=14, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(n_methods + 1, 2, figure=fig, hspace=0.55, wspace=0.35)

bar_width = 0.8 / max(n_methods, 1)

for m_idx, (method_name, records) in enumerate(methods.items()):
    color      = COLORS[m_idx % len(COLORS)]
    num_traces = [r["num_traces"] * r["iteration"] for r in records]
    train_goal = [r["TrainGoalFrac"] for r in records]
    train_fail = [r["TrainFailFrac"] for r in records]
    train_unsafe = [r["TrainUnsafeFrac"] for r in records]
    eval_goal  = [r["EvalGoalFrac"]  for r in records]
    eval_fail  = [r["EvalFailFrac"]  for r in records]
    eval_unsafe = [r["EvalUnsafeFrac"] for r in records]

    ax = fig.add_subplot(gs[m_idx, :])
    ax.plot(num_traces, train_goal, marker="o", label="Train goal", color="#3266ad",     linewidth=2)
    ax.plot(num_traces, train_fail, marker="o", label="Train fail", color="#d9534f", linewidth=2)
    ax.plot(num_traces, train_unsafe, marker="o", label="Train unsafe", color="#0f6e56", linewidth=2)
    ax.plot(num_traces, eval_goal,  marker="s", label="Eval goal",  color="#7f77dd",     linewidth=2, linestyle="--", alpha=0.6)
    ax.plot(num_traces, eval_fail,  marker="s", label="Eval fail",  color="#f0a0a0", linewidth=2, linestyle="--", alpha=0.6)
    ax.plot(num_traces, eval_unsafe, marker="s", label="Eval unsafe", color="#3b6d11", linewidth=2, linestyle="--", alpha=0.6)
    ax.set_xlabel("Number of episodes (cumulative)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(num_traces)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{method_name} — train / eval fractions", fontsize=11)
    ax.grid(True, alpha=0.3)

# ── bottom left: faults (stacked bars per method) ────────────────────────────
ax_faults = fig.add_subplot(gs[n_methods, 0])

for m_idx, (method_name, records) in enumerate(methods.items()):
    color       = COLORS[m_idx % len(COLORS)]
    num_traces  = [r["num_traces"] * r["iteration"] for r in records]
    faults_cum  = [r["num_faults"] for r in records]
    faults_iter = [r["it_faults"]  for r in records]
    faults_base = [c - i for c, i in zip(faults_cum, faults_iter)]

    x_m    = np.arange(len(num_traces))
    offset = (m_idx - n_methods / 2 + 0.5) * bar_width

    # Solid bars for previous (accumulated) faults
    ax_faults.bar(x_m + offset, faults_base, bar_width,
                  label=method_name, color=color, alpha=0.85)

    # Hatched bars for new faults this iteration, same color
    ax_faults.bar(x_m + offset, faults_iter, bar_width, color=color, alpha=0.4,
                  bottom=faults_base, hatch="///", edgecolor=color, linewidth=0.5)

    ax_faults.set_xticks(x_m)
    ax_faults.set_xticklabels(num_traces)

ax_faults.set_xlabel("Number of episodes (cumulative)")
ax_faults.set_ylabel("Faults")
ax_faults.set_title("Faults fixed (solid = prev, hatched = new this iter)", fontsize=11)
ax_faults.legend(fontsize=8, ncol=1)
ax_faults.grid(True, alpha=0.3, axis="y")

# ── bottom right: repair runtime ──────────────────────────────────────────────
ax_rt = fig.add_subplot(gs[n_methods, 1])

for m_idx, (method_name, records) in enumerate(methods.items()):
    color       = COLORS[m_idx % len(COLORS)]
    repair_secs = [r["repair_seconds"] for r in records]
    x_m         = np.arange(len(repair_secs))
    offset      = (m_idx - n_methods / 2 + 0.5) * bar_width
    ax_rt.bar(x_m + offset, repair_secs, bar_width,
              label=method_name, color=color, alpha=0.85)
    ax_rt.set_xticks(x_m)
    ax_rt.set_xticklabels([r["num_traces"] * r["iteration"] for r in records])

ax_rt.set_xlabel("Number of episodes (cumulative)")
ax_rt.set_ylabel("Seconds (log scale)")
ax_rt.set_yscale('log')
ax_rt.set_title("Repair runtime per method", fontsize=11)
ax_rt.legend(fontsize=9)
ax_rt.grid(True, alpha=0.3, axis="y")

plt.savefig("repair_pipeline_comparison.png", dpi=150, bbox_inches="tight")
plt.show()