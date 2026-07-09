"""
Parse and display stats from artifacts/action_shield_multitask.

Usage:
    python parse_shield_stats.py [--dir PATH]

Outputs two tables:
  1. Baseline vs Shielded comparison (goal / failure / cycle rates) per job.
  2. Per-action classifier stats (all-applicable unsafe pairs, policy-selected
     faults, test faults, fixed, missed, fix rate, precision, wrongly blocked).
"""

import argparse
import json
import os
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_pct(v):
    return f"{v:.2f}%" if v is not None else "N/A"

def fmt_int(v):
    return f"{v:,}" if v is not None else "N/A"

def col_width(rows, header, key, fmt_fn=str):
    vals = [fmt_fn(r.get(key)) for r in rows]
    return max(len(header), max((len(v) for v in vals), default=0))

def print_table(headers, rows, fmt_fns=None):
    """Print a simple fixed-width ASCII table."""
    if fmt_fns is None:
        fmt_fns = [str] * len(headers)
    widths = []
    for i, h in enumerate(headers):
        col_vals = [fmt_fns[i](r[i]) for r in rows]
        widths.append(max(len(h), max((len(v) for v in col_vals), default=0)))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    row_fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)
    for r in rows:
        print(row_fmt.format(*[fmt_fns[i](r[i]) for i in range(len(headers))]))
    print(sep)


# ── loader ────────────────────────────────────────────────────────────────────

def load_summary(job_dir: Path):
    summary_path = job_dir / "evaluation" / "merged" / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def collect_jobs(base_dir: Path):
    """Return list of (job_name, summary_dict) for all complete jobs."""
    jobs = []
    for env_dir in sorted(base_dir.iterdir()):
        if not env_dir.is_dir():
            continue
        for run_dir in sorted(env_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary = load_summary(run_dir)
            job_name = f"{env_dir.name}"
            if summary is None:
                print(f"[SKIP] {job_name} — no evaluation/merged/summary.json")
                continue
            jobs.append((job_name, summary))
    return jobs


# ── table 1: baseline vs shielded ────────────────────────────────────────────

def table_baseline_vs_shielded(jobs):
    print("\n" + "=" * 80)
    print("TABLE 1: Baseline vs Shielded — Goal / Failure / Cycle rates (%)")
    print("=" * 80)

    headers = [
        "Job",
        "B: Goal%", "S: Goal%", "Δ Goal",
        "B: Fail%", "S: Fail%", "Δ Fail",
        "B: Cycle%", "S: Cycle%", "Δ Cycle",
        "Episodes",
    ]

    rows = []
    for job_name, s in jobs:
        bm = s.get("baseline_metrics", {})
        sm = s.get("shielded_metrics", {})

        b_goal   = bm.get("goal_percent", 0.0)
        s_goal   = sm.get("goal_percent", 0.0)
        b_fail   = bm.get("avoid_percent", 0.0)
        s_fail   = sm.get("avoid_percent", 0.0)
        b_cycle  = bm.get("cycle_percent", 0.0)
        s_cycle  = sm.get("cycle_percent", 0.0)
        episodes = s.get("processed_start_states", "?")

        rows.append([
            job_name,
            b_goal,   s_goal,   s_goal  - b_goal,
            b_fail,   s_fail,   s_fail  - b_fail,
            b_cycle,  s_cycle,  s_cycle - b_cycle,
            episodes,
        ])

    def pct(v):
        if isinstance(v, str):
            return v
        return f"{v:+.2f}%" if isinstance(v, float) else f"{v:.2f}%"

    def pct_plain(v):
        if isinstance(v, str):
            return v
        return f"{v:.2f}%"

    def intfmt(v):
        if v is None:
            return "N/A"
        return f"{v:,}" if isinstance(v, int) else str(v)

    fmt = [
        str,           # job
        pct_plain, pct_plain, pct,   # goal
        pct_plain, pct_plain, pct,   # fail
        pct_plain, pct_plain, pct,   # cycle
        intfmt,                      # episodes
    ]
    print_table(headers, rows, fmt)

    # Summary legend
    print("  B = Baseline  |  S = Shielded  |  Δ = S − B  (positive = worse for failure/cycle)")


# ── table 2: per-action classifier stats ─────────────────────────────────────

def table_per_action(jobs):
    print("\n" + "=" * 80)
    print("TABLE 2: Per-Action Classifier Stats")
    print("=" * 80)

    headers = [
        "Job", "Action",
        "Kind",
        "All Unsafe Pairs",
        "Policy Faults",
        "Test Faults",
        "Fixed",
        "Missed",
        "Fix Rate%",
        "Precision%",
        "Wrongly Blocked",
        "Runtime Blocks",
    ]

    rows = []
    for job_name, s in jobs:
        per_action = s.get("per_action_evaluation", [])
        for ae in per_action:
            fix_rate  = ae.get("held_out_fix_rate_percent")
            precision = ae.get("held_out_precision_percent")
            rows.append([
                job_name,
                ae.get("action_name", f"action_{ae.get('action')}"),
                ae.get("classifier_kind", "?"),
                ae.get("collected_unique_faults", 0),
                ae.get("policy_selected_unique_faults"),
                ae.get("held_out_test_faults", 0),
                ae.get("held_out_faults_fixed", 0),
                ae.get("held_out_faults_missed", 0),
                fix_rate,
                precision,
                ae.get("held_out_safe_actions_wrongly_blocked", 0),
                ae.get("runtime_block_occurrences", 0),
            ])

    def pct_or_na(v):
        return f"{v:.2f}%" if v is not None else "N/A"

    def intfmt(v):
        if v is None:
            return "N/A"
        return f"{v:,}" if isinstance(v, int) else str(v)

    fmt = [
        str,      # job
        str,      # action name
        str,      # kind
        intfmt,   # unique faults
        intfmt,   # policy-selected unique faults
        intfmt,   # test faults
        intfmt,   # fixed
        intfmt,   # missed
        pct_or_na, # fix rate
        pct_or_na, # precision
        intfmt,   # wrongly blocked
        intfmt,   # runtime blocks
    ]
    print_table(headers, rows, fmt)


# ── table 3: quick shield effectiveness summary ───────────────────────────────

def table_shield_effectiveness(jobs):
    print("\n" + "=" * 80)
    print("TABLE 3: Shield Effectiveness Summary")
    print("=" * 80)

    headers = [
        "Job",
        "B: Fail%", "S: Fail%", "Fail Reduction%",
        "B: Goal%", "S: Goal%", "Goal Improvement%",
        "States w/ Block%",
        "All-Blocked Episodes",
    ]

    rows = []
    for job_name, s in jobs:
        bm = s.get("baseline_metrics", {})
        sm = s.get("shielded_metrics", {})
        b_fail  = bm.get("avoid_percent", 0.0)
        s_fail  = sm.get("avoid_percent", 0.0)
        b_goal  = bm.get("goal_percent", 0.0)
        s_goal  = sm.get("goal_percent", 0.0)

        fail_red = b_fail - s_fail   # positive = good
        goal_imp = s_goal - b_goal   # positive = good

        block_pct      = s.get("states_with_any_classifier_block_percent", 0.0)
        all_blocked_ep = sm.get("all_blocked_episode_percent", 0.0)

        rows.append([
            job_name,
            b_fail, s_fail, fail_red,
            b_goal, s_goal, goal_imp,
            block_pct,
            all_blocked_ep,
        ])

    def pct(v): return f"{v:.2f}%"
    def pct_delta(v): return f"{v:+.2f}%"

    fmt = [
        str,
        pct, pct, pct_delta,
        pct, pct, pct_delta,
        pct,
        pct,
    ]
    print_table(headers, rows, fmt)
    print("  Fail Reduction%  = B_Fail − S_Fail  (positive = shield reduces failures)")
    print("  Goal Improvement% = S_Goal − B_Goal  (positive = shield improves goal rate)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse action_shield_multitask stats")
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).parent / "artifacts" / "action_shield_multitask"),
        help="Path to the action_shield_multitask directory",
    )
    parser.add_argument(
        "--table", choices=["1", "2", "3", "all"], default="all",
        help="Which table(s) to print (default: all)",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return

    print(f"Scanning: {base_dir}")
    jobs = collect_jobs(base_dir)
    print(f"Found {len(jobs)} complete job(s).\n")

    if not jobs:
        print("No jobs with evaluation results found.")
        return

    if args.table in ("1", "all"):
        table_baseline_vs_shielded(jobs)
    if args.table in ("2", "all"):
        table_per_action(jobs)
    if args.table in ("3", "all"):
        table_shield_effectiveness(jobs)


if __name__ == "__main__":
    main()
