#!/usr/bin/env python3
"""Evaluate the original policy with per-action classifier shielding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.repair_wo_domain.action_fault_shield import ActionFaultShield
from experiments.repair_wo_domain.collect_policy_faults import load_policy_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--jani-model", required=True)
    evaluate.add_argument("--jani-property", default="")
    evaluate.add_argument("--start-states", required=True)
    evaluate.add_argument("--policy", required=True)
    evaluate.add_argument("--classifiers-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--max-steps", type=int, default=2000)
    evaluate.add_argument("--max-state-visits", type=int, default=3)
    evaluate.add_argument("--num-shards", type=int, default=1)
    evaluate.add_argument("--shard-id", type=int, default=0)
    evaluate.add_argument("--start-index-limit", type=int, default=None)
    evaluate.add_argument("--progress-every", type=int, default=25)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input-dir", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--expected-shards", type=int, default=None)
    merge.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_env(args: argparse.Namespace, seed_offset: int) -> Any:
    from jani.env import JANIEnv

    return JANIEnv(
        jani_model_path=args.jani_model,
        jani_property_path=args.jani_property,
        start_states_path=args.start_states,
        objective_path="",
        failure_property_path="",
        seed=args.seed + seed_offset,
        goal_reward=1.0,
        failure_reward=-1.0,
        unsafe_reward=-0.01,
        use_oracle=False,
        reduced_memory_mode=True,
    )


def select_action(
    policy: torch.nn.Module,
    observation: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
) -> int:
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
    if not bool(mask.any()):
        raise ValueError("Cannot select an action from an empty mask")
    observation_tensor = torch.as_tensor(
        observation, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        logits = policy(observation_tensor).squeeze(0)
        return int(logits.masked_fill(~mask, float("-inf")).argmax().item())


def terminal_name(done: bool, info: dict[str, Any]) -> str | None:
    if not done:
        return None
    if info.get("reached_goal", False):
        return "goal"
    if info.get("reached_fail", False):
        return "failure"
    return "dead_end"


def baseline_rollout(
    env: Any,
    policy: torch.nn.Module,
    start_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[str, int]:
    observation, _ = env.reset(options={"idx": start_index})
    visits: Counter[bytes] = Counter()
    for step in range(args.max_steps):
        key = np.asarray(observation, dtype=np.float32).tobytes()
        if visits[key] >= args.max_state_visits:
            return "cycle", step
        visits[key] += 1
        applicable = np.asarray(env.action_mask(), dtype=bool)
        if not applicable.any():
            return "dead_end", step
        action = select_action(policy, observation, applicable, device)
        observation, _, done, _, info = env.step(action)
        termination = terminal_name(done, info)
        if termination is not None:
            return termination, step + 1
    return "max_steps", args.max_steps


def shielded_rollout(
    env: Any,
    policy: torch.nn.Module,
    shield: ActionFaultShield,
    start_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[str, int, dict[str, Any]]:
    observation, _ = env.reset(options={"idx": start_index})
    visits: Counter[bytes] = Counter()
    n_actions = int(env.action_space.n)
    stats: dict[str, Any] = {
        "decision_states": 0,
        "states_with_any_classifier_block": 0,
        "blocked_action_occurrences": [0] * n_actions,
        "all_blocked_observation": None,
    }

    for step in range(args.max_steps):
        key = np.asarray(observation, dtype=np.float32).tobytes()
        if visits[key] >= args.max_state_visits:
            return "cycle", step, stats
        visits[key] += 1
        applicable = np.asarray(env.action_mask(), dtype=bool)
        if not applicable.any():
            return "dead_end", step, stats

        stats["decision_states"] += 1
        shielded_mask, blocked, _ = shield.masks(observation, applicable)
        if blocked.any():
            stats["states_with_any_classifier_block"] += 1
            for action in np.flatnonzero(blocked):
                stats["blocked_action_occurrences"][int(action)] += 1
        if not shielded_mask.any():
            if not applicable.any():
                raise AssertionError(
                    "An all-blocked shield state must have an applicable action"
                )
            stats["all_blocked_observation"] = np.asarray(
                observation, dtype=np.float32
            ).copy()
            return "all_blocked", step, stats

        action = select_action(policy, observation, shielded_mask, device)
        observation, _, done, _, info = env.step(action)
        termination = terminal_name(done, info)
        if termination is not None:
            return termination, step + 1, stats
    return "max_steps", args.max_steps, stats


def result_percentages(terminations: Counter[str], episodes: int) -> dict[str, float]:
    return {
        "goal_percent": 100.0 * terminations["goal"] / episodes,
        "avoid_percent": 100.0 * terminations["failure"] / episodes,
        "cycle_percent": 100.0 * terminations["cycle"] / episodes,
        "all_blocked_episode_percent": 100.0
        * terminations["all_blocked"]
        / episodes,
    }


def evaluate(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.max_steps <= 0 or args.max_state_visits <= 0:
        raise ValueError("max-steps and max-state-visits must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    policy_path = Path(args.policy)
    classifiers_dir = Path(args.classifiers_dir)
    policy_digest = sha256_file(policy_path)
    classifier_manifest = json.loads(
        (classifiers_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if classifier_manifest.get("source_policy_sha256") != policy_digest:
        raise ValueError(
            "The classifier shield was not trained from the policy being evaluated"
        )
    policy = load_policy_checkpoint(policy_path, device).eval()
    shield = ActionFaultShield.load(classifiers_dir, device=device)
    baseline_env = make_env(args, seed_offset=0)
    shielded_env = make_env(args, seed_offset=0)
    if len(shield.classifiers) != int(shielded_env.action_space.n):
        raise ValueError("Classifier count does not match environment action count")

    pool_size = baseline_env.get_init_state_pool_size()
    indices = list(range(args.shard_id, pool_size, args.num_shards))
    if args.start_index_limit is not None:
        if args.start_index_limit < 0:
            raise ValueError("start-index-limit cannot be negative")
        indices = indices[: args.start_index_limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.csv"
    episodes_tmp = episodes_path.with_suffix(".csv.tmp")
    baseline_terminations: Counter[str] = Counter()
    shielded_terminations: Counter[str] = Counter()
    blocked_action_occurrences = [0] * int(shielded_env.action_space.n)
    decision_states = 0
    states_with_any_classifier_block = 0
    all_blocked_observations: list[np.ndarray] = []

    with episodes_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start_state_index",
                "baseline_termination",
                "baseline_steps",
                "shielded_termination",
                "shielded_steps",
                "shielded_decision_states",
                "states_with_any_classifier_block",
                "all_actions_blocked",
            ],
        )
        writer.writeheader()
        for position, start_index in enumerate(indices, start=1):
            baseline_result, baseline_steps = baseline_rollout(
                baseline_env, policy, start_index, args, device
            )
            shielded_result, shielded_steps, stats = shielded_rollout(
                shielded_env, policy, shield, start_index, args, device
            )
            baseline_terminations[baseline_result] += 1
            shielded_terminations[shielded_result] += 1
            decision_states += int(stats["decision_states"])
            states_with_any_classifier_block += int(
                stats["states_with_any_classifier_block"]
            )
            for action, count in enumerate(stats["blocked_action_occurrences"]):
                blocked_action_occurrences[action] += int(count)
            if stats["all_blocked_observation"] is not None:
                all_blocked_observations.append(stats["all_blocked_observation"])

            writer.writerow(
                {
                    "start_state_index": start_index,
                    "baseline_termination": baseline_result,
                    "baseline_steps": baseline_steps,
                    "shielded_termination": shielded_result,
                    "shielded_steps": shielded_steps,
                    "shielded_decision_states": stats["decision_states"],
                    "states_with_any_classifier_block": stats[
                        "states_with_any_classifier_block"
                    ],
                    "all_actions_blocked": int(
                        stats["all_blocked_observation"] is not None
                    ),
                }
            )
            if args.progress_every > 0 and (
                position % args.progress_every == 0 or position == len(indices)
            ):
                print(
                    f"[{position}/{len(indices)}] "
                    f"baseline_goal={baseline_terminations['goal']} "
                    f"shield_goal={shielded_terminations['goal']} "
                    f"shield_avoid={shielded_terminations['failure']} "
                    f"all_blocked={shielded_terminations['all_blocked']}",
                    flush=True,
                )
    episodes_tmp.replace(episodes_path)

    obs_dim = int(shielded_env.observation_space.shape[0])
    all_blocked_array = (
        np.stack(all_blocked_observations).astype(np.float32)
        if all_blocked_observations
        else np.empty((0, obs_dim), dtype=np.float32)
    )
    np.savez_compressed(
        output_dir / "all_blocked_states.npz", observations=all_blocked_array
    )
    episodes = len(indices)
    if sum(baseline_terminations.values()) != episodes:
        raise AssertionError("Baseline termination counts do not match episode count")
    if sum(shielded_terminations.values()) != episodes:
        raise AssertionError("Shielded termination counts do not match episode count")
    summary = {
        "format_version": 1,
        "experiment": "per_action_classifier_shield_evaluation",
        "shield_rule": (
            "applicable AND NOT classifier_fault; terminate as all_blocked when "
            "the resulting mask is empty"
        ),
        "jani_model": str(Path(args.jani_model).resolve()),
        "start_states": str(Path(args.start_states).resolve()),
        "policy": str(policy_path.resolve()),
        "policy_sha256": policy_digest,
        "classifiers_dir": str(classifiers_dir.resolve()),
        "classifiers_manifest_sha256": sha256_file(classifiers_dir / "manifest.json"),
        "classifier_metadata": shield.metadata(),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_state_visits": args.max_state_visits,
        "start_state_pool_size": pool_size,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "processed_start_states": episodes,
        "baseline_terminations": dict(sorted(baseline_terminations.items())),
        "baseline_metrics": result_percentages(baseline_terminations, episodes),
        "shielded_terminations": dict(sorted(shielded_terminations.items())),
        "shielded_metrics": result_percentages(shielded_terminations, episodes),
        "shielded_decision_states": decision_states,
        "states_with_any_classifier_block": states_with_any_classifier_block,
        "states_with_any_classifier_block_percent": (
            100.0 * states_with_any_classifier_block / decision_states
            if decision_states
            else 0.0
        ),
        "all_actions_blocked_state_occurrences": int(
            shielded_terminations["all_blocked"]
        ),
        "blocked_action_occurrences": {
            shield.classifiers[action].action_name: count
            for action, count in enumerate(blocked_action_occurrences)
        },
    }
    write_json(output_dir / "summary.json", summary)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def merge(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted(input_dir.glob("shard_*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No evaluation shard summaries found below {input_dir}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    shard_ids = {int(summary["shard_id"]) for summary in summaries}
    if args.expected_shards is not None:
        missing = sorted(set(range(args.expected_shards)).difference(shard_ids))
        if missing and not args.allow_incomplete:
            raise RuntimeError(f"Missing evaluation shards: {missing}")
    else:
        missing = []

    compatibility_fields = (
        "policy_sha256",
        "classifiers_manifest_sha256",
        "jani_model",
        "start_states",
        "start_state_pool_size",
        "num_shards",
        "max_steps",
        "max_state_visits",
    )
    reference = summaries[0]
    for summary in summaries[1:]:
        for field in compatibility_fields:
            if summary.get(field) != reference.get(field):
                raise ValueError(f"Evaluation shard summaries disagree on {field}")

    baseline_terminations: Counter[str] = Counter()
    shielded_terminations: Counter[str] = Counter()
    blocked_action_occurrences: Counter[str] = Counter()
    all_blocked_chunks: list[np.ndarray] = []
    episodes_rows: list[dict[str, str]] = []
    for summary_path, summary in zip(summary_paths, summaries):
        baseline_terminations.update(summary["baseline_terminations"])
        shielded_terminations.update(summary["shielded_terminations"])
        blocked_action_occurrences.update(summary["blocked_action_occurrences"])
        with np.load(summary_path.parent / "all_blocked_states.npz") as data:
            all_blocked_chunks.append(
                np.asarray(data["observations"], dtype=np.float32)
            )
        episodes_rows.extend(read_csv_rows(summary_path.parent / "episodes.csv"))

    episodes_rows.sort(key=lambda row: int(row["start_state_index"]))
    episodes_path = output_dir / "episodes.csv"
    episodes_tmp = episodes_path.with_suffix(".csv.tmp")
    with episodes_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episodes_rows)
    episodes_tmp.replace(episodes_path)

    obs_dim = int(
        all_blocked_chunks[0].shape[1]
        if all_blocked_chunks
        else reference["classifier_metadata"][0].get("input_dim", 0)
    )
    all_blocked = (
        np.concatenate(all_blocked_chunks, axis=0)
        if all_blocked_chunks
        else np.empty((0, obs_dim), dtype=np.float32)
    )
    unique_all_blocked = (
        np.unique(all_blocked, axis=0)
        if len(all_blocked)
        else np.empty((0, obs_dim), dtype=np.float32)
    )
    np.savez_compressed(
        output_dir / "all_blocked_states.npz", observations=unique_all_blocked
    )

    episodes = sum(int(summary["processed_start_states"]) for summary in summaries)
    if sum(baseline_terminations.values()) != episodes:
        raise ValueError("Merged baseline termination counts do not match episode count")
    if sum(shielded_terminations.values()) != episodes:
        raise ValueError("Merged shielded termination counts do not match episode count")
    start_indices = [int(row["start_state_index"]) for row in episodes_rows]
    if len(start_indices) != episodes or len(set(start_indices)) != episodes:
        raise ValueError("Merged episode rows are missing or contain duplicate start states")
    if not missing and episodes != int(reference["start_state_pool_size"]):
        raise ValueError(
            f"Complete evaluation covered {episodes} starts, expected "
            f"{reference['start_state_pool_size']}"
        )
    decision_states = sum(int(summary["shielded_decision_states"]) for summary in summaries)
    states_with_any_block = sum(
        int(summary["states_with_any_classifier_block"]) for summary in summaries
    )
    merged_summary = {
        "format_version": 1,
        "experiment": "per_action_classifier_shield_evaluation_merged",
        "shield_rule": reference["shield_rule"],
        "jani_model": reference["jani_model"],
        "start_states": reference["start_states"],
        "policy": reference["policy"],
        "policy_sha256": reference["policy_sha256"],
        "classifiers_dir": reference["classifiers_dir"],
        "classifiers_manifest_sha256": reference["classifiers_manifest_sha256"],
        "classifier_metadata": reference["classifier_metadata"],
        "expected_shards": args.expected_shards,
        "completed_shards": len(shard_ids),
        "missing_shards": missing,
        "processed_start_states": episodes,
        "start_state_pool_size": int(reference["start_state_pool_size"]),
        "baseline_terminations": dict(sorted(baseline_terminations.items())),
        "baseline_metrics": result_percentages(baseline_terminations, episodes),
        "shielded_terminations": dict(sorted(shielded_terminations.items())),
        "shielded_metrics": result_percentages(shielded_terminations, episodes),
        "shielded_decision_states": decision_states,
        "states_with_any_classifier_block": states_with_any_block,
        "states_with_any_classifier_block_percent": (
            100.0 * states_with_any_block / decision_states if decision_states else 0.0
        ),
        "all_actions_blocked_state_occurrences": int(
            shielded_terminations["all_blocked"]
        ),
        "unique_all_actions_blocked_states": int(len(unique_all_blocked)),
        "blocked_action_occurrences": dict(sorted(blocked_action_occurrences.items())),
    }
    write_json(output_dir / "summary.json", merged_summary)
    print(
        f"Merged {len(shard_ids)} evaluation shards covering {episodes} starts into "
        f"{output_dir.resolve()}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.command == "evaluate":
        evaluate(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
