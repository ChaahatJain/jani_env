#!/usr/bin/env python3
"""Collect and merge oracle-labelled policy faults without repairing the policy.

A collected fault is a unique (observation, action) pair for which the JANI
oracle reports that the source state is safe and at least one successor under
that action is unsafe.  Collection never changes the policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect all oracle-labelled faults seen during fixed-policy rollouts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect one shard of policy rollouts.")
    collect.add_argument("--jani-model", required=True)
    collect.add_argument("--jani-property", default="")
    collect.add_argument("--start-states", required=True)
    collect.add_argument("--policy", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--device", default="cpu")
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--max-steps", type=int, default=2000)
    collect.add_argument("--max-state-visits", type=int, default=3)
    collect.add_argument("--num-shards", type=int, default=1)
    collect.add_argument("--shard-id", type=int, default=0)
    collect.add_argument(
        "--start-index-limit",
        type=int,
        default=None,
        help="Optional test-only limit on the number of starts assigned to this shard.",
    )
    collect.add_argument(
        "--action-scope",
        choices=("policy", "all-applicable"),
        default="policy",
        help=(
            "policy: check every action selected by the fixed policy; "
            "all-applicable: check every applicable action at each policy-visited state."
        ),
    )
    collect.add_argument("--disable-oracle-cache", action="store_true")
    collect.add_argument("--reduced-memory-mode", action="store_true")
    collect.add_argument("--include-decoded-state", action="store_true")
    collect.add_argument("--progress-every", type=int, default=25)

    merge = subparsers.add_parser("merge", help="Merge completed shard datasets.")
    merge.add_argument("--input-dir", required=True, help="Directory containing shard_*/ outputs.")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--expected-shards", type=int, default=None)
    merge.add_argument("--allow-incomplete", action="store_true")

    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_action_names(model_path: Path, n_actions: int) -> list[str]:
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        names = [str(action["name"]) for action in model.get("actions", [])]
    except (OSError, ValueError, KeyError, TypeError):
        names = []
    if len(names) != n_actions:
        return [f"action_{index}" for index in range(n_actions)]
    return names


def load_policy_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    """Load either an SB3 MaskedPPO actor checkpoint or a native repair checkpoint."""
    from dagger.policy import Policy

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch images.
        checkpoint = torch.load(path, map_location=device)

    required = {"input_dim", "output_dim", "hidden_dims", "state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Policy checkpoint is missing keys: {sorted(missing)}")

    state_dict = checkpoint["state_dict"]
    if "mlp_extractor.policy_net.0.weight" in state_dict:
        activation_fn = torch.nn.Tanh
        state_dict = {
            "model.0.weight": state_dict["mlp_extractor.policy_net.0.weight"],
            "model.0.bias": state_dict["mlp_extractor.policy_net.0.bias"],
            "model.2.weight": state_dict["mlp_extractor.policy_net.2.weight"],
            "model.2.bias": state_dict["mlp_extractor.policy_net.2.bias"],
            "model.4.weight": state_dict["action_net.weight"],
            "model.4.bias": state_dict["action_net.bias"],
        }
    else:
        activation_fn = (
            torch.nn.Tanh if checkpoint.get("activation_fn", "relu") == "tanh" else torch.nn.ReLU
        )

    policy = Policy(
        checkpoint["input_dim"],
        checkpoint["output_dim"],
        checkpoint["hidden_dims"],
        activation_fn=activation_fn,
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.to(device)
    policy.eval()
    return policy


def select_policy_action(
    policy: torch.nn.Module,
    observation: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
) -> int:
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
    if not bool(mask.any()):
        raise ValueError("Cannot select an action from an empty action mask")
    observation_tensor = torch.as_tensor(
        observation, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        logits = policy(observation_tensor).squeeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))
        return int(logits.argmax().item())


def observation_values(observation: np.ndarray) -> list[float | int]:
    values: list[float | int] = []
    for value in np.asarray(observation).tolist():
        numeric = float(value)
        values.append(int(numeric) if numeric.is_integer() else numeric)
    return values


def fault_key(observation: Iterable[float | int], action: int) -> tuple[tuple[float | int, ...], int]:
    return tuple(observation), int(action)


def fault_id(key: tuple[tuple[float | int, ...], int]) -> str:
    payload = json.dumps([list(key[0]), key[1]], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_fault_dataset(output_dir: Path, faults: Iterable[dict[str, Any]]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(faults, key=lambda item: item["fault_id"])

    jsonl_path = output_dir / "faults.jsonl"
    jsonl_tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with jsonl_tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    jsonl_tmp.replace(jsonl_path)

    csv_path = output_dir / "faults.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    fields = [
        "fault_id",
        "faulty_action",
        "action_name",
        "occurrences",
        "first_start_state_index",
        "first_step",
        "observation",
        "action_mask",
        "decoded_state",
    ]
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in ordered:
            row = dict(record)
            row["observation"] = json.dumps(record["observation"], separators=(",", ":"))
            row["action_mask"] = json.dumps(record["action_mask"], separators=(",", ":"))
            row.setdefault("decoded_state", "")
            writer.writerow({field: row.get(field, "") for field in fields})
    csv_tmp.replace(csv_path)
    return len(ordered)


def collect(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.max_steps <= 0 or args.max_state_visits <= 0:
        raise ValueError("max-steps and max-state-visits must be positive")

    from jani.env import JANIEnv

    started_at = utc_now()
    started = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    policy_path = Path(args.policy).resolve()
    model_path = Path(args.jani_model).resolve()
    output_dir = Path(args.output_dir)

    policy = load_policy_checkpoint(policy_path, device)
    env = JANIEnv(
        jani_model_path=args.jani_model,
        jani_property_path=args.jani_property,
        start_states_path=args.start_states,
        objective_path="",
        failure_property_path="",
        seed=args.seed,
        goal_reward=1.0,
        failure_reward=-1.0,
        unsafe_reward=-0.01,
        use_oracle=True,
        disable_oracle_cache=args.disable_oracle_cache,
        reduced_memory_mode=args.reduced_memory_mode,
    )
    pool_size = env.get_init_state_pool_size()
    assigned_indices = list(range(args.shard_id, pool_size, args.num_shards))
    if args.start_index_limit is not None:
        if args.start_index_limit < 0:
            raise ValueError("start-index-limit cannot be negative")
        assigned_indices = assigned_indices[: args.start_index_limit]

    n_actions = int(env.action_space.n)
    action_names = load_action_names(model_path, n_actions)
    faults: dict[tuple[tuple[float | int, ...], int], dict[str, Any]] = {}
    classifications: dict[tuple[tuple[float | int, ...], int], bool] = {}
    terminations: Counter[str] = Counter()
    action_fault_occurrences: Counter[int] = Counter()
    total_steps = 0
    action_evaluations = 0
    oracle_queries = 0
    classification_cache_hits = 0
    total_fault_occurrences = 0

    print(
        f"Collecting shard {args.shard_id}/{args.num_shards}: "
        f"{len(assigned_indices)} of {pool_size} start states; action_scope={args.action_scope}",
        flush=True,
    )

    for position, start_index in enumerate(assigned_indices, start=1):
        observation, _ = env.reset(options={"idx": start_index})
        visits: Counter[bytes] = Counter()
        termination = "max_steps"

        for step in range(args.max_steps):
            visits[np.asarray(observation, dtype=np.float32).tobytes()] += 1
            action_mask = np.asarray(env.action_mask(), dtype=np.float32)
            applicable_actions = np.flatnonzero(action_mask).tolist()
            if not applicable_actions:
                termination = "dead_end"
                break

            policy_action = select_policy_action(policy, observation, action_mask, device)
            actions_to_check = (
                applicable_actions if args.action_scope == "all-applicable" else [policy_action]
            )
            obs_values = observation_values(observation)
            mask_values = [int(value) for value in action_mask.tolist()]

            for action in actions_to_check:
                key = fault_key(obs_values, action)
                action_evaluations += 1
                if key in classifications:
                    is_fault = classifications[key]
                    classification_cache_hits += 1
                else:
                    is_fault = bool(env.is_state_action_fault(observation, action))
                    classifications[key] = is_fault
                    oracle_queries += 1

                if not is_fault:
                    continue

                total_fault_occurrences += 1
                action_fault_occurrences[action] += 1
                if key in faults:
                    faults[key]["occurrences"] += 1
                    continue

                record: dict[str, Any] = {
                    "fault_id": fault_id(key),
                    "observation": obs_values,
                    "faulty_action": int(action),
                    "action_name": action_names[action],
                    "action_mask": mask_values,
                    "occurrences": 1,
                    "first_start_state_index": int(start_index),
                    "first_step": int(step),
                }
                if args.include_decoded_state:
                    record["decoded_state"] = env.debug_show_state(observation)
                faults[key] = record

            next_observation, _, done, _, info = env.step(policy_action)
            total_steps += 1
            if info.get("reached_goal", False):
                termination = "goal"
            elif info.get("reached_fail", False):
                termination = "failure"
            elif done:
                termination = "terminal"

            if done:
                break
            next_key = np.asarray(next_observation, dtype=np.float32).tobytes()
            if visits[next_key] >= args.max_state_visits:
                termination = "cycle"
                break
            observation = next_observation

        terminations[termination] += 1
        if args.progress_every > 0 and (
            position % args.progress_every == 0 or position == len(assigned_indices)
        ):
            print(
                f"[{position}/{len(assigned_indices)}] steps={total_steps} "
                f"unique_faults={len(faults)} occurrences={total_fault_occurrences} "
                f"oracle_queries={oracle_queries}",
                flush=True,
            )

    unique_by_action = Counter(record["faulty_action"] for record in faults.values())
    summary = {
        "format_version": FORMAT_VERSION,
        "experiment": "fixed_policy_fault_collection",
        "fault_definition": (
            "The source state is oracle-safe and at least one successor under the action is unsafe."
        ),
        "action_scope": args.action_scope,
        "coverage": (
            "Every policy-selected action at every visited rollout step."
            if args.action_scope == "policy"
            else "Every applicable action at every state visited by the policy rollouts."
        ),
        "policy_was_modified": False,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "jani_model": str(model_path),
        "jani_property": str(Path(args.jani_property).resolve()) if args.jani_property else "",
        "start_states": str(Path(args.start_states).resolve()),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_state_visits": args.max_state_visits,
        "start_state_pool_size": pool_size,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "assigned_start_states": len(assigned_indices),
        "processed_start_states": sum(terminations.values()),
        "total_steps": total_steps,
        "action_evaluations": action_evaluations,
        "oracle_queries": oracle_queries,
        "classification_cache_hits": classification_cache_hits,
        "unique_faults": len(faults),
        "fault_occurrences": total_fault_occurrences,
        "terminations": dict(sorted(terminations.items())),
        "action_names": action_names,
        "unique_faults_by_action": {
            action_names[action]: unique_by_action[action] for action in range(n_actions)
        },
        "fault_occurrences_by_action": {
            action_names[action]: action_fault_occurrences[action] for action in range(n_actions)
        },
    }

    write_fault_dataset(output_dir, faults.values())
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote {len(faults)} unique faults to {output_dir.resolve()}", flush=True)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error


def merge(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    summary_paths = sorted(input_dir.glob("shard_*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No shard summaries found below {input_dir}")

    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    shard_ids = {int(summary["shard_id"]) for summary in summaries}
    if args.expected_shards is not None:
        expected = set(range(args.expected_shards))
        missing = sorted(expected.difference(shard_ids))
        if missing and not args.allow_incomplete:
            raise RuntimeError(f"Missing {len(missing)} shards: {missing}")
    else:
        missing = []

    compatibility_fields = ("policy_sha256", "jani_model", "start_states", "action_scope")
    reference = summaries[0]
    for summary in summaries[1:]:
        for field in compatibility_fields:
            if summary.get(field) != reference.get(field):
                raise ValueError(f"Shard summaries disagree on {field}")

    merged: dict[tuple[tuple[float | int, ...], int], dict[str, Any]] = {}
    for summary_path in summary_paths:
        for record in read_jsonl(summary_path.parent / "faults.jsonl"):
            key = fault_key(record["observation"], record["faulty_action"])
            if key not in merged:
                merged[key] = record
                continue
            merged[key]["occurrences"] += int(record["occurrences"])
            old_first = (
                int(merged[key]["first_start_state_index"]),
                int(merged[key]["first_step"]),
            )
            new_first = (int(record["first_start_state_index"]), int(record["first_step"]))
            if new_first < old_first:
                occurrences = merged[key]["occurrences"]
                merged[key] = record
                merged[key]["occurrences"] = occurrences

    action_names = reference["action_names"]
    unique_by_action = Counter(record["faulty_action"] for record in merged.values())
    occurrences_by_action: Counter[int] = Counter()
    for record in merged.values():
        occurrences_by_action[record["faulty_action"]] += int(record["occurrences"])

    terminations: Counter[str] = Counter()
    for summary in summaries:
        terminations.update(summary.get("terminations", {}))

    merged_summary = {
        "format_version": FORMAT_VERSION,
        "experiment": "fixed_policy_fault_collection_merged",
        "fault_definition": reference["fault_definition"],
        "action_scope": reference["action_scope"],
        "coverage": reference["coverage"],
        "policy_was_modified": False,
        "completed_at_utc": utc_now(),
        "jani_model": reference["jani_model"],
        "jani_property": reference["jani_property"],
        "start_states": reference["start_states"],
        "policy": reference["policy"],
        "policy_sha256": reference["policy_sha256"],
        "start_state_pool_size": reference["start_state_pool_size"],
        "expected_shards": args.expected_shards,
        "completed_shards": len(shard_ids),
        "shard_ids": sorted(shard_ids),
        "missing_shards": missing,
        "processed_start_states": sum(int(item["processed_start_states"]) for item in summaries),
        "total_steps": sum(int(item["total_steps"]) for item in summaries),
        "action_evaluations": sum(int(item["action_evaluations"]) for item in summaries),
        "oracle_queries": sum(int(item["oracle_queries"]) for item in summaries),
        "classification_cache_hits": sum(
            int(item["classification_cache_hits"]) for item in summaries
        ),
        "unique_faults": len(merged),
        "fault_occurrences": sum(int(item["occurrences"]) for item in merged.values()),
        "terminations": dict(sorted(terminations.items())),
        "action_names": action_names,
        "unique_faults_by_action": {
            action_names[action]: unique_by_action[action] for action in range(len(action_names))
        },
        "fault_occurrences_by_action": {
            action_names[action]: occurrences_by_action[action]
            for action in range(len(action_names))
        },
    }

    write_fault_dataset(output_dir, merged.values())
    write_json(output_dir / "summary.json", merged_summary)
    print(
        f"Merged {len(shard_ids)} shards into {len(merged)} unique faults at {output_dir.resolve()}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "merge":
        merge(args)
    else:  # pragma: no cover
        raise AssertionError(f"Unexpected command: {args.command}")


if __name__ == "__main__":
    main()
