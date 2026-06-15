#!/usr/bin/env python3
"""Compare deterministic masked rollouts before and after policy repair."""

import argparse
import csv
import json
from pathlib import Path
import sys

# Running this file directly makes Python use this subdirectory as sys.path[0].
# Add the repository root so local packages such as dagger and jani are importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from dagger.fault_collector import OracleFaultCollector
from dagger.policy_wrapper import NNPolicyWrapper
from dagger.sampler import StandardTraceSampler
from jani.env import JANIEnv
from pipeline import load_policy_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jani_model", required=True)
    parser.add_argument("--jani_property", default="")
    parser.add_argument("--start_states", required=True)
    parser.add_argument("--before_policy", required=True)
    parser.add_argument("--after_policy", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--max_state_visits", type=int, default=100)
    parser.add_argument("--search_states", type=int, default=5000)
    parser.add_argument("--fault_scan_states", type=int, default=1000)
    return parser.parse_args()


def make_env(args: argparse.Namespace, use_oracle: bool) -> JANIEnv:
    return JANIEnv(
        jani_model_path=args.jani_model,
        jani_property_path=args.jani_property,
        start_states_path=args.start_states,
        objective_path="",
        failure_property_path="",
        seed=args.seed,
        goal_reward=1.0,
        failure_reward=-10.0,
        unsafe_reward=-0.5,
        step_reward=-0.005,
        cycle_reward=-0.1,
        use_oracle=use_oracle,
    )


def policy_decision(policy: torch.nn.Module, observation: np.ndarray, mask: np.ndarray) -> dict:
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
    with torch.no_grad():
        logits = policy(obs_tensor)[0]
        masked_logits = logits.masked_fill(~mask_tensor, float("-inf"))
        probabilities = torch.softmax(masked_logits, dim=-1)
    return {
        "action": int(masked_logits.argmax().item()),
        "logits": logits.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
    }


def rollout(env: JANIEnv, policy: torch.nn.Module, start_idx: int, args: argparse.Namespace) -> tuple[str, list[dict]]:
    observation, _ = env.reset(options={"idx": start_idx})
    visits: dict[bytes, int] = {}
    trace = []

    for step in range(args.max_steps):
        key = observation.tobytes()
        visits[key] = visits.get(key, 0) + 1
        if visits[key] > args.max_state_visits:
            return "cycle", trace

        mask = env.action_mask().astype(bool)
        if not mask.any():
            return "dead_end", trace
        decision = policy_decision(policy, observation, mask)
        trace.append(
            {
                "step": step,
                "observation": observation.copy(),
                "mask": mask.copy(),
                **decision,
            }
        )

        observation, _, done, _, info = env.step(decision["action"])
        if done:
            if info.get("reached_goal", False):
                return "goal", trace
            if info.get("reached_fail", False):
                return "failure", trace
            return "dead_end", trace

    return "timeout", trace


def write_trace(path: Path, trace: list[dict], env: JANIEnv) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "action", "valid_actions", "observation", "decoded_state", "logits", "probabilities"],
        )
        writer.writeheader()
        for item in trace:
            writer.writerow(
                {
                    "step": item["step"],
                    "action": item["action"],
                    "valid_actions": np.flatnonzero(item["mask"]).tolist(),
                    "observation": item["observation"].tolist(),
                    "decoded_state": env.debug_show_state(item["observation"]),
                    "logits": np.round(item["logits"], 6).tolist(),
                    "probabilities": np.round(item["probabilities"], 6).tolist(),
                }
            )


def first_divergence(before_trace: list[dict], after_trace: list[dict]) -> dict | None:
    for before, after in zip(before_trace, after_trace):
        same_state = np.array_equal(before["observation"], after["observation"])
        if not same_state or before["action"] != after["action"]:
            return {
                "step": before["step"],
                "same_state": same_state,
                "before_observation": before["observation"].tolist(),
                "after_observation": after["observation"].tolist(),
                "before_action": before["action"],
                "after_action": after["action"],
                "valid_actions_before": np.flatnonzero(before["mask"]).tolist(),
                "valid_actions_after": np.flatnonzero(after["mask"]).tolist(),
                "before_logits": np.round(before["logits"], 6).tolist(),
                "after_logits": np.round(after["logits"], 6).tolist(),
            }
    return None


def parameter_differences(before: torch.nn.Module, after: torch.nn.Module) -> dict:
    before_state = before.state_dict()
    after_state = after.state_dict()
    differences = {}
    total_changed = 0
    total_parameters = 0

    for name, before_tensor in before_state.items():
        after_tensor = after_state[name]
        delta = (after_tensor - before_tensor).abs()
        changed = int(torch.count_nonzero(delta).item())
        total_changed += changed
        total_parameters += delta.numel()
        differences[name] = {
            "changed_parameters": changed,
            "num_parameters": delta.numel(),
            "max_abs_change": float(delta.max().item()) if delta.numel() else 0.0,
            "mean_abs_change": float(delta.mean().item()) if delta.numel() else 0.0,
        }

    return {
        "total_changed_parameters": total_changed,
        "total_parameters": total_parameters,
        "layers": differences,
    }


def reconstruct_initial_faults(
    env: JANIEnv,
    before: torch.nn.Module,
    after: torch.nn.Module,
    indices: np.ndarray,
    args: argparse.Namespace,
    output_path: Path,
) -> int:
    sampler = StandardTraceSampler()
    collector = OracleFaultCollector()
    wrapper = NNPolicyWrapper(before, device=torch.device("cpu"))
    faults = []

    for start_idx in indices[: args.fault_scan_states]:
        trace = sampler.sample_trace(
            env=env,
            policy=wrapper,
            init_state_idx=int(start_idx),
            max_steps=args.max_steps,
            max_state_visits=args.max_state_visits,
        )
        if not trace["is_safe_trajectory"]:
            for fault in collector.collect_faults(trace, env):
                faults.append((int(start_idx), fault))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "start_idx", "fault_step", "observation", "decoded_state", "valid_actions",
            "faulty_action_before", "action_after", "after_action_is_fault",
            "before_logits", "after_logits",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for start_idx, fault in faults:
            observation = np.asarray(fault["observation"], dtype=np.float32)
            mask = env.action_mask_for_obs(observation).astype(bool)
            before_decision = policy_decision(before, observation, mask)
            after_decision = policy_decision(after, observation, mask)
            writer.writerow(
                {
                    "start_idx": start_idx,
                    "fault_step": fault["step"],
                    "observation": observation.tolist(),
                    "decoded_state": env.debug_show_state(observation),
                    "valid_actions": np.flatnonzero(mask).tolist(),
                    "faulty_action_before": fault["faulty_action"],
                    "action_after": after_decision["action"],
                    "after_action_is_fault": env.is_state_action_fault(observation, after_decision["action"]),
                    "before_logits": np.round(before_decision["logits"], 6).tolist(),
                    "after_logits": np.round(after_decision["logits"], 6).tolist(),
                }
            )
    return len(faults)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Before policy: {args.before_policy}", flush=True)
    print(f"After policy: {args.after_policy}", flush=True)
    before = load_policy_checkpoint(Path(args.before_policy), torch.device("cpu")).eval()
    after = load_policy_checkpoint(Path(args.after_policy), torch.device("cpu")).eval()
    print("Both policy checkpoints loaded.", flush=True)
    before_env = make_env(args, use_oracle=False)
    after_env = make_env(args, use_oracle=False)

    indices = np.random.permutation(before_env.get_init_state_pool_size())
    search_limit = min(args.search_states, len(indices))
    print(f"Searching {search_limit} start states for a goal regression...", flush=True)
    weight_differences = parameter_differences(before, after)
    print(
        "Changed checkpoint parameters: "
        f"{weight_differences['total_changed_parameters']}/"
        f"{weight_differences['total_parameters']}",
        flush=True,
    )
    selected = None
    first_behavior_change = None
    before_result = after_result = ""
    before_trace: list[dict] = []
    after_trace: list[dict] = []
    before_outcomes = {name: 0 for name in ("goal", "failure", "cycle", "dead_end", "timeout")}
    after_outcomes = {name: 0 for name in ("goal", "failure", "cycle", "dead_end", "timeout")}
    different_outcome_count = 0
    different_first_action_count = 0

    for search_position, start_idx in enumerate(indices[:search_limit], start=1):
        before_result, candidate_before = rollout(before_env, before, int(start_idx), args)
        after_result, candidate_after = rollout(after_env, after, int(start_idx), args)
        before_outcomes[before_result] += 1
        after_outcomes[after_result] += 1
        if before_result != after_result:
            different_outcome_count += 1
        if candidate_before and candidate_after and candidate_before[0]["action"] != candidate_after[0]["action"]:
            different_first_action_count += 1
        divergence = first_divergence(candidate_before, candidate_after)
        if first_behavior_change is None and divergence is not None:
            first_behavior_change = {
                "start_idx": int(start_idx),
                "before_result": before_result,
                "after_result": after_result,
                "before_trace": candidate_before,
                "after_trace": candidate_after,
                "divergence": divergence,
            }
        if before_result == "goal" and after_result != "goal":
            selected = int(start_idx)
            before_trace = candidate_before
            after_trace = candidate_after
            print(
                f"Found regression at start index {selected}: "
                f"before={before_result}, after={after_result}",
                flush=True,
            )
            break
        if search_position % 100 == 0:
            print(
                f"Checked {search_position}/{search_limit}: "
                f"before={before_outcomes}, after={after_outcomes}, "
                f"different_outcomes={different_outcome_count}",
                flush=True,
            )

    if selected is None and first_behavior_change is not None:
        selected = first_behavior_change["start_idx"]
        before_result = first_behavior_change["before_result"]
        after_result = first_behavior_change["after_result"]
        before_trace = first_behavior_change["before_trace"]
        after_trace = first_behavior_change["after_trace"]
        print(
            f"No goal regression found; saving first action/trajectory divergence "
            f"at start index {selected}.",
            flush=True,
        )

    summary = {
        "selected_start_idx": selected,
        "before_result": before_result if selected is not None else None,
        "after_result": after_result if selected is not None else None,
        "before_steps": len(before_trace),
        "after_steps": len(after_trace),
        "first_divergence": first_divergence(before_trace, after_trace) if selected is not None else None,
        "before_outcomes": before_outcomes,
        "after_outcomes": after_outcomes,
        "different_outcome_count": different_outcome_count,
        "different_first_action_count": different_first_action_count,
        "parameter_differences": weight_differences,
    }

    if selected is not None:
        write_trace(output_dir / "trace_before.csv", before_trace, before_env)
        write_trace(output_dir / "trace_after.csv", after_trace, after_env)
    else:
        print("No goal regression found in the requested search range.", flush=True)

    print(
        f"Reconstructing faults from up to {args.fault_scan_states} original-policy traces...",
        flush=True,
    )
    oracle_env = make_env(args, use_oracle=True)
    summary["reconstructed_initial_faults"] = reconstruct_initial_faults(
        oracle_env, before, after, indices, args, output_dir / "fault_comparison.csv"
    )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
