import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from dagger.buffer import DAggerBuffer
from dagger.fault_collector import OracleFaultCollector
from dagger.policy import Policy
from dagger.policy_wrapper import NNPolicyWrapper
from dagger.sampler import StandardTraceSampler
from dagger.updater import SupervisedPolicyUpdater
from jani.env import JANIEnv
from mask_ppo.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end DAgger pipeline: bootstrap policy, detect faults, update policy, repeat."
    )

    # Environment files
    parser.add_argument("--jani_model", type=str, required=True, help="Path to JANI model.")
    parser.add_argument("--jani_property", type=str, default="", help="Path to JANI property.")
    parser.add_argument("--start_states", type=str, required=True, help="Path to start states.")
    parser.add_argument("--objective", type=str, default="", help="Path to objective file.")
    parser.add_argument("--failure_property", type=str, default="", help="Path to failure property file.")

    # Rewards and env settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal_reward", type=float, default=1.0)
    parser.add_argument("--failure_reward", type=float, default=-1.0)
    parser.add_argument("--unsafe_reward", type=float, default=-0.01)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--disable_oracle_cache", action="store_true")
    parser.add_argument("--reduced_memory_mode", action="store_true")

    # Bootstrap training settings
    parser.add_argument(
        "--initial_policy",
        type=str,
        default="",
        help="Checkpoint (.pth) to start from. If missing, policy is bootstrapped with MaskedPPO.",
    )
    parser.add_argument("--bootstrap_timesteps", type=int, default=20000)
    parser.add_argument("--bootstrap_n_steps", type=int, default=256)
    parser.add_argument("--bootstrap_use_oracle", action="store_true")

    # DAgger loop settings
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--traces_per_iteration", type=int, default=100)
    parser.add_argument("--updater_batch_size", type=int, default=256)
    parser.add_argument("--updater_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--accumulate_faults", action="store_true")

    # Sampling settings
    parser.add_argument(
        "--sample_from_init_pool",
        action="store_true",
        help="If set, sample traces from random initial-state indices when available.",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="artifacts/pipeline")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def build_env(args: argparse.Namespace, use_oracle: bool) -> JANIEnv:
    return JANIEnv(
        jani_model_path=args.jani_model,
        jani_property_path=args.jani_property,
        start_states_path=args.start_states,
        objective_path=args.objective,
        failure_property_path=args.failure_property,
        seed=args.seed,
        goal_reward=args.goal_reward,
        failure_reward=args.failure_reward,
        unsafe_reward=args.unsafe_reward,
        use_oracle=use_oracle,
        disable_oracle_cache=args.disable_oracle_cache,
        reduced_memory_mode=args.reduced_memory_mode,
    )


def load_policy_checkpoint(checkpoint_path: Path, device: torch.device) -> Policy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    hidden_dims = checkpoint["hidden_dims"]

    policy = Policy(input_dim, output_dim, hidden_dims)
    state_dict = checkpoint["state_dict"]

    if "mlp_extractor.policy_net.0.weight" in state_dict:
        # MaskedPPO format
        mapped = {
            "model.0.weight": state_dict["mlp_extractor.policy_net.0.weight"],
            "model.0.bias": state_dict["mlp_extractor.policy_net.0.bias"],
            "model.2.weight": state_dict["mlp_extractor.policy_net.2.weight"],
            "model.2.bias": state_dict["mlp_extractor.policy_net.2.bias"],
            "model.4.weight": state_dict["action_net.weight"],
            "model.4.bias": state_dict["action_net.bias"],
        }
        policy.load_state_dict(mapped, strict=True)
    else:
        # Native DAgger format
        policy.load_state_dict(state_dict, strict=True)

    return policy


def get_hidden_dims(policy: Policy) -> list[int]:
    linear_layers = [m for m in policy.model if isinstance(m, torch.nn.Linear)]
    if len(linear_layers) <= 1:
        return []
    return [layer.out_features for layer in linear_layers[:-1]]


def save_policy_checkpoint(policy: Policy, path: Path, input_dim: int, output_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": get_hidden_dims(policy),
            "state_dict": policy.state_dict(),
        },
        path,
    )


def bootstrap_policy_if_needed(args: argparse.Namespace, output_dir: Path) -> Path:
    initial = Path(args.initial_policy) if args.initial_policy else None
    if initial is not None and initial.exists():
        print(f"Using existing initial policy: {initial}")
        return initial

    print("No existing policy checkpoint found. Bootstrapping with MaskedPPO training...")
    bootstrap_dir = output_dir / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    train_args = Namespace(
        jani_model=args.jani_model,
        jani_property=args.jani_property,
        start_states=args.start_states,
        objective=args.objective,
        failure_property=args.failure_property,
        eval_start_states="",
        goal_reward=args.goal_reward,
        failure_reward=args.failure_reward,
        unsafe_reward=args.unsafe_reward,
        use_oracle=args.bootstrap_use_oracle,
        disable_oracle_cache=args.disable_oracle_cache,
        no_memory_reduced_mode=not args.reduced_memory_mode,
        seed=args.seed,
        total_timesteps=args.bootstrap_timesteps,
        n_envs=1,
        max_steps=args.max_steps,
        n_steps=args.bootstrap_n_steps,
        log_dir=str(bootstrap_dir / "logs"),
        log_reward=False,
        model_save_dir=str(bootstrap_dir / "models"),
        use_separate_eval_env=False,
        enumate_all_init_states=False,
        eval_freq=2048,
        n_eval_episodes=10,
        load_policy_path="",
        save_all_checkpoints=False,
        eval_safety=False,
        disable_eval=True,
        wandb_project="jani_rl",
        wandb_entity=None,
        experiment_name="pipeline_bootstrap",
        verbose=1,
        device=args.device,
        disable_wandb=True,
    )

    file_args = {
        "jani_model": args.jani_model,
        "jani_property": args.jani_property,
        "start_states": args.start_states,
        "objective": args.objective,
        "failure_property": args.failure_property,
        "goal_reward": args.goal_reward,
        "failure_reward": args.failure_reward,
        "unsafe_reward": args.unsafe_reward,
        "seed": args.seed,
        "use_oracle": args.bootstrap_use_oracle,
        "max_steps": args.max_steps,
        "disable_oracle_cache": args.disable_oracle_cache,
        "reduced_memory_mode": args.reduced_memory_mode,
    }

    train_model(train_args, file_args)
    bootstrapped = bootstrap_dir / "models" / "final_actor.pth"

    if not bootstrapped.exists():
        raise FileNotFoundError(f"Bootstrapped checkpoint not found at {bootstrapped}")

    print(f"Bootstrapped policy saved at: {bootstrapped}")
    return bootstrapped


def faults_to_tensordict(faults: list[dict[str, Any]], obs_dim: int, n_actions: int) -> tuple[TensorDict, TensorDict]:
    if not faults:
        positive_samples = TensorDict(
            {
                "observation": torch.empty((0, obs_dim), dtype=torch.float32),
                "action": torch.empty((0,), dtype=torch.long),
                "action_mask": torch.empty((0, n_actions), dtype=torch.bool),
            },
            batch_size=[0],
        )
        negative_samples = TensorDict(
            {
                "observation": torch.empty((0, obs_dim), dtype=torch.float32),
                "action": torch.empty((0,), dtype=torch.long),
                "action_mask": torch.empty((0, n_actions), dtype=torch.bool),
            },
            batch_size=[0],
        )
        return positive_samples, negative_samples

    observations = np.stack([np.asarray(f["observation"], dtype=np.float32) for f in faults], axis=0)
    corrected_actions = np.asarray([int(f["action"]) for f in faults], dtype=np.int64)
    masks = np.stack([np.asarray(f["action_mask"], dtype=bool) for f in faults], axis=0)

    negative_samples = TensorDict(
        {
            "observation": torch.tensor(observations, dtype=torch.float32),
            "action": torch.tensor(corrected_actions, dtype=torch.long),
            "action_mask": torch.tensor(masks, dtype=torch.bool),
        },
        batch_size=[len(faults)],
    )

    positive_samples = TensorDict(
        {
            "observation": torch.empty((0, obs_dim), dtype=torch.float32),
            "action": torch.empty((0,), dtype=torch.long),
            "action_mask": torch.empty((0, n_actions), dtype=torch.bool),
        },
        batch_size=[0],
    )

    return positive_samples, negative_samples


def maybe_pick_init_state(env: JANIEnv, rng: np.random.Generator, sample_from_pool: bool) -> int:
    if not sample_from_pool:
        return -1

    pool_size = None
    if hasattr(env.unwrapped, "get_init_state_pool_size"):
        pool_size = env.unwrapped.get_init_state_pool_size()

    if pool_size is None or pool_size <= 0:
        return -1

    return int(rng.integers(0, pool_size))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    policy_path = bootstrap_policy_if_needed(args, output_dir)
    policy_model = load_policy_checkpoint(policy_path, device=device)
    policy_model.to(device)

    env = build_env(args, use_oracle=True)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = NNPolicyWrapper(policy_model, device=device)
    sampler = StandardTraceSampler()
    collector = OracleFaultCollector()

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.learning_rate)
    updater = SupervisedPolicyUpdater(
        optimizer=optimizer,
        batch_size=args.updater_batch_size,
        steps_per_iteration=args.updater_steps,
        device=device,
    )

    replay_buffer = DAggerBuffer(buffer_size=args.buffer_size)
    rng = np.random.default_rng(args.seed)

    metrics_file = logs_dir / "iterations.jsonl"
    converged = False

    print("Starting DAgger pipeline loop...")
    print(f"Initial policy: {policy_path}")

    for iteration in range(1, args.max_iterations + 1):
        traces = []
        all_faults = []
        total_steps = 0

        for _ in range(args.traces_per_iteration):
            init_state_idx = maybe_pick_init_state(env, rng, args.sample_from_init_pool)
            trace = sampler.sample_trace(
                env=env,
                policy=policy,
                init_state_idx=init_state_idx,
                max_steps=args.max_steps,
            )
            traces.append(trace)
            total_steps += len(trace["observations"])
            all_faults.extend(collector.collect_faults(trace))

        num_faults = len(all_faults)
        print(
            f"Iteration {iteration}: traces={len(traces)}, steps={total_steps}, faults={num_faults}"
        )

        metrics = {
            "iteration": iteration,
            "num_traces": len(traces),
            "total_steps": total_steps,
            "num_faults": num_faults,
        }

        if num_faults == 0:
            converged = True
            with metrics_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            print("No faults found. Pipeline converged.")
            break

        if not args.accumulate_faults:
            replay_buffer.empty()

        positive, negative = faults_to_tensordict(all_faults, obs_dim=obs_dim, n_actions=n_actions)
        replay_buffer.add_samples(positive, negative)

        update_info = updater.update_policy(policy_model, replay_buffer)
        metrics.update({"update_loss": float(update_info["loss"])})

        ckpt_path = checkpoints_dir / f"policy_iter_{iteration:03d}.pth"
        save_policy_checkpoint(policy_model, ckpt_path, input_dim=obs_dim, output_dim=n_actions)
        metrics.update({"checkpoint": str(ckpt_path)})

        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        print(f"Updated policy saved: {ckpt_path}")
        print(f"Iteration {iteration} update loss: {update_info['loss']:.6f}")

    final_path = checkpoints_dir / "final_policy.pth"
    save_policy_checkpoint(policy_model, final_path, input_dim=obs_dim, output_dim=n_actions)

    print("\n=== Pipeline summary ===")
    print(f"Converged: {converged}")
    print(f"Final policy: {final_path}")
    print(f"Iteration logs: {metrics_file}")


if __name__ == "__main__":
    main()
