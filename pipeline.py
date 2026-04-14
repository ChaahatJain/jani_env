import argparse
import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any
import os

import numpy as np
import torch
from tensordict import TensorDict

from dagger.buffer import DAggerBuffer
from dagger.fault_collector import OracleFaultCollector
from dagger.policy import Policy
from dagger.policy_wrapper import NNPolicyWrapper
from dagger.sampler import StandardTraceSampler
from updater.retain_aware_unlearning import RetainAwareUnlearningUpdater
from updater.supervised import SupervisedPolicyUpdater
from updater.goldberger import MILPPolicyUpdater
from updater.spec_repair import SpecRepairPolicyUpdater
from jani.env import JANIEnv
from mask_ppo.train import train_model
from torchrl.modules.distributions import MaskedCategorical

# python pipeline.py --jani_model benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/model.jani --jani_property benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/model.jani --initial_policy artifacts/pipeline/two_way_line_det/two_way_line_80_40/bootstrap/models/final_actor.pth --start_states benchmarks_generator/benchmarks/two_way_line_det/two_way_line_80_40/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 10 --output_dir artifacts/pipeline/two_way_line_det/two_way_line_80_40/ --device cpu --accumulate_faults --repair_method milp

# python pipeline.py --jani_model benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/model.jani --jani_property benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/model.jani --initial_policy artifacts/pipeline/one_way_line_det/one_way_line_80_40/bootstrap/models/final_actor.pth --start_states benchmarks_generator/benchmarks/one_way_line_det/one_way_line_80_40/pa_model_random_starts_100000.jani --objective "" --failure_property "" --max_steps 100 --traces_per_iteration 100 --max_iterations 10 --output_dir artifacts/pipeline/one_way_line_det/one_way_line_80_40/ --device cpu --accumulate_faults --repair_method milp

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
    parser.add_argument("--bootstrap_timesteps", type=int, default=500_000)
    parser.add_argument("--bootstrap_n_steps", type=int, default=256)
    parser.add_argument("--bootstrap_use_oracle", action="store_true")
    
    parser.add_argument("--repair_method", type=str, default="milp") # All options are milp, dagger, spec, retain_unlearn

    # Repair settings
    parser.add_argument("--max_iterations", type=int, default=1000)
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
    
def repair_metrics(policy: torch.nn.Module, dataset: Any, verbose = False):
        states = torch.tensor([f["observation"] for f in dataset], dtype=torch.float32) 
        applicable_actions = [[i for i, a in enumerate(f["action_mask"]) if int(a) == 1] for f in dataset]       
        faults = [int(f["faulty_action"]) for f in dataset]
        logits = policy(states)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for i, valid_actions in enumerate(applicable_actions):
            mask[i, valid_actions] = True
        masked_logits = logits.masked_fill(~mask, float('-inf'))
        predicted_actions = torch.log_softmax(masked_logits, dim=-1).argmax(dim=-1)
        faults_tensor = torch.tensor(faults)
        fixed_count = (predicted_actions != faults_tensor).sum().item()
        
        if verbose:
            print(f"{'#':<6} {'Faulty Action':<16} {'New Action':<12} {'Fixed?':<8} {'Observation'}")
            print("-" * 80)
            for i, (obs, fault, pred) in enumerate(zip(dataset, faults, predicted_actions.tolist())):
                fixed = "✓" if pred != fault else "✗"
                print(f"{i:<6} {fault:<16} {pred:<12} {fixed:<8} {obs['observation']}")
            print("-" * 80)
            print(f"Fixed: {fixed_count} / {len(dataset)}")
        
        return fixed_count
    
def is_policy_safe(
        env: JANIEnv, 
        policy: Policy, 
        observation: np.ndarray, 
        visited_obs: dict, 
        deterministic: bool=True,
        depth: int=0) -> bool:
    # This is a strict definition of unsafety where we exhaustively check the entire policy envelope. This is expensive but gives a more accurate measure of whether the policy can crash from a given start state.
    # print("  " * depth + f"Checking safety for observation: {env.debug_show_state(observation)}")

    if env.unwrapped.obs_reach_goal(observation):
        # print("  " * depth + "Reached goal state during safety check.")
        return True
    if env.unwrapped.obs_reach_failure(observation):
        # print("  " * depth + "Reached failure state during safety check.")
        return False
    
    # Cache the visited obs
    visited_obs[observation.tobytes()] = 0

    # Get policy's action
    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    action_mask = env.unwrapped.action_mask_for_obs(observation).astype(int)
    action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        logits = policy(obs_tensor)
        action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
        if deterministic:
            action = action_dist.probs.argmax(dim=-1).squeeze(0).item()  # Get the most likely action
        else:
            action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension

    # Recursively check all successor states
    successor_obs = env.unwrapped.get_successor_obs(observation, action)
    # print("  " * depth + f"Policy action: {action}, Successor count: {len(successor_obs)}")
    for succ_obs in successor_obs:
        if succ_obs.tobytes() in visited_obs:
            # print("  " * depth + "Already visited successor, skipping to avoid cycles.")
            if visited_obs[succ_obs.tobytes()] == -1:
                # print("  " * depth + "But it's currently being explored, so we have a cycle. Marking unsafe.")
                visited_obs[observation.tobytes()] = -1
                return False
            else:
                continue
        if not is_policy_safe(env, policy, succ_obs, visited_obs, deterministic, depth + 1):
            # print("  " * depth + "Found unsafe successor.")
            visited_obs[observation.tobytes()] = -1
            return False
    visited_obs[observation.tobytes()] = 1
    return True

def evaluate_policy(
        env: JANIEnv, 
        policy: torch.nn.Module, 
        init_state_indices: list[int], 
        max_steps: int = 1100,
        deterministic: bool = True
    ) -> dict[str, Any]:
    assert deterministic, "We always look at deterministic policies when evaluating these metrics"
    num_total_states = len(init_state_indices)
    num_unsafe_runs = 0 # Runs where we reach an unsafe state: State where no safe policy exists.
    num_goal_reached = 0 # Runs where we reach a goal state
    num_failed_runs = 0 # Runs where we reach a fail state. This met
    
    from tqdm import tqdm
    
    for idx in tqdm(init_state_indices):
        seen_obs = set() # cache for cycle detection
        obs, _ = env.reset(options={"idx": idx})
        if not is_policy_safe(env, policy, obs, {}, deterministic):
            num_unsafe_runs += 1
        done = False
        step_count = 0
        last_reward = None # Keep track of the last reward to determine if we reached the goal at the end of the episode
        while not done and step_count < max_steps:
            obs_key = obs.tobytes()
            if obs_key in seen_obs:
                break # Early termination cycle
            seen_obs.add(obs_key)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            action_mask = env.unwrapped.action_mask().astype(int)
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                logits = policy(obs_tensor)
                action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
                if deterministic:
                    action = action_dist.probs.argmax(dim=-1).squeeze(0).item()  # Get the most likely action
                else:
                    action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension            
            # Step the environment
            obs, reward, done, _, _ = env.step(action)
            step_count += 1
            last_reward = reward

        if last_reward == 1.0: # We reached the goal
            num_goal_reached += 1
        elif last_reward == -1.0: # We reached a failure state
            num_failed_runs += 1

    frac_unsafe = num_unsafe_runs / num_total_states if num_total_states > 0 else 0.0
    frac_goal = num_goal_reached / num_total_states if num_total_states > 0 else 0.0
    frac_failure = num_failed_runs / num_total_states if num_total_states > 0 else 0.0

    return {
        "frac_unsafe": frac_unsafe,
        "frac_goal": frac_goal,
        "frac_failure": frac_failure,
    }


def load_policy_checkpoint(checkpoint_path: Path, device: torch.device) -> Policy:
    print("Loading policy from path:", checkpoint_path)
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
    bootstrap_dir = output_dir / "rl_training"
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
        eval_freq=100,
        n_eval_episodes=10,
        load_policy_path="",
        save_all_checkpoints=True,
        eval_safety=False,
        disable_eval=True,
        wandb_project="jani_rl",
        wandb_entity=None,
        experiment_name="pipeline_bootstrap",
        verbose=0,
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

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize stuff
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "repair_checkpoints" / f"{args.repair_method}"
    logs_dir = output_dir / "repair_logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Train and load the policy
    policy_path = bootstrap_policy_if_needed(args, output_dir)
    policy_model = load_policy_checkpoint(policy_path, device=device)
    policy_model.to(device)
    policy = NNPolicyWrapper(policy_model, device=device)

    # Create the environment
    env = build_env(args, use_oracle=True)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    num_start_states = env.get_init_state_pool_size()
    all_indices = np.random.permutation(num_start_states)
    if num_start_states >= 2*args.traces_per_iteration:
        repair_indices = all_indices[:args.traces_per_iteration]
        eval_indices   = all_indices[args.traces_per_iteration:2*args.traces_per_iteration]
    else:
        repair_indices = all_indices
        eval_indices = all_indices
        
    sampler = StandardTraceSampler()
    collector = OracleFaultCollector()

    # Create repair method
    ml_repair_method = args.repair_method != "milp"
    if ml_repair_method:
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.learning_rate)
        if args.repair_method == "dagger":
            updater = SupervisedPolicyUpdater(
                optimizer=optimizer,
                batch_size=args.updater_batch_size,
                steps_per_iteration=args.updater_steps,
                device=device,
            )
        elif args.repair_method == "retain_unlearn":
            updater = RetainAwareUnlearningUpdater(
            optimizer=optimizer,
            batch_size=args.updater_batch_size,
            steps_per_iteration=args.updater_steps,
            device=device,
            )
        elif args.repair_method == "spec":
            updater = SpecRepairPolicyUpdater(optimizer=optimizer, batch_size=args.updater_batch_size, device=device)
    else:
        updater = MILPPolicyUpdater()

    replay_buffer = DAggerBuffer(buffer_size=args.buffer_size)

    metrics_file = logs_dir / f"{args.repair_method}.jsonl"
    converged = False
    
    print("Starting fault analysis and repair loop...")
    print(f"Initial policy: {policy_path}")

    all_faults = []
    total_steps = 0
    fault_cache = set()
    fixed = True
    for iteration in range(1, args.max_iterations + 1):
        traces = []
        new_faults = []
        sampling_seconds = 0.0
        oracle_seconds = 0.0
        repair_seconds = 0.0
        performance_evaluation_seconds = 0.0
        
        # if iteration > 1:
        #     print("NEW ITERATION REACHED HERE")
        #     it = iteration - 1
        #     policy_path = checkpoints_dir / f"policy_iter_{it:03d}.pth"
        #     policy_model = load_policy_checkpoint(policy_path, device=device)
        #     policy_model.to(device)
        #     policy = NNPolicyWrapper(policy_model, device=device)
        #     repair_metrics(policy_model, all_faults)
        
        performance_evaluation_time = time.perf_counter()
        train_pol_performance = evaluate_policy(env, policy_model, repair_indices)
        eval_pol_performance = evaluate_policy(env, policy_model, eval_indices)
        performance_evaluation_seconds = time.perf_counter() - performance_evaluation_time
        duplicates = 0
        for i in range(args.traces_per_iteration):
            trace_start_time = time.perf_counter()
            init_state_idx = repair_indices[i] # Randomly sample initial state from repair start states
            # Sample a trace and store it along with whether we reached safety or unsafety.
            trace = sampler.sample_trace(
                env=env,
                policy=policy,
                init_state_idx=init_state_idx,
                max_steps=args.max_steps,
                verbose = False
            )
            traces.append(trace) 
            sampling_seconds += time.perf_counter() - trace_start_time
            total_steps += len(trace["observations"])
            
            if not trace["is_safe_trajectory"]: # We run fault analysis on the unsafe traces found here to get more informative fixes
                trace_fa_time = time.perf_counter()
                faults = collector.collect_faults(trace, env)
                oracle_seconds += time.perf_counter() - trace_fa_time

                new = [f for f in faults if (tuple(f["observation"]), f["faulty_action"]) not in fault_cache]
                new_faults.extend(new)
                all_faults.extend(new)
                duplicates += len(faults) - len(new)
                fault_cache.update((tuple(f["observation"]), f["faulty_action"]) for f in new)
        num_new_faults = len(new_faults)
        num_faults = len(all_faults)
        print(
            f"Iteration {iteration}: traces={len(traces)}, steps={total_steps}, faults={num_faults} total {num_new_faults} new {duplicates} duplicates, "
            f"sampling_seconds={sampling_seconds:.2f}, oracle_seconds={oracle_seconds:.2f}, evaluation_seconds={performance_evaluation_seconds:.2f}"
        )

        metrics = {
            "iteration": iteration,
            "num_traces": len(traces),
            "total_steps": total_steps,
            "num_faults": num_faults,
            "it_faults": num_new_faults,
            "duplicate_faults": duplicates,
            "sampling_seconds": sampling_seconds,
            "oracle_seconds": oracle_seconds,
            "evaluation_seconds": performance_evaluation_seconds,
            "TrainGoalFrac": train_pol_performance["frac_goal"],
            "TrainFailFrac": train_pol_performance["frac_failure"],
            "TrainUnsafeFrac": train_pol_performance["frac_unsafe"],
            "EvalGoalFrac": eval_pol_performance["frac_goal"],
            "EvalFailFrac": eval_pol_performance["frac_failure"],
            "EvalUnsafeFrac": eval_pol_performance["frac_unsafe"],            
        }

        if num_new_faults == 0 and fixed:
            metrics.update({"num_faults_fixed": num_faults})
            converged = True
            with metrics_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            print("No faults found. Pipeline converged.")
            break
        # Collect faults
        if not args.accumulate_faults:
            replay_buffer.empty()
            assert False, "This should never be called. Unless we decide to later for experimental reasons."
        positive, negative = faults_to_tensordict(all_faults, obs_dim=obs_dim, n_actions=n_actions)
        replay_buffer.add_samples(positive, negative)

        # Repair policy based on collected faults
        repair_time = time.perf_counter()
        update_info = updater.update_policy(policy_model, replay_buffer) if args.repair_method == "dagger" else updater.update_policy(policy_model, all_faults)
        repair_seconds += time.perf_counter() - repair_time 
        num_fixed = repair_metrics(policy_model, all_faults)
        metrics.update({"repair_seconds": repair_seconds})
        metrics.update({"num_faults_fixed": num_fixed})
        fixed = num_fixed >= num_faults
        if ml_repair_method:
            metrics.update({"update_loss": float(update_info["loss"])}) # Is NONE for MILP policy repair
        
        # Save intermediate policies
        ckpt_path = checkpoints_dir / f"policy_iter_{iteration:03d}.pth"
        save_policy_checkpoint(policy_model, ckpt_path, input_dim=obs_dim, output_dim=n_actions)
        metrics.update({"checkpoint": str(ckpt_path)})

        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        print(f"Updated policy saved: {ckpt_path}")
        if ml_repair_method:
            print(f"Iteration {iteration} update loss: {update_info['loss']:.6f}")

    final_path = checkpoints_dir / "final_policy.pth"
    save_policy_checkpoint(policy_model, final_path, input_dim=obs_dim, output_dim=n_actions)

    print("\n=== Pipeline summary ===")
    print(f"Converged: {converged}")
    print(f"Final policy: {final_path}")
    print(f"Iteration logs: {metrics_file}")


if __name__ == "__main__":
    main()
