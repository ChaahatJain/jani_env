import json
import argparse
import torch
import time
import numpy as np

from pathlib import Path
from tensordict import TensorDict

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from dagger.policy_wrapper import NNPolicyWrapper
from jani.enum_env import JANIEnumEnv
from callbacks import SaveActorCallback, save_policy
from utils import mask_fn

from dagger.updater import MILPPolicyUpdater
from dagger.policy import Policy
from dagger.fault_collector import OracleFaultCollector
from dagger.sampler import StandardTraceSampler
from dagger.buffer import DAggerBuffer

from torchrl.modules.distributions import MaskedCategorical


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic JANI RL trainer — mask_ppo | ppo_lag | safe_dqn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Environment (all algorithms) ─────────────────────────────────────────
    env = parser.add_argument_group("Environment")
    env.add_argument("--jani_model",             type=str,   required=True)
    env.add_argument("--jani_property",          type=str,   default="")
    env.add_argument("--start_states",           type=str,   default="")
    env.add_argument("--objective",              type=str,   default="")
    env.add_argument("--failure_property",       type=str,   default="")
    env.add_argument("--eval_start_states",      type=str,   default="")
    env.add_argument("--goal_reward",            type=float, default=1.0)
    env.add_argument("--failure_reward",         type=float, default=-1.0)
    env.add_argument("--unsafe_reward",          type=float, default=-0.01)
    env.add_argument("--use_oracle",             action="store_true")
    env.add_argument("--disable_oracle_cache",   action="store_true")
    env.add_argument("--no_memory_reduced_mode", action="store_true")
    env.add_argument("--max_steps",              type=int,   default=256)
    env.add_argument("--max_training_idx",       type=int,   default=80000)

    # ── Shared hyperparameters ───────────────────────────────────────────────
    hp = parser.add_argument_group("Hyperparameters (shared across all algorithms)")
    hp.add_argument("--learning_rate", type=float, default=3e-4,
                    help="Learning rate (all algorithms).")
    hp.add_argument("--gamma",         type=float, default=0.99,
                    help="Reward discount factor.")
    hp.add_argument("--batch_size",    type=int,   default=64)
    hp.add_argument("--hidden_dims",   type=int, nargs="+", default=[64, 64],
                    help="Hidden layer sizes for actor/critic (ppo_lag) or "
                         "feature network (safe_dqn). Passed as net_arch for mask_ppo.")
    hp.add_argument("--cost_gamma",    type=float, default=0.99,
                    help="Discount factor for cost returns (ppo_lag / safe_dqn).")
    hp.add_argument("--cost_limit",    type=float, default=0.1,
                    help="Cost budget (ppo_lag Lagrangian) / safety threshold (safe_dqn).")
    
    # ── Repairing hyperparameters ─────────────────────────────────────────────
    repair = parser.add_argument_group("Repairing-specific hyperparameters")
    repair.add_argument("--max_iterations", type=int, default=150)
    repair.add_argument("--num_traces_per_iter", type=int, default=100)

    # ── PPO hyperparameters (mask_ppo + ppo_lag) ─────────────────────────────
    ppo = parser.add_argument_group("PPO hyperparameters (mask_ppo / ppo_lag)")
    ppo.add_argument("--n_steps",       type=int,   default=256)
    ppo.add_argument("--n_epochs",      type=int,   default=5)
    ppo.add_argument("--gae_lambda",    type=float, default=0.95)
    ppo.add_argument("--clip_range",    type=float, default=0.2)
    ppo.add_argument("--ent_coef",      type=float, default=0.0)
    ppo.add_argument("--vf_coef",       type=float, default=0.5)
    ppo.add_argument("--max_grad_norm", type=float, default=0.5)

    # ── PPO-Lag specific ─────────────────────────────────────────────────────
    lag = parser.add_argument_group("PPO-Lagrangian specific")
    lag.add_argument("--cost_vf_coef", type=float, default=0.5,
                     help="Weight on cost value-function loss.")
    lag.add_argument("--init_lambda",  type=float, default=0.0,
                     help="Initial Lagrange multiplier.")
    lag.add_argument("--lr_lambda",    type=float, default=0.01,
                     help="Learning rate for Lagrange multiplier.")
    lag.add_argument("--pi_net_arch",   type=int, nargs="+", default=[64, 64])
    lag.add_argument("--vf_net_arch",   type=int, nargs="+", default=[64, 64])

    # ── Safe-DQN specific ────────────────────────────────────────────────────
    dqn = parser.add_argument_group("Safe-DQN specific")
    dqn.add_argument("--buffer_capacity",    type=int,   default=100_000)
    dqn.add_argument("--tau",                type=float, default=0.005,
                     help="Soft update coefficient for target network.")
    dqn.add_argument("--eps_start",          type=float, default=1.0)
    dqn.add_argument("--eps_end",            type=float, default=0.05)
    dqn.add_argument("--eps_decay_steps",    type=int,   default=50_000)
    dqn.add_argument("--learning_starts",    type=int,   default=1_000)
    dqn.add_argument("--train_freq",         type=int,   default=4)
    dqn.add_argument("--target_update_freq", type=int,   default=1_000)

    # ── Training (shared) ────────────────────────────────────────────────────
    tr = parser.add_argument_group("Training")
    tr.add_argument("--seed",             type=int, default=42)
    tr.add_argument("--total_timesteps",  type=int, default=1_000_000)
    tr.add_argument("--n_envs",           type=int, default=1,
                    help="Parallel envs — mask_ppo only; ignored otherwise.")
    tr.add_argument("--device",           type=str, default="cpu")
    tr.add_argument("--load_policy_path", type=str, default="")

    # ── Evaluation (shared) ──────────────────────────────────────────────────
    ev = parser.add_argument_group("Evaluation")
    ev.add_argument("--disable_eval",            action="store_true")
    ev.add_argument("--eval_freq",               type=int, default=2048)
    ev.add_argument("--n_eval_episodes",         type=int, default=50)
    ev.add_argument("--use_separate_eval_env",   action="store_true")
    ev.add_argument("--enumate_all_init_states", action="store_true")
    ev.add_argument("--eval_safety",             action="store_true")
    ev.add_argument("--save_all_checkpoints",    action="store_true")
    ev.add_argument("--log_reward",              action="store_true")

    # ── Logging (shared) ─────────────────────────────────────────────────────
    lg = parser.add_argument_group("Logging")
    lg.add_argument("--save_freq",       type=int, default=256)
    lg.add_argument("--log_dir",         type=str, default="./logs")
    lg.add_argument("--perf_file",       type=str, default="./performance.csv")
    lg.add_argument("--model_save_dir",  type=str, default="./models")
    lg.add_argument("--repair_save_dir", type=str, default="./repairs")
    lg.add_argument("--experiment_name", type=str, default="")
    lg.add_argument("--verbose",         type=int, default=1)
    lg.add_argument("--disable_wandb",   action="store_true")
    lg.add_argument("--wandb_project",   type=str, default="jani_rl")
    lg.add_argument("--wandb_entity",    type=str, default=None)

    return parser


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


def _build_file_args(args) -> dict:
    return {
        "jani_model":           args.jani_model,
        "jani_property":        args.jani_property,
        "start_states":         args.start_states,
        "objective":            args.objective,
        "failure_property":     args.failure_property,
        "goal_reward":          args.goal_reward,
        "failure_reward":       args.failure_reward,
        "unsafe_reward":        args.unsafe_reward,
        "seed":                 args.seed,
        "use_oracle":           args.use_oracle,
        "max_steps":            args.max_steps,
        "disable_oracle_cache": args.disable_oracle_cache,
        "reduced_memory_mode":  not args.no_memory_reduced_mode,
        "max_training_idx":     args.max_training_idx,
    }


def create_enum_env(
        file_args: dict, 
        n_envs: int = 1, 
        monitor: bool = False, 
        time_limited: bool = True) -> JANIEnumEnv:
    """Create JANI environment with specified parameters."""
    def make_env():
        env = JANIEnumEnv(
            jani_model_path=file_args["jani_model"],
            jani_property_path=file_args["jani_property"],
            start_states_path=file_args["start_states"],
            objective_path=file_args["objective"],
            failure_property_path=file_args["failure_property"],
            seed=file_args["seed"],
            goal_reward=file_args["goal_reward"],
            use_oracle=file_args.get("use_oracle", False),
            failure_reward=file_args["failure_reward"],
            unsafe_reward=file_args.get("unsafe_reward", -0.01),
            disable_oracle_cache=file_args.get("disable_oracle_cache", False),
            reduced_memory_mode=file_args.get("reduced_memory_mode", False),
            max_training_idx=file_args.get("max_training_idx", 80000)
        ) 

        if time_limited:
            env = TimeLimit(env, max_episode_steps=file_args["max_steps"])
        # Apply action masking
        env = ActionMasker(env, mask_fn)
        if monitor:
            env = Monitor(env)
        return env
    
    env = None
    if n_envs == 1:
        env = make_env()
    else:
        env = make_vec_env(make_env, n_envs=n_envs)
        if monitor:
            env = VecMonitor(env)

    return env

def faults_to_tensordict(faults: list[dict[str, any]], obs_dim: int, n_actions: int) -> tuple[TensorDict, TensorDict]:
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

def is_policy_safe(
        env: JANIEnumEnv, 
        policy: Policy, 
        observation: np.ndarray, 
        visited_obs: dict, 
        deterministic: bool=True,
        depth: int=0) -> bool:
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
        env: JANIEnumEnv,
        policy: torch.nn.Module, 
        init_state_indices: list[int], 
        max_steps: int = 256,
        deterministic: bool = True
    ) -> dict[str, any]:
    num_total_states = len(init_state_indices)
    num_unsafe_runs = 0
    num_goal_reached = 0
    num_failed_runs = 0

    from tqdm import tqdm
    for idx in tqdm(init_state_indices):
        obs, _ = env.reset(options={"idx": idx})
        if isinstance(env.unwrapped, JANIEnumEnv):
            # When the environment is a JANIEnumEnv, call the custom safety checker
            if not is_policy_safe(env, policy, obs, {}, deterministic):
                num_unsafe_runs += 1
        done = False
        truncated = False
        step_count = 0
        last_reward = None # Keep track of the last reward to determine if we reached the goal at the end of the episode
        while (not done) and (not truncated) and (step_count < max_steps):
            # print(f"Step {step_count}: Current observation: {env.debug_show_state(obs)}")
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
            # print(f"Step {step_count}: Action taken: {action}")
            
            # Step the environment
            obs, reward, done, truncated, _ = env.step(action)
            step_count += 1
            last_reward = reward

        if isinstance(env.unwrapped, JANIEnumEnv):
            if last_reward == 1.0: # We reached the goal
                num_goal_reached += 1
            elif last_reward == -1.0: # We reached a failure state
                num_failed_runs += 1
        else:
            if done:
                assert not truncated, "Episode ended due to truncation, which should not happen."
                # For Lava Env, episode done while the last reward is 0 means failure
                if last_reward == 0.0:
                    num_failed_runs += 1
                    num_unsafe_runs += 1
                else:
                    num_goal_reached += 1
                

    frac_unsafe = num_unsafe_runs / num_total_states if num_total_states > 0 else 0.0
    frac_goal = num_goal_reached / num_total_states if num_total_states > 0 else 0.0
    frac_failure = num_failed_runs / num_total_states if num_total_states > 0 else 0.0

    return {
        "frac_unsafe": frac_unsafe,
        "frac_goal": frac_goal,
        "frac_failure": frac_failure,
    }    

def main():
    parser = build_parser()
    args = parser.parse_args()
    file_args = _build_file_args(args)
    training_env = create_enum_env(
        file_args, 
        n_envs=args.n_envs, 
        monitor=args.verbose > 0, 
        time_limited=True
    )
    print(f"Created environment: {training_env}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = training_env.observation_space.shape[0]
    n_actions = training_env.action_space.n

    hyperparams = {
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_range': args.clip_range,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'max_grad_norm': args.max_grad_norm,
    }

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        training_env,
        verbose=args.verbose,
        device=args.device,
        **hyperparams
    )

    callbacks = []
    save_actor_callback = SaveActorCallback(
        save_freq=args.save_freq,
        save_path=Path(args.model_save_dir),
        verbose=args.verbose,
    )
    callbacks.append(save_actor_callback)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )

    # TODO: Iterate through all checkpoints and evaluate the safety results on them

    # TODO: Load the actor_iter_0 which is the randomly initialized policy
    checkpoint_path = Path(args.model_save_dir) / "actor_iter_0.pth"
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    network_params = {
        "input_dim": checkpoint["input_dim"],
        "output_dim": checkpoint["output_dim"],
        "hidden_dims": checkpoint["hidden_dims"],
    }
    actor_model = load_policy_checkpoint(checkpoint_path, device=args.device)
    actor_wrapper = NNPolicyWrapper(actor_model, device=args.device)
    policy_repairer = MILPPolicyUpdater()
    repairer_env = JANIEnumEnv(
        jani_model_path=file_args["jani_model"],
        jani_property_path=file_args["jani_property"],
        start_states_path=file_args["start_states"],
        objective_path=file_args["objective"],
        failure_property_path=file_args["failure_property"],
        seed=file_args["seed"],
        goal_reward=file_args["goal_reward"],
        use_oracle=file_args.get("use_oracle", False),
        failure_reward=file_args["failure_reward"],
        unsafe_reward=file_args.get("unsafe_reward", -0.01),
        disable_oracle_cache=file_args.get("disable_oracle_cache", False),
        reduced_memory_mode=file_args.get("reduced_memory_mode", False),
        max_training_idx=file_args.get("max_training_idx", 80000)
    ) 

    sampler = StandardTraceSampler()
    collector = OracleFaultCollector()

    # replay_buffer = DAggerBuffer(buffer_size=args.buffer_size)
    rng = np.random.default_rng(args.seed)

    all_faults = []
    step_count = 0
    num_saved_checkpoints = 0
    repairer_metric_log_file = log_dir / f"repairer_metrics.json"
    for i in range(args.max_iterations):
        print(f"Repair iteration {i + 1}/{args.max_iterations}")

        trajectories = []
        new_faults = []

        total_sampling_time = 0.0
        total_oracle_time = 0.0

        for _ in range(args.num_traces_per_iter):
            sample_start_time = time.perf_counter()

            trajectory = sampler.sample_trace(
                env=repairer_env,
                policy=actor_wrapper,
                max_steps=args.max_steps,
            )

            sample_end_time = time.perf_counter()
            time_for_sampling = sample_end_time - sample_start_time
            total_sampling_time += time_for_sampling

            if not trajectory["is_safe_trajectory"]: # We run fault analysis on the unsafe traces found here to get more informative fixes
                oracle_start_time = time.perf_counter()
                faults = collector.collect_faults(trajectory, repairer_env)
                oracle_end_time = time.perf_counter()
                time_for_oracle = oracle_end_time - oracle_start_time
                total_oracle_time += time_for_oracle

                new_faults.extend(faults)
                all_faults.extend(faults)

            trajectory_len = len(trajectory["observations"])
            step_count += trajectory_len

            if step_count > (args.save_freq * num_saved_checkpoints):
                print(f"Saving repaired policy at step count {step_count}...")
                save_policy(actor_model, network_params, Path(args.repair_save_dir), f"actor_iter_{num_saved_checkpoints}")
                num_saved_checkpoints += 1

            trajectories.append(trajectory)

        num_faults = len(all_faults)
        num_new_faults = len(new_faults)

        metrics = {
            "iteration": i,
            "num_traces": len(trajectories),
            "total_steps": step_count,
            "num_faults": num_faults,
            "it_faults": num_new_faults,
            "sampling_time": total_sampling_time,
            "oracle_time": total_oracle_time,
        }

        if num_new_faults == 0:
            converged = True
            print("No faults found. Pipeline converged.")
            break

        # Collect faults
        # if not args.accumulate_faults:
        #     replay_buffer.empty()
        #     assert False, "This should never be called. Unless we decide to later for experimental reasons."
        # positive, negative = faults_to_tensordict(all_faults, obs_dim=obs_dim, n_actions=n_actions)
        # replay_buffer.add_samples(positive, negative)

        # Repair policy based on collected faults
        _ = policy_repairer.update_policy(actor_model, all_faults)

        with repairer_metric_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    # Evaluate every checkpoint of the rl policy
    eval_env = create_enum_env(
        file_args, 
        n_envs=1, 
        monitor=False, 
        time_limited=True
    )
    eval_indices = list(range(args.max_training_idx + 1, eval_env.unwrapped.get_init_state_pool_size()))

    print("Evaluating RL policies on all checkpoints...")
    rl_metric_result_file = log_dir / f"rl_results.json"
    rl_checkpoints = sorted(Path(args.model_save_dir).glob("actor_iter_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    for checkpoint_path in rl_checkpoints:
        actor_model = load_policy_checkpoint(checkpoint_path, device=args.device)
        results = evaluate_policy(
            env=eval_env,
            policy=actor_model,
            init_state_indices=eval_indices,
            max_steps=args.max_steps,
            deterministic=True,
        )
        with rl_metric_result_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(results) + "\n")

    print("Evaluating repaired policies on all checkpoints...")
    repairer_result_file = log_dir / f"repairer_results.json"
    repair_checkpoints = sorted(Path(args.repair_save_dir).glob("actor_iter_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    for checkpoint_path in repair_checkpoints:
        actor_model = load_policy_checkpoint(checkpoint_path, device=args.device)
        results = evaluate_policy(
            env=eval_env,
            policy=actor_model,
            init_state_indices=eval_indices,
            max_steps=args.max_steps,
            deterministic=True,
        )
        with repairer_result_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(results) + "\n")

if __name__ == "__main__":
    main()