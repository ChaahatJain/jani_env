import argparse
import json
import torch
import numpy as np

from pathlib import Path

from torchrl.modules.distributions import MaskedCategorical
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from sb3_contrib.common.wrappers import ActionMasker

from jani.enum_env import JANIEnumEnv
from dagger.policy import Policy
from utils import mask_fn


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
    parser = argparse.ArgumentParser(description="Evaluate repaired policies on JANI environments.")
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

    ev = parser.add_argument_group("Evaluation")
    ev.add_argument("--repair_save_dir", type=str, required=True)
    ev.add_argument("--log_dir", type=str, required=True)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_args = _build_file_args(args)
    eval_env = create_enum_env(
        file_args, 
        n_envs=1, 
        monitor=False, 
        time_limited=True
    )
    eval_indices = list(range(args.max_training_idx + 1, eval_env.unwrapped.get_init_state_pool_size()))

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