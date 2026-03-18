"""
Utility functions for training policies with stable-baselines3.
"""
import sys
import numpy as np
import torch

from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase

from jani.env import JANIEnv

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from sb3_contrib.common.wrappers import ActionMasker

from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeRemainingColumn, 
    TimeElapsedColumn
)


def create_env(file_args: dict, n_envs: int = 1, monitor: bool = False, time_limited: bool = True) -> JANIEnv:
    """Create JANI environment with specified parameters."""
    def make_env():
        env = JANIEnv(
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
            reduced_memory_mode=file_args.get("reduced_memory_mode", False)
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

def create_eval_file_args(file_args: Dict[str, Any], use_separate_eval_env: bool = False) -> Dict[str, Any]:
    """Create file arguments for evaluation environment."""
    eval_file_args = file_args.copy()
    # Modify any parameters specific to evaluation if needed
    if use_separate_eval_env:
        eval_start_states = file_args.get("eval_start_states", "")
        assert eval_start_states != "", "Evaluation start states file must be provided when using separate eval env."
        eval_file_args["start_states"] = eval_start_states # use different start states for evaluation
    eval_file_args["seed"] += 1000  # offset seed for evaluation
    eval_file_args["use_oracle"] = False  # disable oracle during evaluation
    return eval_file_args

def create_safety_eval_file_args(file_args: Dict[str, Any], args: Dict[str, Any], use_oracle: bool = True) -> Dict[str, Any]:
    """Create file arguments for safety evaluation environment."""
    safety_eval_file_args = file_args.copy()
    # Modify any parameters specific to safety evaluation if needed
    eval_start_states = args.get("eval_start_states", "")
    if eval_start_states == "":
        eval_start_states = file_args.get("start_states", "")
    safety_eval_file_args["start_states"] = eval_start_states # use different start states for safety evaluation
    safety_eval_file_args["seed"] += 2000  # offset seed for safety evaluation
    safety_eval_file_args["use_oracle"] = use_oracle  # enable or disable oracle during safety evaluation
    return safety_eval_file_args

def mask_fn(env) -> np.ndarray:
    """Action masking function for the environment."""
    return env.unwrapped.action_mask()

def save_network(network: torch.nn.Module, network_paras: dict, save_path: Path, name: str):
    """Save the network to the specified path."""
    save_path.mkdir(parents=True, exist_ok=True)
    actor_path = save_path / f"{name}.pth"
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'input_dim': network_paras.get('input_dim'),
        'output_dim': network_paras.get('output_dim'),
        'hidden_dims': network_paras.get('hidden_dims'),
        'state_dict': network.state_dict()
    }, actor_path)


def safety_evaluation(
    env: JANIEnv,
    actor: TensorDictModule,
    max_steps: int = 256,
    progress: Progress = None,
    task_id = None
) -> Dict[str, float]:
    """Evaluate the safety of the current policy.

    Args:
        env: The environment to evaluate on.
        actor: The actor module to evaluate.
        max_steps: Maximum steps per episode.
        progress: Optional Progress instance to use for progress bar.
        task_id: Optional task ID within the progress instance.
    """
    num_init_states = env.get_init_state_pool_size()

    num_unsafe = 0
    rewards = []

    def _run_evaluation(prog, tid):
        nonlocal num_unsafe
        for idx in range(num_init_states):
            episode_reward = 0.0
            td_reset = TensorDict({"idx": torch.tensor(idx)}, batch_size=())
            obs_td = env.reset(td_reset)
            done = False
            keep_using_oracle = True
            step_count = 0
            while not done and step_count < max_steps:
                assert "observation" in obs_td, "Observation key missing in reset output"
                # Action selection using the actor
                td_action = actor(obs_td)
                assert "action" in td_action, "Action key missing in actor output"
                action = td_action.get("action").item()

                # Check whether the action is a safe action under the current state
                if keep_using_oracle:
                    is_safe = env.is_state_action_safe(action)
                    if not is_safe:
                        num_unsafe += 1
                        # If an unsafe action is found, no need to keep using the oracle
                        keep_using_oracle = False

                # Step the environment
                next_td = env.step(td_action)
                done = next_td.get(("next", "done")).item()
                episode_reward += next_td.get(("next", "reward")).item()
                obs_td = next_td.get("next")
                step_count += 1
            assert episode_reward == 0 or episode_reward == env._goal_reward or episode_reward == env._failure_reward, "Unexpected episode reward: {}".format(episode_reward)
            rewards.append(episode_reward)
            prog.update(tid, advance=1)

    if progress is not None and task_id is not None:
        # Use the provided progress bar
        progress.reset(task_id, total=num_init_states, visible=True)
        _run_evaluation(progress, task_id)
        progress.update(task_id, visible=False)
    else:
        # Create a standalone progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            disable=not sys.stdout.isatty(),
        ) as prog:
            tid = prog.add_task("Evaluating safety...", total=num_init_states)
            _run_evaluation(prog, tid)

    # Compute safety rate and average reward
    safety_rate = 1 - num_unsafe / num_init_states
    average_reward = sum(rewards) / len(rewards)
    results = {
        "safety_rate": safety_rate,
        "average_reward": average_reward,
    }
    return results