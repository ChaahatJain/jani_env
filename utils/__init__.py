"""
Utility functions for training policies with stable-baselines3.
"""

import numpy as np
import torch

from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from jani.env import JANIEnv

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from sb3_contrib.common.wrappers import ActionMasker


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