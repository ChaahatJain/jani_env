""" 
Callback functions for training and evaluating policies using stable-baselines3. 
"""

from dataclasses import dataclass, field
import numpy as np
import torch
import pandas as pd
import os, psutil, tracemalloc
from pathlib import Path
from typing import Optional
from pathlib import Path

# Optional imports for advanced features
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


p = psutil.Process(os.getpid())
tracemalloc.start()

def snap(tag):
    rss = p.memory_info().rss / 1024**2
    current, peak = tracemalloc.get_traced_memory()
    print(f"{tag}: RSS={rss:.1f}MB | py_current={current/1024**2:.1f}MB | py_peak={peak/1024**2:.1f}MB")



def save_policy(policy: torch.nn.Module, network_paras: dict, save_path: Path, name: str):
    """Save the policy network to the specified path."""
    save_path.mkdir(parents=True, exist_ok=True)
    actor_path = save_path / f"{name}.pth"
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'input_dim': network_paras.get('input_dim'),
        'output_dim': network_paras.get('output_dim'),
        'hidden_dims': network_paras.get('hidden_dims'),
        'state_dict': policy.state_dict()
    }, actor_path)


def compute_mean_reward(eval_env, model, n_eval_episodes=10, enumate_all_init_states=False) -> float:
    rewards = []
    num_iter = n_eval_episodes

    # If enumerating all initial states for evaluation, set num_iter accordingly
    if enumate_all_init_states:
        # unwrap the environment to get JANIEnv
        unwrapped_env = eval_env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        if hasattr(unwrapped_env, 'unwrapped'):
            unwrapped_env = unwrapped_env.unwrapped
        # Set the number of iterations to size of initial state pool
        num_iter = unwrapped_env.get_init_state_pool_size()

    for i in range(num_iter):
        if enumate_all_init_states:
            obs, _ = eval_env.reset(options={"idx": i})
        else:
            obs, _ = eval_env.reset()
        done = False
        truncated = False
        episode_rewards = 0.0
        while not done and not truncated:
            action_masks = get_action_masks(eval_env)
            action_masks = np.expand_dims(action_masks, axis=0)  # shape (1, n_actions)
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, truncated, _ = eval_env.step(action)
            episode_rewards += reward
        rewards.append(episode_rewards)
    mean_reward = np.mean(rewards)
    return mean_reward


class SaveActorCallback(BaseCallback):
    """Callback for saving the model at regular intervals."""
    def __init__(self, save_freq: int, save_path: Path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.save_freq > 0 and (self.n_calls - 1) % self.save_freq == 0:
            policy = self.model.policy
            network_paras = {
                'input_dim': self.training_env.observation_space.shape[0],
                'output_dim': self.training_env.action_space.n,
                'hidden_dims': policy.net_arch['pi']
            }
            save_policy(
                policy, network_paras, self.save_path, 
                f"actor_iter_{self.n_calls // self.save_freq}"
            )
        return True
    

class LoggingCallback(BaseCallback):
    """Custom logging callback."""
    def __init__(self, log_dir, log_freq, eval_env, enumate_all_init_states=False, n_eval_episodes=100, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.eval_env = eval_env
        self.enumate_all_init_states = enumate_all_init_states
        self.n_eval_episodes = n_eval_episodes
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "avg_rewards.txt"
        open(self.log_file, 'w').close()  # Create or clear log file

    def _on_step(self) -> bool:
        # Custom logging logic can be added here
        if self.log_freq > 0 and self.n_calls % self.log_freq == 0:
            mean_reward = compute_mean_reward(self.eval_env, self.model, self.n_eval_episodes, self.enumate_all_init_states)
            with open(self.log_file, 'a') as f:
                f.write(f"{self.num_timesteps}\t{mean_reward}\n")
        return True


class EvalCallback(BaseCallback):
    """Custom evaluation callback."""
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, best_model_save_path: Optional[str] = None, enumate_all_init_states=False):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.enumate_all_init_states = enumate_all_init_states
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -float('inf')
        self.last_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            mean_reward = compute_mean_reward(self.eval_env, self.model, self.n_eval_episodes, self.enumate_all_init_states)
            self.last_mean_reward = mean_reward
            if self.best_model_save_path is not None:
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    policy = self.model.policy
                    network_paras = {
                        'input_dim': self.training_env.observation_space.shape[0],
                        'output_dim': self.training_env.action_space.n,
                        'hidden_dims': policy.net_arch['pi']
                    }
                    save_policy(
                        policy, network_paras, 
                        Path(self.best_model_save_path), "best_model"
                    )
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'eval/mean_reward': mean_reward,
                    'eval/best_mean_reward': self.best_mean_reward,
                    'eval/timesteps': self.n_calls
                })
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/best_mean_reward', self.best_mean_reward)
            self.logger.record('eval/timesteps', self.n_calls)
        return True


class SafetyEvalCallback(BaseCallback):
    """Custom safety evaluation callback."""

    def __init__(self, safety_eval_env, eval_freq: int, log_dir: Optional[Path] = None):
        super().__init__()
        self.safety_eval_env = safety_eval_env
        self.eval_freq = eval_freq
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = log_dir / "safety_eval.txt"
            open(self.log_file, 'w').close()  # Create or clear log file
        else:
            self.log_file = None

    def _unwrap_to_jani_env(self, env):
        """Helper method to unwrap environment to get JaniEnv."""
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        if hasattr(unwrapped_env, 'unwrapped'):
            unwrapped_env = unwrapped_env.unwrapped
        return unwrapped_env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print(f"Starting safety evaluation... (Timesteps: {self.n_calls})")
            if hasattr(self.safety_eval_env, 'envs'):
                # Vectorized environment - get first individual environment
                unwrapped_env = self._unwrap_to_jani_env(self.safety_eval_env.envs[0])
            else:
                # Single environment
                unwrapped_env = self._unwrap_to_jani_env(self.safety_eval_env)
            init_pool_size = unwrapped_env.get_init_state_pool_size()
            num_unsafe_episode = 0 # count number of episodes with unsafe steps
            episode_rewards = []
            for idx in range(init_pool_size):  
                obs, _ = self.safety_eval_env.reset(options={"idx": idx})
                done = False
                truncated = False
                rewards = []
                keep_using_oracle = True
                while not done and not truncated:
                    # snap("        Inside eval step ")
                    action_masks = get_action_masks(self.safety_eval_env)
                    action_masks = np.expand_dims(action_masks, axis=0)  # shape (1, n_actions)
                    action, _ = self.model.predict(obs, action_masks=action_masks)
                    if keep_using_oracle:
                        is_action_safe = unwrapped_env.is_current_state_action_safe(action)
                        if not is_action_safe:
                            keep_using_oracle = False  # Stop using oracle for the rest of this episode
                            num_unsafe_episode += 1
                    obs, reward, done, truncated, _ = self.safety_eval_env.step(action)
                    rewards.append(reward)
                # Compute average reward for this episode
                episode_reward = sum(rewards)
                assert episode_reward == rewards[-1], f"Episode reward {episode_reward} should equal the last step reward {rewards[-1]} in JANIEnv where intermediate rewards are 0 and only terminal reward is non-zero."
                episode_rewards.append(episode_reward)
    
            safe_episode_rate = (init_pool_size - num_unsafe_episode) / init_pool_size
            avg_reward = np.mean(episode_rewards)
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'safety_eval/safe_episodes_rate': safe_episode_rate,
                    'safety_eval/timesteps': self.n_calls
                })
            self.logger.record('safety_eval/safe_episodes_rate', safe_episode_rate)
            self.logger.record('safety_eval/timesteps', self.n_calls)
            with open(self.log_file, 'a') as f:
                f.write(f"{self.n_calls}\t{safe_episode_rate}\t{avg_reward}\n")
        return True


class WandbCallback(BaseCallback):
    """Custom callback for Weights & Biases logging."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log metrics to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            # Log training environment rewards
            if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                if len(self.model.ep_info_buffer) > 0:
                    ep_info = self.model.ep_info_buffer[-1]
                    self.episode_rewards.append(ep_info['r'])
                    self.episode_lengths.append(ep_info['l'])
                    
                    # Log individual episode metrics
                    wandb.log({
                        'train/episode_reward': ep_info['r'],
                        'train/episode_length': ep_info['l'],
                        'train/episode_time': ep_info['t'],
                        'train/timesteps': self.num_timesteps
                    })
            
            # Log training statistics every 100 steps
            if self.n_calls % 100 == 0 and len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                wandb.log({
                    'train/mean_reward_last_10': np.mean(recent_rewards),
                    'train/std_reward_last_10': np.std(recent_rewards),
                    'train/max_reward': np.max(self.episode_rewards),
                    'train/min_reward': np.min(self.episode_rewards),
                    'train/total_episodes': len(self.episode_rewards),
                    'train/timesteps': self.num_timesteps
                })
            
            # Log model training metrics
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                log_dict = {}
                for key, value in self.model.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        log_dict[f'train/{key}'] = value
                if log_dict:
                    wandb.log(log_dict)
        
        return True