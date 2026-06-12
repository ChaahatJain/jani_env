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

from dagger.sampler import StandardTraceSampler
from dagger.fault_collector import OracleFaultCollector
from updater.goldberger import MILPPolicyUpdater

from dagger.policy import Policy
from dagger.policy_wrapper import NNPolicyWrapper

import torch
from typing import Any


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


def compute_mean_reward(eval_env, model, n_eval_episodes=10, enumate_all_init_states=False, goal_reward=1.0, failure_reward=-1.0, max_state_visits=3) -> float:
    rewards = []
    num_iter = n_eval_episodes
    goal_count = avoid_count = cycle_count = 0
    
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
        
        seen_obs = {}  # state -> visit count
        episode_reward = 0.0
        # break only when same state visited more than this many times (configurable via max_state_visits)
        
        while True:
            obs_key = obs.tobytes()
            visit_count = seen_obs.get(obs_key, 0)
            if visit_count >= max_state_visits:
                cycle_count += 1
                break # Genuine deterministic loop — policy cannot escape
            action_masks = get_action_masks(eval_env)
            action_masks = np.expand_dims(action_masks, axis=0)  # shape (1, n_actions)
            action, _ = model.predict(obs, action_masks=action_masks)
            seen_obs[obs_key] = visit_count + 1
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            
            if truncated:
                break
            if done:
                if info.get("reached_goal", False):
                    goal_count += 1
                elif info.get("reached_fail", False):
                    avoid_count += 1
                break
        
        rewards.append(episode_reward)
    mean_reward = np.mean(rewards)
    # print("Evaluation statistics:", mean_reward, goal_count, avoid_count, num_iter)
    return mean_reward, goal_count / num_iter, avoid_count / num_iter, cycle_count / num_iter

import csv
import time
class SaveActorCallback(BaseCallback):
    """Callback for saving the model at regular intervals."""
    def __init__(self, eval_env, n_eval_episodes : int, enumate_all_init_states: bool, save_freq: int, save_path: Path, log_path: Path, verbose=0, goal_reward = 1.0, failure_reward = -1.0, use_timestep_freq: bool = False, max_state_visits: int = 3):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.use_timestep_freq = use_timestep_freq  # if True, save_freq is in timesteps; if False, in episodes
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.start_time = None
        self.log_path = log_path
        self.episodes_since_last_save = 0
        self.total_episodes = 0
        self.last_checkpoint_timestep = 0
        self.checkpoint_idx = 0
        self.eval_env = eval_env
        self.enumate_all_init_states = enumate_all_init_states
        self.n_eval_episodes = n_eval_episodes
        self.goal_reward = goal_reward
        self.failure_reward = failure_reward
        self.max_state_visits = max_state_visits

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.episodes_count = 0
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["Checkpoint", "Timestep", "Elapsed(s)", "MeanReward", "GoalFrac", "AvoidFrac", "CycleFrac", "Episodes"])
            
    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for d in dones:
            if d:
                self.episodes_since_last_save += 1
                self.total_episodes += 1

        should_checkpoint = (
            self.num_timesteps - self.last_checkpoint_timestep >= self.save_freq
            if self.use_timestep_freq
            else self.episodes_since_last_save >= self.save_freq
        )
        if should_checkpoint:
            policy = self.model.policy
            network_paras = {
                'input_dim': self.training_env.observation_space.shape[0],
                'output_dim': self.training_env.action_space.n,
                'hidden_dims': policy.net_arch['pi']
            }
            save_policy(
                policy, network_paras, self.save_path,
                f"actor_iter_{self.checkpoint_idx}"
            )

            elapsed = time.time() - self.start_time
            avg_reward, goal_frac, failure_frac, cycle_frac = compute_mean_reward(self.eval_env, self.model, self.n_eval_episodes, self.enumate_all_init_states, self.goal_reward, self.failure_reward, self.max_state_visits)

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.checkpoint_idx, self.num_timesteps, round(elapsed, 2),
                    round(avg_reward, 4), round(goal_frac, 4), round(failure_frac, 4),
                    round(cycle_frac, 4), self.episodes_since_last_save
                ])
                f.flush()

            if self.verbose:
                print(
                    f"[SaveActorCallback] checkpoint={self.checkpoint_idx} | t={self.num_timesteps} | "
                    f"elapsed={elapsed:.1f}s | avg_reward={avg_reward:.4f} | "
                    f"goal={goal_frac:.2%} | failure={failure_frac:.2%} | cycle={cycle_frac:.2%} | "
                    f"episodes={self.episodes_since_last_save}"
                )

            self.episodes_since_last_save = 0
            self.last_checkpoint_timestep = self.num_timesteps
            self.checkpoint_idx += 1

        return True

class ModelRepairCallback(BaseCallback):
    """
    Periodically samples traces from the repair environment, collects faults
    on unsafe trajectories, and repairs the policy's final layer via MILP.
    Maintains a fault cache across all repair rounds to avoid duplicates and
    use the full fault history for each repair.
    """

    def __init__(
        self,
        repair_env,
        repair_freq: int,
        n_episodes_for_repair: int,
        save_actor_callback: SaveActorCallback,
        max_steps : int,
        log_file : Path,
        verbose: bool,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.repair_env = repair_env
        self.repair_freq = repair_freq
        self.n_episodes_for_repair = n_episodes_for_repair
        self.save_actor_callback = save_actor_callback
        self.max_steps = max_steps
        self.device = device

        self.sampler = StandardTraceSampler()
        self.collector = OracleFaultCollector()
        self.updater = MILPPolicyUpdater()

        self.total_episodes = 0
        self.episodes_since_last_repair = 0
        self.verbose = verbose

        self.log_file = log_file
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w", newline="") as f:
            csv.writer(f).writerow(["Episodes", "RepairTime", "NumFaultsTotal", "NumFaultsThisRound"])

        # Fault cache — persists across all repair rounds
        self.all_faults: list = []
        self.fault_cache: set = set()

    def _build_policy_wrapper(self) -> NNPolicyWrapper:
        """Extract current SB3 policy weights into a Policy wrapper for the sampler."""
        sb3_policy = self.model.policy
        state_dict = sb3_policy.state_dict()

        input_dim = self.model.observation_space.shape[0]
        output_dim = self.model.action_space.n
        hidden_dims = sb3_policy.net_arch["pi"]

        policy_model = Policy(input_dim, output_dim, hidden_dims, activation_fn=torch.nn.Tanh)
        mapped = {
            "model.0.weight": state_dict["mlp_extractor.policy_net.0.weight"],
            "model.0.bias":   state_dict["mlp_extractor.policy_net.0.bias"],
            "model.2.weight": state_dict["mlp_extractor.policy_net.2.weight"],
            "model.2.bias":   state_dict["mlp_extractor.policy_net.2.bias"],
            "model.4.weight": state_dict["action_net.weight"],
            "model.4.bias":   state_dict["action_net.bias"],
        }
        policy_model.load_state_dict(mapped, strict=True)
        policy_model.to(self.device)
        return NNPolicyWrapper(policy_model, device=self.device)

    def _collect_new_faults(self, traces: list) -> list:
        """
        Run fault analysis on unsafe traces, deduplicate against the cache,
        and update the persistent fault store.
        """
        new_faults_this_round = []
        duplicates = 0

        for trace in traces:
            if not trace["is_safe_trajectory"]:
                faults = self.collector.collect_faults(trace, self.repair_env)
                new = [
                    f for f in faults
                    if (tuple(f["observation"]), f["faulty_action"]) not in self.fault_cache
                ]
                duplicates += len(faults) - len(new)
                new_faults_this_round.extend(new)
                self.all_faults.extend(new)
                self.fault_cache.update(
                    (tuple(f["observation"]), f["faulty_action"]) for f in new
                )

        if self.verbose:
            print(
                f"[ModelRepairCallback] faults this round: {len(new_faults_this_round)} new, "
                f"{duplicates} duplicates | total in cache: {len(self.all_faults)}"
            )

        return new_faults_this_round

    def _on_step(self) -> bool:
        # --- Track episode completions ---
        dones = self.locals.get("dones", [])
        for d in dones:
            if d:
                self.episodes_since_last_repair += 1
                self.total_episodes += 1

        if self.episodes_since_last_repair < self.repair_freq:
            return True

        self.episodes_since_last_repair = 0

        # 1. Wrap live SB3 weights into the Policy interface the sampler expects
        policy_wrapper = self._build_policy_wrapper()
        policy_model = policy_wrapper.model
        # 2. Sample traces
        traces = []
        for _ in range(self.n_episodes_for_repair):
            trace = self.sampler.sample_trace(
                env=self.repair_env,
                policy=policy_wrapper,
                init_state_idx=-1,
                max_steps=self.max_steps,
                verbose=False,
            )
            traces.append(trace)
            
        if not traces:
            return True

        # 3. Collect faults from unsafe traces, deduplicated against full history
        new_faults = self._collect_new_faults(traces)
        print("New faults found:", len(new_faults))

        # 4. Skip repair if no new faults found this round — policy is safe on sampled traces
        if not new_faults:
            if self.verbose:
                print("[ModelRepairCallback] No new faults found this round — skipping repair.")
            return True

        # 5. Repair using the full fault history — mutates self.model.policy in-place
        repair_time_start = time.perf_counter()
        update_info = self.updater.update_policy(policy_model, self.all_faults)
        repair_time = time.perf_counter() - repair_time_start
        # 6. Sync repaired weights back into SB3 policy
        repaired_state = policy_model.state_dict()
        sb3_state = self.model.policy.state_dict()
        sb3_state["mlp_extractor.policy_net.0.weight"] = repaired_state["model.0.weight"]
        sb3_state["mlp_extractor.policy_net.0.bias"]   = repaired_state["model.0.bias"]
        sb3_state["mlp_extractor.policy_net.2.weight"] = repaired_state["model.2.weight"]
        sb3_state["mlp_extractor.policy_net.2.bias"]   = repaired_state["model.2.bias"]
        sb3_state["action_net.weight"]                 = repaired_state["model.4.weight"]
        sb3_state["action_net.bias"]                   = repaired_state["model.4.bias"]
        self.model.policy.load_state_dict(sb3_state)

        optimizer = self.model.policy.optimizer
        # Find and reset state only for action_net parameters
        for name, param in self.model.policy.named_parameters():
            if "action_net" in name or "policy_net" in name:
                if param in optimizer.state:
                    optimizer.state[param] = {}

        if self.verbose:
            print(f"[ModelRepairCallback] update_info: {update_info}")
            # repair_metrics(self._build_policy_wrapper().model, self.all_faults, self.verbose)

        with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.total_episodes, round(repair_time, 2), len(self.all_faults), len(new_faults)
                ])
                f.flush()

        

        # 6. Inflate SaveActorCallback counter so it fires on next step
        self.save_actor_callback.episodes_since_last_save += self.n_episodes_for_repair

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