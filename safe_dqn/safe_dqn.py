# python -m safe_dqn.train --jani_model examples/one_way_line_15_10/model.jani --jani_property examples/one_way_line_15_10/property.jani --start_states examples/one_way_line_15_10/property.jani --eval_start_states examples/one_way_line_15_10/eval_start_states.jani --goal_reward 1.0 --failure_reward -1.0 --unsafe_reward -0.01   --max_steps 256 --total_timesteps 1000 --n_eval_episodes 100 --experiment_name one_way_line_15_10_det   --log_dir /jani_env/logs/ppo/one_way_line_15_10 --model_save_dir /jani_env/models/ppo/one_way_line_15_10 --disable_eval --enumate_all_init_states --log_reward --eval_freq 1025 --eval_safety --disable_wandb --verbose 1 --device cpu --seed 50

# TODO: @Songtuan please review

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import deque
import random


from utils import create_env, create_eval_file_args
from dagger.sampler import StandardTraceSampler
from dagger.policy import Policy
from dagger.policy_wrapper import NNPolicyWrapper
from dagger.fault_collector import OracleFaultCollector

from updater.goldberger import MILPPolicyUpdater
from updater.spec_repair import SpecRepairPolicyUpdater

import csv
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available. Advanced logging will be disabled.")


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------

class SafeDQNNetwork(nn.Module):
    """
    Dueling-style network with two heads:
      - Q-value head  : standard action-value estimates
      - Safety head   : per-action cost estimates (C-values)
    Both share a common feature extractor.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(negative_slope=0.01)]
            prev = h
        self.feature_net = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev, output_dim)
        self.c_head = nn.Linear(prev, output_dim)   # cost / safety head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(x)
        return self.q_head(features), self.c_head(features)

# Create a wrapper around your SafeDQNNetwork that only exposes Q-values
class QValueOnlyWrapper(nn.Module):
    """
    Wraps SafeDQNNetwork to expose only Q-values as logits for the policy.
    """
    def __init__(self, safe_dqn_net: SafeDQNNetwork):
        super().__init__()
        self.safe_dqn_net = safe_dqn_net
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_vals, _ = self.safe_dqn_net(x)
        return q_vals  # Return only Q-values as the "logits"
    
    @property
    def model(self):
        """Make policy.model work in the updater."""
        return self
    
    def __getitem__(self, idx):
        """Support policy.model[:-1] and policy.model[-1] indexing."""
        if idx == -1:
            # Return the Q-head as the final layer
            return self.safe_dqn_net.q_head
        elif isinstance(idx, slice) and idx == slice(None, -1, None):
            # Return feature extractor (everything before the final layer)
            return self.safe_dqn_net.feature_net
        else:
            raise IndexError(f"Only -1 (head) and [:-1] (feature extractor) are supported")

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple uniform replay buffer storing (s, a, r, c, s', done, mask)."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, cost, next_state, done, action_mask, next_action_mask):
        self.buffer.append((state, action, reward, cost, next_state, done, action_mask, next_action_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, costs, next_states, dones, masks, next_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(costs, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(masks, dtype=np.float32),
            np.array(next_masks, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Safe-DQN agent
# ---------------------------------------------------------------------------

class SafeDQNAgent:
    """
    Safe DQN with action masking and a cost-value head.

      1. Invalid actions are masked out (set to -inf).
      2. Among valid actions, any action whose estimated cost C(s,a) exceeds
         `cost_threshold` is further masked unless *all* valid actions are unsafe
         (in which case we fall back to the least-cost valid action).

    The cost head is trained with a separate Bellman target using the same
    discount factor as the reward head.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int],
        lr: float,
        gamma: float,
        cost_threshold: float,
        cost_gamma: float,
        tau: float,                 # soft-update coefficient for target network
        device: str,
    ):
        self.act_dim = act_dim
        self.gamma = gamma
        self.cost_threshold = cost_threshold
        self.cost_gamma = cost_gamma
        self.tau = tau
        self.device = torch.device(device)

        self.online_net = SafeDQNNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
        self.target_net = SafeDQNNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        """ε-greedy selection with validity + safety masking."""
        valid = torch.tensor(action_mask, dtype=torch.bool, device=self.device)

        if random.random() < epsilon:
            # Random among valid actions only
            valid_indices = valid.nonzero(as_tuple=True)[0].cpu().numpy()
            return int(np.random.choice(valid_indices))

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals, c_vals = self.online_net(obs_t)
        q_vals = q_vals.squeeze(0)
        c_vals = c_vals.squeeze(0)

        # Mask invalid actions
        q_vals[~valid] = -float("inf")

        # Safety masking: penalise actions that exceed cost threshold
        safe = valid & (c_vals <= self.cost_threshold)
        if safe.any():
            q_vals[~safe] = -float("inf")
        # else: all valid actions are unsafe → fall back to greedy over valid actions

        return int(q_vals.argmax().item())

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def update(self, batch) -> Dict[str, float]:
        states, actions, rewards, costs, next_states, dones, masks, next_masks = batch

        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device).unsqueeze(1)
        rewards_t     = torch.tensor(rewards,     device=self.device).unsqueeze(1)
        costs_t       = torch.tensor(costs,       device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device).unsqueeze(1)
        next_masks_t  = torch.tensor(next_masks,  dtype=torch.bool, device=self.device)

        # Current Q and C estimates
        q_vals, c_vals = self.online_net(states_t)
        q_pred = q_vals.gather(1, actions_t)
        c_pred = c_vals.gather(1, actions_t)

        with torch.no_grad():
            # Double-DQN: action selected by online net, evaluated by target net
            next_q_online, next_c_online = self.online_net(next_states_t)

            # Mask invalid next actions
            INF = float("inf")
            next_q_online[~next_masks_t] = -INF

            # Safety masking for next-action selection
            next_c_for_mask = next_c_online.clone()
            next_safe = next_masks_t & (next_c_for_mask <= self.cost_threshold)
            if next_safe.any(dim=1).all():
                next_q_online[~next_safe] = -INF

            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target, next_c_target = self.target_net(next_states_t)
            next_q = next_q_target.gather(1, next_actions)
            next_c = next_c_target.gather(1, next_actions)

            q_target = rewards_t + self.gamma      * (1 - dones_t) * next_q
            c_target = costs_t   + self.cost_gamma * (1 - dones_t) * next_c

        q_loss = self.loss_fn(q_pred, q_target)
        c_loss = self.loss_fn(c_pred, c_target)
        loss   = q_loss + c_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {
            "loss/total": loss.item(),
            "loss/q":     q_loss.item(),
            "loss/cost":  c_loss.item(),
        }

    def soft_update_target(self):
        for online_p, target_p in zip(self.online_net.parameters(), self.target_net.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1 - self.tau) * target_p.data)

    def save(self, path: Path):
        torch.save({
            "state_dict": self.online_net.state_dict(),
            "input_dim":  next(self.online_net.parameters()).shape[-1],   # approx
            "act_dim":    self.act_dim,
        }, path)


# ---------------------------------------------------------------------------
# Epsilon schedule helpers
# ---------------------------------------------------------------------------

def linear_epsilon_schedule(step: int, eps_start: float, eps_end: float, eps_decay_steps: int) -> float:
    fraction = min(step / max(eps_decay_steps, 1), 1.0)
    return eps_start + fraction * (eps_end - eps_start)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

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

def train_model(args, file_args: Dict[str, str], hyperparams: Optional[Dict[str, Any]] = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"safe_dqn_{timestamp}"

    log_dir       = Path(args.log_dir)
    model_save_dir = Path(args.model_save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # ----- environments -----
    print("Creating training environment...")
    print(f"🤖 Oracle enabled: {file_args.get('use_oracle', False)}")
    train_env = create_env(file_args, n_envs=1, monitor=False, time_limited=True)

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.n

    # ----- hyperparameters -----
    if hyperparams is None:
        hyperparams = {
            "lr":               args.lr,
            "gamma":            args.gamma,
            "cost_gamma":       args.cost_gamma,
            "cost_threshold":   args.cost_threshold,
            "hidden_dims":      args.hidden_dims,
            "buffer_capacity":  args.buffer_capacity,
            "batch_size":       args.batch_size,
            "tau":              args.tau,
            "eps_start":        args.eps_start,
            "eps_end":          args.eps_end,
            "eps_decay_steps":  args.eps_decay_steps,
            "learning_starts":  args.learning_starts,
            "train_freq":       args.train_freq,
            "target_update_freq": args.target_update_freq,
        }

    print(f"Training with hyperparameters: {hyperparams}")

    agent = SafeDQNAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=hyperparams["hidden_dims"],
        lr=hyperparams["lr"],
        gamma=hyperparams["gamma"],
        cost_threshold=hyperparams["cost_threshold"],
        cost_gamma=hyperparams["cost_gamma"],
        tau=hyperparams["tau"],
        device=args.device,
    )

    # Load pre-trained policy if specified
    if args.load_policy_path:
        print(f"Loading pre-trained policy from {args.load_policy_path}...")
        checkpoint = torch.load(args.load_policy_path, map_location=args.device, weights_only=False)
        saved_sd = checkpoint["state_dict"]

        # Detect simple Policy checkpoint (keys like "model.0.weight")
        if any(k.startswith("model.") for k in saved_sd):
            print("Detected simple Policy checkpoint – remapping weights to SafeDQNNetwork...")
            new_sd = agent.online_net.state_dict()
            # Identify hidden vs output layers in simple Policy
            layer_keys = sorted(
                [k for k in saved_sd if k.startswith("model.") and "weight" in k],
                key=lambda k: int(k.split(".")[1]),
            )
            hidden_indices = [int(k.split(".")[1]) for k in layer_keys[:-1]]
            output_index = int(layer_keys[-1].split(".")[1])

            # Hidden layers → feature_net
            for feat_idx, src_idx in enumerate(hidden_indices):
                for suffix in ("weight", "bias"):
                    new_sd[f"feature_net.{feat_idx * 2}.{suffix}"] = saved_sd[f"model.{src_idx}.{suffix}"]
            # Output layer → q_head (c_head stays random-initialized)
            for suffix in ("weight", "bias"):
                new_sd[f"q_head.{suffix}"] = saved_sd[f"model.{output_index}.{suffix}"]

            agent.online_net.load_state_dict(new_sd)
            agent.target_net.load_state_dict(new_sd)
        else:
            agent.online_net.load_state_dict(saved_sd)
            agent.target_net.load_state_dict(saved_sd)

    if args.enable_repair:
        sampler = StandardTraceSampler()
        collector = OracleFaultCollector()
        if args.repair_algo == "spec":
            optimizer = optim.Adam(agent.online_net.parameters(), lr=hyperparams["lr"])
            updater = SpecRepairPolicyUpdater(optimizer=optimizer, batch_size=hyperparams["batch_size"], device=args.device)
        else:
            updater = MILPPolicyUpdater()
        all_faults: list = []
        fault_cache: set = set()

    replay_buffer = ReplayBuffer(hyperparams["buffer_capacity"])

    # ----- wandb -----
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            config={**vars(args), **hyperparams, "file_args": file_args},
        )

    # ----- optional eval env -----
    eval_env = None
    if not args.disable_eval:
        eval_file_args = create_eval_file_args(file_args, args.use_separate_eval_env)
        eval_env = create_env(eval_file_args, 1, monitor=False, time_limited=True)

    # ----- training loop -----
    obs, _ = train_env.reset()
    episode_reward  = 0.0
    episode_cost    = 0.0
    episode_len     = 0
    episode_count   = 0
    episodes_since_last_eval = 0
    episodes_since_last_repair = 0
    best_eval_reward = -float("inf")

    metrics_window = deque(maxlen=100)  # rolling episode stats'

    time_start = time.time()
    checkpoint_idx = 0

    for global_step in range(1, args.total_timesteps + 1):
        epsilon = linear_epsilon_schedule(
            global_step,
            hyperparams["eps_start"],
            hyperparams["eps_end"],
            hyperparams["eps_decay_steps"],
        )

        # Get action mask from environment (sb3-contrib compatible interface)
        action_mask = train_env.action_masks()

        action = agent.select_action(obs, action_mask, epsilon)
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated

        # Cost signal: info dict may supply 'cost'; fall back to 0.
        cost = float(info.get("cost", 0.0)) if isinstance(info, dict) else 0.0

        next_mask = train_env.action_masks()

        replay_buffer.push(obs, action, reward, cost, next_obs, done, action_mask, next_mask)

        obs = next_obs if not done else train_env.reset()[0]
        episode_reward += reward
        episode_cost   += cost
        episode_len    += 1

        if done:
            metrics_window.append({
                "ep_reward": episode_reward,
                "ep_cost":   episode_cost,
                "ep_len":    episode_len,
            })
            episode_count  += 1
            episodes_since_last_eval += 1
            episodes_since_last_repair += 1
            episode_reward  = 0.0
            episode_cost    = 0.0
            episode_len     = 0

        # ----- learning -----
        if (global_step >= hyperparams["learning_starts"]
                and len(replay_buffer) >= hyperparams["batch_size"]
                and global_step % hyperparams["train_freq"] == 0):

            batch   = replay_buffer.sample(hyperparams["batch_size"])
            metrics = agent.update(batch)

            if WANDB_AVAILABLE and not args.disable_wandb and global_step % 100 == 0:
                wandb.log({"train/epsilon": epsilon, **metrics, "timestep": global_step})

        # Soft-update target network
        if global_step % hyperparams["target_update_freq"] == 0:
            agent.soft_update_target()

        # ----- evaluation -----
        if not args.disable_eval and eval_env is not None and episodes_since_last_eval >= args.eval_freq:
            episodes_since_last_eval = 0
            mean_reward, goal_fraction, avoid_fraction = _evaluate(agent, eval_env, n_episodes=args.n_eval_episodes, enumate_all_init_states=args.enumate_all_init_states, goal_reward=file_args.get("goal_reward", 1.0), failure_reward=file_args.get("failure_reward", -1.0))
            performance_file = file_args.get("performance_file", None)

            if WANDB_AVAILABLE and not args.disable_wandb:
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "timestep":         global_step,
                })

            with open(performance_file, "a", newline="") as f:
                    elapsed = time.time() - time_start
                    writer = csv.writer(f)
                    writer.writerow([
                        checkpoint_idx, global_step, round(elapsed, 2),
                        round(mean_reward, 4), round(goal_fraction, 4), round(avoid_fraction, 4),
                        episode_count
                    ])
                    f.flush()
            if args.save_all_checkpoints:
                print("Here", model_save_dir / f"actor_iter_{checkpoint_idx}.pth")
                agent.save(model_save_dir / f"actor_iter_{checkpoint_idx}.pth")
            checkpoint_idx += 1

            print(f"  ↳ eval  reward={mean_reward:.3f}  goal_frac={goal_fraction}, avoid_frac={avoid_fraction}")

        # ----- repair ------
        if args.enable_repair and episodes_since_last_repair >= args.repair_freq:
            episodes_since_last_repair = 0
            # Placeholder for repair logic - to be implemented
            q_value_wrapper = QValueOnlyWrapper(agent.online_net)
            policy_wrapper = NNPolicyWrapper(q_value_wrapper, device=args.device)

            # 2. Sample traces
            traces = []
            for _ in range(args.repair_episodes):
                trace = sampler.sample_trace(
                    env=train_env.env.env, # Unwrap time and action masker
                    policy=policy_wrapper,
                    init_state_idx=-1,
                    max_steps=args.max_steps,
                    verbose=False,
                )
                traces.append(trace)
                global_step += len(trace["observations"])  # Account for the additional timesteps used for sampling traces
                
            if not traces:
                continue

            # 3. Collect faults from unsafe traces, deduplicated against full history
            new_faults_this_round = []
            duplicates = 0

            for trace in traces:
                if not trace["is_safe_trajectory"]:
                    faults = collector.collect_faults(trace, train_env.env.env)  # Unwrap time and action masker
                    new =  new = [f for f in faults if (tuple(f["observation"]), f["faulty_action"]) not in fault_cache]
                    duplicates += len(faults) - len(new)
                    new_faults_this_round.extend(new)
                    all_faults.extend(new)
                    fault_cache.update((tuple(f["observation"]), f["faulty_action"]) for f in new)
            if args.verbose > 0:
                print(f"Repair iteration: Collected {len(new_faults_this_round)} new faults, {duplicates} duplicates (total unique faults: {len(fault_cache)})")
            

            
            repair_time_start = time.perf_counter()
            _ = updater.update_policy(q_value_wrapper, all_faults)
            repair_time = time.perf_counter() - repair_time_start

            with open(args.repair_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode_count, round(repair_time, 2), len(all_faults), len(new_faults_this_round)
                ])
                f.flush()
            episode_count += args.repair_episodes
            episodes_since_last_eval += args.repair_episodes

            # repair_metrics(QValueOnlyWrapper(agent.online_net), all_faults, verbose=args.verbose > 0)
    # ----- final save -----
    final_path = model_save_dir / "final_actor.pth"
    agent.save(final_path)
    print(f"Final actor model saved to {final_path}")

    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.finish()

def _evaluate(agent, eval_env, n_episodes=10, enumate_all_init_states=False, goal_reward=1.0, failure_reward=-1.0) -> Tuple[float, float, float]:
        num_iter = n_episodes
        rewards = []
        if enumate_all_init_states:
            # unwrap the environment to get JANIEnv
            unwrapped_env = eval_env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
            if hasattr(unwrapped_env, 'unwrapped'):
                unwrapped_env = unwrapped_env.unwrapped
            # Set the number of iterations to size of initial state pool
            num_iter = unwrapped_env.get_init_state_pool_size()
        goal_count = 0
        avoid_count = 0
        for i in range(num_iter):
            if enumate_all_init_states:
                e_obs, _ = eval_env.reset(options={"idx": i})
            else:
                e_obs, _ = eval_env.reset()
            ep_r = 0.0
            
            seen_obs = set()
            while True:
                e_mask = eval_env.action_masks()
                e_act  = agent.select_action(e_obs, e_mask, epsilon=0.0)
                seen_obs.add(e_obs.tobytes())
                e_obs, r, terminated, truncated, _ = eval_env.step(e_act)
                ep_r += float(r)
                if e_obs.tobytes() in seen_obs:
                    break
                
                if terminated or truncated:
                    if r == goal_reward:
                        goal_count += 1
                    if r == failure_reward:
                        avoid_count += 1
                    break
            rewards.append(ep_r)

        return float(np.mean(rewards)), goal_count / num_iter, avoid_count / num_iter

def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    performance_file = Path(args.perf_file)
    performance_file.parent.mkdir(parents=True, exist_ok=True)
    if performance_file.exists():
        performance_file.unlink()

    with open(performance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Checkpoint", "Timestep", "Elapsed(s)", "MeanReward", "GoalFrac", "AvoidFrac", "Episodes"])
        f.flush()

    repair_file = Path(args.repair_log_file)
    repair_file.parent.mkdir(parents=True, exist_ok=True)
    if repair_file.exists():
        repair_file.unlink()

    with open(repair_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episodes", "RepairTime", "NumFaultsTotal", "NumFaultsThisRound"])
        f.flush()

    file_args = {
        "jani_model":        args.jani_model,
        "jani_property":     args.jani_property,
        "start_states":      args.start_states,
        "objective":         args.objective,
        "failure_property":  args.failure_property,
        "goal_reward":       args.goal_reward,
        "failure_reward":    args.failure_reward,
        "unsafe_reward":     args.unsafe_reward,
        "seed":              args.seed,
        "use_oracle":        args.use_oracle,
        "max_steps":         args.max_steps,
        "disable_oracle_cache": args.disable_oracle_cache,
        "reduced_memory_mode":  not args.no_memory_reduced_mode,
        "faulty_states_path": args.faulty_states_path,
        "faulty_state_reset_prob": args.faulty_state_reset_prob,
        "performance_file": str(performance_file),
        "repair_file": str(repair_file)
    }
    
    train_model(args, file_args)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Safe-DQN on JANI Environments")

    # Environment settings (mirrors train_ppo.py)
    parser.add_argument("--jani_model",        type=str, required=True)
    parser.add_argument("--jani_property",     type=str, default="")
    parser.add_argument("--start_states",      type=str, default="")
    parser.add_argument("--objective",         type=str, default="")
    parser.add_argument("--failure_property",  type=str, default="")
    parser.add_argument("--eval_start_states", type=str, default="")
    parser.add_argument("--goal_reward",       type=float, default=1.0)
    parser.add_argument("--failure_reward",    type=float, default=-1.0)
    parser.add_argument("--unsafe_reward",     type=float, default=-0.01)
    parser.add_argument("--use_oracle",        action="store_true")
    parser.add_argument("--disable_oracle_cache", action="store_true")
    parser.add_argument("--no_memory_reduced_mode", action="store_true")
    parser.add_argument("--faulty_states_path", type=str, default="",
                        help="JSON state pool produced by repair for training resets.")
    parser.add_argument("--faulty_state_reset_prob", type=float, default=0.0,
                        help="Probability that a training episode starts from the faulty-state pool.")

    # DQN-specific hyperparameters
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--gamma",              type=float, default=0.99)
    parser.add_argument("--cost_gamma",         type=float, default=0.99,
                        help="Discount factor for cost Bellman target.")
    parser.add_argument("--cost_threshold",     type=float, default=0.1,
                        help="Max tolerated per-action cost for safety masking.")
    parser.add_argument("--hidden_dims",        type=int,   nargs="+", default=[64, 64])
    parser.add_argument("--buffer_capacity",    type=int,   default=400_000)
    parser.add_argument("--batch_size",         type=int,   default=64)
    parser.add_argument("--tau",                type=float, default=0.005,
                        help="Soft update coefficient for target network.")
    parser.add_argument("--eps_start",          type=float, default=1.0)
    parser.add_argument("--eps_end",            type=float, default=0.05)
    parser.add_argument("--eps_decay_steps",    type=int,   default=200_000)
    parser.add_argument("--learning_starts",    type=int,   default=500)
    parser.add_argument("--train_freq",         type=int,   default=4)
    parser.add_argument("--target_update_freq", type=int,   default=5_000)

    # Training / evaluation settings (mirrors train_ppo.py)
    parser.add_argument("--seed",                    type=int,  default=42)
    parser.add_argument("--total_timesteps",         type=int,  default=1_000_000)
    parser.add_argument("--max_steps",               type=int,  default=1000)
    parser.add_argument("--log_dir",                 type=str,  default="./logs")
    parser.add_argument("--log_reward",              action="store_true")
    parser.add_argument("--model_save_dir",          type=str,  default="./models")
    parser.add_argument("--use_separate_eval_env",   action="store_true")
    parser.add_argument("--enumate_all_init_states", action="store_true")
    parser.add_argument("--eval_freq",               type=int,  default=2048)
    parser.add_argument("--n_eval_episodes",         type=int,  default=50)
    parser.add_argument("--load_policy_path",        type=str,  default="")
    parser.add_argument("--save_all_checkpoints",    action="store_true")
    parser.add_argument("--eval_safety",             action="store_true")
    parser.add_argument("--disable_eval",            action="store_true")

    # Logging
    parser.add_argument("--wandb_project",    type=str,  default="jani_rl")
    parser.add_argument("--wandb_entity",     type=str,  default=None)
    parser.add_argument("--experiment_name",  type=str,  default="")
    parser.add_argument("--verbose",          type=int,  default=1)
    parser.add_argument("--device",           type=str,  default="cpu")
    parser.add_argument("--disable_wandb",    action="store_true")
    parser.add_argument("--perf_file",       type=str,  default="performance_metrics.csv")
    
    # Repair
    parser.add_argument("--enable_repair", action="store_true")
    parser.add_argument("--repair_freq",    type=int, default=10_000)
    parser.add_argument("--repair_episodes",   type=int, default=100)
    parser.add_argument("--repair_algo", type=str)
    parser.add_argument("--repair_log_file", type=str, default="safe_dqn_repair_log.csv")

    args = parser.parse_args()
    run(args)
    

if __name__ == "__main__":
    main()
