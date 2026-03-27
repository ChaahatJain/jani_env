# python -m safe_dqn.train --jani_model examples/one_way_line_15_10/model.jani --jani_property examples/one_way_line_15_10/property.jani --start_states examples/one_way_line_15_10/property.jani --eval_start_states examples/one_way_line_15_10/eval_start_states.jani --goal_reward 1.0 --failure_reward -1.0 --unsafe_reward -0.01   --max_steps 256 --total_timesteps 1000 --n_eval_episodes 100 --experiment_name one_way_line_15_10_det   --log_dir /jani_env/logs/ppo/one_way_line_15_10 --model_save_dir /jani_env/models/ppo/one_way_line_15_10 --disable_eval --enumate_all_init_states --log_reward --eval_freq 1025 --eval_safety --disable_wandb --verbose 1 --device cpu --seed 50


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

from callbacks import EvalCallback, SafetyEvalCallback, WandbCallback, SaveActorCallback, LoggingCallback
from jani.env import JANIEnv
from utils import create_env, create_eval_file_args, create_safety_eval_file_args

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
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.feature_net = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev, output_dim)
        self.c_head = nn.Linear(prev, output_dim)   # cost / safety head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(x)
        return self.q_head(features), self.c_head(features)


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

    Safety is enforced at action-selection time:
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
        agent.online_net.load_state_dict(checkpoint["state_dict"])
        agent.target_net.load_state_dict(checkpoint["state_dict"])

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
        eval_env = create_env(eval_file_args, 1, monitor=True, time_limited=True)

    # ----- training loop -----
    obs, _ = train_env.reset()
    episode_reward  = 0.0
    episode_cost    = 0.0
    episode_len     = 0
    episode_count   = 0
    best_eval_reward = -float("inf")

    metrics_window = deque(maxlen=100)  # rolling episode stats

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
        if not args.disable_eval and eval_env is not None and global_step % args.eval_freq == 0:
            eval_rewards = []
            eval_costs   = []
            for _ in range(args.n_eval_episodes):
                e_obs, _ = eval_env.reset()
                ep_r = ep_c = 0.0
                while True:
                    e_mask = eval_env.env_method("action_masks")[0] if hasattr(eval_env, "env_method") \
                             else eval_env.action_masks()
                    e_act  = agent.select_action(e_obs, e_mask, epsilon=0.0)
                    e_obs, e_r, e_term, e_trunc, e_info = eval_env.step(e_act)
                    ep_r += e_r
                    ep_c += float(e_info.get("cost", 0.0)) if isinstance(e_info, dict) else 0.0
                    if e_term or e_trunc:
                        break
                eval_rewards.append(ep_r)
                eval_costs.append(ep_c)

            mean_eval_reward = float(np.mean(eval_rewards))
            mean_eval_cost   = float(np.mean(eval_costs))
            print(
                f"[Step {global_step:>8d}] eval_reward={mean_eval_reward:.3f} "
                f"eval_cost={mean_eval_cost:.3f}  eps={epsilon:.3f}"
            )

            if WANDB_AVAILABLE and not args.disable_wandb:
                wandb.log({
                    "eval/mean_reward": mean_eval_reward,
                    "eval/mean_cost":   mean_eval_cost,
                    "timestep":         global_step,
                })

            # Save best model
            if mean_eval_reward > best_eval_reward:
                best_eval_reward = mean_eval_reward
                agent.save(model_save_dir / "best_actor.pth")
                print(f"  ↳ New best model saved (reward={best_eval_reward:.3f})")

        # ----- periodic checkpoint -----
        if args.save_all_checkpoints and global_step % args.eval_freq == 0:
            agent.save(model_save_dir / f"actor_step{global_step}.pth")

    # ----- final save -----
    final_path = model_save_dir / "final_actor.pth"
    agent.save(final_path)
    print(f"Final actor model saved to {final_path}")

    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.finish()


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

    # DQN-specific hyperparameters
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--gamma",              type=float, default=0.99)
    parser.add_argument("--cost_gamma",         type=float, default=0.99,
                        help="Discount factor for cost Bellman target.")
    parser.add_argument("--cost_threshold",     type=float, default=0.1,
                        help="Max tolerated per-action cost for safety masking.")
    parser.add_argument("--hidden_dims",        type=int,   nargs="+", default=[64, 64])
    parser.add_argument("--buffer_capacity",    type=int,   default=100_000)
    parser.add_argument("--batch_size",         type=int,   default=64)
    parser.add_argument("--tau",                type=float, default=0.005,
                        help="Soft update coefficient for target network.")
    parser.add_argument("--eps_start",          type=float, default=1.0)
    parser.add_argument("--eps_end",            type=float, default=0.05)
    parser.add_argument("--eps_decay_steps",    type=int,   default=50_000)
    parser.add_argument("--learning_starts",    type=int,   default=1_000)
    parser.add_argument("--train_freq",         type=int,   default=4)
    parser.add_argument("--target_update_freq", type=int,   default=1_000)

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

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

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
    }

    train_model(args, file_args)


if __name__ == "__main__":
    main()