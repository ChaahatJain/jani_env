"""
PPO-Lagrangian trainer for JANI environments.

Design principles:
  - No SB3 internals. Pure PyTorch rollout + update loop.
  - Environments are always wrapped with ActionMasker, so masks are obtained
    via env.action_masks() at every step.
  - The cost signal is extracted from info["next_state_safety"] (False = unsafe)
    as produced by JANIEnv when use_oracle=True.  If the key is absent the cost
    is 0 (safe), so the code works with oracle disabled too.
  - Lagrangian dual variable is updated once per rollout by gradient ascent on
    (mean_episode_cost - cost_limit).
"""

# TODO: @Songtuan please review

# python -m ppo_lag.train --jani_model examples/one_way_line_15_10/model.jani --jani_property examples/one_way_line_15_10/property.jani --start_states examples/one_way_line_15_10/property.jani --eval_start_states examples/one_way_line_15_10/eval_start_states.jani --goal_reward 1.0 --failure_reward -1.0 --unsafe_reward -0.01   --max_steps 256 --total_timesteps 1000 --n_eval_episodes 100 --experiment_name one_way_line_15_10_det --model_save_dir /jani_env/models/ppo/one_way_line_15_10 --disable_eval --disable_wandb --verbose 1 --device cpu --seed 50

import argparse
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import time
import csv

from utils import create_env, create_eval_file_args
from dagger.sampler import StandardTraceSampler
from dagger.fault_collector import OracleFaultCollector
from updater.goldberger import MILPPolicyUpdater
from updater.spec_repair import SpecRepairPolicyUpdater
from dagger.policy_wrapper import NNPolicyWrapper
from dagger.policy import Policy

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available.")


# ---------------------------------------------------------------------------
# Policy network  (actor + reward critic + cost critic)
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Separate MLP heads for policy logits, reward value, cost value.
    No shared trunk — keeps the cost critic gradient isolated from the actor
    except through the explicit Lagrangian penalty term.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 pi_hidden: List[int], vf_hidden: List[int]):
        super().__init__()
        self.actor     = _mlp(obs_dim, pi_hidden, act_dim)
        self.reward_vf = _mlp(obs_dim, vf_hidden, 1)
        self.cost_vf   = _mlp(obs_dim, vf_hidden, 1)

    def get_values(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reward_value, cost_value), both shape (B, 1)."""
        return self.reward_vf(obs), self.cost_vf(obs)

    def act(self, obs: torch.Tensor, mask: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action under the mask.
        Returns (action, log_prob, reward_value, cost_value).
        """
        logits = self.actor(obs).masked_fill(~mask, -float('inf'))
        if not mask.any():
            logits = torch.zeros_like(logits)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        rv, cv = self.get_values(obs)
        return action, dist.log_prob(action), rv, cv

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor,
                 mask: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions for the policy update.
        Returns (log_prob, entropy, reward_value, cost_value).
        """
        logits = self.actor(obs).masked_fill(~mask, -float('inf'))
        dist   = torch.distributions.Categorical(logits=logits)
        rv, cv = self.get_values(obs)
        return dist.log_prob(action), dist.entropy(), rv, cv


# ---------------------------------------------------------------------------
# GAE helper
# ---------------------------------------------------------------------------

def compute_gae(rewards:    np.ndarray,
                values:     np.ndarray,
                dones:      np.ndarray,
                last_value: float,
                gamma:      float,
                gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard GAE computation.
    All inputs are flat arrays of length T.
    Returns (advantages, returns).
    """
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae   = 0.0
    next_val   = last_value
    for t in reversed(range(T)):
        non_term      = 1.0 - float(dones[t])
        delta         = rewards[t] + gamma * next_val * non_term - values[t]
        last_gae      = delta + gamma * gae_lambda * non_term * last_gae
        advantages[t] = last_gae
        next_val      = values[t]
    return advantages, advantages + values


# ---------------------------------------------------------------------------
# Lagrangian multiplier
# ---------------------------------------------------------------------------

class LagrangianMultiplier:
    """λ ← max(0, λ + lr * (mean_ep_cost - cost_limit))"""

    def __init__(self, init_lambda: float, lr: float, cost_limit: float):
        self.value      = max(0.0, init_lambda)
        self.lr         = lr
        self.cost_limit = cost_limit

    def update(self, mean_episode_cost: float):
        self.value = max(0.0, self.value + self.lr * (mean_episode_cost - self.cost_limit))

    def __float__(self):
        return float(self.value)


# ---------------------------------------------------------------------------
# PPO-Lagrangian
# ---------------------------------------------------------------------------

class PPOLagrangian:

    def __init__(
        self,
        env,
        pi_hidden:      List[int],
        vf_hidden:      List[int],
        lr:             float,
        max_steps:      int,
        n_steps:        int,
        batch_size:     int,
        n_epochs:       int,
        gamma:          float,
        cost_gamma:     float,
        gae_lambda:     float,
        clip_range:     float,
        ent_coef:       float,
        vf_coef:        float,
        cost_vf_coef:   float,
        max_grad_norm:  float,
        init_lambda:    float,
        lr_lambda:      float,
        cost_limit:     float,
        device:         str,
        verbose:        int = 1,
        goal_reward:    float = 1.0,
        failure_reward: float = -1.0,
    ):
        self.env           = env
        self.max_steps     = max_steps
        self.n_steps       = n_steps
        self.batch_size    = batch_size
        self.n_epochs      = n_epochs
        self.gamma         = gamma
        self.cost_gamma    = cost_gamma
        self.gae_lambda    = gae_lambda
        self.clip_range    = clip_range
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.cost_vf_coef  = cost_vf_coef
        self.max_grad_norm = max_grad_norm
        self.verbose       = verbose
        self.goal_reward   = goal_reward
        self.failure_reward= failure_reward
        self.device        = torch.device(device)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.policy    = ActorCritic(obs_dim, act_dim, pi_hidden, vf_hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.lagrangian = LagrangianMultiplier(init_lambda, lr_lambda, cost_limit)

    # ------------------------------------------------------------------
    # Mask helper  (ActionMasker always exposes action_masks())
    # ------------------------------------------------------------------

    def _mask(self, env) -> torch.Tensor:
        """Returns bool tensor shape (1, act_dim)."""
        m = env.action_masks()   # np.ndarray (act_dim,)
        return torch.tensor(m, dtype=torch.bool, device=self.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, obs: np.ndarray):
        self.policy.eval()
        act_dim = self.env.action_space.n
        obs_shape = self.env.observation_space.shape

        obs_buf  = np.zeros((self.n_steps,) + obs_shape, dtype=np.float32)
        act_buf  = np.zeros(self.n_steps, dtype=np.int64)
        logp_buf = np.zeros(self.n_steps, dtype=np.float32)
        rew_buf  = np.zeros(self.n_steps, dtype=np.float32)
        cost_buf = np.zeros(self.n_steps, dtype=np.float32)
        rv_buf   = np.zeros(self.n_steps, dtype=np.float32)
        cv_buf   = np.zeros(self.n_steps, dtype=np.float32)
        done_buf = np.zeros(self.n_steps, dtype=np.float32)
        mask_buf = np.zeros((self.n_steps, act_dim), dtype=bool)
        num_completed = 0
        episode_costs: List[float] = []
        ep_cost_acc = 0.0

        for t in range(self.n_steps):
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = self._mask(self.env)

            with torch.no_grad():
                action, logp, rv, cv = self.policy.act(obs_t, mask_t)

            action_int = int(action.item())
            next_obs, reward, terminated, truncated, info = self.env.step(action_int)
            done = terminated or truncated

            # Cost: 1 if oracle reports next state is unsafe, else 0.
            # Gracefully falls back to 0 when oracle is disabled.
            if isinstance(info, dict):
                cost = 0.0 if info.get("next_state_safety", True) else 1.0
            else:
                cost = 0.0

            obs_buf[t]  = obs
            act_buf[t]  = action_int
            logp_buf[t] = logp.item()
            rew_buf[t]  = float(reward)
            cost_buf[t] = cost
            rv_buf[t]   = rv.item()
            cv_buf[t]   = cv.item()
            done_buf[t] = float(done)
            mask_buf[t] = mask_t.squeeze(0).cpu().numpy()

            ep_cost_acc += cost
            obs = next_obs

            if done:
                episode_costs.append(ep_cost_acc)
                ep_cost_acc = 0.0
                obs, _ = self.env.reset()
                num_completed += 1

        # Bootstrap
        with torch.no_grad():
            obs_t      = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_rv, last_cv = self.policy.get_values(obs_t)

        r_adv, r_ret = compute_gae(rew_buf,  rv_buf, done_buf,
                                   last_rv.item(), self.gamma,      self.gae_lambda)
        c_adv, c_ret = compute_gae(cost_buf, cv_buf, done_buf,
                                   last_cv.item(), self.cost_gamma, self.gae_lambda)

        mean_ep_cost = float(np.mean(episode_costs)) if episode_costs else ep_cost_acc

        return dict(obs=obs_buf, actions=act_buf, old_logp=logp_buf,
                    r_adv=r_adv, r_ret=r_ret, c_adv=c_adv, c_ret=c_ret,
                    masks=mask_buf, completed=num_completed), obs, mean_ep_cost

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def _update(self, rollout: dict, lam: float) -> Dict[str, float]:
        self.policy.train()

        obs      = torch.tensor(rollout["obs"],      dtype=torch.float32, device=self.device)
        actions  = torch.tensor(rollout["actions"],  dtype=torch.long,    device=self.device)
        old_logp = torch.tensor(rollout["old_logp"], dtype=torch.float32, device=self.device)
        r_adv    = torch.tensor(rollout["r_adv"],    dtype=torch.float32, device=self.device)
        r_ret    = torch.tensor(rollout["r_ret"],    dtype=torch.float32, device=self.device)
        c_adv    = torch.tensor(rollout["c_adv"],    dtype=torch.float32, device=self.device)
        c_ret    = torch.tensor(rollout["c_ret"],    dtype=torch.float32, device=self.device)
        masks    = torch.tensor(rollout["masks"],    dtype=torch.bool,    device=self.device)

        T = obs.shape[0]
        totals   = dict(pg=0., vf=0., cvf=0., ent=0., total=0.)
        n_updates = 0

        for _ in range(self.n_epochs):
            idx = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                mb = idx[start: start + self.batch_size]

                logp, entropy, rv, cv = self.policy.evaluate(obs[mb], actions[mb], masks[mb])

                # Normalise advantages within mini-batch
                mb_r = r_adv[mb]; mb_r = (mb_r - mb_r.mean()) / (mb_r.std() + 1e-8)
                mb_c = c_adv[mb]; mb_c = (mb_c - mb_c.mean()) / (mb_c.std() + 1e-8)

                ratio = torch.exp(logp - old_logp[mb])

                # Clipped reward PG (maximise → negate for gradient descent)
                pg_loss = -torch.mean(torch.min(
                    mb_r * ratio,
                    mb_r * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
                ))

                # Clipped cost PG — pessimistic upper bound, added with weight λ
                cost_pg_loss = torch.mean(torch.max(
                    mb_c * ratio,
                    mb_c * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
                ))

                vf_loss  = nn.functional.mse_loss(rv.flatten(), r_ret[mb])
                cvf_loss = nn.functional.mse_loss(cv.flatten(), c_ret[mb])
                ent_loss = -torch.mean(entropy)

                loss = (pg_loss
                        + lam             * cost_pg_loss
                        + self.vf_coef    * vf_loss
                        + self.cost_vf_coef * cvf_loss
                        + self.ent_coef   * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                totals["pg"]    += pg_loss.item()
                totals["vf"]    += vf_loss.item()
                totals["cvf"]   += cvf_loss.item()
                totals["ent"]   += ent_loss.item()
                totals["total"] += loss.item()
                n_updates       += 1

        d = max(n_updates, 1)
        return {k: v / d for k, v in totals.items()}

    def _repair_init(self, repair_algo="milp"):
        self.sampler = StandardTraceSampler()
        self.collector = OracleFaultCollector()
        if repair_algo == "spec":
            optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=1e-3)
            self.updater = SpecRepairPolicyUpdater(optimizer=optimizer, batch_size=64, device=self.device)
        else:
            self.updater = MILPPolicyUpdater()
        # Fault cache — persists across all repair rounds
        self.all_faults: list = []
        self.fault_cache: set = set()

    def _build_policy_wrapper(self) -> NNPolicyWrapper:
        """Extract current SB3 policy weights into a Policy wrapper for the sampler."""
        return NNPolicyWrapper(self.policy.actor, device=self.device)

    def _collect_new_faults(self, traces: list) -> list:
        """
        Run fault analysis on unsafe traces, deduplicate against the cache,
        and update the persistent fault store.
        """
        new_faults_this_round = []
        duplicates = 0

        for trace in traces:
            if not trace["is_safe_trajectory"]:
                faults = self.collector.collect_faults(trace, self.env.env.env)
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
    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, model_save_dir: Path,
              eval_args: Optional[Dict] = None, repair_args: Optional[Dict] = None):
        obs, _           = self.env.reset()
        timestep         = 0
        best_eval_reward = -float("inf")
        episodes = 0
        episodes_since_last_eval = 0
        episodes_since_last_repair = 0
        checkpoint_idx = 0
        if repair_args is not None:
            self._repair_init(repair_args.get("repair_algo", "milp"))
            self.log_file = repair_args["repair_file"]
        time_now = time.time()
        while timestep < total_timesteps:
            rollout, obs, mean_ep_cost = self._collect_rollout(obs)
            timestep += self.n_steps
            episodes += rollout["completed"]
            episodes_since_last_eval += rollout["completed"]
            episodes_since_last_repair += rollout["completed"]

            print("Episode", episodes, "timesteps", timestep)
            self.lagrangian.update(mean_ep_cost)
            lam = float(self.lagrangian)
            m   = self._update(rollout, lam)

            if self.verbose >= 1 and timestep % (self.n_steps * 10) == 0:
                print(f"[{timestep:>8d}] λ={lam:.4f}  cost={mean_ep_cost:.4f}  "
                      f"pg={m['pg']:.4f}  vf={m['vf']:.4f}  cvf={m['cvf']:.4f}")

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"train/mean_ep_cost": mean_ep_cost, "train/lambda": lam,
                           "loss/policy": m["pg"], "loss/value": m["vf"],
                           "loss/cost_value": m["cvf"], "loss/entropy": m["ent"],
                           "loss/total": m["total"], "timestep": timestep})

            if (eval_args is not None
                    and not eval_args.get("disable_eval", False)
                    and episodes_since_last_eval >= eval_args["eval_freq"]):
                episodes_since_last_eval = 0
                performance_file = eval_args.get("performance_file", None)
                mean_r, goal_frac, avoid_frac = self._evaluate(eval_args["eval_env"],
                                                eval_args["n_eval_episodes"])
                with open(performance_file, "a", newline="") as f:
                    elapsed = time.time() - time_now
                    writer = csv.writer(f)
                    writer.writerow([
                        checkpoint_idx, timestep, round(elapsed, 2),
                        round(mean_r, 4), round(goal_frac, 4), round(avoid_frac, 4),
                        episodes
                    ])
                    f.flush()
                if eval_args.get("save_all_checkpoints", False):
                    print("Here", model_save_dir / f"actor_iter_{checkpoint_idx}.pth")
                    self._save(model_save_dir / f"actor_iter_{checkpoint_idx}.pth")
                checkpoint_idx += 1

                print(f"  ↳ eval  reward={mean_r:.3f}  goal_frac={goal_frac}, avoid_frac={avoid_frac}")
                # assert False, "Stop after first eval for debugging. Remove this line to continue training."

            if (repair_args is not None and episodes_since_last_repair >= repair_args["repair_freq"]): ## Repair the policy by sampling repair_episodes, running fault analysis and updating the policy 
                episodes_since_last_repair = 0
                print("Starting policy repair phase...")
                # 1. Build a policy wrapper around the current SB3 policy for the sampler and updater to use
                policy_wrapper = self._build_policy_wrapper()

                # 2. Sample traces
                traces = []
                for _ in range(repair_args["repair_episodes"]):
                    trace = self.sampler.sample_trace(
                        env=self.env.env.env, # Unwrap time and action masker
                        policy=policy_wrapper,
                        init_state_idx=-1,
                        max_steps=self.max_steps,
                        verbose=False,
                    )
                    traces.append(trace)
                    timestep += len(trace["observations"])  # Account for the additional timesteps used for sampling traces
                    
                if not traces:
                    continue

                # 3. Collect faults from unsafe traces, deduplicated against full history
                new_faults = self._collect_new_faults(traces)
                print("New faults found:", len(new_faults))

                # 4. Skip repair if no new faults found this round — policy is safe on sampled traces
                if not new_faults:
                    if self.verbose:
                        print("[ModelRepairCallback] No new faults found this round — skipping repair.")
                    continue

                # 5. Repair using the full fault history — mutates self.model.policy in-place
                repair_time_start = time.perf_counter()
                _ = self.updater.update_policy(self.policy.actor, self.all_faults)
                repair_time = time.perf_counter() - repair_time_start

                with open(repair_args["repair_file"], "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episodes, round(repair_time, 2), len(self.all_faults), len(new_faults)
                    ])
                    f.flush()
                episodes += repair_args["repair_episodes"]  # Account for the additional episodes used for repair
                episodes_since_last_eval += repair_args["repair_episodes"]
                
        self._save(model_save_dir / "final_actor.pth")
        print(f"Saved final actor → {model_save_dir / 'final_actor.pth'}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, eval_env, n_episodes=10, enumate_all_init_states=False) -> Tuple[float, float, float]:
        self.policy.eval()
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
                obs_t  = torch.tensor(e_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                seen_obs.add(e_obs.tobytes())
                mask_t = self._mask(eval_env)
                with torch.no_grad():
                    action, _, _, _ = self.policy.act(obs_t, mask_t)
                e_obs, r, terminated, truncated, _ = eval_env.step(int(action.item()))
                ep_r += float(r)
                if e_obs.tobytes() in seen_obs:
                    break
                
                if terminated or truncated:
                    if r == self.goal_reward:
                        goal_count += 1
                    if r == self.failure_reward:
                        avoid_count += 1
                    break
            rewards.append(ep_r)
        return float(np.mean(rewards)), goal_count / num_iter, avoid_count / num_iter

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"obs_dim": self.env.observation_space.shape[0],
                    "act_dim": self.env.action_space.n,
                    "state_dict": self.policy.state_dict()}, path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        saved_sd = ckpt["state_dict"]

        # Detect simple Policy checkpoint (keys like "model.0.weight")
        if any(k.startswith("model.") for k in saved_sd):
            print("Detected simple Policy checkpoint – remapping weights to ActorCritic...")
            new_sd = self.policy.state_dict()
            # Map simple model layers → actor layers (same structure)
            for k, v in saved_sd.items():
                actor_key = k.replace("model.", "actor.", 1)
                if actor_key in new_sd:
                    new_sd[actor_key] = v
            # reward_vf and cost_vf keep random init (no value info in simple Policy)
            self.policy.load_state_dict(new_sd)
        else:
            self.policy.load_state_dict(saved_sd)

def run(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    file_args = {
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
    }

    model_save_dir  = Path(args.model_save_dir)
    experiment_name = args.experiment_name or f"ppo_lag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    print("Creating training environment...")
    train_env = create_env(file_args, n_envs=1, monitor=False, time_limited=True)

    trainer = PPOLagrangian(
        env           = train_env,
        pi_hidden     = args.pi_net_arch,
        vf_hidden     = args.vf_net_arch,
        lr            = args.learning_rate,
        max_steps     = args.max_steps,
        n_steps       = args.n_steps,
        batch_size    = args.batch_size,
        n_epochs      = args.n_epochs,
        gamma         = args.gamma,
        cost_gamma    = args.cost_gamma,
        gae_lambda    = args.gae_lambda,
        clip_range    = args.clip_range,
        ent_coef      = args.ent_coef,
        vf_coef       = args.vf_coef,
        cost_vf_coef  = args.cost_vf_coef,
        max_grad_norm = args.max_grad_norm,
        init_lambda   = args.init_lambda,
        lr_lambda     = args.lr_lambda,
        cost_limit    = args.cost_limit,
        device        = args.device,
        verbose       = args.verbose,
        goal_reward   = args.goal_reward,
        failure_reward= args.failure_reward,
    )

    if args.load_policy_path:
        print(f"Loading pre-trained policy from {args.load_policy_path}...")
        trainer.load(Path(args.load_policy_path))

    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=experiment_name, config=vars(args))

    eval_args = None
    if not args.disable_eval:
        eval_file_args = create_eval_file_args(file_args, args.use_separate_eval_env)
        eval_env       = create_env(eval_file_args, n_envs=1, monitor=False, time_limited=True)
        eval_args = {
            "eval_env":             eval_env,
            "eval_freq":            args.eval_freq,
            "n_eval_episodes":      args.n_eval_episodes,
            "disable_eval":         args.disable_eval,
            "save_all_checkpoints": args.save_all_checkpoints,
            "performance_file":       performance_file,
        }

    repair_args = None
    if args.enable_repair:
        repair_args = {
            "repair_freq": args.repair_freq,
            "repair_episodes": args.repair_episodes,
            "repair_algo": args.repair_algo,
            "repair_file": repair_file
        }

    trainer.learn(total_timesteps=args.total_timesteps,
                  model_save_dir=model_save_dir,
                  eval_args=eval_args, repair_args=repair_args)

    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.finish()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO-Lagrangian on JANI Environments")

    # Environment
    parser.add_argument("--jani_model",             type=str,   required=True)
    parser.add_argument("--jani_property",          type=str,   default="")
    parser.add_argument("--start_states",           type=str,   default="")
    parser.add_argument("--objective",              type=str,   default="")
    parser.add_argument("--failure_property",       type=str,   default="")
    parser.add_argument("--eval_start_states",      type=str,   default="")
    parser.add_argument("--goal_reward",            type=float, default=1.0)
    parser.add_argument("--failure_reward",         type=float, default=-1.0)
    parser.add_argument("--unsafe_reward",          type=float, default=-0.01)
    parser.add_argument("--use_oracle",             action="store_true")
    parser.add_argument("--disable_oracle_cache",   action="store_true")
    parser.add_argument("--no_memory_reduced_mode", action="store_true")
    parser.add_argument("--max_steps",              type=int,   default=1000)

    # PPO
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps",       type=int,   default=2048)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--n_epochs",      type=int,   default=10)
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--cost_gamma",    type=float, default=0.99)
    parser.add_argument("--gae_lambda",    type=float, default=0.95)
    parser.add_argument("--clip_range",    type=float, default=0.2)
    parser.add_argument("--ent_coef",      type=float, default=0.0)
    parser.add_argument("--vf_coef",       type=float, default=0.5)
    parser.add_argument("--cost_vf_coef",  type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--pi_net_arch",   type=int, nargs="+", default=[64, 64])
    parser.add_argument("--vf_net_arch",   type=int, nargs="+", default=[64, 64])

    # Lagrangian
    parser.add_argument("--cost_limit",  type=float, default=0.1)
    parser.add_argument("--init_lambda", type=float, default=0.0)
    parser.add_argument("--lr_lambda",   type=float, default=0.01)

    # Training / eval
    parser.add_argument("--seed",                  type=int,  default=42)
    parser.add_argument("--total_timesteps",       type=int,  default=1_000_000)
    parser.add_argument("--model_save_dir",        type=str,  default="./models")
    parser.add_argument("--use_separate_eval_env", action="store_true")
    parser.add_argument("--eval_freq",             type=int,  default=10_000)
    parser.add_argument("--n_eval_episodes",       type=int,  default=50)
    parser.add_argument("--disable_eval",          action="store_true")
    parser.add_argument("--save_all_checkpoints",  action="store_true")
    parser.add_argument("--load_policy_path",      type=str,  default="")

    # Logging
    parser.add_argument("--wandb_project",   type=str, default="jani_rl")
    parser.add_argument("--wandb_entity",    type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--disable_wandb",   action="store_true")
    parser.add_argument("--verbose",         type=int, default=1)
    parser.add_argument("--device",          type=str, default="cpu")
    parser.add_argument("--perf_file", type=str, default="ppo_lag_performance.csv")

    # Interleaving repair
    parser.add_argument("--enable_repair", action="store_true")
    parser.add_argument("--repair_freq",    type=int, default=10_000)
    parser.add_argument("--repair_episodes",   type=int, default=100)
    parser.add_argument("--repair_algo", type=str)
    parser.add_argument("--repair_log_file", type=str, default="ppo_lag_repair_log.csv")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
