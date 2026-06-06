"""
Generic training interface for JANI RL algorithms.

Supports:
  mask_ppo  — Maskable PPO (SB3-contrib)
  ppo_lag   — PPO-Lagrangian (pure PyTorch)
  safe_dqn  — Safe DQN with cost head (pure PyTorch)

Usage:
  python learning.py --algo mask_ppo --jani_model model.jani [...]
  python learning.py --algo ppo_lag  --jani_model model.jani [...]
  python learning.py --algo safe_dqn --jani_model model.jani [...]

All algorithms share --learning_rate, --gamma, --batch_size, --hidden_dims,
--cost_gamma, --cost_limit, and all env/eval/logging flags.
Algorithm-specific flags that don't apply to the chosen algo are silently ignored.
"""

import argparse
import torch
import numpy as np
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Registry of available domain-specific expert policies.
_DOMAIN_POLICY_REGISTRY: dict = {}


def _load_domain_policy(name: str):
    """Lazy-load a domain-specific policy callable by name."""
    if not name:
        return None
    if name not in _DOMAIN_POLICY_REGISTRY:
        if name == "transport":
            from domain_specific.manual_policies.transport_policy import transport_policy
            _DOMAIN_POLICY_REGISTRY["transport"] = transport_policy
        elif name == "one_way_line":
            from domain_specific.manual_policies.one_way_line_policy import one_way_line_policy
            _DOMAIN_POLICY_REGISTRY["one_way_line"] = one_way_line_policy
        elif name == "two_way_line":
            from domain_specific.manual_policies.two_way_line_policy import two_way_line_policy
            _DOMAIN_POLICY_REGISTRY["two_way_line"] = two_way_line_policy
        elif name == "beluga":
            from domain_specific.manual_policies.beluga_policy import beluga_policy
            _DOMAIN_POLICY_REGISTRY["beluga"] = beluga_policy
        else:
            raise ValueError(f"Unknown domain policy: '{name}'")
    return _DOMAIN_POLICY_REGISTRY[name]


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
        "step_reward":          args.step_reward,
        "cycle_reward":         args.cycle_reward,
        "seed":                 args.seed,
        "use_oracle":           args.use_oracle,
        "max_steps":            args.max_steps,
        "disable_oracle_cache": args.disable_oracle_cache,
        "reduced_memory_mode":  not args.no_memory_reduced_mode,
        "domain_policy":        _load_domain_policy(args.domain_policy),
        "policy_match_reward":  args.policy_match_reward,
    }


def _experiment_name(args) -> str:
    return args.experiment_name or f"{args.algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

def _run_mask_ppo(args, file_args: dict):
    import mask_ppo.train as mask_ppo_train
    mask_ppo_train.train_model(args, file_args)


def _run_ppo_lag(args, file_args: dict):
    import ppo_lag.ppo_lag as ppo_lag_train
    ppo_lag_train.run(args)


def _run_safe_dqn(args, file_args: dict):
    # safe_dqn.train_model reads args.lr; inject from the unified --learning_rate
    args.lr = args.learning_rate
    # safe_dqn.train_model reads args.cost_threshold; map from unified --cost_limit
    args.cost_threshold = args.cost_limit
    import safe_dqn.safe_dqn as safe_dqn_train
    safe_dqn_train.run(args)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic JANI RL trainer — mask_ppo | ppo_lag | safe_dqn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Algorithm ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--algo", type=str, required=True,
        choices=["mask_ppo", "ppo_lag", "safe_dqn"],
        help="RL algorithm to use.",
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
    env.add_argument("--step_reward",            type=float, default=0.0,
                    help="Reward added on every step (use negative to penalise long episodes).")
    env.add_argument("--cycle_reward",           type=float, default=0.0,
                    help="Extra reward when the agent revisits a state in the same episode.")
    env.add_argument("--use_oracle",             action="store_true")
    env.add_argument("--disable_oracle_cache",   action="store_true")
    env.add_argument("--no_memory_reduced_mode", action="store_true")
    env.add_argument("--max_steps",              type=int,   default=1000)
    env.add_argument("--domain_policy",          type=str,   default="",
                    choices=["", "transport", "one_way_line", "beluga"],
                    help="Name of the domain-specific expert policy to use for reward shaping.")
    env.add_argument("--policy_match_reward",    type=float, default=0.0,
                    help="Bonus reward added when the agent's action matches the expert policy.")

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

    # ── PPO hyperparameters (mask_ppo + ppo_lag) ─────────────────────────────
    ppo = parser.add_argument_group("PPO hyperparameters (mask_ppo / ppo_lag)")
    ppo.add_argument("--n_steps",       type=int,   default=256)
    ppo.add_argument("--n_epochs",      type=int,   default=10)
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
    ev.add_argument("--max_state_visits",         type=int, default=3,
                    help='Max visits to same state in eval before episode counted as cycle')
    ev.add_argument("--use_separate_eval_env",   action="store_true")
    ev.add_argument("--enumate_all_init_states", action="store_true")
    ev.add_argument("--eval_safety",             action="store_true")
    ev.add_argument("--save_all_checkpoints",    action="store_true")
    ev.add_argument("--log_reward",              action="store_true")

    # ── Logging (shared) ─────────────────────────────────────────────────────
    lg = parser.add_argument_group("Logging")
    lg.add_argument("--log_dir",         type=str, default="./logs")
    lg.add_argument("--perf_file", type=str, default="./performance.csv")
    lg.add_argument("--model_save_dir",  type=str, default="./models")
    lg.add_argument("--experiment_name", type=str, default="")
    lg.add_argument("--verbose",         type=int, default=1)
    lg.add_argument("--disable_wandb",   action="store_true")
    lg.add_argument("--wandb_project",   type=str, default="jani_rl")
    lg.add_argument("--wandb_entity",    type=str, default=None)

    # ── Interleaving repair (shared) ───────────────────────────────────────────────────────
    repair = parser.add_argument_group("Interleaving repair")
    repair.add_argument("--enable_repair", action="store_true")
    repair.add_argument("--repair_freq",    type=int, default=10_000)
    repair.add_argument("--repair_episodes",   type=int, default=100)
    repair.add_argument("--repair_algo", type=str)
    repair.add_argument("--repair_log_file", type=str, default="./repair_log.csv")

    # ── Unique-states logging (mask_ppo) ──────────────────────────────────────
    us = parser.add_argument_group("Unique-states logging (mask_ppo)")
    us.add_argument("--unique_states_log", type=str, default="",
                    help="Path to CSV file for logging unique states count over training.")
    us.add_argument("--unique_states_log_freq", type=int, default=10_000,
                    help="Timestep frequency at which to log unique states count.")
    return parser


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(args):
    if args.use_separate_eval_env or args.eval_safety:
        assert args.eval_start_states != "", \
            "--eval_start_states is required with --use_separate_eval_env / --eval_safety."
    if args.algo != "mask_ppo" and args.n_envs != 1:
        print(f"Warning: --n_envs={args.n_envs} is ignored for {args.algo} (always 1 env).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DISPATCH = {
    "mask_ppo": _run_mask_ppo,
    "ppo_lag":  _run_ppo_lag,
    "safe_dqn": _run_safe_dqn,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    _set_seeds(args.seed)
    _validate(args)

    print(args.perf_file)
    file_args = _build_file_args(args)

    print(f"=== Algorithm : {args.algo}")
    print(f"    Model     : {args.jani_model}")
    print(f"    Device    : {args.device}  |  Timesteps: {args.total_timesteps:,}  |  Seed: {args.seed}")
    if args.enable_repair:
        print(f"    Repair    : {args.repair_algo} every {args.repair_freq} steps for {args.repair_episodes} episodes")

    DISPATCH[args.algo](args, file_args)


if __name__ == "__main__":
    main()