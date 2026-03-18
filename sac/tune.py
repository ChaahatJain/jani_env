import sys
import argparse
import torch
import torch.nn as nn
import optuna
from optuna.trial import Trial
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import DiscreteSACLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from jani.torchrl_env import JANIEnv
from utils import safety_evaluation


def create_actor_module(hyperparams: Dict[str, Any], env: JANIEnv) -> TensorDictModule:
    """Create the actor network for the policy."""
    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("actor_hidden_sizes", [64, 64])
    dropout = hyperparams.get("actor_dropout", 0.0)
    activation_fn = hyperparams.get("activation_fn", nn.Tanh)

    actor_backbone = MLP(
        in_features=input_size,
        out_features=n_actions,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    actor_module = TensorDictModule(
        module=actor_backbone,
        in_keys=["observation"],
        out_keys=["logits"],
    )
    return actor_module


def create_qvalue_network(hyperparams: Dict[str, Any], env: JANIEnv) -> ValueOperator:
    """Create the Q-value network for SAC."""
    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("critic_hidden_sizes", [64, 128])
    dropout = hyperparams.get("critic_dropout", 0.0)
    activation_fn = hyperparams.get("activation_fn", nn.Tanh)

    qvalue_backbone = MLP(
        in_features=input_size,
        out_features=n_actions,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    qvalue_module = ValueOperator(
        module=qvalue_backbone,
        in_keys=["observation"],
        out_keys=["action_value"],
    )
    return qvalue_module


def create_replay_buffer(hyperparams: Dict[str, Any]) -> ReplayBuffer:
    """Create a replay buffer for off-policy SAC training."""
    buffer_size = hyperparams.get("replay_buffer_size", 100000)
    storage = LazyTensorStorage(
        max_size=buffer_size,
        device=hyperparams.get("device", "cpu"),
    )
    sampler = RandomSampler()
    replay_buffer = ReplayBuffer(
        storage=storage,
        sampler=sampler,
    )
    return replay_buffer


def train_and_evaluate(hyperparams: Dict[str, Any], env: JANIEnv, eval_env: JANIEnv, trial: Trial) -> float:
    """Train the Discrete SAC agent and return the evaluation metric."""

    # Hyperparameters
    total_timesteps = hyperparams.get("total_timesteps", 100000)
    n_steps = hyperparams.get("n_steps", 256)
    batch_size = hyperparams.get("batch_size", 64)
    lr_actor = hyperparams.get("learning_rate_actor", 3e-4)
    lr_critic = hyperparams.get("learning_rate_critic", 3e-4)
    max_grad_norm = hyperparams.get("max_grad_norm", 1.0)
    n_updates_per_step = hyperparams.get("n_updates_per_step", 1)
    target_update_freq = hyperparams.get("target_update_freq", 1)
    tau = hyperparams.get("tau", 0.005)
    warmup_steps = hyperparams.get("warmup_steps", 1000)
    alpha_init = hyperparams.get("alpha_init", 0.1)
    fixed_alpha = hyperparams.get("fixed_alpha", True)

    # Create actor and Q-value networks
    actor_module = create_actor_module(hyperparams, env)
    actor_rollout = ProbabilisticActor(
        module=actor_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )
    actor_train = ProbabilisticActor(
        module=actor_module,
        in_keys=['logits'],
        out_keys=['action'],
        distribution_class=Categorical,
        return_log_prob=True,
    )
    qvalue_network = create_qvalue_network(hyperparams, env)

    # Create loss module
    loss_module = DiscreteSACLoss(
        actor_network=actor_train,
        qvalue_network=qvalue_network,
        action_space="categorical",
        num_actions=env.n_actions,
        delay_qvalue=True,
        alpha_init=alpha_init,
        fixed_alpha=fixed_alpha,
        target_entropy="auto",
    )

    # Create data collector
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=actor_rollout,
        total_frames=total_timesteps,
        frames_per_batch=n_steps,
        split_trajs=False,
    )

    # Create replay buffer
    replay_buffer = create_replay_buffer(hyperparams)

    # Create optimizers
    actor_optim = torch.optim.Adam(actor_train.parameters(), lr=lr_actor)
    qvalue_optim = torch.optim.Adam(qvalue_network.parameters(), lr=lr_critic)
    alpha_optim = torch.optim.Adam([loss_module.log_alpha], lr=lr_actor)

    # Training loop
    total_steps_collected = 0
    best_safety_rate = 0.0
    eval_interval = hyperparams.get("eval_interval", 10)

    for i, td_data in enumerate(collector):
        data_view = td_data.reshape(-1)
        replay_buffer.extend(data_view)
        total_steps_collected += n_steps

        # Skip training until we have enough data
        if total_steps_collected < warmup_steps:
            continue

        # Training updates
        for _ in range(n_updates_per_step):
            if len(replay_buffer) < batch_size:
                continue

            batch_data = replay_buffer.sample(batch_size)
            loss_dict = loss_module(batch_data)

            # Update Q-value network
            qvalue_loss = loss_dict["loss_qvalue"]
            qvalue_optim.zero_grad()
            qvalue_loss.backward()
            torch.nn.utils.clip_grad_norm_(qvalue_network.parameters(), max_grad_norm)
            qvalue_optim.step()

            # Update actor network
            actor_loss = loss_dict["loss_actor"]
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_train.parameters(), max_grad_norm)
            actor_optim.step()

            # Update alpha if not fixed
            if not fixed_alpha:
                alpha_loss = loss_dict["loss_alpha"]
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

            # Soft update target networks
            if (i + 1) % target_update_freq == 0:
                with torch.no_grad():
                    for param, target_param in zip(
                        loss_module.qvalue_network_params.values(),
                        loss_module.target_qvalue_network_params.values()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1 - tau) * target_param.data
                        )

        # Periodic evaluation
        if i % eval_interval == 0 and i > 0:
            eval_result = safety_evaluation(
                eval_env,
                actor_rollout,
                max_steps=hyperparams.get("max_steps", 256),
            )

            safety_rate = eval_result["safety_rate"]
            best_safety_rate = max(best_safety_rate, safety_rate)

            # Report intermediate value for pruning
            trial.report(safety_rate, i)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            print(f"  Step {total_steps_collected}: Safety Rate = {safety_rate:.4f}, Best = {best_safety_rate:.4f}")

    # Final evaluation
    final_result = safety_evaluation(
        eval_env,
        actor_rollout,
        max_steps=hyperparams.get("max_steps", 256),
    )

    return final_result["safety_rate"]


def sample_hyperparams(trial: Trial) -> Dict[str, Any]:
    """Sample hyperparameters for a trial."""

    # Network architecture
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    hidden_sizes = [hidden_size] * n_layers

    activation_name = trial.suggest_categorical("activation", ["ReLU", "Tanh", "LeakyReLU"])
    activation_fn = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "LeakyReLU": nn.LeakyReLU}[activation_name]

    # Learning rates
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 1e-2, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-5, 1e-2, log=True)

    # SAC-specific
    alpha_init = trial.suggest_float("alpha_init", 0.01, 0.5, log=True)
    fixed_alpha = trial.suggest_categorical("fixed_alpha", [True, False])
    tau = trial.suggest_float("tau", 0.001, 0.1, log=True)

    # Training dynamics
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_updates_per_step = trial.suggest_int("n_updates_per_step", 1, 16)
    target_update_freq = trial.suggest_int("target_update_freq", 1, 10)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 2.0)

    # Buffer and warmup
    replay_buffer_size = trial.suggest_categorical("replay_buffer_size", [50000, 100000, 200000])
    warmup_steps = trial.suggest_int("warmup_steps", 500, 5000, step=500)

    # Regularization
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    return {
        "actor_hidden_sizes": hidden_sizes,
        "critic_hidden_sizes": hidden_sizes,
        "activation_fn": activation_fn,
        "learning_rate_actor": lr_actor,
        "learning_rate_critic": lr_critic,
        "alpha_init": alpha_init,
        "fixed_alpha": fixed_alpha,
        "tau": tau,
        "batch_size": batch_size,
        "n_updates_per_step": n_updates_per_step,
        "target_update_freq": target_update_freq,
        "max_grad_norm": max_grad_norm,
        "replay_buffer_size": replay_buffer_size,
        "warmup_steps": warmup_steps,
        "actor_dropout": dropout,
        "critic_dropout": dropout,
    }


def objective(trial: Trial, args: argparse.Namespace) -> float:
    """Optuna objective function."""

    # Sample hyperparameters
    hyperparams = sample_hyperparams(trial)

    # Add fixed hyperparameters from args
    hyperparams.update({
        "total_timesteps": args.tuning_timesteps,
        "n_steps": 256,
        "gamma": 0.99,
        "device": args.device,
        "max_steps": args.max_steps,
        "eval_interval": args.eval_interval,
    })

    print(f"\nTrial {trial.number}: {trial.params}")

    # Create environments
    file_args = {
        'jani_model_path': args.jani_model,
        'jani_property_path': args.jani_property,
        'start_states_path': args.start_states,
        'objective_path': args.objective,
        'failure_property_path': args.failure_property,
        'goal_reward': args.goal_reward,
        'failure_reward': args.failure_reward,
        'seed': args.seed + trial.number,  # Different seed per trial
        'use_oracle': args.use_oracle,
        'unsafe_reward': args.unsafe_reward,
        'reduced_memory_mode': not args.no_memory_reduced_mode,
    }
    env = JANIEnv(**file_args)

    eval_file_args = file_args.copy()
    eval_file_args["start_states_path"] = args.eval_start_states
    eval_file_args["use_oracle"] = False
    eval_env = JANIEnv(**eval_file_args)

    try:
        safety_rate = train_and_evaluate(hyperparams, env, eval_env, trial)
        print(f"Trial {trial.number} finished with safety_rate: {safety_rate:.4f}")
        return safety_rate
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.TrialPruned()


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Discrete SAC using Optuna")

    # Environment arguments
    parser.add_argument('--jani_model', type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument('--jani_property', type=str, default="", help="Path to the JANI property file.")
    parser.add_argument('--start_states', type=str, default="", help="Path to the start states file.")
    parser.add_argument('--eval_start_states', type=str, required=True, help="Path to the evaluation start states file.")
    parser.add_argument('--objective', type=str, default="", help="Path to the objective file.")
    parser.add_argument('--failure_property', type=str, default="", help="Path to the failure property file.")
    parser.add_argument('--goal_reward', type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument('--failure_reward', type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument('--use_oracle', action='store_true', help="Use Tarjan oracle for unsafe state detection.")
    parser.add_argument('--unsafe_reward', type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument('--no_memory_reduced_mode', action='store_true', help="Disable memory reduced mode.")
    parser.add_argument('--max_steps', type=int, default=256, help="Max steps per episode.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use (cpu or cuda).")

    # Optuna arguments
    parser.add_argument('--n_trials', type=int, default=100, help="Number of Optuna trials.")
    parser.add_argument('--tuning_timesteps', type=int, default=100000, help="Timesteps per trial (shorter for tuning).")
    parser.add_argument('--eval_interval', type=int, default=10, help="Evaluation interval (in collector iterations).")
    parser.add_argument('--study_name', type=str, default="sac_tuning", help="Optuna study name.")
    parser.add_argument('--storage', type=str, default=None, help="Optuna storage URL (e.g., sqlite:///study.db).")
    parser.add_argument('--pruner', type=str, default="median", choices=["median", "hyperband", "none"], help="Pruner type.")
    parser.add_argument('--n_startup_trials', type=int, default=10, help="Number of trials before pruning starts.")
    parser.add_argument('--n_warmup_steps', type=int, default=5, help="Number of steps before pruning in each trial.")

    args = parser.parse_args()

    # Create pruner
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
        )
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",  # Maximize safety rate
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"Starting hyperparameter tuning with {args.n_trials} trials...")
    print(f"Each trial runs for {args.tuning_timesteps} timesteps")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best safety rate: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params to file (both txt and json)
    best_params_path = Path(f"{args.study_name}_best_params.txt")
    with open(best_params_path, "w") as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best safety rate: {study.best_value:.4f}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"\nBest parameters saved to: {best_params_path}")

    # Also save as JSON for easy loading
    import json
    best_params_json_path = Path(f"{args.study_name}_best_params.json")
    with open(best_params_json_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Best parameters (JSON) saved to: {best_params_json_path}")

    # Print top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False).head(5)
    print(trials_df[["number", "value", "state"]].to_string())


def run_tuning(
    jani_model: str,
    eval_start_states: str,
    start_states: str = "",
    jani_property: str = "",
    objective_path: str = "",
    failure_property: str = "",
    goal_reward: float = 1.0,
    failure_reward: float = -1.0,
    use_oracle: bool = False,
    unsafe_reward: float = -0.01,
    no_memory_reduced_mode: bool = False,
    max_steps: int = 256,
    seed: int = 42,
    device: str = "cpu",
    n_trials: int = 50,
    tuning_timesteps: int = 100000,
    eval_interval: int = 10,
    study_name: str = "sac_tuning",
    storage: str = None,
    pruner: str = "median",
    n_startup_trials: int = 10,
    n_warmup_steps: int = 5,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """Run hyperparameter tuning and return best parameters.

    This function can be called programmatically from train.py.
    Returns the best hyperparameters as a dictionary.
    """
    import json as json_module

    # Create a namespace object to mimic argparse
    class Args:
        pass

    args = Args()
    args.jani_model = jani_model
    args.jani_property = jani_property
    args.start_states = start_states
    args.eval_start_states = eval_start_states
    args.objective = objective_path
    args.failure_property = failure_property
    args.goal_reward = goal_reward
    args.failure_reward = failure_reward
    args.use_oracle = use_oracle
    args.unsafe_reward = unsafe_reward
    args.no_memory_reduced_mode = no_memory_reduced_mode
    args.max_steps = max_steps
    args.seed = seed
    args.device = device
    args.tuning_timesteps = tuning_timesteps
    args.eval_interval = eval_interval

    # Create pruner
    if pruner == "median":
        pruner_obj = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        )
    elif pruner == "hyperband":
        pruner_obj = optuna.pruners.HyperbandPruner()
    else:
        pruner_obj = optuna.pruners.NopPruner()

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=pruner_obj,
        load_if_exists=True,
    )

    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    print(f"Each trial runs for {tuning_timesteps} timesteps")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best safety rate: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params to files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_params_json_path = output_path / f"{study_name}_best_params.json"
    with open(best_params_json_path, "w") as f:
        json_module.dump(study.best_params, f, indent=2)
    print(f"\nBest parameters (JSON) saved to: {best_params_json_path}")

    return study.best_params, str(best_params_json_path)


if __name__ == "__main__":
    main()
