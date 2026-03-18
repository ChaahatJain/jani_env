"""Recovery RL training with Discrete SAC for task policy.

This module implements Recovery RL where:
- Task policy uses Discrete SAC (instead of PPO)
- Recovery policy and Q-risk module are loaded from pretrained checkpoints
- The recovery actor switches between task and recovery actions based on risk assessment
"""
import sys
import json
import torch
import torch.nn as nn

from pathlib import Path
from tensordict.nn import TensorDictModule

from torch.distributions import Categorical
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import DiscreteSACLoss
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from typing import Dict, Any
from collections import defaultdict
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)

from jani.torchrl_env import JANIEnv

from .utils import (
    create_data_collector,
    load_recovery_policy_module,
    load_q_risk_backbone,
    load_replay_buffer,
    safety_evaluation
)
from .actor import RecoveryActor


def load_hyperparams_from_file(file_path: str) -> Dict[str, Any]:
    """Load hyperparameters from a JSON file.

    The file should contain a JSON object with hyperparameter key-value pairs.
    Activation functions should be specified as strings: "ReLU", "Tanh", "LeakyReLU".
    """
    with open(file_path, 'r') as f:
        params = json.load(f)

    # Convert activation function string to actual class
    if "activation" in params:
        activation_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU,
        }
        params["activation_fn"] = activation_map.get(params.pop("activation"), nn.ReLU)

    # Convert hidden_size and n_layers to hidden_sizes lists if present
    if "hidden_size" in params and "n_layers" in params:
        hidden_sizes = [params.pop("hidden_size")] * params.pop("n_layers")
        params["actor_hidden_sizes"] = hidden_sizes
        params["critic_hidden_sizes"] = hidden_sizes

    # Map tuning param names to training param names
    param_mapping = {
        "lr_actor": "learning_rate_actor",
        "lr_critic": "learning_rate_critic",
        "dropout": "actor_dropout",
    }
    for old_key, new_key in param_mapping.items():
        if old_key in params:
            params[new_key] = params.pop(old_key)

    # If dropout is set, apply to both actor and critic
    if "actor_dropout" in params and "critic_dropout" not in params:
        params["critic_dropout"] = params["actor_dropout"]

    return params


def create_actor_module(hyperparams: Dict[str, Any], env: JANIEnv) -> TensorDictModule:
    """Create the actor network for the task policy."""
    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("actor_hidden_sizes", [256, 256])
    dropout = hyperparams.get("actor_dropout", 0.0)
    activation_fn = hyperparams.get("activation_fn", nn.ReLU)
    # Build the actor network
    actor_backbone = MLP(
        in_features=input_size,
        out_features=n_actions,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    # Wrap in TensorDictModule
    actor_module = TensorDictModule(
        module=actor_backbone,
        in_keys=["observation"],
        out_keys=["logits"],
    )
    return actor_module


def create_qvalue_network(hyperparams: Dict[str, Any], env: JANIEnv) -> ValueOperator:
    """Create the Q-value network for SAC.

    For discrete SAC, the Q-network outputs Q-values for all actions given a state.
    Output shape: (batch_size, n_actions)
    """
    n_actions = env.n_actions
    input_size = env.observation_spec["observation"].shape[0]
    hidden_sizes = hyperparams.get("critic_hidden_sizes", [256, 256])
    dropout = hyperparams.get("critic_dropout", 0.0)
    activation_fn = hyperparams.get("activation_fn", nn.ReLU)
    # Build the Q-value network (outputs Q-value for each action)
    qvalue_backbone = MLP(
        in_features=input_size,
        out_features=n_actions,
        num_cells=hidden_sizes,
        dropout=dropout,
        activation_class=activation_fn,
    )
    # Wrap in ValueOperator
    qvalue_module = ValueOperator(
        module=qvalue_backbone,
        in_keys=["observation"],
        out_keys=["action_value"],
    )
    return qvalue_module


def create_replay_buffer(hyperparams: Dict[str, Any]) -> ReplayBuffer:
    """Create a replay buffer for off-policy SAC training."""
    buffer_size = hyperparams.get("replay_buffer_size", 100000)
    # Create the storage
    storage = LazyTensorStorage(
        max_size=buffer_size,
        device=hyperparams.get("device", "cpu"),
    )
    # Create the sampler (random sampling for off-policy)
    sampler = RandomSampler()
    # Create the replay buffer
    replay_buffer = ReplayBuffer(
        storage=storage,
        sampler=sampler,
    )
    return replay_buffer


def create_task_loss_module(
    hyperparams: Dict[str, Any],
    actor: ProbabilisticActor,
    qvalue_network: ValueOperator,
    env: JANIEnv
) -> DiscreteSACLoss:
    """Create the loss module for Discrete SAC (task policy)."""
    alpha_init = hyperparams.get("alpha_init", 0.05)
    target_entropy = hyperparams.get("target_entropy", "auto")

    loss_module = DiscreteSACLoss(
        actor_network=actor,
        qvalue_network=qvalue_network,
        action_space="categorical",
        num_actions=env.n_actions,
        delay_qvalue=True,
        alpha_init=alpha_init,
        fixed_alpha=True,
        target_entropy=target_entropy,
    )
    return loss_module


def train(hyperparams: Dict[str, Any], args: Dict[str, Any], env: JANIEnv, eval_env: JANIEnv) -> None:
    """Train the Recovery RL agent with Discrete SAC task policy."""
    logs = defaultdict(list)

    # Hyperparameters
    total_timesteps = hyperparams.get("total_timesteps", 1024000)
    n_steps = hyperparams.get("n_steps", 256)
    batch_size = hyperparams.get("batch_size", 128)
    # Support separate learning rates for actor and critic
    lr = hyperparams.get("learning_rate", 1e-3)
    lr_actor = hyperparams.get("learning_rate_actor", lr)
    lr_critic = hyperparams.get("learning_rate_critic", lr)
    max_grad_norm = hyperparams.get("max_grad_norm", 0.5)
    n_updates_per_step = hyperparams.get("n_updates_per_step", 8)
    target_update_freq = hyperparams.get("target_update_freq", 1)
    tau = hyperparams.get("tau", 0.01)  # Soft update coefficient

    warmup_steps = hyperparams.get("warmup_steps", 2000)
    risk_threshold = hyperparams.get("risk_threshold", -0.35)

    # Setup wandb logging
    use_wandb = not args.get("disable_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=args.get("wandb_project", "recovery-rl-sac"),
            entity=args.get("wandb_entity"),
            name=args.get("experiment_name"),
            config={**hyperparams, **args},
        )

    log_dir = args.get("log_dir", "")
    log_results = False
    if log_dir != "":
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "safety_evaluation_results.txt"
        log_file_path.open("w").close()  # Create an empty log file
        log_results = True

    model_save_dir = args.get("model_save_dir", "")
    save_models = False
    if model_save_dir != "":
        model_save_dir = Path(model_save_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        save_models = True

    # Create task policy actor and Q-value networks
    print("Creating task policy (SAC) actor and Q-value networks...")
    task_actor_module = create_actor_module(hyperparams, env)

    # Task policy for collecting data (outputs "task_action")
    task_policy = ProbabilisticActor(
        module=task_actor_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["task_action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )

    # Task policy for training (outputs "action" for DiscreteSACLoss)
    task_policy_training = ProbabilisticActor(
        module=task_actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    # Q-value network for task policy
    task_qvalue_network = create_qvalue_network(hyperparams, env)

    # Load recovery policy and risk module from pretrained models
    print("Loading pretrained recovery policy and risk module...")
    rec_policy_path = args.get("recover_policy_path", "")
    assert rec_policy_path != "", "Path to pretrained recovery policy must be provided in args with key 'recover_policy_path'"

    q_risk_path = args.get("q_risk_path", "")
    assert q_risk_path != "", "Path to pretrained q_risk module must be provided in args with key 'q_risk_path'"

    # Load the recovery policy backbone and create the recovery policy module
    recovery_policy_module = load_recovery_policy_module(rec_policy_path)
    # Recovery policy for collecting data (outputs "recovery_action")
    recovery_policy = ProbabilisticActor(
        module=recovery_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["recovery_action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )
    # Recovery policy for training (outputs "action")
    recovery_policy_training = ProbabilisticActor(
        module=recovery_policy_module,
        in_keys={"logits": "logits"},
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    # Load the q_risk backbone and create the q_risk module
    q_risk_backbone = load_q_risk_backbone(q_risk_path)
    # Q-risk module for collecting data (outputs "q_risk_value")
    q_risk_module = ValueOperator(
        module=q_risk_backbone,
        in_keys=["observation"],
        out_keys=["q_risk_value"],
    )
    # Q-risk module for training (outputs "action_value")
    q_risk_module_training = ValueOperator(
        module=q_risk_backbone,
        in_keys=["observation"],
        out_keys=["action_value"],
    )

    # Create the combined recovery actor for data collection
    recovery_actor = RecoveryActor(
        task_policy=task_policy,
        recovery_policy=recovery_policy,
        q_risk_module=q_risk_module,
        risk_threshold=risk_threshold
    )

    # Create replay buffer for task policy training (SAC is off-policy)
    print("Creating replay buffer for task policy...")
    task_replay_buffer = create_replay_buffer(hyperparams)

    # Load offline replay buffer for q_risk and recovery policy training
    print("Loading offline replay buffer for recovery policy and q_risk module...")
    offline_buffer_path = args.get("offline_buffer_path", "")
    assert offline_buffer_path != "", "Path to offline replay buffer must be provided in args with key 'offline_buffer_path'"
    offline_replay_buffer = load_replay_buffer(offline_buffer_path, hyperparams)

    # Create loss modules
    print("Creating loss modules...")
    # Task policy loss module (Discrete SAC)
    task_loss_module = create_task_loss_module(
        hyperparams,
        task_policy_training,
        task_qvalue_network,
        env
    )

    # Risk module loss (Discrete SAC loss with only Q-value loss for risk module)
    risk_loss_module = DiscreteSACLoss(
        actor_network=recovery_policy_training,
        qvalue_network=q_risk_module_training,
        action_space="categorical",
        num_actions=env.n_actions,
        fixed_alpha=True,
        alpha_init=1e-12,  # Disable entropy regularization for risk module
    )

    # Create data collector
    print("Creating data collector...")
    collector = create_data_collector(hyperparams, env, recovery_actor)

    # Create optimizers (separate for task actor, task Q-value, recovery components)
    task_actor_optim = torch.optim.Adam(task_policy_training.parameters(), lr=lr_actor)
    task_qvalue_optim = torch.optim.Adam(task_qvalue_network.parameters(), lr=lr_critic)
    risk_optim = torch.optim.Adam(
        list(recovery_policy_training.parameters()) + list(q_risk_module_training.parameters()),
        lr=lr
    )

    # Learning rate schedulers
    task_actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        task_actor_optim, total_timesteps // n_steps, 1e-5
    )
    task_qvalue_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        task_qvalue_optim, total_timesteps // n_steps, 1e-5
    )
    risk_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        risk_optim, total_timesteps // n_steps, 1e-5
    )

    # Training loop
    total_steps_collected = 0
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=False,
            disable=not sys.stdout.isatty(),
        ) as progress:
        training_task = progress.add_task("Training Recovery RL (SAC) Agent", total=total_timesteps)
        eval_task = progress.add_task("Policy Safety Evaluation", total=100, visible=False)

        for i, td_data in enumerate(collector):
            # Process collected data for task policy training
            td_task_data = td_data.clone(recurse=True).detach()
            # Set the task action as the main action for task policy training
            td_task_data = td_task_data.select(
                "observation",
                "action",
                "task_action",
                "task_action_log_prob",
                "action_mask",
                "next"
            )
            td_task_data.set_("action", td_task_data.get("task_action"))
            td_task_data.rename_key_("task_action_log_prob", "sample_log_prob")
            del td_task_data["task_action"]

            # Add task data to replay buffer
            data_view = td_task_data.reshape(-1)
            task_replay_buffer.extend(data_view)
            total_steps_collected += n_steps

            # Process collected data for risk module training
            td_risk_data = td_data.clone(recurse=True).detach()
            td_risk_data = td_risk_data.select(
                "observation",
                "action",
                "action_mask",
                "next"
            )
            # Relabel all goal reached rewards to 0 for risk module training
            td_risk_data["next", "reward"] = torch.where(
                td_risk_data["next", "reward"] == env._goal_reward,
                torch.tensor(0.0, dtype=torch.float32),
                td_risk_data["next", "reward"]
            )
            # Add the data to the offline replay buffer for risk module training
            offline_replay_buffer.extend(td_risk_data.reshape(-1))

            # Skip training until we have enough data (warmup)
            if total_steps_collected < warmup_steps:
                progress.update(training_task, advance=n_steps)
                continue

            # Training updates
            for _ in range(n_updates_per_step):
                # Ensure we have enough samples
                if len(task_replay_buffer) < batch_size:
                    continue

                # Sample batches for task and risk training
                batch_task_data = task_replay_buffer.sample(batch_size)
                batch_risk_data = offline_replay_buffer.sample(batch_size)

                # Compute task policy losses (SAC)
                task_loss_dict = task_loss_module(batch_task_data)

                # Update task Q-value network
                task_qvalue_loss = task_loss_dict["loss_qvalue"]
                task_qvalue_optim.zero_grad()
                task_qvalue_loss.backward()
                torch.nn.utils.clip_grad_norm_(task_qvalue_network.parameters(), max_grad_norm)
                task_qvalue_optim.step()

                # Update task actor network
                task_actor_loss = task_loss_dict["loss_actor"]
                task_actor_optim.zero_grad()
                task_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(task_policy_training.parameters(), max_grad_norm)
                task_actor_optim.step()

                # Compute risk module losses
                risk_loss_dict = risk_loss_module(batch_risk_data)
                risk_loss = risk_loss_dict["loss_actor"] + risk_loss_dict["loss_qvalue"]
                risk_optim.zero_grad()
                risk_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(recovery_policy_training.parameters()) + list(q_risk_module_training.parameters()),
                    max_grad_norm
                )
                risk_optim.step()

                # Soft update target networks for task Q-value
                if (i + 1) % target_update_freq == 0:
                    with torch.no_grad():
                        for param, target_param in zip(
                            task_loss_module.qvalue_network_params.values(),
                            task_loss_module.target_qvalue_network_params.values()
                        ):
                            target_param.data.copy_(
                                tau * param.data + (1 - tau) * target_param.data
                            )

            # Logging
            logs["loss_task_actor"].append(task_loss_dict["loss_actor"].item())
            logs["loss_task_qvalue"].append(task_loss_dict["loss_qvalue"].item())
            logs["loss_task_alpha"].append(task_loss_dict["loss_alpha"].item())
            logs["alpha"].append(task_loss_dict["alpha"].item())
            logs["loss_risk_actor"].append(risk_loss_dict["loss_actor"].item())
            logs["loss_risk_qvalue"].append(risk_loss_dict["loss_qvalue"].item())
            logs["reward"].append(td_data["next", "reward"].mean().item())

            # Evaluate policy's safety
            with torch.no_grad():
                eval_result = safety_evaluation(
                    eval_env,
                    recovery_actor,
                    max_steps=hyperparams.get("max_steps", 256),
                    progress=progress,
                    task_id=eval_task
                )

            # Log to wandb
            if use_wandb:
                wandb.log({
                    "loss/task_actor": logs["loss_task_actor"][-1],
                    "loss/task_qvalue": logs["loss_task_qvalue"][-1],
                    "loss/task_alpha": logs["loss_task_alpha"][-1],
                    "loss/risk_actor": logs["loss_risk_actor"][-1],
                    "loss/risk_qvalue": logs["loss_risk_qvalue"][-1],
                    "train/alpha": logs["alpha"][-1],
                    "train/reward": logs["reward"][-1],
                    "safety/safety_rate": eval_result["safety_rate"],
                    "safety/average_reward": eval_result["average_reward"],
                    "train/timesteps": total_steps_collected,
                })

            if log_results:
                with log_file_path.open("a") as f:
                    f.write(f"{eval_result['safety_rate'] * 100:.2f}, {eval_result['average_reward']:.4f}\n")

            progress.console.print(
                f"Step {total_steps_collected}: Safety Rate: {eval_result['safety_rate']:.4f}, "
                f"Avg Reward: {eval_result['average_reward']:.4f}"
            )
            progress.console.print(
                f"  Task: Actor={logs['loss_task_actor'][-1]:.4f} Q={logs['loss_task_qvalue'][-1]:.4f} "
                f"α={logs['alpha'][-1]:.4f} | Risk: Actor={logs['loss_risk_actor'][-1]:.4f} Q={logs['loss_risk_qvalue'][-1]:.4f}"
            )

            # Save model checkpoints
            if save_models:
                model_save_path = model_save_dir / f"checkpoint_{i}"
                model_save_path.mkdir(parents=True, exist_ok=True)

                # Save task policy
                task_policy_path = model_save_path / "task_policy.pth"
                task_policy_params = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "output_dim": env.n_actions,
                    "hidden_dims": hyperparams.get("actor_hidden_sizes", [256, 256]),
                    "dropout": hyperparams.get("actor_dropout", 0.0),
                    "activation_fn": hyperparams.get("activation_fn", nn.ReLU),
                    "state_dict": task_policy.state_dict()
                }
                torch.save(task_policy_params, task_policy_path)

                # Save task Q-value network
                task_qvalue_path = model_save_path / "task_qvalue.pth"
                task_qvalue_params = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "output_dim": env.n_actions,
                    "hidden_dims": hyperparams.get("critic_hidden_sizes", [256, 256]),
                    "dropout": hyperparams.get("critic_dropout", 0.0),
                    "activation_fn": hyperparams.get("activation_fn", nn.ReLU),
                    "state_dict": task_qvalue_network.state_dict()
                }
                torch.save(task_qvalue_params, task_qvalue_path)

                # Save recovery policy
                recovery_policy_path = model_save_path / "recovery_policy.pth"
                recovery_policy_init = torch.load(rec_policy_path)
                recovery_policy_params = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "output_dim": env.n_actions,
                    "hidden_dims": recovery_policy_init["hidden_dims"],
                    "state_dict": recovery_policy.state_dict()
                }
                torch.save(recovery_policy_params, recovery_policy_path)

                # Save q_risk module
                q_risk_module_path = model_save_path / "q_risk_module.pth"
                q_risk_init = torch.load(q_risk_path)
                q_risk_params = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "output_dim": env.n_actions,
                    "hidden_dims": q_risk_init["hidden_dims"],
                    "state_dict": q_risk_module.state_dict()
                }
                torch.save(q_risk_params, q_risk_module_path)

            progress.update(training_task, advance=n_steps)
            progress.refresh()
            task_actor_scheduler.step()
            task_qvalue_scheduler.step()
            risk_scheduler.step()

    # Finish wandb logging
    if use_wandb:
        wandb.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Recovery RL with Discrete SAC on JANI Environments")
    parser.add_argument(
        '--jani_model',
        type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument(
        '--jani_property',
        type=str, required=True, help="Path to the JANI property file.")
    parser.add_argument(
        '--start_states',
        type=str, required=True, help="Path to the start states file.")
    parser.add_argument(
        '--eval_start_states',
        type=str, default="", help="Path to the evaluation start states file.")
    parser.add_argument(
        '--recover_policy_path',
        type=str, required=True, help="Path to the pretrained recovery policy.")
    parser.add_argument(
        '--q_risk_path',
        type=str, required=True, help="Path to the pretrained q_risk module.")
    parser.add_argument(
        '--offline_buffer_path',
        type=str, required=True, help="Path to the offline replay buffer for recovery policy and q_risk module.")
    parser.add_argument(
        '--objective',
        type=str, default="", help="Path to the objective file.")
    parser.add_argument(
        '--failure_property',
        type=str, default="", help="Path to the failure property file.")
    parser.add_argument(
        '--goal_reward',
        type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument(
        '--failure_reward',
        type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument(
        '--use_oracle',
        action='store_true', help="Use Tarjan oracle for unsafe state detection.")
    parser.add_argument(
        '--unsafe_reward',
        type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument(
        '--seed',
        type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        '--total_timesteps',
        type=int, default=1_000_000, help="Total timesteps for training.")
    parser.add_argument(
        '--max_steps',
        type=int, default=1000, help="Max steps per episode.")
    parser.add_argument(
        '--no_memory_reduced_mode',
        action='store_true', help="Disable memory reduced mode in the environment")
    parser.add_argument(
        '--log_dir',
        type=str, default="", help="Directory for logging.")
    parser.add_argument(
        '--model_save_dir',
        type=str, default="", help="Directory to save models.")
    parser.add_argument(
        '--disable_wandb',
        action='store_true', help="Disable logging to Weights & Biases.")
    parser.add_argument(
        '--wandb_project',
        type=str, default="recovery-rl-sac", help="Weights & Biases project name.")
    parser.add_argument(
        '--experiment_name',
        type=str, default=None, help="Experiment name for Weights & Biases logging.")
    parser.add_argument(
        '--wandb_entity',
        type=str, default=None, help="Weights & Biases entity name.")
    parser.add_argument(
        '--verbose',
        type=int, default=1, help="Verbosity level.")
    parser.add_argument(
        '--device',
        type=str, default='auto', help="Device to use for training (cpu or cuda).")
    parser.add_argument(
        '--hyperparams_file',
        type=str, default="", help="Path to JSON file with hyperparameters.")
    parser.add_argument(
        '--risk_threshold',
        type=float, default=-0.35, help="Risk threshold for switching to recovery policy.")

    # Tuning arguments
    parser.add_argument(
        '--tune',
        action='store_true', help="Run hyperparameter tuning before training.")
    parser.add_argument(
        '--n_trials',
        type=int, default=50, help="Number of Optuna trials for tuning.")
    parser.add_argument(
        '--tuning_timesteps',
        type=int, default=100000, help="Timesteps per trial during tuning (shorter than full training).")
    parser.add_argument(
        '--tuning_eval_interval',
        type=int, default=10, help="Evaluation interval during tuning (in collector iterations).")
    parser.add_argument(
        '--study_name',
        type=str, default="recovery_rl_sac_tuning", help="Optuna study name for tuning.")
    parser.add_argument(
        '--tuning_storage',
        type=str, default=None, help="Optuna storage URL for tuning (e.g., sqlite:///study.db).")
    parser.add_argument(
        '--tuning_pruner',
        type=str, default="median", choices=["median", "hyperband", "none"], help="Pruner type for tuning.")

    args = parser.parse_args()

    # Additional PyTorch seeding for full reproducibility
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)

    # # Make PyTorch deterministic (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Create training environment
    file_args = {
        'jani_model_path': args.jani_model,
        'jani_property_path': args.jani_property,
        'start_states_path': args.start_states,
        'objective_path': args.objective,
        'failure_property_path': args.failure_property,
        'goal_reward': args.goal_reward,
        'failure_reward': args.failure_reward,
        'seed': args.seed,
        'use_oracle': args.use_oracle,
        'unsafe_reward': args.unsafe_reward,
        'reduced_memory_mode': not args.no_memory_reduced_mode,
    }
    env = JANIEnv(**file_args)

    # Create evaluation environment
    eval_file_args = file_args.copy()
    eval_file_args["start_states_path"] = args.eval_start_states if args.eval_start_states else args.start_states
    eval_file_args["use_oracle"] = False  # Disable oracle for evaluation
    eval_env = JANIEnv(**eval_file_args)

    # Run hyperparameter tuning if requested
    tuned_params_file = None
    if args.tune:
        print("=" * 60)
        print("STARTING HYPERPARAMETER TUNING")
        print("=" * 60)

        from .tune_sac import run_tuning

        # Determine output directory for tuning results
        tuning_output_dir = args.log_dir if args.log_dir else "."

        _, tuned_params_file = run_tuning(
            jani_model=args.jani_model,
            jani_property=args.jani_property,
            start_states=args.start_states,
            eval_start_states=args.eval_start_states if args.eval_start_states else args.start_states,
            recover_policy_path=args.recover_policy_path,
            q_risk_path=args.q_risk_path,
            offline_buffer_path=args.offline_buffer_path,
            objective_path=args.objective,
            failure_property=args.failure_property,
            goal_reward=args.goal_reward,
            failure_reward=args.failure_reward,
            use_oracle=args.use_oracle,
            unsafe_reward=args.unsafe_reward,
            no_memory_reduced_mode=args.no_memory_reduced_mode,
            max_steps=args.max_steps,
            seed=args.seed,
            device=args.device,
            n_trials=args.n_trials,
            tuning_timesteps=args.tuning_timesteps,
            eval_interval=args.tuning_eval_interval,
            study_name=args.study_name,
            storage=args.tuning_storage,
            pruner=args.tuning_pruner,
            output_dir=tuning_output_dir,
        )

        print("\n" + "=" * 60)
        print("TUNING COMPLETE - STARTING FULL TRAINING")
        print("=" * 60 + "\n")

    # Define default hyperparameters for SAC-based Recovery RL
    hyperparams = {
        'total_timesteps': args.total_timesteps,
        'n_steps': 256,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'tau': 0.01,  # Soft update coefficient
        'target_update_freq': 1,
        'alpha_init': 0.05,
        'target_entropy': 'auto',
        'replay_buffer_size': 100000,
        'warmup_steps': 2000,
        'n_updates_per_step': 8,
        'max_grad_norm': 0.5,
        'max_steps': args.max_steps,
        'device': args.device,
        'actor_hidden_sizes': [256, 256],
        'critic_hidden_sizes': [256, 256],
        'actor_dropout': 0.0,
        'critic_dropout': 0.0,
        'activation_fn': nn.ReLU,
        'risk_threshold': args.risk_threshold,
    }

    # Load hyperparameters from tuning or file (tuning takes precedence, then file)
    if tuned_params_file:
        file_params = load_hyperparams_from_file(tuned_params_file)
        hyperparams.update(file_params)
        print(f"Using tuned hyperparameters from: {tuned_params_file}")
    elif args.hyperparams_file:
        file_params = load_hyperparams_from_file(args.hyperparams_file)
        hyperparams.update(file_params)
        print(f"Loaded hyperparameters from: {args.hyperparams_file}")

    print(f"Training with hyperparameters: {hyperparams}")

    # Start training
    train(hyperparams, vars(args), env, eval_env)


if __name__ == "__main__":
    main()
