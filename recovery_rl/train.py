import sys
import torch
import torch.nn as nn

from pathlib import Path
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import DiscreteSACLoss
from torchrl.envs.utils import ExplorationType, set_exploration_type

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
from utils import create_safety_eval_file_args

from .utils import (
    create_actor_module, 
    create_critic, 
    create_data_collector, 
    create_rollout_buffer,
    create_advantage_module,
    create_loss_module,
    load_recovery_policy_module,
    load_q_risk_backbone,
    load_replay_buffer,
    safety_evaluation
)
from .actor import RecoveryActor


def train(hyperparams: Dict[str, Any], args: dict[str, any], env: JANIEnv, eval_env: JANIEnv) -> None:
    """Train the PPO agent."""
    # Some hyperparameters
    total_timesteps = hyperparams.get("total_timesteps", 1024000)
    n_steps = hyperparams.get("n_steps", 256)
    batch_size = hyperparams.get("batch_size", 64)
    lr = hyperparams.get("learning_rate", 3e-4)
    n_epoches = hyperparams.get("n_epoches", 5)

    # Setup wandb logging
    use_wandb = not args.get("disable_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=args.get("wandb_project", "recovery-rl"),
            entity=args.get("wandb_entity"),
            name=args.get("experiment_name"),
            config={**hyperparams, **args},
        )

    log_dir = args.get("log_dir", "")
    log_safety_results = False
    if log_dir != "":
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "safety_evaluation_results.txt"
        log_file_path.open("w").close() # Create an empty log file
        log_safety_results = True

    model_save_dir = args.get("model_save_dir", "")
    save_models = False
    if model_save_dir != "":
        model_save_dir = Path(model_save_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        save_models = True

    # Create actor and critic (i.e., q_value) networks
    print("Creating task policy and critic networks...")
    task_policy_module = create_actor_module(hyperparams, env)
    # Task policy for collecting data
    task_policy = ProbabilisticActor(
        module=task_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["task_action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True, # Not sure whether this is actually need
    )
    # Training view of the task policy (outputs "action" instead of "task_action")
    task_policy_training = ProbabilisticActor(
        module=task_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True, # Not sure whether this is actually need
    )
    # Critic for task policy value estimation
    critic = create_critic(hyperparams, env)

    # Load recovery policy and risk module from pretrained models
    print("Loading pretrained recovery policy and risk module...")
    rec_policy_path = args.get("recover_policy_path", "")
    assert rec_policy_path != "", "Path to pretrained recovery policy must be provided in args with key 'recover_policy_path'"

    q_risk_path = args.get("q_risk_path", "")
    assert q_risk_path != "", "Path to pretrained q_risk module must be provided in args with key 'q_risk_path'"

    # Load the recovery policy backbone and create the recovery policy module
    recovery_policy_module = load_recovery_policy_module(rec_policy_path)
    # recovery policy for collecting data
    recovery_policy = ProbabilisticActor(
        module=recovery_policy_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["recovery_action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True, # Not sure whether this is actually need
    )
    # recovery policy for training (outputs "action" instead of "recovery_action")
    recovery_policy_training = ProbabilisticActor(
        module=recovery_policy_module,
        in_keys={"logits": "logits"},
        out_keys=["action"],
        distribution_class=Categorical, # DiscreteSACLoss does not support action mask
        return_log_prob=True, # Not sure whether this is actually need
    )

    # Load the q_risk backbone and create the q_risk module
    q_risk_backbone = load_q_risk_backbone(q_risk_path)
    # q_risk module for collecting data
    q_risk_module = ValueOperator(
        module=q_risk_backbone,
        in_keys=["observation"],
        out_keys=["q_risk_value"],
    )
    # q_risk module for training (outputs "action_value" instead of "q_risk_value")
    q_risk_module_training = ValueOperator(
        module=q_risk_backbone,
        in_keys=["observation"],
        out_keys=["action_value"],
    )

    # Create the combined recovery actor
    recovery_actor = RecoveryActor(
        task_policy=task_policy,
        recovery_policy=recovery_policy,
        q_risk_module=q_risk_module,
        risk_threshold=-0.35
    )

    # Create replay buffer for training task policy and critic
    print("Creating rollout buffer...")
    rollout_buffer = create_rollout_buffer(hyperparams)

    # Load replay buffer for q_risk and recovery policy
    print("Loading offline replay buffer for recovery policy and q_risk module...")
    offline_buffer_path = args.get("offline_buffer_path", "")
    assert offline_buffer_path != "", "Path to offline replay buffer must be provided in args with key 'offline_buffer_path'"
    offline_replay_buffer = load_replay_buffer(offline_buffer_path, hyperparams)
    # offline_replay_buffer = create_replay_buffer(hyperparams)
    
    # Create loss module
    print("Creating loss modules...")
    # Task policy loss module (PPO)
    ppo_loss_module = create_loss_module(
        hyperparams, 
        task_policy_training, 
        critic)
    
    # Risk module loss (Discrete SAC loss with only Q-value loss)
    risk_loss_module = DiscreteSACLoss(
        actor_network = recovery_policy_training,
        qvalue_network = q_risk_module_training,
        action_space = "categorical",
        num_actions = env.n_actions,
        fixed_alpha=True,
        alpha_init=1e-12, # Disable entropy regularization to simulate normal average value learning for the risk module
    )

    # Create advantage module
    print("Creating advantage module...")
    advantage_module = create_advantage_module(hyperparams, critic)

    # Create data collector
    print("Creating data collector...")
    collector = create_data_collector(hyperparams, env, recovery_actor)

    # Create optimizer
    optim = torch.optim.Adam(
        list(ppo_loss_module.parameters()) + list(risk_loss_module.parameters()), 
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_timesteps // n_steps, 0.0
    )

    # Training loop
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
        training_task = progress.add_task("Training Recovery RL Agent", total=hyperparams.get("total_timesteps", 1024000))
        safety_eval_task = progress.add_task("Evaluating safety...", total=100, visible=False)

        for i, td_data in enumerate(collector):
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
            # Rename "task_action_log_prob" to "action_log_prob" for PPO loss module
            td_task_data.rename_key_("task_action_log_prob", "action_log_prob")
            del td_task_data["task_action"]

            # Compute advantages for PPO loss
            with torch.no_grad():
                advantage_module(td_task_data)

            # Add the data to the replay buffer
            rollout_buffer.empty() # Clear the rollout buffer before adding new data
            rollout_buffer.extend(td_task_data)

            td_risk_data = td_data.clone(recurse=True).detach()
            # The final action is the main action for risk policy training
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
            # Add the data to the replay buffer for risk module training
            offline_replay_buffer.extend(td_risk_data)

            for _ in range(n_epoches):
                # Sample batches for task and risk training
                for _ in range(n_steps // batch_size):
                    batch_task_data = rollout_buffer.sample(batch_size)
                    batch_risk_data = offline_replay_buffer.sample(batch_size)

                    task_loss = ppo_loss_module(batch_task_data)
                    risk_loss = risk_loss_module(batch_risk_data)
                    loss_value = (
                        task_loss["loss_objective"]
                        + task_loss["loss_critic"]
                        + task_loss["loss_entropy"]
                        + risk_loss["loss_actor"]
                        + risk_loss["loss_qvalue"] # Only q_value loss for the risk module
                    )

                    # Optimizer step
                    optim.zero_grad()
                    loss_value.backward()

                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(
                        list(ppo_loss_module.parameters()) + list(risk_loss_module.parameters()),
                        max_norm=1.0
                    )
                    
                    optim.step()
            
            # Evaluate policy's safety
            with torch.no_grad():
                safety_results = safety_evaluation(eval_env, recovery_actor, progress=progress, task_id=safety_eval_task)

            # Log to wandb
            if use_wandb:
                wandb.log({
                    "loss/total": loss_value.item(),
                    "loss/ppo_objective": task_loss["loss_objective"].item(),
                    "loss/ppo_critic": task_loss["loss_critic"].item(),
                    "loss/ppo_entropy": task_loss["loss_entropy"].item(),
                    "loss/risk_actor": risk_loss["loss_actor"].item(),
                    "loss/risk_qvalue": risk_loss["loss_qvalue"].item(),
                    "safety/safety_rate": safety_results["safety_rate"],
                    "safety/average_reward": safety_results["average_reward"],
                    "train/reward": td_data["next", "reward"].mean().item(),
                    "train/timesteps": (i + 1) * n_steps,
                })

            if log_safety_results:
                with log_file_path.open("a") as f:
                    f.write(f"{safety_results['safety_rate'] * 100:.2f}, {safety_results['average_reward']:.4f}\n")
            
            # Save model
            if save_models:
                # Set up directory for saving the model checkpoint
                model_save_path = model_save_dir / f"checkpoint_{i}"
                model_save_path.mkdir(parents=True, exist_ok=True)

                # Save task policy
                task_policy_path = model_save_path / "task_policy.pth"
                task_policy_paras = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "output_dim": env.n_actions,
                    "hidden_dims": hyperparams.get("actor_hidden_sizes", [64, 128]),
                    "dropout": hyperparams.get("actor_dropout", 0.2),
                    "activation_fn": hyperparams.get("activation_fn", nn.Tanh),
                    "state_dict": task_policy.state_dict()
                }
                torch.save(task_policy_paras, task_policy_path)

                # Save critic
                critic_path = model_save_path / "critic.pth"
                critic_paras = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "hidden_dims": hyperparams.get("critic_hidden_sizes", [64, 128]),
                    "dropout": hyperparams.get("critic_dropout", 0.2),
                    "activation_fn": hyperparams.get("activation_fn", nn.Tanh),
                    "state_dict": critic.state_dict()
                }
                torch.save(critic_paras, critic_path)

                # Save recovery policy
                recovery_policy_path = model_save_path / "recovery_policy.pth"
                recovery_policy_init = torch.load(rec_policy_path)
                recovery_policy_paras = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "output_dim": env.n_actions,
                    "hidden_dims": recovery_policy_init["hidden_dims"],
                    "state_dict": recovery_policy.state_dict()
                }
                torch.save(recovery_policy_paras, recovery_policy_path)

                # Save q_risk module
                q_module_path = model_save_path / "q_risk_module.pth"
                q_risk_init = torch.load(q_risk_path)
                q_risk_paras = {
                    "input_dim": env.observation_spec["observation"].shape[0],
                    "output_dim": env.n_actions,
                    "hidden_dims": q_risk_init["hidden_dims"],
                }
                torch.save(q_risk_paras, q_module_path)

            progress.console.print(f"Percentage of safe runs: {safety_results['safety_rate']}; Average reward : {safety_results['average_reward']}")

            progress.update(training_task, advance=n_steps)
            progress.refresh()
            scheduler.step()

    # Finish wandb logging
    if use_wandb:
        wandb.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Masked PPO on JANI Environments")
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
        type=str, default="", help="Path to the start states file for evaluation environment.")
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
        type=str, default="recovery-rl", help="Weights & Biases project name.")
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

    args = parser.parse_args()
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
    eval_file_args["start_states_path"] = args.eval_start_states
    eval_file_args["use_oracle"] = False  # Disable oracle for evaluation
    eval_env = JANIEnv(**eval_file_args)

    # Define hyperparameters
    hyperparams = {
        'total_timesteps': args.total_timesteps,
        'n_steps': 256,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'gae_lambda': 0.95,
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'ent_coef': 1e-4,
        'critic_coeff': 1.0,
        'max_grad_norm': 0.5,
        'n_epochs': 10,
        'device': args.device,
    }
    print(f"Training with hyperparameters: {hyperparams}")
    # Start training
    train(hyperparams, vars(args), env, eval_env)

if __name__ == "__main__":
    main()