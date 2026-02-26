import argparse
import torch
import numpy as np

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from callbacks import EvalCallback, SafetyEvalCallback, WandbCallback, SaveActorCallback, LoggingCallback
from jani.env import JANIEnv
from utils import create_env, create_eval_file_args, create_safety_eval_file_args

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# Optional imports for advanced features
try:
    import optuna
    OPTUNA_AVAILABLE = True
    try:
        from optuna.integration import WeightsAndBiasesCallback
        OPTUNA_WANDB_CALLBACK_AVAILABLE = True
    except ImportError:
        OPTUNA_WANDB_CALLBACK_AVAILABLE = False
except ImportError:
    OPTUNA_AVAILABLE = False
    OPTUNA_WANDB_CALLBACK_AVAILABLE = False
    print("Warning: Optuna not available. Hyperparameter tuning will be disabled.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available. Advanced logging will be disabled.")


def train_model(args, file_args: Dict[str, str], hyperparams: Optional[Dict[str, Any]] = None):
    """Train the model with given hyperparameters."""
    # Set up logging directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"jani_training_{timestamp}"
    
    log_dir = Path(args.log_dir)
    model_save_dir = Path(args.model_save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environments
    print("Creating training environment...")
    print(f"🤖 Oracle enabled: {file_args.get('use_oracle', False)}")
    train_env = create_env(file_args, args.n_envs, monitor=False, time_limited=True)

    # Default hyperparameters if not provided
    if hyperparams is None:
        hyperparams = {
            'learning_rate': 3e-4,
            'n_steps': args.n_steps,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    
    print(f"Training with hyperparameters: {hyperparams}")

    # Initialize MaskablePPO model
    model = MaskablePPO(
            MaskableActorCriticPolicy,
            train_env,
            tensorboard_log=str(log_dir),
            verbose=args.verbose,
            device=args.device,
            seed=args.seed,
            **hyperparams
        )
    reset_timesteps = True
    # Load pre-trained policy if specified
    if args.load_policy_path:
        print(f"Loading pre-trained policy from {args.load_policy_path}...")
        checkpoint = torch.load(args.load_policy_path, map_location=args.device, weights_only=False)
        model.policy.load_state_dict(checkpoint['state_dict'])
    
    # Placeholder for callbacks
    callbacks = []

    # Initialize wandb if available and not disabled
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            config={
                **vars(args),
                **(hyperparams or {}),
                'file_args': file_args
            }
        )
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)

    # Create evaluation environment and callback
    if not args.disable_eval:
        eval_file_args = create_eval_file_args(file_args, args.use_separate_eval_env)
        eval_env = create_env(eval_file_args, 1, monitor=True, time_limited=True)
        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            enumate_all_init_states=args.enumate_all_init_states,
            best_model_save_path=str(model_save_dir)
        )
        callbacks.append(eval_callback)

    # Checkpoint callback to save actor at intervals
    if args.save_all_checkpoints:
        save_actor_callback = SaveActorCallback(
            save_freq=args.eval_freq,
            save_path=model_save_dir,
            verbose=args.verbose
        )
        callbacks.append(save_actor_callback)

    # Logging callback for rewards
    if args.log_reward:
        safety_eval_file_args = create_safety_eval_file_args(file_args, vars(args), use_oracle=False)
        safety_eval_env = create_env(safety_eval_file_args, 1, monitor=True, time_limited=True)
        logging_callback = LoggingCallback(
            log_dir=log_dir,
            log_freq=args.eval_freq,
            eval_env=safety_eval_env,
            n_eval_episodes=args.n_eval_episodes,
            enumate_all_init_states=args.enumate_all_init_states,
            verbose=args.verbose
        )
        callbacks.append(logging_callback)

    # Create safety evaluation environment and callback
    if args.eval_safety:
        assert args.eval_start_states != "", "Evaluation start states file must be provided for safety evaluation."
        safety_eval_file_args = create_safety_eval_file_args(file_args, vars(args), use_oracle=False)
        safety_eval_env = create_env(safety_eval_file_args, 1, monitor=True, time_limited=True)
        safety_eval_callback = SafetyEvalCallback(
            safety_eval_env=safety_eval_env,
            eval_freq=args.eval_freq,
            log_dir=log_dir
        )
        callbacks.append(safety_eval_callback)

    # Start training
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="PPO",
        reset_num_timesteps=reset_timesteps
    )

    # Save the final model (actor only)
    policy = model.policy
    hidden_dims = policy.net_arch['pi']
    actor_path = model_save_dir / "final_actor.pth"
    torch.save({
        'input_dim': train_env.observation_space.shape[0],
        'output_dim': train_env.action_space.n,
        'hidden_dims': hidden_dims,
        'state_dict': policy.state_dict()
    }, actor_path)
    print(f"Final actor model saved to {actor_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Masked PPO on JANI Environments")
    # Environment settings
    parser.add_argument('--jani_model', type=str, required=True, 
                        help="Path to the JANI model file.")
    parser.add_argument('--jani_property', type=str, default="", 
                        help="Path to the JANI property file.")
    parser.add_argument('--start_states', type=str, default="", 
                        help="Path to the start states file.")
    parser.add_argument('--objective', type=str, default="", 
                        help="Path to the objective file.")
    parser.add_argument('--failure_property', type=str, default="", 
                        help="Path to the failure property file.")
    parser.add_argument('--eval_start_states', type=str, default="", 
                        help="Path to the evaluation start states file.")
    parser.add_argument('--goal_reward', type=float, default=1.0, 
                        help="Reward for reaching the goal.")
    parser.add_argument('--failure_reward', type=float, default=-1.0, 
                        help="Reward for reaching failure state.")
    parser.add_argument('--unsafe_reward', type=float, default=-0.01, 
                        help="Reward for unsafe states when using oracle.")
    parser.add_argument('--use_oracle', action='store_true', 
                        help="Use Tarjan oracle for unsafe state detection.")
    parser.add_argument('--disable_oracle_cache', action='store_true', 
                        help="Disable caching in the oracle.")
    
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility.")
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, 
                        help="Total timesteps for training.")
    parser.add_argument('--n_envs', type=int, default=1, 
                        help="Number of parallel environments.")
    parser.add_argument('--max_steps', type=int, default=1000, 
                        help="Max steps per episode.")
    parser.add_argument('--n_steps', type=int, default=256, 
                        help="Number of steps per update.")
    
    # Training and evaluation settings
    parser.add_argument('--log_dir', type=str, default="./logs", 
                        help="Directory for logging.")
    parser.add_argument('--log_reward', action='store_true', 
                        help="Log episode rewards.")
    parser.add_argument('--model_save_dir', type=str, default="./models", 
                        help="Directory to save models.")
    parser.add_argument('--use_separate_eval_env', action='store_true', 
                        help="Use a separate environment for evaluation.")
    parser.add_argument('--enumate_all_init_states', action='store_true', 
                        help="Evaluate on all initial states during evaluation.")
    parser.add_argument('--eval_freq', type=int, default=2048, 
                        help="Evaluation frequency in timesteps.")
    parser.add_argument('--n_eval_episodes', type=int, default=50, 
                        help="Number of episodes for each evaluation.")
    parser.add_argument('--load_policy_path', type=str, default="", 
                        help="Path to a pre-trained policy to load before training.")
    parser.add_argument('--save_all_checkpoints', action='store_true', 
                        help="Save model checkpoints at each evaluation.")
    parser.add_argument('--eval_safety', action='store_true', 
                        help="Enable safety evaluation during training.")
    parser.add_argument('--disable_eval', action='store_true', 
                        help="Disable evaluation during training.")
    
    # Others
    parser.add_argument('--wandb_project', type=str, default="jani_rl", 
                        help="Weights & Biases project name.")
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help="Weights & Biases entity name.")
    parser.add_argument('--experiment_name', type=str, default="", 
                        help="Name of the experiment.")
    parser.add_argument('--verbose', type=int, default=1, 
                        help="Verbosity level.")
    parser.add_argument('--device', type=str, default='cpu', 
                        help="Device to use for training (cpu or cuda).")
    parser.add_argument('--disable_wandb', action='store_true', 
                        help="Disable Weights & Biases logging.")
    

    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Additional PyTorch seeding for full reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    file_args = {
        'jani_model': args.jani_model,
        'jani_property': args.jani_property,
        'start_states': args.start_states,
        'objective': args.objective,
        'failure_property': args.failure_property,
        'goal_reward': args.goal_reward,
        'failure_reward': args.failure_reward,
        'unsafe_reward': args.unsafe_reward,
        'seed': args.seed,
        'use_oracle': args.use_oracle,
        'max_steps': args.max_steps,
        'disable_oracle_cache': args.disable_oracle_cache
    }
    if args.use_separate_eval_env:
        assert args.eval_start_states != "", "Evaluation start states file must be provided when using separate eval env."

    train_model(args, file_args)

if __name__ == "__main__":
    main()