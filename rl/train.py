#!/usr/bin/env python3
"""
Comprehensive training script for JANI environment with hyperparameter tuning.

This script provides advanced training capabilities including:
- Command line argument parsing for JANI model files
- Hyperparameter optimization using Optuna
- TensorBoard logging
- Weights & Biases integration
- Model checkpointing and best model saving
- Support for both property files and individual start/goal/safe files
"""

# @Chaahat: Sample use: python -m rl.train --model_file examples/transport/model.jani --goal_file examples/transport/goal.jani --start_file examples/transport/start.jani --safe_file examples/transport/safe.jani --experiment_name testing_stuff --disable-wandb --total_timesteps 20000

import argparse
import sys
import warnings
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import torch

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

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

from jani.environment import JaniEnv
from .callbacks import ClassifierDebugCallback, ClassifierMonitorCallback, WandbCallback, EvalCallback
from .buffer import RolloutBufferWithLB
from classifier import load_trained_model

from gymnasium.wrappers import TimeLimit


def mask_fn(env) -> np.ndarray:
    """Action masking function for the environment."""
    return env.unwrapped.action_mask()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Comprehensive training script for JANI environments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # JANI model file arguments
    parser.add_argument('--model_file', type=str, required=True,
                       help='Path to the JANI model file (mandatory)')
    parser.add_argument('--start_file', type=str, 
                       help='Path to the start state file')
    parser.add_argument('--goal_file', type=str,
                       help='Path to the goal condition file') 
    parser.add_argument('--safe_file', type=str,
                       help='Path to the safe condition file')
    parser.add_argument('--property_file', type=str,
                       help='Path to the property file (if provided, overrides start/goal/safe files)')

    # overwrite initial state generator to a pure random one
    parser.add_argument('--random_init', action='store_true',
                       help='Use random initial state generator instead of the one defined in the model')
    
    # Training arguments
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total number of training timesteps')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency (in timesteps)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    
    # Hyperparameter tuning
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of hyperparameter tuning trials')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Optuna study name')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for logs and tensorboard')
    parser.add_argument('--model_save_dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--wandb_project', type=str, default='jani-training',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity/team name')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    
    # Control flags
    parser.add_argument('--disable-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--disable-eval', action='store_true',
                       help='Disable policy evaluation during training')
    
    # Model loading options
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to a trained model to load and continue training from')
    parser.add_argument('--no_reset_timesteps', action='store_true',
                       help='Continue from saved timesteps when loading a model (default: reset timesteps)')
    
    # Safety classifier arguments
    parser.add_argument('--use_classifier', action='store_true',
                       help='Enable safety classifier for intermediate rewards')
    parser.add_argument('--classifier_model', type=str, default=None,
                       help='Path to trained safety classifier model (.pth file)')
    parser.add_argument('--safe_reward', type=float, default=0.0,
                       help='Reward for safe states when using classifier (default: 0.0)')
    parser.add_argument('--unsafe_reward', type=float, default=-0.01,
                       help='Reward for unsafe states when using classifier (default: -0.01)')
    parser.add_argument('--debug_classifier', action='store_true',
                       help='Enable classifier debugging mode: check classifier predictions against Tarjan oracle and log accuracy to wandb')
    
    # Classifier monitor callback arguments
    parser.add_argument('--monitor_classifier', action='store_true',
                       help='Enable classifier monitoring callback to evaluate performance against oracle during training')
    parser.add_argument('--monitor_freq', type=int, default=10000,
                       help='Frequency (in timesteps) for classifier monitoring (default: 10000)')
    parser.add_argument('--monitor_episodes', type=int, default=30,
                       help='Number of episodes to run for each monitoring evaluation (default: 30)')
    parser.add_argument('--monitor_csv_output', type=str, default=None,
                       help='Path to CSV file for writing monitoring data (states, actions, safety labels)')
    parser.add_argument('--tune_classifier_online', action='store_true',
                       help='Enable online fine-tuning of classifier using new oracle-labeled data')
    parser.add_argument('--classifier_data_dir', type=str, default=None,
                       help='Directory containing classifier training data (required for online fine-tuning)')
    parser.add_argument('--disable_monitor_model_saving', action='store_true',
                       help='Disable saving policy models during classifier monitoring')
    
    # Floored advantage estimation arguments
    parser.add_argument('--use_floored_advantages', action='store_true',
                       help='Enable floored advantage estimation using classifier lower bounds')
    parser.add_argument('--floored_alpha', type=float, default=1.0,
                       help='Scaling factor for lower bounds (default: 1.0). Safe states get 0, unsafe get -alpha')
    
    # JANI constraints generator options
    parser.add_argument('--no_block_previous', action='store_true',
                       help='Disable blocking previously generated values in ConstraintsGenerator (default: block_previous=True)')
    parser.add_argument('--block_all', action='store_true',
                       help='Enable blocking all previously generated models in ConstraintsGenerator (default: False)')
    
    return parser.parse_args()


def validate_file_arguments(args) -> Tuple[Dict[str, str], bool]:
    """Validate and process file arguments."""
    file_args = {}
    
    # Check if model file exists
    if not Path(args.model_file).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    file_args['model_file'] = args.model_file
    
    # Validate classifier arguments
    if args.use_classifier:
        if args.classifier_model is None:
            raise ValueError("--classifier_model must be provided when --use_classifier is enabled")
        if not Path(args.classifier_model).exists():
            raise FileNotFoundError(f"Classifier model file not found: {args.classifier_model}")
        print(f"🤖 Safety classifier enabled: {args.classifier_model}")
        print(f"   Safe reward: {args.safe_reward}, Unsafe reward: {args.unsafe_reward}")
    
    # Validate classifier monitoring arguments
    if args.monitor_classifier:
        print(f"📊 State monitoring enabled: freq={args.monitor_freq}, episodes={args.monitor_episodes}")
        if args.monitor_csv_output:
            print(f"   CSV output: {args.monitor_csv_output}")
        
        # Validate fine-tuning arguments (only if classifier is used)
        if args.tune_classifier_online:
            if not args.use_classifier:
                raise ValueError("--tune_classifier_online requires --use_classifier to be enabled")
            if args.classifier_data_dir is None:
                raise ValueError("--classifier_data_dir must be provided when --tune_classifier_online is enabled")
            if not Path(args.classifier_data_dir).exists():
                raise FileNotFoundError(f"Classifier data directory not found: {args.classifier_data_dir}")
            print(f"🔧 Online classifier fine-tuning enabled with data from: {args.classifier_data_dir}")
        
        if args.use_classifier:
            print(f"   Will evaluate classifier performance against oracle")
        else:
            print(f"   Will track unsafe and failure states using oracle only")
    
    # Validate floored advantages arguments
    if args.use_floored_advantages:
        if args.classifier_model is None:
            raise ValueError("--classifier_model must be provided when --use_floored_advantages is enabled")
        if not Path(args.classifier_model).exists():
            raise FileNotFoundError(f"Classifier model file not found: {args.classifier_model}")
        print(f"🔄 Floored advantage estimation enabled with alpha={args.floored_alpha}")
        print(f"   Lower bounds: Safe states=0, Unsafe states=-{args.floored_alpha}")
    
    # Handle property file vs individual files
    if args.property_file:
        if not Path(args.property_file).exists():
            raise FileNotFoundError(f"Property file not found: {args.property_file}")
        
        # Warn if other files are also provided
        other_files_provided = any([args.start_file, args.goal_file, args.safe_file])
        if other_files_provided:
            warnings.warn(
                "Property file provided along with start/goal/safe files. "
                "Only property file will be used, other files will be ignored.",
                UserWarning
            )
        
        file_args['property_file'] = args.property_file
        if args.random_init:
            file_args['random_init'] = args.random_init
        return file_args, True
    
    else:
        # Check individual files
        required_files = ['start_file', 'goal_file', 'safe_file']
        missing_files = []
        
        for file_type in required_files:
            file_path = getattr(args, file_type)
            if not file_path:
                missing_files.append(file_type)
            elif not Path(file_path).exists():
                raise FileNotFoundError(f"{file_type} not found: {file_path}")
            else:
                file_args[file_type] = file_path
        
        if missing_files:
            raise ValueError(
                f"When property_file is not provided, all of start_file, goal_file, and safe_file must be provided. "
                f"Missing: {', '.join(missing_files)}"
            )
        if args.random_init:
            file_args['random_init'] = args.random_init
        return file_args, False


def add_classifier_args(file_args: Dict[str, str], args) -> Dict[str, str]:
    """Add classifier arguments to file_args."""
    file_args['use_classifier'] = args.use_classifier
    if args.use_classifier:
        file_args['classifier_model'] = args.classifier_model
        file_args['safe_reward'] = args.safe_reward
        file_args['unsafe_reward'] = args.unsafe_reward
    # Add seed for reproducible environments
    if hasattr(args, 'seed') and args.seed is not None:
        file_args['seed'] = args.seed
    # Add JANI constraints generator parameters
    file_args['block_previous'] = not args.no_block_previous  # Convert no_block_previous to block_previous
    file_args['block_all'] = args.block_all
    return file_args


def create_seeded_file_args(file_args: Dict[str, str], seed_offset: int, env_name: str = "environment") -> Dict[str, str]:
    """Create file_args with a different seed by adding an offset."""
    seeded_args = file_args.copy()
    if 'seed' in seeded_args and seeded_args['seed'] is not None:
        new_seed = seeded_args['seed'] + seed_offset
        seeded_args['seed'] = new_seed
        print(f"🎯 {env_name}: Using seed {new_seed} (training seed + {seed_offset})")
    return seeded_args

def create_eval_file_args(file_args: Dict[str, str]) -> Dict[str, str]:
    """Create file_args for evaluation environment with classifier disabled."""
    eval_args = create_seeded_file_args(file_args, 10000, "Evaluation environment")
    
    eval_args['use_classifier'] = False
    # Remove classifier-specific keys to avoid passing them to JaniEnv
    eval_args.pop('classifier_model', None)
    eval_args.pop('safe_reward', None) 
    eval_args.pop('unsafe_reward', None)
    
    # Print message if classifier was originally enabled
    if file_args.get('use_classifier', False):
        print("📊 Evaluation environment: Classifier disabled for true reward assessment")
    
    return eval_args


def create_env(file_args: Dict[str, str], n_envs: int = 1, monitor: bool = True, timelimit: bool = True):
    """Create the training environment."""
    def make_env():
        env = JaniEnv(**file_args)
        if timelimit:
            env = TimeLimit(env, max_episode_steps=2048)
        env = ActionMasker(env, mask_fn)
        if monitor:
            env = Monitor(env)
        return env
    
    if n_envs == 1:
        env = make_env()
    else:
        # Create vectorized environment - make_vec_env will handle seeding internally
        # when the model is created with a seed parameter
        env = make_vec_env(make_env, n_envs=n_envs)
        if monitor:
            env = VecMonitor(env)
    
    return env


def create_model_with_buffer(env, args, hyperparams: Dict[str, Any], **model_kwargs) -> MaskablePPO:
    """Create MaskablePPO model with optional custom buffer for floored advantages."""
    model_params = {
        'policy': MaskableActorCriticPolicy,
        'env': env,
        'device': args.device,
        'seed': args.seed,
        **hyperparams,
        **model_kwargs
    }
    
    # Use custom buffer if floored advantages are enabled
    if args.use_floored_advantages:
        # Load classifier and scaler
        classifier, scaler = load_trained_model(args.classifier_model)
        
        # Create custom buffer
        buffer_kwargs = {
            'classifier': classifier,
            'scaler': scaler,
            'alpha': args.floored_alpha
        }
        model_params['rollout_buffer_class'] = RolloutBufferWithLB
        model_params['rollout_buffer_kwargs'] = buffer_kwargs
    
    return MaskablePPO(**model_params)


def suggest_hyperparameters(trial) -> Dict[str, Any]:
    """Suggest hyperparameters for Optuna optimization."""
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [32, 64, 128, 256, 512, 1024, 2048]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 30),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 5.0),
    }


def objective(trial, args, file_args: Dict[str, str]) -> float:
    """Objective function for hyperparameter optimization."""
    # Get suggested hyperparameters
    params = suggest_hyperparameters(trial)
    
    # Create environments
    train_env = create_env(file_args, args.n_envs, monitor=False)
    # For hyperparameter tuning, we always need evaluation regardless of disable_eval flag
    eval_env = create_env(create_eval_file_args(file_args), 1, timelimit=True)  # Disable classifier for evaluation
    
    try:
        # Create model with suggested hyperparameters
        model = create_model_with_buffer(
            train_env, 
            args, 
            params, 
            verbose=0
        )
        
        # Train for a subset of timesteps for hyperparameter tuning
        tuning_timesteps = min(args.total_timesteps // 4, 100000)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=max(tuning_timesteps // 10, 1000),
            n_eval_episodes=args.eval_episodes,
            best_model_save_path=None,  # Don't save during tuning
        )
        
        # Train model
        model.learn(total_timesteps=tuning_timesteps, callback=eval_callback)
        
        # Return mean reward from evaluation
        mean_reward = eval_callback.last_mean_reward
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        mean_reward = -float('inf')
    
    finally:
        # Clean up environments
        train_env.close()
        eval_env.close()
    
    return mean_reward


def train_model(args, file_args: Dict[str, str], hyperparams: Optional[Dict[str, Any]] = None):
    """Train the model with given hyperparameters."""
    # Set up logging directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"jani_training_{timestamp}"
    
    log_dir = Path(args.log_dir) / experiment_name
    model_save_dir = Path(args.model_save_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create environments
    print("Creating training environment...")
    train_env = create_env(file_args, args.n_envs, monitor=False, timelimit=True)
    
    # Create evaluation environment only if evaluation is not disabled
    eval_env = None
    if not args.disable_eval:
        eval_env = create_env(create_eval_file_args(file_args), 1, timelimit=True)  # Disable classifier for evaluation
    
    # Default hyperparameters if not provided
    if hyperparams is None:
        hyperparams = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
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
    
    # Create or load model
    if args.load_model:
        if not Path(args.load_model).exists():
            raise FileNotFoundError(f"Model file not found: {args.load_model}")
        print(f"Loading existing model from: {args.load_model}")
        model = MaskablePPO.load(
            args.load_model,
            env=train_env,
            device=args.device,
            **hyperparams
        )
        # Reset environment after loading model
        model.set_env(train_env)
        reset_timesteps = not args.no_reset_timesteps
        print(f"Model loaded. Reset timesteps: {reset_timesteps}")
    else:
        print("Creating new model from scratch")
        model = create_model_with_buffer(
            train_env,
            args,
            hyperparams,
            tensorboard_log=str(log_dir),
            verbose=args.verbose
        )
        reset_timesteps = True
    
    # Set up callbacks
    callbacks = []
    eval_callback = None
    
    # Evaluation callback (only if evaluation is not disabled)
    if not args.disable_eval and eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_save_dir / "best_model"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
        )
        callbacks.append(eval_callback)
    
    # Wandb callback (only if available and not disabled)
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb_callback = WandbCallback(verbose=args.verbose)
        callbacks.append(wandb_callback)
    
    # Classifier debug callback (only if debug_classifier is enabled and classifier is used)
    if args.debug_classifier and args.use_classifier:
        debug_callback = ClassifierDebugCallback(verbose=args.verbose)
        callbacks.append(debug_callback)
        if args.verbose >= 1:
            print("🔍 Classifier debugging enabled - will compare predictions with Tarjan oracle")
    elif args.debug_classifier and not args.use_classifier:
        print("⚠️ Warning: --debug_classifier enabled but --use_classifier is not. Debug callback will not be added.")
    
    # Classifier monitor callback (if monitor_classifier is enabled)
    if args.monitor_classifier:
        # Create an independent monitoring environment identical to training environment
        print("Creating independent monitoring environment...")
        monitor_file_args = create_seeded_file_args(file_args, 20000, "Monitor environment")
        monitor_env = create_env(monitor_file_args, n_envs=1, monitor=False, timelimit=True)
        
        monitor_callback = ClassifierMonitorCallback(
            env=monitor_env,
            monitor_freq=args.monitor_freq,
            num_episodes=args.monitor_episodes,
            tune_classifier=args.tune_classifier_online,
            data_dir=args.classifier_data_dir,
            csv_output_path=args.monitor_csv_output,
            model_save_dir=None if args.disable_monitor_model_saving else str(model_save_dir / "monitor_checkpoints"),
            verbose=args.verbose
        )
        callbacks.append(monitor_callback)
        if args.verbose >= 1:
            if args.use_classifier:
                print("📊 Classifier monitoring enabled - will evaluate classifier performance against Tarjan oracle")
                if args.tune_classifier_online:
                    print("🔧 Online classifier fine-tuning enabled")
            else:
                print("📊 State monitoring enabled - will track unsafe and failure states using Tarjan oracle only")
            
            if not args.disable_monitor_model_saving:
                print(f"💾 Policy model saving enabled - models will be saved to: {model_save_dir / 'monitor_checkpoints'}")
            else:
                print("💾 Policy model saving during monitoring is disabled")
    
    # Early stopping on reward threshold (optional)
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    # callbacks.append(stop_callback)
    
    print(f"Starting training for {args.total_timesteps} timesteps...")
    
    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="PPO",
        reset_num_timesteps=reset_timesteps
    )
    
    # Save final model
    final_model_path = model_save_dir / "final_model"
    model.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}")
    
    # Save hyperparameters
    hyperparams_path = model_save_dir / "hyperparameters.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Clean up
    train_env.close()
    if eval_env is not None:
        eval_env.close()
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.finish()
    
    # Return best reward if evaluation was performed, otherwise return 0
    best_reward = eval_callback.best_mean_reward if eval_callback is not None else 0.0
    return model, best_reward


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
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
    
    # Validate file arguments
    try:
        file_args, using_property_file = validate_file_arguments(args)
        file_args = add_classifier_args(file_args, args)
        print(f"Using {'property file' if using_property_file else 'individual files'}: {file_args}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    best_hyperparams = None
    best_reward = -float('inf')
    
    # Hyperparameter tuning
    if args.tune_hyperparams:
        if not OPTUNA_AVAILABLE:
            print("Error: Optuna not available. Cannot perform hyperparameter tuning.")
            print("Install optuna with: pip install optuna")
            print("Proceeding with default hyperparameters...")
        else:
            print(f"Starting hyperparameter tuning with {args.n_trials} trials...")
            
            study_name = args.study_name or f"jani_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize wandb for hyperparameter tuning if not disabled
            if WANDB_AVAILABLE and not args.disable_wandb:
                wandb.init(
                    project=f"{args.wandb_project}_tuning",
                    entity=args.wandb_entity,
                    name=f"{study_name}_tuning"
                )
            
            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
            
            # Run optimization
            try:
                study.optimize(
                    lambda trial: objective(trial, args, file_args),
                    n_trials=args.n_trials,
                    timeout=None
                )
                
                print("Hyperparameter tuning completed!")
                print(f"Best trial: {study.best_trial.number}")
                print(f"Best reward: {study.best_value}")
                print(f"Best params: {study.best_params}")
                
                best_hyperparams = study.best_params
                best_reward = study.best_value
                
                # Log best hyperparameters to wandb if not disabled
                if WANDB_AVAILABLE and not args.disable_wandb:
                    wandb.log({"best_reward": best_reward})
                    wandb.config.update(best_hyperparams)
                
            except Exception as e:
                print(f"Hyperparameter tuning failed: {e}")
                print("Proceeding with default hyperparameters...")
            
            if WANDB_AVAILABLE and not args.disable_wandb:
                wandb.finish()
    
    # Train final model with best hyperparameters
    print("Training final model...")
    final_model, final_reward = train_model(args, file_args, best_hyperparams)
    
    print(f"Training completed!")
    print(f"Final model reward: {final_reward}")
    if best_hyperparams:
        print(f"Best hyperparameters used: {best_hyperparams}")


if __name__ == "__main__":
    main()