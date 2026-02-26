import argparse
import torch
import torch.nn as nn

from pathlib import Path
from tensordict import TensorDict
from torchrl.modules import MaskedCategorical

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from jani import JANIEnv
from utils import create_env, create_safety_eval_file_args, create_eval_file_args

from .buffer import collect_trajectory, collect_trajectory_with_stricted_rule, DAggerBuffer
from .policy import Policy, evaluate_policy_safety_on_state

import sys
DISABLE_BAR = not sys.stdout.isatty()

RAY_AVAILABLE = True
try:
    import ray
    from . import ray_workers
except Exception:
    RAY_AVAILABLE = False




def evaluate_policy(
        env: JANIEnv, 
        policy: nn.Module, 
        num_episodes: int = 100,
        max_steps: int = 1024, 
        device: torch.device = torch.device("cpu")) -> float:
    """Evaluate the policy over a number of episodes and return the average reward."""
    policy.to(device)
    policy.eval()
    total_reward = 0.0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done and step_count < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
            action_mask = env.unwrapped.action_mask().astype(int)
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                logits = policy(obs_tensor)
                action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
                action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension
            obs, reward, done, _, _ = env.step(action)
            step_count += 1
        episode_reward += reward # Consider only reward at the end of episode

        total_reward += episode_reward

    average_reward = total_reward / num_episodes
    return average_reward


def evaluate_policy_safety(args: dict, hyperparams: dict, file_args: dict, network_paras: dict, policy: nn.Module) -> tuple[float, float]:
    """Evaluate the safety rate of a policy under all initial states."""
    safety_eval_env = create_env(file_args, n_envs=1, monitor=False, time_limited=True)
    num_init_states = safety_eval_env.unwrapped.get_init_state_pool_size()
    
    if args.get("use_multiprocessors", False) and RAY_AVAILABLE:
        print(f"Evaluating policy safety using {hyperparams.get('num_workers', 200)} Ray workers...")
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=False)

        network_state_dict = ray_workers.to_cpu_state_dict(policy)
        # Create PolicySafetyEvaluator actors
        evaluators = [ray_workers.PolicySafetyEvaluator.remote(file_args, network_paras, network_state_dict) for _ in range(hyperparams.get("num_workers", 200))]
        futures = []
        for idx in range(num_init_states):
            evaluator = evaluators[idx % len(evaluators)]
            futures.append(evaluator.evaluate_safety.remote(idx, file_args["max_steps"]))

        # Gather results
        results = ray.get(futures)
    else:
        print("Evaluating policy safety sequentially...")
        results = []
        for idx in range(num_init_states):
            is_safe, final_reward = evaluate_policy_safety_on_state(
                safety_eval_env, policy, idx, file_args["max_steps"], torch.device("cpu")
            )
            results.append((is_safe, final_reward))
    num_unsafe_trajectories = sum([1 for is_safe, _ in results if not is_safe])
    safety_rate = 1.0 - (num_unsafe_trajectories / num_init_states)
    total_reward = sum([reward for _, reward in results])
    avg_reward = total_reward / num_init_states

    return safety_rate, avg_reward


def train_step(
        rb: DAggerBuffer, 
        policy: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        batch_size: int, 
        device: torch.device = torch.device("cpu")) -> float:
    """Perform a single training step for the policy network."""
    policy.train()
    policy.to(device)
    # Sample a batch of data from the replay buffer
    batch = rb.sample(batch_size)
    observations = batch["observation"].to(device)  # Shape: (batch_size, obs_dim)
    actions = batch["action"].to(device)  # Shape: (batch_size,)
    action_masks = batch["action_mask"].to(device)  # Shape: (batch_size, n_actions)

    # Forward pass through the policy network to get predicted actions
    logits = policy(observations)  # Shape: (batch_size, n_actions)
    masked_logits = logits.masked_fill(~action_masks.bool(), float('-inf'))  # Mask invalid actions
    logp = torch.log_softmax(masked_logits, dim=-1)

    # Compute loss between predicted actions and expert actions
    loss = torch.nn.NLLLoss()(logp, actions.long())

    # Backpropagation and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def load_policy(checkpoint) -> nn.Module:
    """Load a policy network from a checkpoint."""
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    hidden_dims = checkpoint['hidden_dims']
    policy = Policy(input_dim, output_dim, hidden_dims)
    # Mapping sb3 state dict to our Policy state dict
    mapped = {
        "model.0.weight": checkpoint['state_dict']["mlp_extractor.policy_net.0.weight"],
        "model.0.bias":   checkpoint['state_dict']["mlp_extractor.policy_net.0.bias"],
        "model.2.weight": checkpoint['state_dict']["mlp_extractor.policy_net.2.weight"],
        "model.2.bias":   checkpoint['state_dict']["mlp_extractor.policy_net.2.bias"],
        "model.4.weight": checkpoint['state_dict']["action_net.weight"],
        "model.4.bias":   checkpoint['state_dict']["action_net.bias"],
    }
    policy.load_state_dict(mapped, strict=True)
    return policy



def train(args: dict, file_args: dict, hyperparams: dict, device: torch.device = torch.device("cpu")):
    """Main training loop for DAgger."""
    assert args["policy_path"] is not None, "Initial policy path must be provided for DAgger."
    policy_path = Path(args["policy_path"])
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    # Load initial policy
    print(f"Loading initial policy from {policy_path}")
    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
    policy = load_policy(checkpoint)
    print("Initial policy loaded.")

    # Set up logging directory
    log_dir = Path(args.get("log_directory", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    safety_log_file = log_dir / "safety_rates.txt"
    rewards_log_file = log_dir / "average_rewards.txt"
    starting_info_file = log_dir / "starting_info.txt"
    final_info_file = log_dir / "final_info.txt"
    open(safety_log_file, 'w').close() # Clear existing log file
    open(rewards_log_file, 'w').close() # Clear existing log file
    open(starting_info_file, 'w').close() # Clear existing log file
    open(final_info_file, 'w').close() # Clear existing log file

    # Initialize Weights & Biases logging if available
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        print("Warning: Weights & Biases not available. Advanced logging will be disabled.")
    if WANDB_AVAILABLE and (not args.get("disable_wandb", False)):
        wandb.init(
            project=args.get("wandb_project", "dagger"),
            entity=args.get("wandb_entity", ""),
            name=args.get("experiment_name", "dagger_experiment"),
            config={
                **args,
                **(hyperparams or {}),
                'file_args': file_args
            }
        ) 

    # Create environment for sequential rollout collection
    if args.get("use_strict_rule", False):
        print("Using strict rules for trajectory collection.")
        safety_eval_file_args = create_safety_eval_file_args(file_args, args, use_oracle=False)
    else:
        print("Using normal safety rule for trajectory collection.")
        safety_eval_file_args = create_safety_eval_file_args(file_args, args)
    safety_eval_env = create_env(safety_eval_file_args, n_envs=1, monitor=False, time_limited=True)

    safety_coverage_file_args = create_safety_eval_file_args(file_args, args, use_oracle=False)

    # Create environment for normal policy evaluation
    eval_file_args = create_eval_file_args(file_args)
    env = create_env(eval_file_args, n_envs=1, monitor=False, time_limited=True)

    # Initialize optimizer
    learning_rate = hyperparams.get("learning_rate", 1e-3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Initialize replay buffer
    rb_capacity = hyperparams.get("replay_buffer_capacity", 10000)
    rb = DAggerBuffer(buffer_size=rb_capacity)

    if not args.get("use_strict_rule", False):
        # Evaluate initial safety coverage of the policy (only requiired when not using strict rule)
        print("Evaluating starting safety coverage")
        starting_safety_coverage, starting_avg_reward = evaluate_policy_safety(
            args=args, 
            hyperparams=hyperparams, 
            file_args=safety_coverage_file_args, 
            network_paras={
                'input_dim': checkpoint['input_dim'],
                'output_dim': checkpoint['output_dim'],
                'hidden_dims': checkpoint['hidden_dims']
            }, 
            policy=policy)
        print(f"Initial Safety Coverage: {starting_safety_coverage*100.0:.2f}%, Average Reward: {starting_avg_reward:.2f}")
        # print(f"Average Reward before training: {evaluate_policy(env, policy, num_episodes=100, device=device):.2f}")
        with open(starting_info_file, 'w') as f:
            f.write(f"{starting_safety_coverage:.4f}\t{starting_avg_reward:.2f}")


    safety_rates = [] # To track safety rates over iterations
    avg_rewards = []  # To track average rewards over iterations

    # Main training loop
    num_iterations = hyperparams.get("num_iterations", 10000)
    if RAY_AVAILABLE and args.get("use_multiprocessors", False):
        rollout_manager = ray_workers.RolloutManager(
            file_args=safety_eval_file_args,
            network_paras={
                'input_dim': checkpoint['input_dim'],
                'output_dim': checkpoint['output_dim'],
                'hidden_dims': checkpoint['hidden_dims']
            },
            policy=policy,
            num_workers=hyperparams.get("num_workers", 200),
            use_strict_rule=args.get("use_strict_rule", False),
            max_horizon=hyperparams.get("max_horizon", 1024)
        )
    for iter in range(num_iterations):
        if args.get("empty_buffer", False):
            rb.empty() # Optionally empty the buffer at each iteration
        # Collect new trajectories and add to replay buffer
        
        rollouts = []
        init_state_size = safety_eval_env.unwrapped.get_init_state_pool_size()
        init_state_size = min(args.get("num_init_states", 10000), init_state_size)
        if args.get("use_multiprocessors", False) and RAY_AVAILABLE:
            print(f"Collecting trajectories using {hyperparams.get('num_workers', 200)} Ray workers...")
            # Use Ray workers to collect trajectories in parallel
            rollout_manager.update_policy(policy)
            rollouts = rollout_manager.run_rollouts(init_state_size)
        else:
            print("Collecting trajectories sequentially...")
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
                disable=DISABLE_BAR
            ) as progress:
                task = progress.add_task("Collecting Trajectories for Training", total=3000)
                # Collect trajectories sequentially
                for _ in range(3000):
                    if args.get("use_strict_rule", False):
                        rollout = collect_trajectory_with_stricted_rule(
                            env=env, 
                            policy=policy, 
                            idx=-1, 
                            max_horizon=hyperparams.get("max_horizon", 1024))
                    else:
                        rollout = collect_trajectory(
                            env=env, 
                            policy=policy, idx=idx,
                            max_horizon=hyperparams.get("max_horizon", 1024))
                    rollouts.append(rollout)
                    progress.advance(task, advance=1)

            eval_rollouts = []
            # Display progress bar for trajectory collection when not using multiprocessors
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
                disable=DISABLE_BAR
            ) as progress:
                task = progress.add_task("Collecting Trajectories for Evaluation", total=init_state_size)
                # Collect trajectories sequentially
                for idx in range(init_state_size):
                    if args.get("use_strict_rule", False):
                        eval_rollout = collect_trajectory_with_stricted_rule(
                            env=safety_eval_env, 
                            policy=policy, 
                            idx=idx, 
                            max_horizon=hyperparams.get("max_horizon", 1024))
                    else:
                        eval_rollout = collect_trajectory(
                            env=safety_eval_env, 
                            policy=policy, idx=idx,
                            max_horizon=hyperparams.get("max_horizon", 1024))
                    eval_rollouts.append(eval_rollout)
                    progress.advance(task, advance=1)

        # Check whether all trajectories are safe
        all_eval_safe = all([eval_rollout[1]["is_safe_trajectory"] for eval_rollout in eval_rollouts])
        all_train_safe = all([rollout[1]["is_safe_trajectory"] for rollout in rollouts])
        all_safe = all_eval_safe or all_train_safe
        percentage_safe = sum([eval_rollout[1]["is_safe_trajectory"] for eval_rollout in eval_rollouts]) / len(eval_rollouts) * 100.0
        avg_reward = sum([eval_rollout[1]["final_reward"] for eval_rollout in eval_rollouts]) / len(eval_rollouts)
        # Log safety statistics
        safety_rates.append(percentage_safe)
        avg_rewards.append(avg_reward)
        with open(safety_log_file, 'a') as f:
            f.write(f"{iter}\t{percentage_safe}\n")
        with open(rewards_log_file, 'a') as f:
            f.write(f"{iter}\t{avg_reward}\n")
        print(f"Before iteration {iter}: {percentage_safe:.2f}% of collected trajectories are safe with average reward {avg_reward:.2f}.")

        if WANDB_AVAILABLE and (not args.get("disable_wandb", False)) and wandb.run is not None:
            wandb.log({
                'safe_eval/percentage_safe_trajectories': percentage_safe,
                'safe_eval/average_reward': avg_reward,
                'safe_eval/iteration': iter
            })

        if all_safe:
            # Save final policy
            model_save_dir = Path(args.get("model_save_dir", "./models"))
            model_save_dir.mkdir(parents=True, exist_ok=True)
            save_path = model_save_dir / f"dagger_policy_final_iter_{iter}.pth"
            torch.save({
                'input_dim': checkpoint['input_dim'],
                'output_dim': checkpoint['output_dim'],
                'hidden_dims': checkpoint['hidden_dims'],
                'state_dict': policy.state_dict()
            }, save_path)
            print(f"Saved final policy to {save_path}")
            
            final_safety_coverage, final_avg_reward = evaluate_policy_safety(
                args=args, 
                hyperparams=hyperparams, 
                file_args=safety_coverage_file_args, 
                network_paras={
                    'input_dim': checkpoint['input_dim'],
                    'output_dim': checkpoint['output_dim'],
                    'hidden_dims': checkpoint['hidden_dims']
                }, 
                policy=policy
            )
            print(f"Final Safety Coverage: {final_safety_coverage*100.0:.2f}%, Average Reward: {final_avg_reward:.2f}")
            with open(final_info_file, 'w') as f:
                f.write(f"{final_safety_coverage:.4f}\t{final_avg_reward:.2f}")
            return
        # Process and add rollouts to replay buffer
        rb.add_rollouts(rollouts)

        # Perform training steps
        total_loss = 0.0
        for s in range(hyperparams.get("steps_per_iteration", 5)):
            batch_size = hyperparams.get("batch_size", 256)
            loss = train_step(rb, policy, optimizer, batch_size, device)
            total_loss += loss
            # print(f"Iteration {iter} step {s}: Loss = {loss:.4f}")
            if WANDB_AVAILABLE and (not args.get("disable_wandb", False)) and wandb.run is not None:
                wandb.log({
                    'train/loss': loss,
                    'train/iteration': iter,
                })
        avg_loss = total_loss / hyperparams.get("steps_per_iteration", 5)
        print(f"Iteration {iter} completed. Average Training Loss: {avg_loss:.4f}")

        # Save intermediate policy
        if (iter + 1) % args.get("save_interval", 10) == 0:
            model_save_dir = Path(args.get("model_save_dir", "./models"))
            model_save_dir.mkdir(parents=True, exist_ok=True)
            save_path = model_save_dir / f"dagger_policy_iter_{iter+1}.pth"
            torch.save({
                'input_dim': checkpoint['input_dim'],
                'output_dim': checkpoint['output_dim'],
                'hidden_dims': checkpoint['hidden_dims'],
                'state_dict': policy.state_dict()
            }, save_path)
            print(f"Saved intermediate policy at iteration {iter+1} to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Masked PPO on JANI Environments")
    parser.add_argument(
        '--jani_model', 
        type=str, required=True, help="Path to the JANI model file.")
    parser.add_argument(
        '--jani_property', 
        type=str, default="", help="Path to the JANI property file.")
    parser.add_argument(
        '--start_states', 
        type=str, required=True, help="Path to the start states file.")
    parser.add_argument(
        '--objective', 
        type=str, default="", help="Path to the objective file.")
    parser.add_argument(
        '--failure_property', 
        type=str, default="", help="Path to the failure property file.")
    parser.add_argument(
        '--eval_start_states', 
        type=str, default="", help="Path to the evaluation start states file.")
    parser.add_argument(
        '--policy_path', 
        type=str, required=True, help="Path to the initial policy file for DAgger.")
    parser.add_argument(
        '--goal_reward', 
        type=float, default=1.0, help="Reward for reaching the goal.")
    parser.add_argument(
        '--failure_reward', 
        type=float, default=-1.0, help="Reward for reaching failure state.")
    parser.add_argument(
        '--disable_oracle_cache', 
        action='store_true', help="Disable caching in the oracle.")
    parser.add_argument(
        '--reduced_memory_mode', 
        action='store_true', help="Enable reduced memory mode in the oracle (clears cache after each query).")
    parser.add_argument(
        '--unsafe_reward', 
        type=float, default=-0.01, help="Reward for unsafe states when using oracle.")
    parser.add_argument(
        '--num_init_states', 
        type=int, default=10000, help="Number of initial states to sample from.")
    parser.add_argument(
        '--num_iterations', 
        type=int, default=200, help="Number of DAgger iterations to perform.")
    parser.add_argument(
        '--steps_per_iteration', 
        type=int, default=5, help="Number of training steps per DAgger iteration.")
    parser.add_argument(
        '--use_strict_rule', 
        action='store_true', help="Use strict rules for trajectory collection.")
    parser.add_argument(
        '--use_multiprocessors', 
        action='store_true', help="Use multiprocessors for rollout collection.")
    parser.add_argument(
        '--num_workers', 
        type=int, default=8, help="Number of workers for multiprocessor rollout collection.")
    parser.add_argument(
        '--empty_buffer', 
        action='store_true', help="Empty the replay buffer at each iteration.")
    parser.add_argument(
        '--device', 
        type=str, default="cpu", help="Device to use for training (cpu or cuda).")
    parser.add_argument(
        '--seed', 
        type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        '--max_steps', 
        type=int, default=1000, help="Max steps per episode.")
    parser.add_argument(
        '--wandb_project', 
        type=str, default="dagger", help="Weights & Biases project name.")
    parser.add_argument(
        '--wandb_entity', 
        type=str, default=None, help="Weights & Biases entity name.")
    parser.add_argument(
        '--experiment_name', 
        type=str, default="", help="Name of the experiment.")
    parser.add_argument(
        '--log_directory', 
        type=str, default="./logs", help="Directory to save logs and checkpoints.")
    parser.add_argument(
        '--model_save_dir', 
        type=str, default="./models", help="Directory to save trained models.")
    parser.add_argument(
        '--disable_wandb', 
        action='store_true', help="Disable Weights & Biases logging.")
    args = parser.parse_args()

    file_args = {
        'jani_model': args.jani_model,
        'jani_property': args.jani_property,
        'start_states': args.start_states,
        'objective': args.objective,
        'failure_property': args.failure_property,
        'goal_reward': args.goal_reward,
        'failure_reward': args.failure_reward,
        'unsafe_reward': args.unsafe_reward,
        'disable_oracle_cache': args.disable_oracle_cache,
        'reduced_memory_mode': args.reduced_memory_mode,
        'seed': args.seed,
        'use_oracle': True, # Always use oracle during DAgger training
        'max_steps': args.max_steps
    }

    import os
    hyperparams = {
        'learning_rate': 1e-3,
        'replay_buffer_capacity': 10000,
        'num_iterations': args.num_iterations,
        'batch_size': 256,
        'num_workers': args.num_workers, # Ensure not to exceed available CPUs
        'steps_per_iteration': args.steps_per_iteration,
        'max_horizon': args.max_steps
    }

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_dir = Path(args.log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    train(vars(args), file_args, hyperparams, device)


if __name__ == "__main__":
    main()