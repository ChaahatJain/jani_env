import torch
import torch.nn as nn

from torchrl.modules import MaskedCategorical

from jani.env import JANIEnv



class Policy(nn.Module):
    """A simple feedforward policy network."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [64, 64], activation_fn: type = nn.ReLU):
        super(Policy, self).__init__()
        self.activation_fn = activation_fn
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation_fn())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def evaluate_policy_safety_on_state(env: JANIEnv, policy: nn.Module, state_idx: int, max_steps: int, device: torch.device) -> tuple[bool, float]:
    """Evaluate the safety of a policy starting from a specific state index."""
    policy.eval()
    policy.to(device)
    obs, _ = env.reset(options={"idx": state_idx})
    done = False
    step_count = 0
    episode_reward = 0.0
    keep_using_oracle = True
    is_safe = True
    while not done and step_count < max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
        action_mask = env.unwrapped.action_mask().astype(int)
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            logits = policy(obs_tensor)
            action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
            # action = action_dist.probs.argmax(dim=-1).squeeze(0).item()  # Sample action and remove batch dimension
            action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension
        if keep_using_oracle:
            is_state_action_safe = env.unwrapped.is_current_state_action_safe(action)
            if not is_state_action_safe:
                is_safe = False
                keep_using_oracle = False
        obs, reward, done, _, info = env.step(action)
        step_count += 1
    episode_reward = reward # Only using the final reward
    
    return is_safe, episode_reward


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate policy on JANI environment.")
    parser.add_argument("--jani_model", type=str, default="path_to_jani_model.jani", help="Path to the JANI model file.")
    parser.add_argument("--jani_property", type=str, default="", help="Path to the JANI property file.")
    parser.add_argument("--start_states", type=str, default="", help="Path to the start states file.")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to the saved policy file.")
    parser.add_argument("--max_steps", type=int, default=256, help="Maximum number of steps to evaluate the policy.")
    parser.add_argument("--num_episodes", type=int, default=300, help="Number of episodes to evaluate.")
    args = parser.parse_args()

    from utils import create_env

    file_args = {
        'jani_model': args.jani_model,
        'jani_property': args.jani_property,
        'start_states': args.start_states,
        'objective': "", # Not needed for evaluation
        'failure_property': "",
        'goal_reward': 1.0,
        'failure_reward': -1.0,
        'unsafe_reward': -0.01,
        'disable_oracle_cache': True,
        'seed': 1024,
        'use_oracle': False, # Always use oracle during DAgger training
        'max_steps': args.max_steps
    }
    env = create_env(file_args, n_envs=1, monitor=False, time_limited=True)

    
    checkpoint = torch.load(args.policy_path, map_location=torch.device('cpu'), weights_only=False)
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    hidden_dims = checkpoint['hidden_dims']
    policy = Policy(input_dim, output_dim, hidden_dims)
    policy.load_state_dict(checkpoint['state_dict'], strict=True)

    rewards = []

    for _ in range(args.num_episodes):
        num_steps = 0
        obs, _ = env.reset()
        done = False
        while not done and num_steps < args.max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            action_mask = env.unwrapped.action_mask().astype(int)
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                logits = policy(obs_tensor)
                action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
                action = action_dist.sample().squeeze(0).item()  # Sample action and remove batch dimension
            obs, reward, done, _, info = env.step(action)
            num_steps += 1
        rewards.append(reward)
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward over {args.num_episodes} episodes: {avg_reward}")