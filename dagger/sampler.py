from typing import Any, Dict
from .interfaces import TraceSamplerInterface, PolicyInterface

class StandardTraceSampler(TraceSamplerInterface):
    """
    Samples a trace given a policy and an environment state.
    Tracks trajectory details and determines whether the entire trace is safe or unsafe.
    Also captures per-step safety information for fault detection.
    """
    def sample_trace(self, env: Any, policy: PolicyInterface, init_state_idx: int = -1, max_steps: int = 1024) -> Dict[str, Any]:
        # Initialize environment state
        if init_state_idx != -1:
            obs, info = env.reset(options={"init_state_idx": init_state_idx})
        else:
            obs, info = env.reset()

        done = False
        step_count = 0

        observations = []
        actions = []
        action_masks = []
        rewards = []

        # Per-step safety tracking
        state_safety = []  # Was this state safe before taking the action?
        safe_actions = []  # What was the safe action at this state? (-1 if none)
        next_state_safety = []  # Is the resulting state safe?

        is_safe_trajectory = True

        while not done and step_count < max_steps:
            action_mask = env.unwrapped.action_mask()

            # Fetch action dynamically (supports NN, Shields, etc.)
            action = policy.get_action(obs, action_mask)

            observations.append(obs)
            actions.append(action)
            action_masks.append(action_mask)
            
            next_obs, reward, done, truncated, info = env.step(action)
            print("Next observation:", next_obs, "Done", done, "reward", reward)

            # Determine if the current step rendered the trace unsafe
            if done and reward == env._failure_reward:
                print("This happens!")
                is_safe_trajectory = False

            rewards.append(reward)
            obs = next_obs
            step_count += 1

        return {
            "observations": observations,
            "actions": actions,
            "action_masks": action_masks,
            "rewards": rewards,
            "state_safety": state_safety,
            "safe_actions": safe_actions,
            "next_state_safety": next_state_safety,
            "is_safe_trajectory": is_safe_trajectory,
            "final_reward": rewards[-1] if rewards else 0.0
        }