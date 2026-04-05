from typing import Any, Dict
from .interfaces import TraceSamplerInterface, PolicyInterface

class StandardTraceSampler(TraceSamplerInterface):
    """
    Samples a trace given a policy and an environment state.

    Records observations, actions, masks, rewards, and raw oracle responses.
    The fault determination LOGIC is NOT here - that's in the FaultCollector.

    This sampler only records what the oracle says at each step (if available).
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


        is_safe_trajectory = True

        while not done and step_count < max_steps:
            action_mask = env.unwrapped.action_mask()

            # Fetch action dynamically (supports NN, Shields, etc.)
            action = policy.get_action(obs, action_mask)

            observations.append(obs)
            actions.append(action)
            action_masks.append(action_mask)

            next_obs, reward, done, truncated, info = env.step(action)

            # Track overall trajectory safety from environment info
            if info.get("is_unsafe", False) or not info.get("next_state_safety", True):
                is_safe_trajectory = False

            rewards.append(reward)
            obs = next_obs
            step_count += 1

        return {
            "observations": observations,
            "actions": actions,
            "action_masks": action_masks,
            "rewards": rewards,
            "is_safe_trajectory": is_safe_trajectory,
            "final_reward": rewards[-1] if rewards else 0.0
        }