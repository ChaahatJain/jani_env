from typing import Any, Dict
from .interfaces import TraceSamplerInterface, PolicyInterface
import time

class StandardTraceSampler(TraceSamplerInterface):
    """
    Samples a trace given a policy and an environment state.

    Records observations, actions, masks, rewards, and raw oracle responses.
    The fault determination LOGIC is NOT here - that's in the FaultCollector.

    This sampler only records what the oracle says at each step (if available).
    """
    def sample_trace(self, env: Any, policy: PolicyInterface, init_state_idx: int = -1, max_steps: int = 1024, verbose = False) -> Dict[str, Any]:
        t = time.perf_counter()
        # Initialize environment state
        if init_state_idx != -1:
            obs, info = env.reset(options={"idx": init_state_idx})
            if verbose:
                print("Inside sample trace with obs:", obs)
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
            action_mask = env.action_mask()
            # Fetch action dynamically (supports NN, Shields, etc.)
            action = policy.get_action(obs, action_mask)
            
            observations.append(obs)
            actions.append(action)
            action_masks.append(action_mask)
            next_obs, reward, done, _, info = env.step(action)
            
            # Track overall trajectory safety from environment info
            if info.get("reached_fail", False):
                is_safe_trajectory = False
                
            is_goal_trajectory = info.get("reached_goal", False)

            rewards.append(reward)
            obs = next_obs
            step_count += 1
        if verbose:
            print(observations)
            print(actions)
        return {
            "observations": observations,
            "actions": actions,
            "action_masks": action_masks,
            "rewards": rewards,
            "is_safe_trajectory": is_safe_trajectory,
            "final_reward": rewards[-1] if rewards else 0.0,
            "is_goal_trajectory": is_goal_trajectory
        }