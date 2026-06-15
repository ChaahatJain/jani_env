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
    def sample_trace(
        self,
        env: Any,
        policy: PolicyInterface,
        init_state_idx: int = -1,
        max_steps: int = 1024,
        max_state_visits: int = 3,
        verbose = False,
    ) -> Dict[str, Any]:
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
        visited = {}

        is_safe_trajectory = True
        is_goal_trajectory = False
        termination_reason = "max_steps"
        final_observation = obs

        while not done and step_count < max_steps:
            action_mask = env.action_mask()
            # Fetch action dynamically (supports NN, Shields, etc.)
            action = policy.get_action(obs, action_mask)
            obs_key = obs.tobytes()
            visited[obs_key] = visited.get(obs_key, 0) + 1
            
            observations.append(obs)
            actions.append(action)
            action_masks.append(action_mask)
            next_obs, reward, done, _, info = env.step(action)
            step_count += 1
            final_observation = next_obs

            # Track overall trajectory safety from environment info
            if info.get("reached_fail", False):
                is_safe_trajectory = False
                termination_reason = "failure"

            is_goal_trajectory = info.get("reached_goal", False)
            if is_goal_trajectory:
                termination_reason = "goal"

            rewards.append(reward)
            next_obs_key = next_obs.tobytes()
            if not done and visited.get(next_obs_key, 0) >= max_state_visits:
                termination_reason = "cycle"
                break  # Early termination if we revisit a state (cycle detected)
            if done:
                if termination_reason == "max_steps":
                    termination_reason = "terminal"
                break
            obs = next_obs
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
            "is_goal_trajectory": is_goal_trajectory,
            "termination_reason": termination_reason,
            "final_observation": final_observation,
        }
