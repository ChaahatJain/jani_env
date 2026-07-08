from typing import TYPE_CHECKING, Any, Dict, List
from .interfaces import FaultCollectorInterface

if TYPE_CHECKING:
    from jani.env import JANIEnv

class OracleFaultCollector(FaultCollectorInterface):
    """
    Given an unsafe trace identify the faults on the trace.

    A fault is a (state, action) pair whose source state is safe and whose
    action transitions to the unsafe region according to the oracle. It is not
    merely any action that differs from one safe action returned by the oracle.
    """
    def collect_faults(self, trace: Dict[str, Any], env: "JANIEnv") -> List[Dict[str, Any]]:
        """
        Collect faults from a trace using oracle state-action fault labels.

        Args:
            trace: A rollout containing observations, actions, and action masks.
            env: Environment exposing is_state_action_fault.

        Returns:
            List of fault dictionaries
        """
        faults = []
        observations = trace["observations"]
        actions = trace["actions"]
        action_masks = trace["action_masks"]

        base_env = getattr(env, "unwrapped", env)
        uses_blocksworld_shortcut = getattr(
            base_env, "uses_blocksworld_safety_shortcut", lambda: False
        )()
        if uses_blocksworld_shortcut:
            # In Blocksworld every non-failure state is safe, so an unsafe
            # sampled path has exactly one fault: its transition into failure.
            # The sampler records the pre-transition state/action at the final
            # index, which lets us avoid all oracle queries.
            if trace.get("is_safe_trajectory", True) or not actions:
                return []
            step = len(actions) - 1
            return [{
                "step": step,
                "observation": observations[step],
                "action_mask": action_masks[step],
                "faulty_action": actions[step],
                "action": -1,
            }]

        for step in range(len(observations) - 1, -1, -1): # Traverse policy paths backwards.
            obs = observations[step]
            action = actions[step]
            mask = action_masks[step]

            if hasattr(env, 'is_state_action_fault'):
                is_fault = env.is_state_action_fault(obs, action)
            else:
                assert False, "Wrong environment initialization. The environment does not have any method of identifiyng faults."
            
            if is_fault:
                faults.append({
                    "step": step,
                    "observation": obs,
                    "action_mask": mask,
                    "faulty_action": action,  # The action that was taken
                    "action" : -1,
                })
                break; # Break after finding the first fault in a trace

        return faults
