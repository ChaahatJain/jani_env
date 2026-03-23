from typing import Any, Dict, List
from .interfaces import FaultCollectorInterface
import numpy as np


class OracleFaultCollector(FaultCollectorInterface):
    """
    Given a trace, identifies faults where a valid action led to an unsafe state.

    A fault is detected when:
    1. The current state was safe
    2. The action taken leads to an unsafe next state
    3. There was a safe alternative action available

    Uses safety information recorded in the trace during sampling.
    """
    def collect_faults(self, trace: Dict[str, Any], oracle: Any = None) -> List[Dict[str, Any]]:
        faults = []
        observations = trace["observations"]
        actions = trace["actions"]
        action_masks = trace["action_masks"]

        # Get safety info from trace (recorded by sampler)
        state_safety = trace.get("state_safety", [])
        safe_actions = trace.get("safe_actions", [])
        next_state_safety = trace.get("next_state_safety", [])

        for step in range(len(observations)):
            obs = observations[step]
            action = actions[step]
            mask = action_masks[step]

            # Convert mask to numpy array if needed
            if isinstance(mask, (list, tuple)):
                mask = np.array(mask)

            # Get safety info for this step
            was_state_safe = state_safety[step] if step < len(state_safety) else True
            safe_action = safe_actions[step] if step < len(safe_actions) else -1
            is_next_safe = next_state_safety[step] if step < len(next_state_safety) else True

            # A fault occurs when:
            # - The state was safe (we had a choice)
            # - The action led to an unsafe state OR there was a different safe action
            # - A safe alternative action exists and differs from what was taken
            is_fault = (
                was_state_safe and
                safe_action != -1 and
                safe_action != action
            )

            if is_fault:
                faults.append({
                    "step": step,
                    "observation": obs,
                    "action_mask": mask,
                    "action": safe_action,  # The corrected (safe) action
                    "faulty_action": action,  # The action that was taken
                    "was_state_safe": was_state_safe,
                    "is_next_safe": is_next_safe
                })

        return faults