from typing import Any, Dict, List
from .interfaces import FaultCollectorInterface, OracleInterface
import numpy as np


class OracleFaultCollector(FaultCollectorInterface):
    """
    Given a trace with recorded oracle responses, identifies faults.

    The fault determination LOGIC is here:
    - A fault occurs when the oracle indicated the state was safe (had safe options)
    - But the policy took a different action than the oracle's safe action

    The raw oracle data (is_state_safe, safe_action) was recorded during sampling.
    This collector only applies the fault determination logic.
    """
    def collect_faults(self, trace: Dict[str, Any], oracle: OracleInterface = None) -> List[Dict[str, Any]]:
        """
        Collect faults from a trace using recorded oracle data.

        Args:
            trace: Must contain 'oracle_is_state_safe' and 'oracle_safe_action' lists
            oracle: Not used - oracle was already queried during sampling

        Returns:
            List of fault dictionaries
        """
        faults = []
        observations = trace["observations"]
        actions = trace["actions"]
        action_masks = trace["action_masks"]

        # Get recorded oracle data from trace
        oracle_is_state_safe = trace.get("oracle_is_state_safe", [])
        oracle_safe_action = trace.get("oracle_safe_action", [])

        if not oracle_is_state_safe or not oracle_safe_action:
            # No oracle data recorded - can't detect faults
            return faults

        for step in range(len(observations)):
            obs = observations[step]
            action = actions[step]
            mask = action_masks[step]

            # Convert mask to numpy array if needed
            if isinstance(mask, (list, tuple)):
                mask = np.array(mask)

            # Get oracle data for this step
            is_state_safe = oracle_is_state_safe[step] if step < len(oracle_is_state_safe) else True
            safe_action = oracle_safe_action[step] if step < len(oracle_safe_action) else -1

            # FAULT LOGIC: A fault occurs when:
            # 1. The state was safe (oracle said there were safe options)
            # 2. There was a specific safe action available
            # 3. But the policy took a DIFFERENT action
            is_fault = (
                is_state_safe and
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
                })

        return faults