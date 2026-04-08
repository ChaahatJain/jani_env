from typing import Any, Dict, List
from .interfaces import FaultCollectorInterface, OracleInterface
import numpy as np
from jani.env import JANIEnv

class OracleFaultCollector(FaultCollectorInterface):
    """
    Given an unsafe trace identify the faults on the trace.

    The fault determination LOGIC is here:
    - A fault occurs when the oracle indicated the state was safe (had a safe policy)
    - But the policy took a different action leading to a state which does not have a safe policy

    The raw oracle data (is_state_safe, safe_action) was recorded during sampling.
    This collector only applies the fault determination logic.
    """
    def collect_faults(self, trace: Dict[str, Any], env: JANIEnv) -> List[Dict[str, Any]]:
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

        for step in range(len(observations) - 1, -1, -1): # Traverse policy paths backwards.
            obs = observations[step]
            action = actions[step]
            mask = action_masks[step]

            if hasattr(env, 'is_state_action_fault'):
                is_fault = env.is_state_action_fault(obs, action) # Fault if state is safe but we're not taking the safe action 
            else:
                assert False, "Wrong environment initialization. The environment does not have any method of identifiyng faults."
            
            if is_fault:
                faults.append({
                    "step": step,
                    "observation": obs,
                    "action_mask": mask,
                    "faulty_action": action,  # The action that was taken
                    "action" : -1, # No safe action at the moment. TODO: @Songtuan, anyway to change this?
                })
                break; # Break after finding the first fault in a trace

        return faults