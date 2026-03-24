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
    def collect_faults(self, trace: Dict[str, Any], oracle: Any) -> List[Dict[str, Any]]:
        faults = []
        observations = trace["observations"]
        actions = trace["actions"]
        action_masks = trace["action_masks"]
        
        for obs, action, mask in zip(observations, actions, action_masks):
            # Check if action is a fault (leads to unsafe state)
            is_fault = oracle.is_state_action_fault(obs, action)
            print("Action", action, "is", ("a" if is_fault else "not a"), "fault for state", obs)
            
            
            
            if is_fault:
                # Find a safe action by checking all possible actions
                state_vector = obs.tolist() if isinstance(obs, np.ndarray) else obs
                safe_action = None
                for candidate_action in range(len(mask)):
                    if mask[candidate_action] > 0:  # Action is allowed by mask
                        if not oracle.is_state_action_fault(obs, candidate_action):
                            safe_action = candidate_action
                            break
                
                # If we found a safe action, record the fault
                if safe_action is not None:
                    faults.append({
                    "observation": obs,
                    "action_mask": mask,
                    "action": safe_action,
                    "faulty_action": action
                })
                    
                else: # TODO: This is bogus. Gotta remove it eventually.
                    faults.append({
                    "observation": obs,
                    "action_mask": mask,
                    "faulty_action": action
                })
        
        return faults