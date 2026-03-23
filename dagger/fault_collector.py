from typing import Any, Dict, List
from .interfaces import FaultCollectorInterface

class OracleFaultCollector(FaultCollectorInterface):
    """
    Given a trace, runs the oracle on each (state, action) pair.
    If an action is deemed "unsafe", it collects it as a fault along with a corrected action.
    """
    def collect_faults(self, trace: Dict[str, Any], oracle: Any) -> List[Dict[str, Any]]:
        faults = []
        observations = trace["observations"]
        actions = trace["actions"]
        action_masks = trace["action_masks"]
        
        for obs, action, mask in zip(observations, actions, action_masks):
            # Oracle acts as the expert here, providing safety status and fallback/corrected action
            is_safe, corrected_action = oracle.evaluate_and_correct(obs, action, mask)
            
            if not is_safe:
                # Save fault to append to the supervised learning dataset
                faults.append({
                    "observation": obs,
                    "action_mask": mask,
                    "action": corrected_action,  # Corrected action to train on
                    "faulty_action": action
                })
                
        return faults