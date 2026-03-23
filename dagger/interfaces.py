from abc import ABC, abstractmethod
from typing import Any, Dict, List

class PolicyInterface(ABC):
    @abstractmethod
    def get_action(self, state: Any, action_mask: Any = None) -> Any:
        """
        Returns an action given a state and an optional action mask.
        This interface is deliberately decoupled from PyTorch, allowing 
        the use of different representations such as STL shields or heuristics.
        """
        pass

class TraceSamplerInterface(ABC):
    @abstractmethod
    def sample_trace(self, env: Any, policy: PolicyInterface, init_state_idx: int = -1, max_steps: int = 1024) -> Dict[str, Any]:
        """Samples a trace from the environment using the given policy, returning the trace and safety metrics."""
        pass

class FaultCollectorInterface(ABC):
    @abstractmethod
    def collect_faults(self, trace: Dict[str, Any], oracle: Any) -> List[Dict[str, Any]]:
        """
        Iterates through a trace and runs the oracle on each (state, action) pair.
        Returns a list of faults (unsafe states and their corrected actions).
        """
        pass

class PolicyUpdaterInterface(ABC):
    @abstractmethod
    def update_policy(self, policy: Any, dataset: Any) -> Dict[str, float]:
        """Updates the policy using a supervised dataset of corrected faults."""
        pass