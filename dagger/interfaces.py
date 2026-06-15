from abc import ABC, abstractmethod
from typing import Any, Dict, List
from jani.env import JANIEnv
class PolicyInterface(ABC):
    @abstractmethod
    def get_action(self, state: Any, action_mask: Any = None) -> Any:
        """
        Returns an action given a state and an optional action mask.
        This interface is deliberately decoupled from PyTorch, allowing
        the use of different representations such as STL shields or heuristics.
        """
        pass

class OracleInterface(ABC):
    @abstractmethod
    def evaluate_action(self, obs: Any, action: Any, mask: Any = None) -> tuple:
        """
        Evaluates a (state, action) pair and determines if it's a fault.

        Args:
            obs: The observation/state
            action: The action taken
            mask: Optional action mask

        Returns:
            Tuple of (is_fault: bool, corrected_action: int or -1)
            - is_fault: True if the action is a fault (unsafe)
            - corrected_action: The safe action to take instead (-1 if no correction available)
        """
        pass

class TraceSamplerInterface(ABC):
    @abstractmethod
    def sample_trace(self, env: Any, policy: PolicyInterface, init_state_idx: int = -1, max_steps: int = 1024) -> Dict[str, Any]:
        """Samples a trace from the environment using the given policy, returning the trace and safety metrics."""
        pass

class FaultCollectorInterface(ABC):
    @abstractmethod
    def collect_faults(self, trace: Dict[str, Any], env: JANIEnv) -> List[Dict[str, Any]]:
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
