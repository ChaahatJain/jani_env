import numpy as np

from .env import JANIEnv


class JANIEnumEnv(JANIEnv):
    def __init__(self, 
                 jani_model_path: str, 
                 jani_property_path: str = "",
                 start_states_path: str = "",
                 objective_path: str = "",
                 failure_property_path: str = "",
                 seed: int = 42,
                 goal_reward: float = 1.0,
                 failure_reward: float = -1.0,
                 use_oracle: bool = False,
                 use_strict_rules: bool = False, 
                 unsafe_reward: float = -0.01,
                 disable_oracle_cache: bool = False,
                 reduced_memory_mode: bool = False) -> None:
        super().__init__(jani_model_path, 
                         jani_property_path, 
                         start_states_path, 
                         objective_path, 
                         failure_property_path, 
                         seed, 
                         goal_reward, 
                         failure_reward, 
                         use_oracle, 
                         use_strict_rules, 
                         unsafe_reward, 
                         disable_oracle_cache, 
                         reduced_memory_mode)
        
        self._init_idx = 0 # Track the current index for enumeration of initial states

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        if self._init_idx >= self._engine.get_num_init_states():
            self._init_idx = 0 # Reset index if it exceeds the number of initial states

        # Set the initial state to the specified one based on the current index    
        state_vec = self._engine.reset_with_index(self._init_idx)
        
        self._init_idx += 1 # Increment index for the next reset
        self._reseted = True

        return np.array(state_vec, dtype=np.float32), {}