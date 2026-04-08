import sys
import numpy as np
import gymnasium as gym

from pathlib import Path
from typing import Optional

# Dynamically add the JANI engine binding directory to sys.path
current_dir = Path(__file__).resolve().parent
binding_dir = current_dir / "engine" / "build"
sys.path.append(str(binding_dir))

from backend import JANIEngine, TarjanOracle


class JANIEnv(gym.Env):
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
                 use_strict_rules: bool = False, # Whether to use strict safety rules (i.e., consider actions's safety but not next state safety when determining unsafe actions)
                 unsafe_reward: float = -0.01,
                 disable_oracle_cache: bool = False,
                 reduced_memory_mode: bool = False) -> None:
        super().__init__()
        print(f"DEBUG: Initializing JANIEnv with model: {jani_model_path}, property: {jani_property_path}, start states: {start_states_path}, objective: {objective_path}, failure property: {failure_property_path}, seed: {seed}")
        self._engine = JANIEngine(jani_model_path, 
                                  jani_property_path, 
                                  start_states_path, 
                                  objective_path, 
                                  failure_property_path, 
                                  seed)
        self._goal_reward: float = goal_reward
        self._failure_reward: float = failure_reward
        self._oracle: Optional[TarjanOracle] = None
        self._use_oracle: bool = use_oracle
        print(f"DEBUG: Setting up oracle with disable_cache={disable_oracle_cache}, reduced_memory_mode={reduced_memory_mode}")
        self._oracle = TarjanOracle(self._engine, disable_oracle_cache, reduced_memory_mode) # Always setup the oracle
        self._use_strict_rules: bool = use_strict_rules
        if self._use_strict_rules:
            assert self._use_oracle, "Strict rules require oracle to be enabled."
        self._unsafe_reward: Optional[float] = None
        if self._use_oracle:
            self._unsafe_reward = unsafe_reward
            assert self._oracle is not None, "Oracle must be set up if use_oracle is True."
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(self._engine.get_num_actions())
        lower_bounds = self._engine.get_lower_bounds()
        upper_bounds = self._engine.get_upper_bounds()
        self.observation_space = gym.spaces.Box(low=np.array(lower_bounds), 
                                                high=np.array(upper_bounds), 
                                                dtype=np.float32)
        # Initialize reset flag
        self._reseted = False

        # For debugging
        self._prev_state_safe = False
        self._prev_safe_action = -1
        self._prev_obs = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if options is not None and "idx" in options:
            state_vec = self._engine.reset_with_index(options["idx"])
        else:
            state_vec = self._engine.reset()
        self._reseted = True
        assert not self._engine.reach_goal_current(), "Initial state should not be a goal state."
        reset_info = {}
        self._prev_obs = state_vec
        return np.array(state_vec, dtype=np.float32), reset_info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self._reseted:
            raise RuntimeError("Environment must be reset before stepping.")
        
        if self._use_oracle:
            assert self._prev_obs == self._engine.get_current_state_vector(), "Current observation does not match engine state. Expected {}, got {}".format(self._prev_obs, self._engine.get_current_state_vector())
            assert self._engine.get_current_state_vector() == self._oracle.get_engine_current_state_vector(), "Engine state does not match oracle state. Engine {}, Oracle {}".format(self._engine.get_current_state_vector(), self._oracle.get_engine_current_state_vector())

        next_state_vec = self._engine.step(action) # The current state should be automatically updated in the engine

        info = {}
        
        # Compute reward and done flag
        reward = None
        done = None
        if self._engine.reach_goal_current():
            reward = self._goal_reward
            done = True
            info["reached_goal"] = True
        elif self._engine.reach_failure_current():
            reward = self._failure_reward
            done = True
            info["reached_fail"] = True
        elif np.sum(self.action_mask()) == 0.0:
            reward = 0.0
            done = True
        else:
            reward = 0.0
            done = False
        self._prev_obs = next_state_vec
        return np.array(next_state_vec, dtype=np.float32), reward, done, False, info
    
    def obs_reach_goal(self, obs: np.ndarray) -> bool:
        return self._engine.reach_goal_state_vector(obs.tolist())
    
    def obs_reach_failure(self, obs: np.ndarray) -> bool:
        return self._engine.reach_failure_state_vector(obs.tolist())
    
    def debug_show_state(self, obs: np.ndarray) -> str:
        return self._engine.debug_show_state(obs.tolist())

    def action_mask(self) -> np.ndarray:
        if not self._reseted:
            raise RuntimeError("Environment must be reset before getting action mask.")
        mask = self._engine.get_current_action_mask()
        return np.array(mask, dtype=np.float32)

    def action_mask_for_obs(self, obs: np.ndarray):
        # print(f"DEBUG: Getting action mask for obs: {obs}")
        # print(f"DEBUG: Obs shape: {obs.shape}, Obs dtype: {obs.dtype}")
        return np.array(self._engine.get_action_mask_for_obs(obs.tolist()), dtype=np.float32)
    
    def get_init_state_pool_size(self) -> int:
        return self._engine.get_init_state_pool_size()
    
    def get_unsafe_reward(self) -> Optional[float]:
        assert self._unsafe_reward is not None, "Unsafe reward is not defined."
        return self._unsafe_reward

    def get_failure_reward(self) -> float:
        return self._failure_reward
    
    def get_successor_obs(self, obs: np.ndarray, action: int) -> list[np.ndarray]:
        successor_obs = self._engine.get_all_successor_states_as_vectors(obs.tolist(), action)
        return [np.array(succ_obs, dtype=np.float32) for succ_obs in successor_obs]

    def is_current_state_action_safe(self, action: int) -> bool:
        if self._oracle is None:
            raise RuntimeError("Oracle is not enabled in this environment.")
        is_safe = self._oracle.is_engine_state_action_safe(action)
        return is_safe
    
    def current_state_safety_with_action(self, action: int) -> tuple[bool, int]:
        if self._oracle is None:
            raise RuntimeError("Oracle is not enabled in this environment.")
        safety_result = self._oracle.engine_state_safety_with_action(action)
        return safety_result
    
    def is_state_action_fault(self, obs: np.ndarray, action: int) -> bool:
        """
        Check if a (state, action) pair is a fault (leads to unsafe successor).
        Returns True if the action from this state leads to an unsafe state.
        """
        if self._oracle is None:
            raise RuntimeError("Oracle is not enabled in this environment.")
        # Convert observation to list for the oracle call
        state_vector = obs.tolist() if isinstance(obs, np.ndarray) else obs
        is_fault = self._oracle.state_action_is_fault(state_vector, action)
        return is_fault