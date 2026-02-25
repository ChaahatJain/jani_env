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
        # print(f"DEBUG: Initializing JANIEnv with model: {jani_model_path}, property: {jani_property_path}, start states: {start_states_path}, objective: {objective_path}, failure property: {failure_property_path}, seed: {seed}")
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
        if self._use_oracle and (not self._use_strict_rules):
            # If we use strict safety rules, we do not need to provide safety info at reset
            if (options is None) or ("no_safety_info" not in options):
                is_current_state_safe, current_safe_action = self._oracle.engine_state_safety_with_action()
                reset_info["current_state_safety"] = is_current_state_safe
                reset_info["current_safe_action"] = current_safe_action

                self._prev_state_safe = is_current_state_safe
                self._prev_safe_action = current_safe_action
                self._prev_obs = state_vec
        return np.array(state_vec, dtype=np.float32), reset_info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self._reseted:
            raise RuntimeError("Environment must be reset before stepping.")
        
        if self._use_oracle:
            assert self._prev_obs == self._engine.get_current_state_vector(), "Current observation does not match engine state. Expected {}, got {}".format(self._prev_obs, self._engine.get_current_state_vector())
            assert self._engine.get_current_state_vector() == self._oracle.get_engine_current_state_vector(), "Engine state does not match oracle state. Engine {}, Oracle {}".format(self._engine.get_current_state_vector(), self._oracle.get_engine_current_state_vector())

        if self._use_strict_rules:
            # If strict safety rule is used, check whether the current state-action pair is safe
            is_action_safe = self._engine.is_engine_state_action_safe(action)

        next_state_vec = self._engine.step(action) # The current state should be automatically updated in the engine

        info = {}
        if self._use_oracle and (not self._use_strict_rules):
            assert self._unsafe_reward is not None
            assert self._engine.get_current_state_vector() == self._oracle.get_engine_current_state_vector() == next_state_vec, "After step, engine state does not match oracle state. Engine {}, Oracle {}".format(self._engine.get_current_state_vector(), self._oracle.get_engine_current_state_vector())
            is_next_state_safe, next_safe_action = self._oracle.engine_state_safety_with_action()
            info["next_state_safety"] = is_next_state_safe
            info["next_safe_action"] = next_safe_action
        
        # Compute reward and done flag
        reward = None
        done = None
        if self._engine.reach_goal_current():
            reward = self._goal_reward
            done = True
        elif self._engine.reach_failure_current():
            reward = self._failure_reward
            done = True
        elif np.sum(self.action_mask()) == 0.0:
            reward = 0.0
            done = True
        else:
            if self._use_oracle and (not self._use_strict_rules) and (not is_next_state_safe):
                reward = self._unsafe_reward
                if self._prev_state_safe:
                    # Just transitioned from safe to unsafe
                    assert action != self._prev_safe_action, f"From last obs: {self._prev_obs} to current obs: {next_state_vec}: Expect taken action {action} differ from previous safe action {self._prev_safe_action} when transitioning to unsafe state."
            elif self._use_oracle and self._use_strict_rules and (not is_action_safe):
                reward = self._unsafe_reward
            else:
                reward = 0.0
            done = False

        if self._use_oracle and (not self._use_strict_rules):
            self._prev_state_safe = is_next_state_safe
            self._prev_safe_action = next_safe_action
            self._prev_obs = next_state_vec

        return np.array(next_state_vec, dtype=np.float32), reward, done, False, info

    def action_mask(self) -> np.ndarray:
        if not self._reseted:
            raise RuntimeError("Environment must be reset before getting action mask.")
        mask = self._engine.get_current_action_mask()
        return np.array(mask, dtype=np.float32)

    def action_mask_for_obs(self, obs: np.ndarray):
        # print(f"DEBUG: Getting action mask for obs: {obs}")
        # print(f"DEBUG: Obs shape: {obs.shape}, Obs dtype: {obs.dtype}")
        return [self._engine.get_action_mask_for_obs(single_obs.tolist()) for single_obs in obs]
    
    def get_init_state_pool_size(self) -> int:
        return self._engine.get_init_state_pool_size()
    
    def get_unsafe_reward(self) -> Optional[float]:
        assert self._unsafe_reward is not None, "Unsafe reward is not defined."
        return self._unsafe_reward

    def get_failure_reward(self) -> float:
        return self._failure_reward

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