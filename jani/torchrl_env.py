import sys
import torch
import numpy as np

from typing import Optional
from pathlib import Path

from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Binary, Categorical, Composite
from torchrl.envs import EnvBase

# Dynamically add the JANI engine binding directory to sys.path
current_dir = Path(__file__).resolve().parent
binding_dir = current_dir / "engine" / "build"
sys.path.append(str(binding_dir))

from backend import JANIEngine, TarjanOracle


class JANIEnv(EnvBase):
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
                 disable_oracle_cache: bool = False,
                 reduced_memory_mode: bool = False,
                 unsafe_reward: float = -0.01) -> None:
        super().__init__()
        self._engine = JANIEngine(jani_model_path, 
                                  jani_property_path, 
                                  start_states_path, 
                                  objective_path, 
                                  failure_property_path, 
                                  seed)
        self._goal_reward: float = goal_reward
        self._failure_reward: float = failure_reward
        self._oracle: Optional[TarjanOracle] = TarjanOracle(
            self._engine, 
            disable_oracle_cache,
            reduced_memory_mode)
        self._use_oracle = use_oracle
        self._unsafe_reward: Optional[float] = None
        if self._use_oracle:
            self._unsafe_reward = unsafe_reward
        
        self.n_actions = self._engine.get_num_actions()
        self.obs_dim = self._engine.get_num_constants() + self._engine.get_num_variables()

        # Define action and observation space
        lower_bounds = self._engine.get_lower_bounds()
        upper_bounds = self._engine.get_upper_bounds()
        assert len(lower_bounds) == self.obs_dim, "Lower bounds dimension mismatch"
        assert len(upper_bounds) == self.obs_dim, "Upper bounds dimension mismatch"
        self.observation_spec = Composite({
            "observation": Bounded(
                low=np.array(lower_bounds), 
                high=np.array(upper_bounds),
                shape=(self.obs_dim,), 
                dtype=torch.float32
            ), 
            "observation_with_goal": Bounded(
                low=np.array(lower_bounds + [-1e9] * self._engine.get_goal_condition_size()), 
                high=np.array(upper_bounds + [1e9] * self._engine.get_goal_condition_size()),
                shape=(self.obs_dim + self._engine.get_goal_condition_size(),), 
                dtype=torch.float32
            ),
            "action_mask": Binary(
                n = self.n_actions,
                shape=(self.n_actions,),
                dtype = torch.bool
            ),
            "condition": Bounded(
                low=-1e9, # need to be change in the future
                high=1e9, # need to be change in the future
                shape=(self._engine.get_goal_condition_size(),),
                dtype=torch.float32
            )
        })

        # Action specification
        self.action_spec = Categorical(self.n_actions)

        # Reward specification
        self.reward_spec = Bounded(
            low=self._failure_reward, 
            high=self._goal_reward,
            shape=(1,), 
            dtype=torch.float32
        )

        # Done specification - TorchRL expects "done", "terminated" and "truncated" keys
        self.done_spec = Composite({
            "done": Binary(n=1, shape=(1,), dtype=torch.bool),
            "terminated": Binary(n=1, shape=(1,), dtype=torch.bool),
            "truncated": Binary(n=1, shape=(1,), dtype=torch.bool),
        })

        # Initialize reset flag
        self._reseted = False


    def action_mask(self) -> torch.Tensor:
        if not self._reseted:
            raise RuntimeError("Environment must be reset before getting action mask.")
        return torch.tensor(self._engine.get_current_action_mask(), dtype=torch.bool)

    def extract_reached_conditions(self, state_vec: torch.Tensor) -> torch.Tensor:
        state_vec_list = state_vec.tolist() # Convert tensor to list (assuming 1D tensor)
        condition_values = self._engine.extract_condition_from_state_vector(state_vec_list)
        return torch.tensor(condition_values, dtype=torch.float32)
    
    def extract_current_conditions(self) -> torch.Tensor:
        condition_values = self._engine.extract_condition_from_current_state_vector()
        return torch.tensor(condition_values, dtype=torch.float32)

    def _reset(self, td: TensorDictBase) -> TensorDictBase:
        if td is not None and "idx" in td:
            idx = td.get("idx").item()
            state_vec = self._engine.reset_with_index(idx)
        else:
            state_vec = self._engine.reset()
        self._reseted = True
        assert not self._engine.reach_goal_current(), "Initial state should not be a goal state."
        obs = {
            "observation": torch.tensor(state_vec, dtype=torch.float32),
            "observation_with_goal": torch.tensor(state_vec + self._engine.extract_goal_condition(), dtype=torch.float32),
            "action_mask": self.action_mask(),
            "condition": self.extract_current_conditions(),
            "done": torch.tensor([False], dtype=torch.bool),
            "terminated": torch.tensor([False], dtype=torch.bool),
            "truncated": torch.tensor([False], dtype=torch.bool),
        }
        return TensorDict(obs, batch_size=())
    
    def _step(self, td: TensorDictBase) -> TensorDictBase:
        if not self._reseted:
            raise RuntimeError("Environment must be reset before stepping.")
        action = td.get("action").item()
        next_state_vec = self._engine.step(action) # The current state should be automatically updated in the engine
        if self._use_oracle:
            assert self._unsafe_reward is not None
            is_next_state_safe = self._oracle.is_engine_state_safe()
        
        # Compute reward and done flag
        reward = None
        done = None
        if self._engine.reach_goal_current():
            reward = self._goal_reward
            done = True
        elif self._engine.reach_failure_current():
            reward = self._failure_reward
            done = True
        elif torch.sum(self.action_mask()) == 0:
            reward = 0.0
            done = True
        else:
            if self._use_oracle and not is_next_state_safe:
                reward = self._unsafe_reward
            else:
                reward = 0.0
            done = False

        # Construct the next tensordict
        if self._use_oracle is False:
            next_td = TensorDict({
                "observation": torch.tensor(next_state_vec, dtype=torch.float32),
                "observation_with_goal": torch.tensor(next_state_vec + self._engine.extract_goal_condition(), dtype=torch.float32),
                "action_mask": self.action_mask(),
                "done": torch.tensor([done], dtype=torch.bool),
                "terminated": torch.tensor([done], dtype=torch.bool),
                "truncated": torch.tensor([False], dtype=torch.bool),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "condition": self.extract_current_conditions()
            }, batch_size=())
        else:
            next_td = TensorDict({
                "observation": torch.tensor(next_state_vec, dtype=torch.float32),
                "observation_with_goal": torch.tensor(next_state_vec + self._engine.extract_goal_condition(), dtype=torch.float32),
                "action_mask": self.action_mask(),
                "done": torch.tensor([done], dtype=torch.bool),
                "terminated": torch.tensor([done], dtype=torch.bool),
                "truncated": torch.tensor([False], dtype=torch.bool),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "is_safe": torch.tensor(is_next_state_safe, dtype=torch.bool),
                "condition": self.extract_current_conditions()
            }, batch_size=())

        return next_td
    
    def get_init_state_pool_size(self) -> int:
        return self._engine.get_init_state_pool_size()

    def is_state_action_safe(self, action: int) -> bool:
        if self._oracle is None:
            raise RuntimeError("Oracle is not enabled for this environment.")
        return self._oracle.is_engine_state_action_safe(action)
    
    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            seed = torch.seed()
        rng = torch.manual_seed(seed)
        self.rng = rng