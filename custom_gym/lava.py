import numpy as np

from minigrid.core.grid import Grid
from minigrid.envs.lavagap import LavaGapEnv
from minigrid.core.world_object import Goal, Lava

from dataclasses import dataclass


@dataclass
class StateConfig:
    grid: Grid
    agent_pos: np.ndarray
    agent_dir: int
    goal_pos: np.ndarray
    gap_pos: np.ndarray
    mission: str


class CustomLavaGapEnv(LavaGapEnv):
    def __init__(
        self,
        size,
        obstacle_type=Lava,
        max_steps: int | None = None,
        num_init: int | None = None,
        **kwargs
    ):
        # Initialize the environment first to set up observation_space, action_space, etc.
        super().__init__(
            size=size,
            obstacle_type=obstacle_type,
            max_steps=max_steps,
            **kwargs
        )

        self.init_states: list[StateConfig] = []

        # Sample and store initial states after initialization
        if num_init is not None:
            for _ in range(num_init):
                self._gen_grid(size, size)
                state_config = StateConfig(
                    grid=self.grid,
                    agent_pos=self.agent_pos.copy(),
                    agent_dir=self.agent_dir,
                    goal_pos=self.goal_pos.copy(),
                    gap_pos=self.gap_pos.copy(),
                    mission=self.mission
                )
                self.init_states.append(state_config)

    def reset(
        self, 
        *, 
        seed: int | None = None, 
        options: dict | None = None
    ):
        if options is not None and "idx" in options:
            idx = options["idx"]
            assert 0 <= idx < len(self.init_states), "Invalid index for initial states"
            state_config = self.init_states[idx]

            # Set the environment fields according to the selected initial state
            self.grid = state_config.grid
            self.agent_pos = state_config.agent_pos
            self.agent_dir = state_config.agent_dir
            self.goal_pos = state_config.goal_pos
            self.gap_pos = state_config.gap_pos
            self.mission = state_config.mission

            # Copy from the original implementation of reset()
            start_cell = self.grid.get(*self.agent_pos)
            assert start_cell is None or start_cell.can_overlap()

            # Item picked up, being carried, initially nothing
            self.carrying = None

            # Step count since episode start
            self.step_count = 0

            if self.render_mode == "human":
                self.render()

            # Return first observation
            obs = self.gen_obs()
        else:
            obs, _ = super().reset(seed=seed, options=options)

        return obs, {}
    
    def action_mask(self):
        # Mark all actions as applicable
        mask = np.ones(self.action_space.n, dtype=np.float32)
        return mask