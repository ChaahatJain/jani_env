import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from faulty_states import (
    RecentFaultyStatePool,
    load_faulty_states,
    restart_states_from_trace,
    save_faulty_states,
)


def test_faulty_state_file_round_trip(tmp_path):
    path = tmp_path / "faulty_states.json"
    states = [[1.0, 2.0], [3.0, 4.0]]

    save_faulty_states(path, states, metadata={"cycle_traces": 2})

    np.testing.assert_allclose(load_faulty_states(path), states)


def test_recent_pool_is_bounded_and_keeps_recent_states():
    pool = RecentFaultyStatePool(max_size=2)
    pool.add([[1.0], [2.0]])
    pool.add([[1.0], [3.0]])

    assert pool.states() == [[1.0], [3.0]]


def test_restart_states_use_tail_and_include_cycle_entry():
    observations = [np.array([float(i)]) for i in range(4)]
    cycle_trace = {
        "observations": observations,
        "termination_reason": "cycle",
        "final_observation": np.array([1.0]),
    }
    failure_trace = {
        "observations": observations,
        "termination_reason": "failure",
        "final_observation": np.array([99.0]),
    }

    assert [state.tolist() for state in restart_states_from_trace(cycle_trace, 2)] == [[3.0], [1.0]]
    assert [state.tolist() for state in restart_states_from_trace(failure_trace, 2)] == [[2.0], [3.0]]


def _load_env_module_with_fake_backend(monkeypatch):
    backend = types.ModuleType("backend")

    class FakeEngine:
        def __init__(self, *args):
            self.current = [0.0, 0.0]

        def get_num_actions(self):
            return 1

        def get_lower_bounds(self):
            return [-10.0, -10.0]

        def get_upper_bounds(self):
            return [10.0, 10.0]

        def reset(self):
            self.current = [0.0, 0.0]
            return self.current

        def reset_with_index(self, index):
            self.current = [float(index), 0.0]
            return self.current

        def reset_from_state_vector(self, state):
            self.current = list(state)
            return self.current

        def get_current_state_vector(self):
            return self.current

    class FakeOracle:
        def __init__(self, *args):
            pass

    backend.JANIEngine = FakeEngine
    backend.TarjanOracle = FakeOracle
    monkeypatch.setitem(sys.modules, "backend", backend)

    env_path = Path(__file__).parents[1] / "jani" / "env.py"
    spec = importlib.util.spec_from_file_location("test_jani_env_module", env_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_sampler_module(monkeypatch):
    package = types.ModuleType("dagger")
    package.__path__ = [str(Path(__file__).parents[1] / "dagger")]
    interfaces = types.ModuleType("dagger.interfaces")
    interfaces.TraceSamplerInterface = object
    interfaces.PolicyInterface = object
    monkeypatch.setitem(sys.modules, "dagger", package)
    monkeypatch.setitem(sys.modules, "dagger.interfaces", interfaces)

    sampler_path = Path(__file__).parents[1] / "dagger" / "sampler.py"
    spec = importlib.util.spec_from_file_location("dagger.sampler", sampler_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trace_sampler_marks_cycles(monkeypatch):
    module = _load_sampler_module(monkeypatch)

    class CyclingEnv:
        def __init__(self):
            self.step_count = 0

        def reset(self, options=None):
            self.step_count = 0
            return np.array([0.0], dtype=np.float32), {}

        def action_mask(self):
            return np.array([True])

        def step(self, action):
            self.step_count += 1
            value = float(self.step_count % 2)
            return np.array([value], dtype=np.float32), 0.0, False, False, {}

    class Policy:
        def get_action(self, observation, action_mask):
            return 0

    trace = module.StandardTraceSampler().sample_trace(
        CyclingEnv(), Policy(), max_steps=10, max_state_visits=2
    )

    assert trace["termination_reason"] == "cycle"
    np.testing.assert_allclose(trace["final_observation"], [0.0])


def test_env_resets_from_faulty_pool_but_indexed_reset_takes_precedence(tmp_path, monkeypatch):
    module = _load_env_module_with_fake_backend(monkeypatch)
    pool_path = tmp_path / "faulty_states.json"
    save_faulty_states(pool_path, [[4.0, 5.0]])
    env = module.JANIEnv(
        "model.jani",
        faulty_states_path=str(pool_path),
        faulty_state_reset_prob=1.0,
    )

    observation, info = env.reset(seed=7)
    np.testing.assert_allclose(observation, [4.0, 5.0])
    assert info["reset_source"] == "faulty_state"

    observation, info = env.reset(options={"idx": 3})
    np.testing.assert_allclose(observation, [3.0, 0.0])
    assert info["reset_source"] == "initial_state_index"


def test_positive_restart_probability_requires_pool(monkeypatch):
    module = _load_env_module_with_fake_backend(monkeypatch)

    with pytest.raises(ValueError, match="faulty_states_path"):
        module.JANIEnv("model.jani", faulty_state_reset_prob=0.5)
