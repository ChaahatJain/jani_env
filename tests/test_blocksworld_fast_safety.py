import sys
import types

import numpy as np

# The compiled JANI backend is not present in lightweight unit-test
# environments.  These tests exercise Python fast paths only.
if "backend" not in sys.modules:
    backend = types.ModuleType("backend")
    backend.JANIEngine = object
    backend.TarjanOracle = object
    sys.modules["backend"] = backend
if "jani.torchrl_env" not in sys.modules:
    torchrl_env = types.ModuleType("jani.torchrl_env")
    torchrl_env.JANIEnv = object
    sys.modules["jani.torchrl_env"] = torchrl_env
if "torchrl.modules" not in sys.modules:
    torchrl = types.ModuleType("torchrl")
    torchrl_modules = types.ModuleType("torchrl.modules")
    torchrl_modules.MaskedCategorical = object
    torchrl.modules = torchrl_modules
    sys.modules["torchrl"] = torchrl
    sys.modules["torchrl.modules"] = torchrl_modules

from dagger.fault_collector import OracleFaultCollector
from jani.env import JANIEnv


class _NoOracleBlocksworldEnv:
    @property
    def unwrapped(self):
        return self

    def uses_blocksworld_safety_shortcut(self):
        return True

    def is_state_action_fault(self, observation, action):
        raise AssertionError("the Blocksworld trace shortcut must not query the oracle")


class _Engine:
    def __init__(self):
        self.current = [0.0]
        self.successors = {
            (0.0, 0): [[1.0]],
            (0.0, 1): [[99.0]],
        }

    def get_current_state_vector(self):
        return self.current

    def reach_failure_state_vector(self, state):
        return state == [99.0]

    def get_all_successor_states_as_vectors(self, state, action):
        return self.successors[(state[0], action)]

    def get_current_action_mask(self):
        return [True, True]


def _fast_blocksworld_env():
    env = JANIEnv.__new__(JANIEnv)
    env._engine = _Engine()
    env._blocksworld_fast_safety = True
    env._oracle = None
    env._reseted = True
    return env


def test_unsafe_blocksworld_trace_marks_only_last_transition_as_fault():
    trace = {
        "observations": [np.array([0.0]), np.array([1.0]), np.array([2.0])],
        "actions": [0, 1, 2],
        "action_masks": [np.ones(3)] * 3,
        "is_safe_trajectory": False,
        "termination_reason": "failure",
        "final_observation": np.array([99.0]),
    }

    faults = OracleFaultCollector().collect_faults(trace, _NoOracleBlocksworldEnv())

    assert len(faults) == 1
    assert faults[0]["step"] == 2
    assert faults[0]["faulty_action"] == 2


def test_safe_blocksworld_trace_has_no_faults():
    trace = {
        "observations": [np.array([0.0])],
        "actions": [0],
        "action_masks": [np.ones(1)],
        "is_safe_trajectory": True,
    }

    assert OracleFaultCollector().collect_faults(
        trace, _NoOracleBlocksworldEnv()
    ) == []


def test_blocksworld_state_action_fault_only_checks_immediate_failure():
    env = _fast_blocksworld_env()

    assert not env.is_state_action_fault(np.array([0.0]), 0)
    assert env.is_state_action_fault(np.array([0.0]), 1)
    assert env.is_current_state_action_safe(0)
    assert not env.is_current_state_action_safe(1)
    assert env.current_state_safety_with_action(1) == (True, 0)
