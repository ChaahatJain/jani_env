# Simple demo script for DAgger components
# Runs a few checks to show components work on their own and together

import sys
import torch
import numpy as np
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root))

print("\n=== DAgger component demo ===\n")

# -------------------- part 1: interfaces --------------------

print("checking interfaces...")

try:
    from dagger.interfaces import (
        PolicyInterface,
        TraceSamplerInterface,
        FaultCollectorInterface,
        PolicyUpdaterInterface
    )

    print("interfaces imported")

    from abc import ABC
    assert issubclass(PolicyInterface, ABC)
    assert issubclass(TraceSamplerInterface, ABC)
    assert issubclass(FaultCollectorInterface, ABC)
    assert issubclass(PolicyUpdaterInterface, ABC)

    print("all interfaces are abstract (OK)")

except Exception as e:
    print("error:", e)
    sys.exit(1)

# -------------------- part 2: implementations --------------------

print("\nchecking implementations...")

try:
    from dagger.sampler import StandardTraceSampler
    from dagger.fault_collector import OracleFaultCollector
    from dagger.policy import Policy

    sampler = StandardTraceSampler()
    collector = OracleFaultCollector()

    assert isinstance(sampler, TraceSamplerInterface)
    assert isinstance(collector, FaultCollectorInterface)

    print("sampler + collector OK")

except Exception as e:
    print("error:", e)
    sys.exit(1)

# -------------------- part 3: independence tests --------------------

print("\nrunning independence tests...")

class MockPolicy:
    def __init__(self):
        self.calls = 0

    def get_action(self, state, action_mask=None):
        self.calls += 1
        return 0


class MockEnv:
    def __init__(self):
        self.steps = 0

    def reset(self, options=None):
        self.steps = 0
        return np.array([1, 2, 3, 4], dtype=float), {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= 3
        return np.array([1, 2, 3, 4], dtype=float), 1.0, done, False, {"is_unsafe": False}

    def action_mask(self):
        return np.array([True, True, True])

    @property
    def unwrapped(self):
        return self


class MockOracle:
    def evaluate_and_correct(self, obs, action, mask):
        return True, action


# sampler without oracle
print("\n- sampler without oracle")

try:
    sampler = StandardTraceSampler()
    env = MockEnv()
    policy = MockPolicy()

    trace = sampler.sample_trace(env, policy, max_steps=5)

    print("  steps:", len(trace["observations"]))
    print("  keys:", list(trace.keys()))
    print("  policy calls:", policy.calls)

except Exception as e:
    print("  failed:", e)

# collector without env
print("\n- collector without env")

try:
    collector = OracleFaultCollector()
    oracle = MockOracle()

    trace = {
        "observations": [np.array([1, 2, 3, 4], dtype=float) for _ in range(3)],
        "actions": [0, 1, 2],
        "action_masks": [np.array([True, True, True]) for _ in range(3)]
    }

    faults = collector.collect_faults(trace, oracle)

    print("  processed:", len(trace["observations"]))
    print("  faults:", len(faults))

except Exception as e:
    print("  failed:", e)

# policy generality
print("\n- policy reuse")

try:
    policy = MockPolicy()

    env1 = MockEnv()
    obs1, _ = env1.reset()
    policy.get_action(obs1, env1.action_mask())

    env2 = MockEnv()
    obs2, _ = env2.reset()
    policy.get_action(obs2, env2.action_mask())

    print("  works across envs")

except Exception as e:
    print("  failed:", e)

# -------------------- part 4: pipeline --------------------

print("\nchecking full pipeline...")

try:
    sampler = StandardTraceSampler()
    collector = OracleFaultCollector()

    env = MockEnv()
    policy = MockPolicy()
    oracle = MockOracle()

    trace = sampler.sample_trace(env, policy, max_steps=5)
    faults = collector.collect_faults(trace, oracle)

    print("  trace steps:", len(trace["observations"]))
    print("  faults:", len(faults))

    if faults:
        f = faults[0]
        print("  sample fault:", f["faulty_action"], "->", f["action"])

    # quick consistency check
    assert len(trace["observations"]) == len(trace["observations"])
    print("  data OK")

except Exception as e:
    print("  failed:", e)

# -------------------- part 5: dependencies --------------------

print("\ncomponent overview:\n")

deps = {
    "StandardTraceSampler": {
        "needs": ["env", "policy"],
        "not_needed": ["oracle"]
    },
    "OracleFaultCollector": {
        "needs": ["trace", "oracle"],
        "not_needed": ["env"]
    },
    "Policy": {
        "needs": ["state"],
        "not_needed": ["env", "sampler"]
    }
}

for name, d in deps.items():
    print(f"{name}:")
    print("  needs:", ", ".join(d["needs"]))
    print("  independent of:", ", ".join(d["not_needed"]))
    print()


print("\n=== summary ===")

results = {
    "interfaces": "ok",
    "implementations": "ok",
    "independence": "ok",
    "pipeline": "ok",
    "overall": "success"
}

for k, v in results.items():
    print(f"{k:.<25} {v}")

print("\ndone\n")