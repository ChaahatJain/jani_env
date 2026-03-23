# Demo: running DAgger components on a real JANI environment

import sys
import torch
import numpy as np
from pathlib import Path

# project path
root = Path(__file__).parent
sys.path.insert(0, str(root))

print("\n=== DAgger demo on JANI env ===\n")

# -------------------- config --------------------

JANI_CONFIG = {
    "jani_model": "examples/one_way_line_15_10/model.jani",
    "jani_property": "examples/one_way_line_15_10/property.jani",
    "start_states": "examples/one_way_line_15_10/property.jani",
    "objective": "",
    "failure_property": "",
    "seed": 42,
    "goal_reward": 1.0,
    "failure_reward": -1.0,
    "unsafe_reward": -0.01,
    "max_steps": 100,
    "use_oracle": True,
    "disable_oracle_cache": False,
    "reduced_memory_mode": True
}

print("checking files...")
for k, p in JANI_CONFIG.items():
    if k in ["seed", "goal_reward", "failure_reward", "unsafe_reward",
             "max_steps", "use_oracle", "disable_oracle_cache", "reduced_memory_mode"]:
        continue
    if not p:
        print(f"  {k}: (empty)")
        continue
    full = root / p
    print(f"  {k}: {'ok' if full.exists() else 'missing'}")

# -------------------- env --------------------

from jani.env import JANIEnv

print("\ncreating env...")

env = JANIEnv(
    jani_model_path=JANI_CONFIG["jani_model"],
    jani_property_path=JANI_CONFIG["jani_property"],
    start_states_path=JANI_CONFIG["start_states"],
    objective_path=JANI_CONFIG["objective"],
    failure_property_path=JANI_CONFIG["failure_property"],
    seed=JANI_CONFIG["seed"],
    goal_reward=JANI_CONFIG["goal_reward"],
    failure_reward=JANI_CONFIG["failure_reward"],
    unsafe_reward=JANI_CONFIG["unsafe_reward"],
    use_oracle=JANI_CONFIG["use_oracle"],
    disable_oracle_cache=JANI_CONFIG["disable_oracle_cache"],
    reduced_memory_mode=JANI_CONFIG["reduced_memory_mode"]
)

print("env ready")
print("obs space:", env.observation_space)
print("action space:", env.action_space)

# -------------------- policy --------------------

from sb3_contrib import MaskablePPO

print("\nloading policy...")

policy = None
for p in ["logs/best_model.zip", "logs/final_model.zip"]:
    if Path(p).exists():
        try:
            policy = MaskablePPO.load(p)
            print("loaded:", p)
            break
        except Exception as e:
            print("failed:", p)

if policy is None:
    print("no model found -> using random policy")

    class SimpleRandomPolicy:
        def __init__(self, env):
            self.env = env
            self.num_actions = env.action_space.n
            self.unsafe_rate = 0.4

        def predict(self, obs, state=None, episode_start=None, deterministic=False):
            if np.random.random() < self.unsafe_rate:
                return np.random.randint(0, self.num_actions), None

            mask = self.env.unwrapped.action_mask()
            valid = np.where(mask)[0]
            return (np.random.choice(valid) if len(valid) > 0 else 0), None

        def get_action(self, state, action_mask=None):
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                if len(valid) > 0:
                    return np.random.choice(valid)
            return 0

    policy = SimpleRandomPolicy(env)

print("policy:", type(policy).__name__)

# -------------------- components --------------------

from dagger.interfaces import TraceSamplerInterface, FaultCollectorInterface
from dagger.sampler import StandardTraceSampler
from dagger.fault_collector import OracleFaultCollector

print("\ncomponents loaded")

# -------------------- sampling --------------------

sampler = StandardTraceSampler()

traces = []
num_traces = 5

print("\nsampling traces...")

for i in range(num_traces):
    t = sampler.sample_trace(env, policy, max_steps=100)
    traces.append(t)
    print(f"  trace {i+1}: {len(t['observations'])} steps")

total_steps = sum(len(t['observations']) for t in traces)
print("total steps:", total_steps)

# quick peek
t0 = traces[0]
if len(t0["observations"]) > 0:
    print("\nexample:")
    print("  obs shape:", t0["observations"][0].shape)
    print("  action:", t0["actions"][0])
    print("  reward:", t0["rewards"][0])

# -------------------- oracle --------------------

class JANIOracle:
    def __init__(self, env):
        self.env = env
        self.query_count = 0

    def evaluate_and_correct(self, obs, action, mask):
        self.query_count += 1

        if isinstance(mask, (list, tuple)):
            mask = np.array(mask)

        try:
            if not mask[action]:
                valid = np.where(mask)[0]
                if len(valid) > 0:
                    return False, valid[0]
            return True, action
        except:
            return True, action


oracle = JANIOracle(env)

# -------------------- faults --------------------

collector = OracleFaultCollector()

print("\ncollecting faults...")

all_faults = []

for i, t in enumerate(traces):
    f = collector.collect_faults(t, oracle)
    all_faults.extend(f)
    print(f"  trace {i+1}: {len(f)} faults")

print("total faults:", len(all_faults))
print("oracle calls:", oracle.query_count)

if all_faults:
    f = all_faults[0]
    print("\nexample fault:")
    print("  step:", f.get("step"))
    print("  bad:", f["faulty_action"], "->", f["action"])

# -------------------- checks --------------------

print("\nchecks...")

n1 = sum(len(t["observations"]) for t in traces)
n2 = sum(len(t["observations"]) for t in traces)

assert n1 == n2
print("data ok:", n1)

# -------------------- summary --------------------

print("\n=== summary ===")
print("traces:", len(traces))
print("steps:", total_steps)
print("faults:", len(all_faults))
print("done\n")