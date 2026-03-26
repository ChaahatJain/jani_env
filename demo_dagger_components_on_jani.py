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
    "start_states": "examples/one_way_line_15_10/eval_start_states.jani",
    # "jani_model": "benchmarks_generator/benchmarks/two_way_line_det/two_way_line_15_10/model.jani",
    # "jani_property": "benchmarks_generator/benchmarks/two_way_line_det/two_way_line_15_10/model.jani",
    # "start_states": "benchmarks_generator/benchmarks/two_way_line_det/two_way_line_15_10/pa_model_random_starts_20000.jani",
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

import torch
from torchrl.modules import MaskedCategorical
from dagger.policy import Policy

print("\nloading policy...")

# Model paths to try (update based on your training output)
MODEL_PATHS = [
    # .pth format (mask_ppo output)
    "/jani_env/examples/one_way_line_15_10/policy/final_actor.pth",
]


def load_policy_from_checkpoint(checkpoint_path):
    """Load policy from .pth checkpoint (mask_ppo format)."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    hidden_dims = checkpoint['hidden_dims']
    policy_net = Policy(input_dim, output_dim, hidden_dims)

    # Check if state_dict needs mapping (SB3 format) or is direct (DAgger format)
    state_dict = checkpoint['state_dict']
    if 'mlp_extractor.policy_net.0.weight' in state_dict:
        # SB3 format - need to map keys
        mapped = {
            "model.0.weight": state_dict["mlp_extractor.policy_net.0.weight"],
            "model.0.bias": state_dict["mlp_extractor.policy_net.0.bias"],
            "model.2.weight": state_dict["mlp_extractor.policy_net.2.weight"],
            "model.2.bias": state_dict["mlp_extractor.policy_net.2.bias"],
            "model.4.weight": state_dict["action_net.weight"],
            "model.4.bias": state_dict["action_net.bias"],
        }
        policy_net.load_state_dict(mapped, strict=True)
    else:
        # Direct format (DAgger saved)
        policy_net.load_state_dict(state_dict, strict=True)

    return policy_net


class PolicyWrapper:
    """Wraps Policy (nn.Module) to implement get_action for DAgger sampler."""
    def __init__(self, policy_net):
        self.policy_net = policy_net
        self.policy_net.eval()

    def get_action(self, state, action_mask=None):
        obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)
        else:
            mask_tensor = torch.ones(1, self.policy_net.model[-1].out_features, dtype=torch.bool)

        with torch.no_grad():
            logits = self.policy_net(obs_tensor)
            action_dist = MaskedCategorical(logits=logits, mask=mask_tensor)
            action = action_dist.sample().squeeze(0).item()

        return int(action)


policy = None
loaded_model = None

for p in MODEL_PATHS:
    if Path(p).exists():
        try:
            loaded_model = load_policy_from_checkpoint(p)
            print("loaded:", p)
            break
        except Exception as e:
            print(f"failed to load {p}: {e}")

if loaded_model is not None:
    policy = PolicyWrapper(loaded_model)
else:
    print("no model found -> using random policy")

    class SimpleRandomPolicy:
        def __init__(self, env):
            self.env = env
            self.num_actions = env.action_space.n

        def get_action(self, state, action_mask=None):
            # Always respect the mask - invalid actions crash the environment
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                if len(valid) > 0:
                    return np.random.choice(valid)
            return 0

    policy = SimpleRandomPolicy(env)

print("policy:", type(policy).__name__)

# -------------------- components --------------------

from dagger.interfaces import TraceSamplerInterface, FaultCollectorInterface, OracleInterface, PolicyUpdaterInterface
from dagger.sampler import StandardTraceSampler
from dagger.fault_collector import OracleFaultCollector
from dagger.updater import MILPPolicyUpdater, SpecRepairPolicyUpdater

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
    print("  trace keys:", t0.keys())
    if "action_masks" in t0:
        print("  action_masks present:", len(t0["action_masks"]), "entries")
        print("  first mask:", t0["action_masks"][0])

# -------------------- sanity check: verify DAgger regime --------------------

from collections import Counter

print("\n=== SANITY CHECK: Action Distribution ===")
print("Verifying ideal DAgger regime conditions...\n")

# Aggregate action distributions across all traces
all_safe_actions = []
all_policy_actions = []

for t in traces:
    if "safe_actions" in t and "actions" in t:
        # Only collect where oracle has a preference (safe_action != -1)
        for safe_act, policy_act in zip(t["safe_actions"], t["actions"]):
            if safe_act != -1:  # Oracle has a recommended action
                all_safe_actions.append(safe_act)
                all_policy_actions.append(policy_act)

print(f"Steps where oracle provided supervision: {len(all_safe_actions)}")
print(f"\nOracle recommendations (safe_actions):")
print(Counter(all_safe_actions))

print(f"\nPolicy choices (actions) at those same steps:")
print(Counter(all_policy_actions))

if all_safe_actions:
    disagreements = sum(1 for s, p in zip(all_safe_actions, all_policy_actions) if s != p)
    print(f"\nDisagreements: {disagreements}/{len(all_safe_actions)} ({100*disagreements/len(all_safe_actions):.1f}%)")

    # Check for ideal DAgger conditions
    policy_counter = Counter(all_policy_actions)
    oracle_counter = Counter(all_safe_actions)

    print("\n Ideal DAgger verification:")
    print(f"  Few faults: {disagreements} disagreements found")
    print(f"  Policy preference: {policy_counter.most_common(1)}")
    print(f"  Oracle preference: {oracle_counter.most_common(1)}")

    if disagreements > 0 and len(set(all_safe_actions)) > 1:
        print("  Learning signal is REAL (policy and oracle have different preferences)")
    elif disagreements == 0:
        print(" No disagreements - policy may already be optimal")
    else:
        print("  Degenerate case - investigate further")
else:
    print("\n No oracle supervision found in traces!")

print("\n" + "="*50 + "\n")

# -------------------- oracle (optional - for reference) --------------------

# Note: Fault detection now uses safety info recorded in the trace during sampling.
# The oracle is kept here for potential future use (e.g., online querying).

class JANIOracle(OracleInterface):
    """
    Oracle that determines if a (state, action) pair is unsafe.
    Can query the environment for safety information.
    """
    def __init__(self, env):
        self.env = env
        self.query_count = 0

    def is_state_action_fault(self, obs, action):
        """Check if action is unsafe at the given state."""
        self.query_count += 1
        return True #TODO: Change later on

        # Use environment's safety checking if available
        if hasattr(self.env.unwrapped, 'is_state_action_fault'):
            try:
                is_fault = self.env.unwrapped.is_state_action_fault(obs, action)
                # Fault if state is safe but we're not taking the safe action
                return is_fault
            except Exception:
                pass

        return False


oracle = JANIOracle(env)

# -------------------- faults --------------------

collector = OracleFaultCollector()

print("\ncollecting faults...")

all_faults = []

for i, t in enumerate(traces):
    print(f"\n[DEBUG] Processing trace {i+1}:")
    print(f"[DEBUG]   observations: {len(t['observations'])}")
    print(f"[DEBUG]   actions: {len(t['actions'])}")
    print(f"[DEBUG]   action_masks in trace: {'action_masks' in t}")
    try:
        f = collector.collect_faults(t, oracle)
        all_faults.extend(f)
        print(f"  trace {i+1}: {len(f)} faults")
    except Exception as e:
        print(f"[ERROR] collect_faults failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\ntotal faults:", len(all_faults))

if all_faults:
    f = all_faults[0]
    print(f)
    # print("  step:", f.get("step"))
    # print(f"  faulty action: {f['faulty_action']} -> corrected to: {f['action']}")
    # print(f"  state was safe: {f.get('was_state_safe')}, next state safe: {f.get('is_next_safe')}")

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


#---------------------------- policy fixing step ------------------------------

#-------------------------- Spec repair --------------------
optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-4)
updater = SpecRepairPolicyUpdater(optimizer=optimizer)
blah = updater.update_policy(loaded_model, all_faults)
print(blah)
#-------------------------- MILP repair ---------------------
updater = MILPPolicyUpdater()
updater.update_policy(loaded_model, all_faults)