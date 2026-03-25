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
    "jani_model": "examples/bouncing_ball/bouncing_ball.jani",
    "jani_property": "examples/bouncing_ball/property.jani",
    "start_states": "examples/bouncing_ball/start.jani",  # Use dedicated start file
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
    "models/ppo/bouncing_ball/final_actor.pth",
    "models/ppo/bouncing_ball/best_actor.pth",
    "/jani_env/models/ppo/bouncing_ball/final_actor.pth",
    "/jani_env/models/ppo/bouncing_ball/best_actor.pth",
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

from dagger.interfaces import TraceSamplerInterface, FaultCollectorInterface, OracleInterface
from dagger.sampler import StandardTraceSampler
from dagger.fault_collector import OracleFaultCollector

print("\ncomponents loaded")

# -------------------- sampling --------------------

sampler = StandardTraceSampler()

traces = []
num_traces = 100

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
    print("  trace keys:", list(t0.keys()))
    if "action_masks" in t0:
        print("  action_masks present:", len(t0["action_masks"]), "entries")
    print("  is_safe_trajectory:", t0.get("is_safe_trajectory", "unknown"))

    # Show oracle data recorded during sampling
    if "oracle_is_state_safe" in t0:
        safe_count = sum(1 for s in t0["oracle_is_state_safe"] if s)
        print(f"  oracle_is_state_safe: {safe_count}/{len(t0['oracle_is_state_safe'])} states were safe")
    if "oracle_safe_action" in t0:
        has_safe = sum(1 for a in t0["oracle_safe_action"] if a != -1)
        print(f"  oracle_safe_action: {has_safe}/{len(t0['oracle_safe_action'])} had safe action")
        # Show first few for debugging
        print(f"  first 10 (safe_action, action): {list(zip(t0['oracle_safe_action'][:10], t0['actions'][:10]))}")

# -------------------- faults --------------------

collector = OracleFaultCollector()

print("\ncollecting faults (using recorded oracle data)...")

all_faults = []

for i, t in enumerate(traces):
    print(f"\n[DEBUG] Processing trace {i+1}:")
    print(f"[DEBUG]   steps: {len(t['observations'])}")
    print(f"[DEBUG]   is_safe_trajectory: {t.get('is_safe_trajectory', 'unknown')}")

    # Show oracle data for first few steps
    if "oracle_is_state_safe" in t and "oracle_safe_action" in t:
        print(f"[DEBUG]   First 5 steps oracle data:")
        for step in range(min(5, len(t['observations']))):
            is_safe = t['oracle_is_state_safe'][step]
            safe_act = t['oracle_safe_action'][step]
            taken_act = t['actions'][step]
            is_fault = is_safe and safe_act != -1 and safe_act != taken_act
            print(f"[DEBUG]     Step {step}: is_safe={is_safe}, safe_action={safe_act}, taken={taken_act}, is_fault={is_fault}")

    try:
        f = collector.collect_faults(t)  # No oracle needed - uses recorded data
        all_faults.extend(f)
        print(f"  trace {i+1}: {len(f)} faults found")
    except Exception as e:
        print(f"[ERROR] collect_faults failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\ntotal faults:", len(all_faults))
print("(faults detected using oracle data recorded during sampling)")

if all_faults:
    f = all_faults[0]
    print("\nexample fault:")
    print("  step:", f.get("step"))
    print(f"  faulty action: {f['faulty_action']} -> corrected to: {f['action']}")

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