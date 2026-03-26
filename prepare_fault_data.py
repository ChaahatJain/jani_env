"""
Prepare fault data for classifier training.

Collects traces, extracts features from all steps, and labels faults.
Exports to CSV format expected by classifier/train.py
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# project path
root = Path(__file__).parent
sys.path.insert(0, str(root))

print("\n=== Fault Data Preparation ===\n")

# -------------------- config --------------------

JANI_CONFIG = {
    "jani_model": "examples/bouncing_ball/bouncing_ball.jani",
    "jani_property": "examples/bouncing_ball/property.jani",
    "start_states": "examples/bouncing_ball/start.jani",
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

# Data collection settings
NUM_TRACES = 200  # Number of traces to sample
OUTPUT_DIR = root / "data" / "fault_classifier"

# -------------------- env --------------------

from jani.env import JANIEnv

print("creating env...")

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

print(f"env ready - obs dim: {env.observation_space.shape[0]}, actions: {env.action_space.n}")

# -------------------- policy --------------------

from torchrl.modules import MaskedCategorical
from dagger.policy import Policy

MODEL_PATHS = [
    "models/ppo/bouncing_ball/final_actor.pth",
    "models/ppo/bouncing_ball/best_actor.pth",
    "/jani_env/models/ppo/bouncing_ball/final_actor.pth",
    "/jani_env/models/ppo/bouncing_ball/best_actor.pth",
]


def load_policy_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    hidden_dims = checkpoint['hidden_dims']
    policy_net = Policy(input_dim, output_dim, hidden_dims)

    state_dict = checkpoint['state_dict']
    if 'mlp_extractor.policy_net.0.weight' in state_dict:
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
        policy_net.load_state_dict(state_dict, strict=True)

    return policy_net


class PolicyWrapper:
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


print("loading policy...")
policy = None
loaded_model = None

for p in MODEL_PATHS:
    if Path(p).exists():
        try:
            loaded_model = load_policy_from_checkpoint(p)
            print(f"  loaded: {p}")
            break
        except Exception as e:
            print(f"  failed: {p}: {e}")

if loaded_model is not None:
    policy = PolicyWrapper(loaded_model)
else:
    print("  no model found -> using random policy")

    class SimpleRandomPolicy:
        def __init__(self, env):
            self.env = env
            self.num_actions = env.action_space.n

        def get_action(self, state, action_mask=None):
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                if len(valid) > 0:
                    return np.random.choice(valid)
            return 0

    policy = SimpleRandomPolicy(env)

# -------------------- sampling --------------------

from dagger.sampler import StandardTraceSampler

sampler = StandardTraceSampler()

print(f"\nsampling {NUM_TRACES} traces...")

traces = []
for i in range(NUM_TRACES):
    t = sampler.sample_trace(env, policy, max_steps=100)
    traces.append(t)
    if (i + 1) % 2 == 0:
        print(f"  sampled {i + 1}/{NUM_TRACES} traces")

total_steps = sum(len(t['observations']) for t in traces)
print(f"total steps: {total_steps}")

# -------------------- extract features & labels --------------------

print("\nextracting features and labels...")

# Feature construction:
# - observation (state features)
# - action taken (one-hot encoded)
# Label: 1 if fault, 0 otherwise

n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

all_features = []
all_labels = []
fault_count = 0

for trace in traces:
    observations = trace["observations"]
    actions = trace["actions"]
    action_masks = trace["action_masks"]
    oracle_is_state_safe = trace.get("oracle_is_state_safe", [])
    oracle_safe_action = trace.get("oracle_safe_action", [])

    for step in range(len(observations)):
        obs = observations[step]
        action = actions[step]
        mask = action_masks[step]

        # Get oracle data
        is_state_safe = oracle_is_state_safe[step] if step < len(oracle_is_state_safe) else True
        safe_action = oracle_safe_action[step] if step < len(oracle_safe_action) else -1

        # Determine if this is a fault
        is_fault = (
            is_state_safe and
            safe_action != -1 and
            safe_action != action
        )

        # Build feature vector: [observation, one-hot action, action_mask]
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1.0

        mask_arr = np.array(mask, dtype=np.float32) if isinstance(mask, (list, tuple)) else mask.astype(np.float32)

        feature = np.concatenate([obs.flatten(), action_onehot, mask_arr])

        all_features.append(feature)
        all_labels.append(1 if is_fault else 0)

        if is_fault:
            fault_count += 1

print(f"extracted {len(all_features)} samples")
print(f"faults: {fault_count} ({100*fault_count/len(all_features):.2f}%)")
print(f"non-faults: {len(all_features) - fault_count}")

# -------------------- balance dataset --------------------

# Since faults are rare, let's check the class balance
fault_ratio = fault_count / len(all_features)
print(f"\nclass balance: {100*fault_ratio:.2f}% faults, {100*(1-fault_ratio):.2f}% non-faults")

# You may want to balance the dataset (e.g., oversampling faults)
# For now, we keep as-is - the classifier can handle class imbalance

# -------------------- split data --------------------

print("\nsplitting into train/val/test...")

features = np.array(all_features)
labels = np.array(all_labels)

# 70/15/15 split with stratification
X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"  train: {len(X_train)} samples ({sum(y_train)} faults)")
print(f"  val:   {len(X_val)} samples ({sum(y_val)} faults)")
print(f"  test:  {len(X_test)} samples ({sum(y_test)} faults)")

# -------------------- save to CSV --------------------

print("\nsaving to CSV...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Combine features and labels for CSV format
train_data = np.column_stack([X_train, y_train])
val_data = np.column_stack([X_val, y_val])
test_data = np.column_stack([X_test, y_test])

np.savetxt(OUTPUT_DIR / "train.csv", train_data, delimiter=",")
np.savetxt(OUTPUT_DIR / "val.csv", val_data, delimiter=",")
np.savetxt(OUTPUT_DIR / "test.csv", test_data, delimiter=",")

print(f"saved to {OUTPUT_DIR}")
print(f"  train.csv: {train_data.shape}")
print(f"  val.csv: {val_data.shape}")
print(f"  test.csv: {test_data.shape}")

# -------------------- summary --------------------

print("\n=== summary ===")
print(f"traces sampled: {NUM_TRACES}")
print(f"total steps: {total_steps}")
print(f"feature dim: {features.shape[1]}")
print(f"  - observation: {obs_dim}")
print(f"  - action one-hot: {n_actions}")
print(f"  - action mask: {n_actions}")
print(f"total samples: {len(features)}")
print(f"faults: {fault_count} ({100*fault_ratio:.2f}%)")
print("\nready for training with:")
print(f"  python -m classifier.train --data-dir {OUTPUT_DIR} --n-trials 50")
print("\ndone\n")
