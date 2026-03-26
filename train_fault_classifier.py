"""
End-to-end fault classifier training.

1. Samples traces from the environment
2. Extracts features and labels
3. Trains a classifier to predict faults
4. Saves the trained model
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse

root = Path(__file__).parent
sys.path.insert(0, str(root))


def parse_args():
    parser = argparse.ArgumentParser(description="Train fault classifier")
    parser.add_argument('--jani-model', default="examples/bouncing_ball/bouncing_ball.jani")
    parser.add_argument('--jani-property', default="examples/bouncing_ball/property.jani")
    parser.add_argument('--start-states', default="examples/bouncing_ball/start.jani")
    parser.add_argument('--num-traces', type=int, default=200, help='Number of traces to sample')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per trace')
    parser.add_argument('--model-path', default=None, help='Path to policy model (uses random if not provided)')
    parser.add_argument('--output-dir', default='models/fault_classifier', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 64], help='Hidden layer sizes')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


class FaultClassifier(nn.Module):
    """Simple MLP for fault prediction."""
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


def collect_traces(env, policy, num_traces, max_steps):
    """Sample traces from environment."""
    from dagger.sampler import StandardTraceSampler
    sampler = StandardTraceSampler()

    traces = []
    for i in range(num_traces):
        t = sampler.sample_trace(env, policy, max_steps=max_steps)
        traces.append(t)
        if (i + 1) % 2 == 0:
            print(f"  sampled {i + 1}/{num_traces}")

    return traces


def extract_features_labels(traces, n_actions, obs_dim):
    """Extract features and labels from traces."""
    all_features = []
    all_labels = []

    for trace in traces:
        observations = trace["observations"]
        actions = trace["actions"]
        action_masks = trace["action_masks"]
        oracle_is_safe = trace.get("oracle_is_state_safe", [])
        oracle_safe_act = trace.get("oracle_safe_action", [])

        for step in range(len(observations)):
            obs = observations[step]
            action = actions[step]
            mask = action_masks[step]

            is_safe = oracle_is_safe[step] if step < len(oracle_is_safe) else True
            safe_act = oracle_safe_act[step] if step < len(oracle_safe_act) else -1

            # Fault: state was safe, had safe action, but took different action
            is_fault = is_safe and safe_act != -1 and safe_act != action

            # Feature: [obs, action_onehot, mask]
            action_onehot = np.zeros(n_actions)
            action_onehot[action] = 1.0
            mask_arr = np.array(mask, dtype=np.float32) if isinstance(mask, (list, tuple)) else mask.astype(np.float32)

            feature = np.concatenate([obs.flatten(), action_onehot, mask_arr])
            all_features.append(feature)
            all_labels.append(1 if is_fault else 0)

    return np.array(all_features), np.array(all_labels)


def train_classifier(X_train, y_train, X_val, y_val, args, device):
    """Train the fault classifier."""
    input_dim = X_train.shape[1]
    model = FaultClassifier(input_dim, args.hidden_dims).to(device)

    # Compute class weights for imbalanced data
    n_faults = y_train.sum()
    n_nonfaults = len(y_train) - n_faults
    pos_weight = torch.tensor([n_nonfaults / max(n_faults, 1)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Note: Since we have sigmoid in model, we'll use BCELoss but adjust
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_f1 = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()

        # Shuffle and batch
        perm = torch.randperm(len(X_train_t))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_t), args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_pred_binary = (val_pred > 0.5).cpu().numpy()
            val_f1 = f1_score(y_val, val_pred_binary, zero_division=0)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1}: loss={total_loss / n_batches:.4f}, val_f1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate(model, X_test, y_test, device):
    """Evaluate model on test set."""
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(X_test_t)
        pred_binary = (pred > 0.5).cpu().numpy()

    acc = accuracy_score(y_test, pred_binary)
    prec = precision_score(y_test, pred_binary, zero_division=0)
    rec = recall_score(y_test, pred_binary, zero_division=0)
    f1 = f1_score(y_test, pred_binary, zero_division=0)
    cm = confusion_matrix(y_test, pred_binary)

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Fault Classifier Training ===")
    print(f"device: {device}\n")

    # ---- Setup environment ----
    print("setting up environment...")
    from jani.env import JANIEnv

    env = JANIEnv(
        jani_model_path=args.jani_model,
        jani_property_path=args.jani_property,
        start_states_path=args.start_states,
        seed=args.seed,
        use_oracle=True,
        reduced_memory_mode=True
    )
    n_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    print(f"  obs_dim={obs_dim}, n_actions={n_actions}")

    # ---- Setup policy ----
    print("setting up policy...")
    policy = None

    # Try to load model if path provided
    model_paths = [args.model_path] if args.model_path else [
        "models/ppo/bouncing_ball/final_actor.pth",
        "models/ppo/bouncing_ball/best_actor.pth",
        "/jani_env/models/ppo/bouncing_ball/final_actor.pth",
    ]

    from torchrl.modules import MaskedCategorical
    from dagger.policy import Policy

    for path in model_paths:
        if path and Path(path).exists():
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                policy_net = Policy(checkpoint['input_dim'], checkpoint['output_dim'], checkpoint['hidden_dims'])
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

                policy_net.eval()

                class PolicyWrapper:
                    def __init__(self, net):
                        self.net = net

                    def get_action(self, state, action_mask=None):
                        obs_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        mask_t = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0) if action_mask is not None else torch.ones(1, n_actions, dtype=torch.bool)
                        with torch.no_grad():
                            logits = self.net(obs_t)
                            dist = MaskedCategorical(logits=logits, mask=mask_t)
                            return dist.sample().item()

                policy = PolicyWrapper(policy_net)
                print(f"  loaded: {path}")
                break
            except Exception as e:
                print(f"  failed: {path}: {e}")

    if policy is None:
        print("  using random policy")
        class RandomPolicy:
            def get_action(self, state, action_mask=None):
                if action_mask is not None:
                    valid = np.where(action_mask)[0]
                    if len(valid) > 0:
                        return np.random.choice(valid)
                return 0
        policy = RandomPolicy()

    # ---- Collect traces ----
    print(f"\ncollecting {args.num_traces} traces...")
    traces = collect_traces(env, policy, args.num_traces, args.max_steps)
    total_steps = sum(len(t['observations']) for t in traces)
    print(f"  total steps: {total_steps}")

    # ---- Extract features ----
    print("\nextracting features...")
    features, labels = extract_features_labels(traces, n_actions, obs_dim)
    n_faults = labels.sum()
    print(f"  samples: {len(labels)}")
    print(f"  faults: {n_faults} ({100 * n_faults / len(labels):.2f}%)")

    if n_faults == 0:
        print("\nWARNING: No faults found! The policy may already be perfect.")
        print("Try using a worse policy or check if oracle is working correctly.")
        return

    # ---- Split data ----
    print("\nsplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.3, random_state=args.seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )
    print(f"  train: {len(X_train)} ({y_train.sum()} faults)")
    print(f"  val: {len(X_val)} ({y_val.sum()} faults)")
    print(f"  test: {len(X_test)} ({y_test.sum()} faults)")

    # ---- Normalize ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ---- Train ----
    print(f"\ntraining classifier...")
    print(f"  hidden_dims: {args.hidden_dims}")
    print(f"  epochs: {args.epochs}, lr: {args.lr}")
    model = train_classifier(X_train, y_train, X_val, y_val, args, device)

    # ---- Evaluate ----
    print("\nevaluating on test set...")
    metrics = evaluate(model, X_test, y_test, device)
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1-score:  {metrics['f1']:.4f}")
    print(f"  confusion matrix:\n{metrics['confusion_matrix']}")

    # ---- Save model ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "fault_classifier.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': features.shape[1],
        'hidden_dims': args.hidden_dims,
        'scaler': scaler,
        'n_actions': n_actions,
        'obs_dim': obs_dim,
        'metrics': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in metrics.items()},
    }, save_path)
    print(f"\nmodel saved to: {save_path}")

    print("\n=== done ===\n")


if __name__ == "__main__":
    main()
