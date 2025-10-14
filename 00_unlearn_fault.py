import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch 
from sb3_contrib import MaskablePPO
from stable_baselines3.common.policies import obs_as_tensor, load_from_vector

from unlearning import losses 
from unlearning import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unlearning script for a model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_file', type=str, 
                       default='models/testing_stuff/final_model.zip',
                       help='Path to the model file (stable-baseline3)')
    parser.add_argument('--faults_file', type=str, 
                       default='faults/faults.csv',
                       help='Path to the faults csv file')
                       
    return parser.parse_args()

def load_policy(policy_file_path):
    """Load a trained MaskablePPO policy from file."""
    if not Path(policy_file_path).exists():
        raise FileNotFoundError(f"Policy file not found: {policy_file_path}")
    
    try:
        policy = MaskablePPO.load(policy_file_path)
        print(f"Successfully loaded policy from: {policy_file_path}")
        return policy
    except Exception as e:
        raise RuntimeError(f"Failed to load policy from {policy_file_path}: {e}")


def load_faults(faults_path):
    df = pd.read_csv(faults_path, sep=',')
    return df

def main(args):
    p_model = load_policy(args.model_file) #.to(device)
    faults = load_faults(args.faults_file)
    # Iterate over states and ask policy for a prediction:
    for str_state in faults["State"]:
        print(str_state)
        print(p_model.policy.device)
        state = np.fromstring(str_state, sep=",")
        obs = load_from_vector(state)
        print(obs)
        obs = obs_as_tensor(state, p_model.policy.device)
        #probs = p_model.policy.get_distribution(obs)
        action, _states = p_model.predict(obs)
        print("Obs: ", obs)
        print("Probs: ", probs)
        print("Action: ", action)
        print("Next states: ", _states)
    
"""    
# Load model
# Use CPU for development and testing
device = "cpu" #torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = SimpleMLP()
model.load_state_dict(torch.load("models/simple_mlp.pt", weights_only=True))
model.to(device)


# Config
cfg = {
    "learning_rate":0.0001,
    "unlearn_epochs":200,
    "forget_loss_lambda": 0.5,
    "retain_loss_lambda": 1.0,
}

# Load data
x_data = torch.load("data/inputs.pt")
y_data = torch.load("data/outputs.pt")

# Sanity check: run the eval before unlearning:
model.eval()
print("---- Pre-unlearning experiments ----")
for x,y in zip(x_data,y_data):
    print(f"Instance: {x}")
    print(f"Predicted: {torch.argmax(model(x))} ---- True: {torch.argmax(y)}")

# Unlearning instance:
x_unlearn = x_data[0]
y_unlearn = y_data[0]

# Retain set:
x_retain = x_data[1:]
y_retain = y_data[1:]

# Calculate the number of warmup steps and total steps
total_steps = cfg["unlearn_epochs"] * len(x_data)  # Total steps = epochs * steps per epoch
warmup_steps = int(0.1 * total_steps)  # 10% of total steps

# Loss function and optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr= cfg["learning_rate"])  
# Initialize the lr scheduler
scheduler = utils.get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

model.train()
for epoch in range(cfg["unlearn_epochs"]):
    #unlearn_idx = random.randint(0, len(y_unlearn))
    x_u = x_unlearn
    y_u = y_unlearn
    
    retain_idx = random.randint(0, len(y_retain)-1)
    x_r = x_retain[retain_idx]
    y_r = y_retain[retain_idx]
    # Zero the parameter gradients
    optimizer.zero_grad()
    #get the loss terms
    forget_loss, regularization_loss = losses.get_simple_loss(
        model, 
        (x_u, y_u),
        (x_r, y_r) 
    )
    #compute total loss
    loss = cfg["forget_loss_lambda"] * forget_loss + cfg["retain_loss_lambda"] * regularization_loss
    loss.backward()
    
    # Step the scheduler and optimizer
    optimizer.step()
    scheduler.step()
    

# Sanity check: run the eval before unlearning:
model.eval()
print("---- Post-unlearning experiments ----")
print(f"Unlearned instance {x_unlearn} with class {torch.argmax(y_unlearn)}")
for x,y in zip(x_data,y_data):
    print(f"Instance: {x}")
    print(f"Predicted: {torch.argmax(model(x))} ---- True: {torch.argmax(y)}")

torch.save(model.state_dict(), "models/simple_mlp_unlearned.pt")
"""

if __name__ == "__main__":
    args = parse_args()
    main(args)
