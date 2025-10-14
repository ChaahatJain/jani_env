import random

import torch 

from unlearning import losses 
from unlearning import utils
from nn.simple_model import SimpleMLP

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
