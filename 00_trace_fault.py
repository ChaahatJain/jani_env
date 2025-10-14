""" Code for tracking neuron activation for a single forward pass. 
    For each layer, the goal is to identify a "path" using high-level activations 
"""
import os
import copy
import random
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import torch
from torch import nn

from tracing import utils
from tracing import visualization
from tracing.hooked_model import HookSimpleMLP
from tracing.manipulated_model import ManipulatedSimpleMLP
from nn.simple_model import SimpleMLP
from utils import evaluate

device = "cpu" #torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = SimpleMLP()
model.load_state_dict(torch.load("models/simple_mlp.pt", weights_only=True))
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss() # CrossEntropyLoss

# Load data
x_data = torch.load("data/inputs.pt")
y_data = torch.load("data/outputs.pt")

# Tracing instance:
x_trace = x_data[0]
y_trace = y_data[0]

# Remaining data
x_remain = x_data[1:]
y_remain = y_data[1:]

# Example visualization of the hooks (see tracing.visualization)
#visualization.visualize_hooked_model(hooked_trace)
#visualization.visualize_hooked_model(hooked_positive)

max_diff_neurons_forward = utils.trace_max_diff_forward(model, loss_fn, (x_trace, y_trace), (x_remain, y_remain))
max_diff_neurons_backward = utils.trace_max_diff_backward(model, loss_fn, (x_trace, y_trace), (x_remain, y_remain))
max_forward = utils.trace_max_forward(model, loss_fn, (x_trace, y_trace))
max_backward = utils.trace_max_backward(model, loss_fn, (x_trace, y_trace))

output = ["Tracing Method | Weight Changing | F1-Positive | F1-Negative | Success Trace\n"]

for name, neurons in [("Max Forward", max_forward), ("Max Backward", max_backward), ("Max Diff Forward", max_diff_neurons_forward), ("Max Diff Backward", max_diff_neurons_backward)]:
    manip_model = ManipulatedSimpleMLP(model)
    manip_model.set_instances([x_trace],[y_trace])
    for elem in ["value", "increase-value", "decrease-value", "random-uniform", "increase-uniform", "decrease-uniform"]:
        # Set change function 
        if elem == "value":
            # Set 0.0 value for single value change
            manip_model.set_change_val(0.0)
        else:
            # set Change value to 1.0
            manip_model.set_change_val(1.0)
            
        manip_model.set_change_func(elem)
        # Manipulate model params
        manip_model.manipulate_model_params(neurons)
        # fetch manipulated model predictions
        new_model = manip_model.get_manipulated_model()
        f1_pos, f1_neg, success_trace = evaluate(model, new_model, (x_remain, y_remain), ([x_trace], [y_trace]))
        outstring = f"{name} | {elem} | {f1_pos} | {f1_neg} | {success_trace[1]}\n"
        logging.info(outstring)
        output.append(outstring)
        
with open("results.txt", 'w') as f:
    for line in output:
        f.write(line)
        





