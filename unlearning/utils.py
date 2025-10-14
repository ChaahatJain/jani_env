import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

# Define the scheduler
def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a linear warmup scheduler for Unlearning.
    :param optimizer: The optimizer instance.
    :param warmup_steps: Number of steps for the warmup phase.
    :param total_steps: Total number of training steps.
    :return: LambdaLR scheduler.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic operations where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

