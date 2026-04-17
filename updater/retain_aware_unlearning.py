import copy
import random
from typing import Any, Dict, List, Optional, Tuple


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_utils import FaultDataset
from dagger.interfaces import PolicyUpdaterInterface


class RetainAwareUnlearningUpdater(PolicyUpdaterInterface):
    """
    Unlearns faulty actions via gradient ascent while preserving policy behavior.

    Expected sample keys (list-of-dicts or replay-buffer batch dict):
    - observation: tensor/array input to the policy
    - action_mask: optional bool mask of valid actions
    - faulty_action: optional action index to unlearn (forget set)
    - action: optional expert action index (retain supervision)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 256,
        steps_per_iteration: int = 10,
        forget_loss_lambda: float = 1.0,
        retain_loss_lambda: float = 1.0,
        kl_loss_lambda: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.steps_per_iteration = steps_per_iteration
        self.forget_loss_lambda = forget_loss_lambda
        self.retain_loss_lambda = retain_loss_lambda
        self.device = device

    ########## Individual Loss Functions START ##########
    def get_simple_loss(self, model, input_forget, input_retain):
        '''
        This function computes the loss for the unlearning process.
        '''
        # forget_loss
        forget_loss = self.ga_simple_loss(model, input_forget)

        # regularization_loss
        regularization_loss = self.gd_simple_loss(model, input_retain)

        return forget_loss, regularization_loss

    # Forget Loss: GA
    def ga_simple_loss(self, model, input_forget):
        # The first element of the data tuple is the target data
        x_forget, y_forget = input_forget
        # Compute the Cross entropy loss for the answer
        loss_fn = torch.nn.CrossEntropyLoss() 
        y_pred = model(x_forget)
        #reversing the sign for gradient ascent
        forget_loss = -1 * loss_fn(y_forget, y_pred)
        return  forget_loss

    # Regularization Loss: GD
    def gd_simple_loss(self, model, input_retain):
        x_retain, y_retain = input_retain
        # Compute the Cross entropy loss for the answer
        loss_fn = torch.nn.CrossEntropyLoss() 
        y_pred = model(x_retain)   
        retain_loss = loss_fn(y_retain, y_pred)

        return retain_loss

    ########## Individual Loss Functions END ##########
    # updates a policy given the current policy model and a dataset containing only faults
    def update_policy(self, policy: torch.nn.Module, dataset: Any) -> Dict[str, float]:
        policy.train()
        policy.to(self.device)
        
        print(dataset)
        
        total_loss = 0.0
        total_forget_loss = 0.0
        total_retain_loss = 0.0
        steps_with_forget = 0
        
        fault_dataset = FaultDataset(dataset)
        # use dataloader instead of individual batching
        dataloader = DataLoader(fault_dataset,  
           batch_size=self.batch_size, 
           shuffle=False, 
           num_workers=0, 
           collate_fn=None)
        
        for batch in dataloader:
            forget_input = (batch["input"], batch["fault"])
            retain_input = (batch["input"], batch["valid"])

            forget_loss, retain_loss = self.get_simple_loss(policy, forget_input, retain_input)

            loss = self.forget_loss_lambda * forget_loss + self.retain_loss_lambda * retain_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_forget_loss += float(forget_loss.item())
            total_retain_loss += float(retain_loss.item())

        denom = float(max(self.steps_per_iteration, 1))
        return {
            "loss": total_loss / denom,
            "forget_loss": total_forget_loss / denom,
            "retain_supervised_loss": total_retain_loss / denom,
            "steps": int(self.steps_per_iteration),
            "steps_with_forget": int(steps_with_forget),
        }
        

