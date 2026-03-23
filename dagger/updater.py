import torch
from typing import Any, Dict
from .interfaces import PolicyUpdaterInterface

class SupervisedPolicyUpdater(PolicyUpdaterInterface):
    """
    Updates a neural network policy using supervised learning over a dataset of collected faults.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, batch_size: int = 256, steps_per_iteration: int = 5, device: torch.device = torch.device("cpu")):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.steps_per_iteration = steps_per_iteration
        self.device = device
        self.loss_fn = torch.nn.NLLLoss()

    def update_policy(self, policy: torch.nn.Module, dataset: Any) -> Dict[str, float]:
        policy.train()
        policy.to(self.device)
        
        total_loss = 0.0
        for _ in range(self.steps_per_iteration):
            # Sample a batch of fault data from the replay buffer
            batch = dataset.sample(self.batch_size)
            
            observations = batch["observation"].to(self.device)
            expert_actions = batch["action"].to(self.device) # Corrected actions from the oracle
            action_masks = batch["action_mask"].to(self.device)

            # Forward pass through the network and mask invalid actions
            logits = policy(observations) 
            masked_logits = logits.masked_fill(~action_masks.bool(), float('-inf')) 
            logp = torch.log_softmax(masked_logits, dim=-1)

            # Compute supervised learning loss against expert actions
            loss = self.loss_fn(logp, expert_actions.long())

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return {"loss": total_loss / self.steps_per_iteration}