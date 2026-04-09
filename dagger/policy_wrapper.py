import torch
from typing import Any
from torchrl.modules import MaskedCategorical
from .interfaces import PolicyInterface

class NNPolicyWrapper(PolicyInterface):
    """
    A concrete implementation of the PolicyInterface for neural network policies.
    Guarantees the sampler can use a NN policy without being tightly coupled to PyTorch.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    def get_action(self, state: Any, action_mask: Any = None) -> Any:
        self.model.eval()
        self.model.to(self.device)
        
        obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(obs_tensor)
            action_dist = MaskedCategorical(logits=logits, mask=action_mask_tensor)
            return action_dist.probs.argmax(dim=-1).squeeze(0).item()