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
        self.kl_loss_lambda = kl_loss_lambda
        self.device = device

    def _sample_batch(self, dataset: Any) -> Any:
        if hasattr(dataset, "sample"):
            return dataset.sample(self.batch_size)

        if isinstance(dataset, list):
            if len(dataset) == 0:
                raise ValueError("dataset is empty")
            k = min(self.batch_size, len(dataset))
            return random.sample(dataset, k)

        raise TypeError("dataset must be replay-buffer-like (.sample) or list of dict samples")

    def _to_action_index_tensor(self, actions: Optional[Any], batch_size: int) -> torch.Tensor:
        if actions is None:
            return torch.full((batch_size,), -1, dtype=torch.long)

        t = torch.as_tensor(actions)
        if t.ndim == 0:
            return t.long().view(1)
        if t.ndim > 1:
            # Supports one-hot or logits-like arrays by converting to class index.
            if t.shape[-1] > 1:
                return torch.argmax(t, dim=-1).long().view(-1)
            return t.long().view(-1)
        return t.long().view(-1)

    def _normalize_batch(
        self,
        batch: Any,
        policy: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            observations = torch.as_tensor(batch["observation"], dtype=torch.float32)
            if observations.ndim == 1:
                observations = observations.unsqueeze(0)

            bsz = observations.shape[0]
            expert_actions = self._to_action_index_tensor(batch.get("action"), bsz)
            faulty_actions = self._to_action_index_tensor(batch.get("faulty_action"), bsz)

            action_masks_raw = batch.get("action_mask")
            if action_masks_raw is None:
                with torch.no_grad():
                    n_actions = policy(observations[:1].to(self.device)).shape[-1]
                action_masks = torch.ones((bsz, n_actions), dtype=torch.bool)
            else:
                action_masks = torch.as_tensor(action_masks_raw).bool()
                if action_masks.ndim == 1:
                    action_masks = action_masks.unsqueeze(0)
                    
            print(expert_actions)
            print(faulty_actions)

            return observations, expert_actions, faulty_actions, action_masks

        if isinstance(batch, list):
            if len(batch) == 0:
                raise ValueError("sampled batch is empty")

            obs_list: List[torch.Tensor] = []
            masks_list: List[Optional[torch.Tensor]] = []
            expert_list: List[int] = []
            faulty_list: List[int] = []

            for item in batch:
                obs_list.append(torch.as_tensor(item["observation"], dtype=torch.float32))

                mask = item.get("action_mask")
                masks_list.append(None if mask is None else torch.as_tensor(mask).bool())

                expert = item.get("action")
                if expert is None:
                    expert_list.append(-1)
                else:
                    expert_idx = self._to_action_index_tensor(expert, 1)
                    expert_list.append(int(expert_idx.view(-1)[0].item()))

                faulty = item.get("faulty_action", -1)
                faulty_list.append(int(faulty))

            observations = torch.stack(obs_list, dim=0)
            expert_actions = torch.tensor(expert_list, dtype=torch.long)
            faulty_actions = torch.tensor(faulty_list, dtype=torch.long)

            if any(m is None for m in masks_list):
                with torch.no_grad():
                    n_actions = policy(observations[:1].to(self.device)).shape[-1]
                action_masks = torch.ones((len(batch), n_actions), dtype=torch.bool)
            else:
                action_masks = torch.stack([m for m in masks_list if m is not None], dim=0)

            print(expert_actions)
            print(faulty_actions)

            return observations, expert_actions, faulty_actions, action_masks

        raise TypeError("sampled batch must be dict or list")

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
        

    def update_policy_old(self, policy: torch.nn.Module, dataset: Any) -> Dict[str, float]:
        policy.train()
        policy.to(self.device)

        # Frozen reference used for retain regularization.
        reference_policy = copy.deepcopy(policy).to(self.device)
        reference_policy.eval()

        total_loss = 0.0
        total_forget_loss = 0.0
        total_retain_loss = 0.0
        total_kl_loss = 0.0
        steps_with_forget = 0

        for _ in range(self.steps_per_iteration):
            batch = self._sample_batch(dataset)
            observations, expert_actions, faulty_actions, action_masks = self._normalize_batch(batch, policy)

            observations = observations.to(self.device)
            expert_actions = expert_actions.to(self.device)
            faulty_actions = faulty_actions.to(self.device)
            action_masks = action_masks.to(self.device)

            logits = policy(observations)
            masked_logits = logits.masked_fill(~action_masks.bool(), float("-inf"))
            logp = torch.log_softmax(masked_logits, dim=-1)

            with torch.no_grad():
                ref_logits = reference_policy(observations)
                ref_masked_logits = ref_logits.masked_fill(~action_masks.bool(), float("-inf"))
                ref_probs = torch.softmax(ref_masked_logits, dim=-1)

            valid_faulty = faulty_actions >= 0
            valid_expert = expert_actions >= 0
            retain_subset = ~valid_faulty

            if valid_faulty.any():
                forget_nll = F.nll_loss(logp[valid_faulty], faulty_actions[valid_faulty], reduction="mean")
                forget_loss = -forget_nll
                steps_with_forget += 1
            else:
                forget_loss = torch.tensor(0.0, device=self.device)

            if valid_expert.any():
                retain_supervised = F.nll_loss(logp[valid_expert], expert_actions[valid_expert], reduction="mean")
            else:
                retain_supervised = torch.tensor(0.0, device=self.device)

            if retain_subset.any():
                kl_loss = F.kl_div(logp[retain_subset], ref_probs[retain_subset], reduction="batchmean")
            else:
                kl_loss = torch.tensor(0.0, device=self.device)

            retain_loss = retain_supervised + self.kl_loss_lambda * kl_loss
            loss = self.forget_loss_lambda * forget_loss + self.retain_loss_lambda * retain_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_forget_loss += float(forget_loss.item())
            total_retain_loss += float(retain_supervised.item())
            total_kl_loss += float(kl_loss.item())

        denom = float(max(self.steps_per_iteration, 1))
        return {
            "loss": total_loss / denom,
            "forget_loss": total_forget_loss / denom,
            "retain_supervised_loss": total_retain_loss / denom,
            "retain_kl_loss": total_kl_loss / denom,
            "steps": int(self.steps_per_iteration),
            "steps_with_forget": int(steps_with_forget),
        }
