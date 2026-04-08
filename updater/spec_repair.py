import torch
from typing import Any, Dict
from dagger.interfaces import PolicyUpdaterInterface
import time
import numpy as np
import torch.nn as nn
import random
    
class SpecRepairPolicyUpdater(PolicyUpdaterInterface):
    """
    Updates a neural network policy using the SpecRepair L1 penalty method.

    Penalty weights for each counterexample are doubled whenever the fault
    persists after a training step, following SpecRepair's adaptive penalty schedule.
    Training continues until all faults in the batch are resolved or the
    penalty weights exceed a maximum threshold.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, batch_size: int = 256, initial_penalty_weight: float = 1.0, max_penalty_weight: float = 16384.0, margin: float = 0.0, device: torch.device = torch.device("cpu")):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.initial_penalty_weight = initial_penalty_weight
        self.max_penalty_weight = max_penalty_weight
        self.margin = margin
        self.device = device
        self.task_loss_fn = torch.nn.NLLLoss()

    def _check_faults(self, logits: torch.Tensor, faulty_actions: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask [N] that is True where the fault persists,
        i.e. where faulty_action is still the argmax of the logits.
        """
        predicted_actions = logits.argmax(dim=-1)
        return predicted_actions == faulty_actions                  # [N]

    def _penalty_loss(self, logits: torch.Tensor, faulty_actions: torch.Tensor, penalty_weights: torch.Tensor) -> torch.Tensor:
        n = logits.size(0)
        faulty_logits = logits[torch.arange(n), faulty_actions]

        # Mask out the faulty action
        masked_logits = logits.clone()
        masked_logits[torch.arange(n), faulty_actions] = float("-inf")

        # Find the best alternative; if all are -inf, we keep -inf
        best_other_logits = masked_logits.max(dim=-1).values

        # Penalty is only defined where there is at least one valid alternative
        has_alternative = best_other_logits != float("-inf")

        per_sample_penalty = torch.relu(faulty_logits - best_other_logits + self.margin)
        per_sample_penalty = torch.where(has_alternative, per_sample_penalty, torch.tensor(0.0, device=logits.device))

        # Also ignore samples where faulty_action is itself invalid
        faulty_is_valid = faulty_logits != float("-inf")
        per_sample_penalty = per_sample_penalty * faulty_is_valid.float()

        return (penalty_weights * per_sample_penalty).mean()

    def update_policy(self, policy: torch.nn.Module, dataset: Any) -> Dict[str, float]:
        policy.train()
        policy.to(self.device)

        batch = random.sample(dataset, min(self.batch_size, len(dataset)))

        observations = torch.stack([torch.from_numpy(item["observation"]) for item in batch]).to(self.device)
        # expert_actions = torch.stack([torch.from_numpy(item["action"]) for item in batch]).to(self.device)
        action_masks = torch.stack([torch.from_numpy(item["action_mask"]) for item in batch]).to(self.device)
        faulty_actions = torch.tensor([item["faulty_action"] for item in batch], dtype=torch.long).to(self.device)

        # Count how many actions are available per row
        valid_counts = action_masks.sum(dim=1)

        # Keep only rows with more than 1 valid action
        keep = valid_counts > 1

        # Apply filter
        observations = observations[keep]
        action_masks = action_masks[keep]
        faulty_actions = faulty_actions[keep]


        # Per-sample penalty weights, doubled whenever a fault persists
        penalty_weights = torch.full(
            (observations.size(0),), self.initial_penalty_weight,
            device=self.device
        )

        total_loss = 0.0
        total_task_loss = 0.0
        total_penalty_loss = 0.0
        steps = 0
        iterations = 0
        while True and iterations < 1000:
            iterations = iterations + 1
            logits = policy(observations)
            masked_logits = logits.masked_fill(~action_masks.bool(), float("-inf"))
            logp = torch.log_softmax(masked_logits, dim=-1)

            with torch.no_grad():
                still_faulty = self._check_faults(masked_logits, faulty_actions.long())

            if not still_faulty.any():
                break

            print(masked_logits[still_faulty][0], faulty_actions[still_faulty][0])
            # task_loss = self.task_loss_fn(logp, expert_actions.long())
            penalty_loss = self._penalty_loss(masked_logits, faulty_actions.long(), penalty_weights)
            loss = penalty_loss
            print("Penalty Loss iteration", iterations, "is", penalty_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Re-evaluate which faults still persist AFTER the gradient step
            with torch.no_grad():
                post_step_logits = policy(observations)
                post_step_masked = post_step_logits.masked_fill(~action_masks.bool(), float("-inf"))
                still_faulty_after = self._check_faults(post_step_masked, faulty_actions.long())

            # Double penalty weights for samples whose fault persists after the step
            penalty_weights = torch.where(
                still_faulty_after,
                (penalty_weights * 2.0).clamp(max=self.max_penalty_weight),
                penalty_weights
            )

            # total_task_loss += task_loss.item()
            total_penalty_loss += penalty_loss.item()
            total_loss += loss.item()
            steps += 1

            if (penalty_weights >= self.max_penalty_weight).all():
                break

        return {
            "loss": total_loss / max(steps, 1),
            # "task_loss": total_task_loss / max(steps, 1),
            "penalty_loss": total_penalty_loss / max(steps, 1),
            "steps": steps,
            "faults_remaining": still_faulty.sum().item() if steps > 0 else 0,
        }