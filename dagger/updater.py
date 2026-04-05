import torch
from typing import Any, Dict
from .interfaces import PolicyUpdaterInterface
import time
from gurobipy import Model, GRB, quicksum
import numpy as np
import torch.nn as nn
import random

class MILPPolicyUpdater(PolicyUpdaterInterface):
    """Updates a neural network using a Minimal Modifications MILP encoding from Goldberger etal.
    """
    GUROBI_TIMEOUT = 3600*3
    
    def get_linear_problem(self, yforb, applicable_actions, hidden_values, final_layer_weights, biases, eps=1e-4, verbose=False):
        "Builds a linear program to tune weights at the final layer adding constraints."
        from contextlib import redirect_stdout
        from io import StringIO

        # print("Shapes: ")
        # print("\t-phi: ", phi.shape)
        # print("\t-W: ", W.shape)
        t = time.time()

        # 1. Define the optimization problem
        with redirect_stdout(StringIO()): 
            model = Model()

        # 2. Variables to optimize: new leaf values
        M = model.addVar(lb = 0, name="M")
        # weight variables
        n_outputs, n_hidden = final_layer_weights.shape
        print("Num output is:", n_outputs)
        print("Num hidden is", n_hidden)
        w = model.addVars(n_outputs, n_hidden, lb=-GRB.INFINITY, name="w")

        # add constraints: -M <= w_ij <= M
        for i in range(n_outputs):
            for j in range(n_hidden):
                model.addConstr(w[i, j] - final_layer_weights[i, j] <= M, name=f"w_upper_{i}_{j}")
                model.addConstr(w[i, j] - final_layer_weights[i, j] >= -M, name=f"w_lower_{i}_{j}")    

        # 3. Objective function: minimize absolute value difference btw new and old weights (L_infinity distance metric)
        model.setObjective(M, GRB.MINIMIZE)
        
        # 4. Constraints: Avoid given state-value pairs (x, y)
        
        def add_output_constraints(hidden, forbidden_action, app_actions, index=0):
            good_actions = list(set(g for g in app_actions if g != forbidden_action))
            # print(good_actions)
            z = model.addVars(good_actions, vtype=GRB.BINARY, name=f"z_{index}") # binary variables for the disjunction

            # at least one must hold
            model.addConstr(quicksum(z[g] for g in good_actions) >= 1)

            for g in good_actions:
                good_expr = quicksum(w[g,j] * hidden[j] for j in range(n_hidden)) + biases[g]
                bad_expr  = quicksum(w[forbidden_action, j] * hidden[j] for j in range(n_hidden)) + biases[forbidden_action]

                # indicator constraint
                model.addGenConstrIndicator(z[g], True, good_expr >= bad_expr + eps)
        
        index = 0
        for hidden, forbidden_action, app_actions in zip(hidden_values, yforb, applicable_actions):
            add_output_constraints(hidden, forbidden_action, app_actions, index) 
            index += 1          
        
        
        if verbose:
            # print total number of constraints per type
            num_constraints = model.numConstrs
            print(f"\t\t- total constraints: {num_constraints}")
        return model, w


    def solve_linear_problem(self, m, final_layer_weights, new_weights):
        """ Solves the linear problem. """
        m.setParam('OutputFlag', 0) # silent mode
        # m.setParam('FeasibilityTol', 1e-9) # avoid minimal non-faulty leaf changes
        m.setParam('TimeLimit', 3600) # set max time for gurobi
        m.setParam('Seed', 10)  # Set a seed for reproducibility
        # m.setParam("OptimalityTol", 1e-8)
        # m.setParam("IntFeasTol", 1e-9)
        m.setParam('NumericFocus', 3) # 0-3 increasingly more careful for numerical issues
        # we want quick feasible solutions
        m.Params.MIPGap = 0.9  # Allow a 90% gap for suboptimal solutions
        m.setParam('MIPFocus', 1)  # Focus on finding feasible solutions quickly
        m.setParam('Presolve', 2)  # Aggressive presolve
        # m.setParam('Cuts', 0)      # Disable cuts to reduce overhead
        m.setParam('Heuristics', 0.5)   # Increase heuristic effort (default is 0.05)
        m.setParam('Threads', 8)   # Use more threads if available

        print("\t\t- optimization begins")
        m.optimize()

        # init. status return string
        status = "unknown"

        # Check the optimization status and output
        if m.status == GRB.OPTIMAL:
            print(f"\t\t- optimization done -->",
                f"obj. value: {m.objVal:.3f},",
                f"time: {m.Runtime:.3f} s")
            status = "optimal"
        elif m.status == GRB.INFEASIBLE:
            print(f"\t\t- problem INFEASIBLE! (runtime: {m.Runtime:.3f} s)")

            # # Analyze infeasible instances (now in ad hoc file)
            # m.computeIIS()
        
            # # Save to a file (optional)
            # m.write("model.ilp")  # or "model.iis" in LP format
            
            # # Print out the IIS
            # for c in m.getConstrs():
            #     if c.IISConstr:  # This constraint is part of the IIS
            #         print(f"Infeasible constraint: {c.constrName}")
        
            status = "infeasible"
            return status, None
        # elif m.status == GRB.TIME_LIMIT:
        #     print(f"\t\t- gurobi TIMED OUT!")
        #     status = "timeout"
        #     return status, None
        elif m.status == GRB.TIME_LIMIT:
            print(f"\t\t- gurobi TIMED OUT!")
            if m.SolCount > 0:
                print(f"\t\t- suboptimal solution found (obj: {m.ObjVal:.3f})")
                status = "suboptimal"
            else:
                return "timeout", None
        else:
            print(f"\t\t- optimization ended with status {m.status}.")
            raise RuntimeError(f"Optimization ended with unexpected status {m.status}.!")
            # exit()

        # Extract optimized weights
        optimized_weights = np.array(
            [new_weights[i, j].X 
            for i in range(final_layer_weights.shape[0]) 
            for j in range(final_layer_weights.shape[1])]
        ).reshape(final_layer_weights.shape)
        return status, optimized_weights
    
    def __init__(self):
        return

    def update_policy(self, policy: torch.nn.Module, dataset: Any) -> Dict[str, float]:
        states = torch.tensor([f["observation"] for f in dataset], dtype=torch.float32) 
        applicable_actions = [[int(a) for a in f["action_mask"]] for f in dataset]
        faults = [int(f["faulty_action"]) for f in dataset]
        
        feature_extractor = policy.model[:-1]  # [..., Linear(64,64), ReLU]
        head = policy.model[-1]                # Linear(64, output_dim)
        hidden = feature_extractor(states).detach()  # shape: [batch, 64]
        model, new_weights = self.get_linear_problem(yforb=faults, applicable_actions=applicable_actions, hidden_values=hidden, final_layer_weights=head.weight.detach().numpy(), biases=head.bias.detach().numpy(), eps=1e-4, verbose=False)
        status, updated_weights = self.solve_linear_problem(model, head.weight.detach().numpy(), new_weights)  
        if updated_weights is not None:
            head.weight = nn.Parameter(torch.tensor(updated_weights, dtype=torch.float32))      

        def test_fix():
            logits = policy(states)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for i, valid_actions in enumerate(applicable_actions):
                mask[i, valid_actions] = True
            masked_logits = logits.masked_fill(~mask, float('-inf'))
            predicted_actions = torch.log_softmax(masked_logits, dim=-1).argmax(dim=-1)
            faults_tensor = torch.tensor(faults)
            assert (predicted_actions != faults_tensor).all(), "Some faulty actions are still being chosen!"
        
        test_fix()
            
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