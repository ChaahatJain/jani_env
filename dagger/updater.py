import torch
from typing import Any, Dict
from .interfaces import PolicyUpdaterInterface
import time
from gurobipy import Model, GRB, quicksum
import numpy as np
import torch.nn as nn

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
            print(good_actions)
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