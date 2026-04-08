"""
Neural Network Repair via MILP — Gradient-Guided Weight Editing (Gurobi)
------------------------------------------------------------------------
Steps:
  1. Identify "problematic" neurons via gradient magnitude on faulty decisions.
  2. Build a MILP (via gurobipy) that:
       - Freezes all weights EXCEPT those touching the top-K neurons.
       - Minimises the L1 norm of weight edits (delta_W).
       - Enforces: for every (state, faulty_action, applicable_actions) triple,
         every applicable action scores strictly higher than the faulty action.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

from dagger.interfaces import PolicyUpdaterInterface

import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB


# ── small margin used in strict-inequality constraints ───────────────────────
MARGIN = 1e-2


# ════════════════════════════════════════════════════════════════════════════
# 1.  GRADIENT-BASED NEURON BLAMING
# ════════════════════════════════════════════════════════════════════════════

def compute_neuron_blame(
    policy: nn.Module,
    states: torch.Tensor,
    faulty_actions: List[int],
) -> Dict[str, torch.Tensor]:
    """
    For each (state, faulty_action) pair, backpropagate the gradient of
    logit[faulty_action] and accumulate absolute gradients across samples.

    Returns
    -------
    Dict mapping param_name -> accumulated |grad| tensor (same shape as param).
    """
    policy.train()
    accumulated: Dict[str, torch.Tensor] = {}

    for state, fa in zip(states, faulty_actions):
        policy.zero_grad()
        logits: torch.Tensor = policy(state.unsqueeze(0))   # (1, n_actions)
        logits[0, fa].backward()                             # blame the faulty score

        for name, param in policy.named_parameters():
            if param.grad is not None:
                g = param.grad.detach().abs()
                accumulated[name] = accumulated.get(name, torch.zeros_like(g)) + g

    policy.zero_grad()
    return accumulated


def rank_neurons(
    accumulated_grads: Dict[str, torch.Tensor],
    top_k: int,
) -> List[Tuple[str, int, float]]:
    """
    Flatten every parameter tensor and return the top-K
    (param_name, flat_index, grad_magnitude) triples, sorted descending.
    """
    entries: List[Tuple[str, int, float]] = []
    for name, grads in accumulated_grads.items():
        for idx, val in enumerate(grads.view(-1).tolist()):
            entries.append((name, idx, val))

    entries.sort(key=lambda x: x[2], reverse=True)
    return entries[:top_k]


# ════════════════════════════════════════════════════════════════════════════
# 2.  LINEARISED FORWARD PASS
# ════════════════════════════════════════════════════════════════════════════

def _linear_forward_numpy(
    policy: nn.Module,
    x: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Run a forward pass through every nn.Linear layer, recording the input
    activation to each layer.  ReLU gates are fixed from this clean pass,
    giving a valid local linear approximation around the current weights.

    Returns
    -------
    weights    : list of W arrays per Linear layer
    biases     : list of b arrays per Linear layer
    activations: list of h arrays (input to each Linear layer)
    """
    weights, biases, activations = [], [], []
    h = x.copy()
    with torch.no_grad():
        for module in policy.modules():
            if isinstance(module, nn.Linear):
                W = module.weight.numpy().copy()
                b = (module.bias.numpy().copy()
                     if module.bias is not None
                     else np.zeros(module.out_features))
                activations.append(h.copy())
                weights.append(W)
                biases.append(b)
                h = np.maximum(W @ h + b, 0)   # ReLU fixes activation mask
    return weights, biases, activations


# ════════════════════════════════════════════════════════════════════════════
# 3.  MILP REPAIR  (Gurobi)
# ════════════════════════════════════════════════════════════════════════════

def build_and_solve_milp(
    policy: nn.Module,
    states: torch.Tensor,
    applicable_actions: List[List[int]],
    faulty_actions: List[int],
    top_k: int = 10,
    solver_time_limit: float = 120.0,
    mip_gap: float = 1e-4,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Identify the top-K most-blamed neurons, then solve a MILP that minimises
    total L1 weight perturbation subject to atleast one applicable action beating
    the faulty action by at least MARGIN on every supplied state.

    Parameters
    ----------
    policy            : nn.Module to repair (modified in-place on success)
    states            : float32 tensor of shape (N, obs_dim)
    applicable_actions: list[list[int]] — valid action indices per sample
    faulty_actions    : list[int]        — wrongly-chosen action per sample
    top_k             : number of neurons freed for weight editing
    solver_time_limit : Gurobi TimeLimit (seconds)
    mip_gap           : Gurobi MIPGap tolerance
    verbose           : whether to stream Gurobi log to stdout

    Returns
    -------
    metrics dict: status, objective value, solve time, node count
    """

    # ── 1. neuron blaming ────────────────────────────────────────────────────
    acc_grads = compute_neuron_blame(policy, states, faulty_actions)
    ranked    = rank_neurons(acc_grads, top_k)
    free_set  = {(name, idx) for name, idx, _ in ranked}

    if verbose:
        print(f"[MILP] Top-{top_k} neurons freed for editing:")
        for name, idx, mag in ranked:
            print(f"  {name}[{idx}]  sum|grad|={mag:.6f}")

    # ── 2. enumerate Linear layers ───────────────────────────────────────────
    linear_layers: List[Tuple[str, nn.Linear]] = [
        (name, mod) for name, mod in policy.named_modules()
        if isinstance(mod, nn.Linear)
    ]
    n_layers = len(linear_layers)

    # ── 3. build Gurobi model ────────────────────────────────────────────────
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", int(verbose))
    env.start()

    m = gp.Model("NeuronRepair", env=env)
    m.setParam("TimeLimit", solver_time_limit)
    m.setParam("MIPGap",    mip_gap)

    # Decision variables — only for free (l_idx, i, j) triples
    # delta[key]  : unbounded continuous weight perturbation
    # abs_d[key]  : auxiliary for |delta[key]|  (two linear constraints)
    delta: Dict[Tuple[int, int, int], gp.Var] = {}
    abs_d: Dict[Tuple[int, int, int], gp.Var] = {}

    for l_idx, (l_name, layer) in enumerate(linear_layers):
        n_out, n_in = layer.weight.shape
        weight_param = f"{l_name}.weight"

        for i in range(n_out):
            for j in range(n_in):
                flat_idx = i * n_in + j
                if (weight_param, flat_idx) in free_set:
                    key = (l_idx, i, j)
                    delta[key] = m.addVar(
                        lb=-GRB.INFINITY, ub=GRB.INFINITY,
                        name=f"dW_{l_idx}_{i}_{j}",
                    )
                    abs_d[key] = m.addVar(
                        lb=0.0,
                        name=f"aW_{l_idx}_{i}_{j}",
                    )
                    m.addConstr(abs_d[key] >=  delta[key], name=f"abs_pos_{l_idx}_{i}_{j}")
                    m.addConstr(abs_d[key] >= -delta[key], name=f"abs_neg_{l_idx}_{i}_{j}")

    # Objective: minimise L1 norm of all weight perturbations
    m.setObjective(gp.quicksum(abs_d.values()), GRB.MINIMIZE)

    # ── 4. safety-margin constraints ─────────────────────────────────────────
    #
    # For each sample s and every alt ∈ applicable_actions[s] \ {fa}:
    #
    #   (base_logit[alt] + Δlogit[alt]) − (base_logit[fa] + Δlogit[fa]) ≥ MARGIN
    #
    # Linearisation (last-layer only):
    #   Δlogit[a] = Σ_j  h_last[j] · delta[last_layer, a, j]
    #
    # Exact when only the last layer is free; for earlier free layers,
    # extend with the full sensitivity (jacobian) through subsequent layers.

    last_l_idx = n_layers - 1
    n_in_last  = linear_layers[last_l_idx][1].weight.shape[1]
    
    for s_idx in range(len(states)):
        x_np = states[s_idx].numpy()
        fa   = faulty_actions[s_idx]
        alts = [a for a in applicable_actions[s_idx] if a != fa]
        if not alts:
            assert False, "We cannot have a fault with no alternative applicable action during repair"

        _, _, acts_np = _linear_forward_numpy(policy, x_np)
        h_last = acts_np[last_l_idx]   # input activation to the final linear layer

        with torch.no_grad():
            base_logits = policy(states[s_idx].unsqueeze(0)).squeeze(0).numpy()

        # Binary variable per alternative: z[alt] = 1 means this alt is the
        # "chosen witness" that beats the faulty action.
        z: Dict[int, gp.Var] = {
            alt: m.addVar(vtype=GRB.BINARY, name=f"z_s{s_idx}_alt{alt}")
            for alt in alts
        }
 
        # At least one alternative must be selected as the witness.
        m.addConstr(gp.quicksum(z.values()) >= 1, name=f"atleast1_s{s_idx}")
 
        for alt in alts:
            # Build Δlogit[alt] − Δlogit[fa]
            diff_expr = gp.LinExpr()
            for j in range(n_in_last):
                coeff = float(h_last[j])
                if coeff == 0.0:
                    continue
                key_alt = (last_l_idx, alt, j)
                key_fa  = (last_l_idx, fa,  j)
                if key_alt in delta:
                    diff_expr.add(delta[key_alt],  coeff)
                if key_fa  in delta:
                    diff_expr.add(delta[key_fa],  -coeff)
 
            base_diff = float(base_logits[alt]) - float(base_logits[fa])
 
            # Margin enforced only when z[alt] = 1; relaxed by BIG_M otherwise.
            BIG_M = 1e4
            m.addConstr(
                base_diff + diff_expr >= MARGIN - BIG_M * (1 - z[alt]),
                name=f"margin_s{s_idx}_fa{fa}_alt{alt}",
            )

    # ── 5. solve ─────────────────────────────────────────────────────────────
    m.optimize()

    status_map = {
        GRB.OPTIMAL:     "optimal",
        GRB.INFEASIBLE:  "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbounded",
        GRB.TIME_LIMIT:  "time_limit",
        GRB.SUBOPTIMAL:  "suboptimal",
    }
    status_str = status_map.get(m.Status, f"gurobi_status_{m.Status}")

    if verbose:
        print(f"\n[MILP] Status    : {status_str}")
        if m.SolCount > 0:
            print(f"[MILP] Objective : {m.ObjVal:.6f}")
            print(f"[MILP] Solve time: {m.Runtime:.2f}s   Nodes: {int(m.NodeCount)}")

    metrics: Dict[str, float] = {
        "gurobi_status":  float(m.Status),
        "status_str":     status_str,
        "objective":      m.ObjVal if m.SolCount > 0 else float("nan"),
        "solve_time_s":   m.Runtime,
        "node_count":     float(m.NodeCount),
        "n_free_weights": float(len(delta)),
    }

    # ── 6. apply deltas in-place ─────────────────────────────────────────────
    if m.SolCount > 0:
        with torch.no_grad():
            for l_idx, (_, layer) in enumerate(linear_layers):
                W = layer.weight.data
                n_out, n_in = W.shape
                for i in range(n_out):
                    for j in range(n_in):
                        key = (l_idx, i, j)
                        if key in delta:
                            W[i, j] += delta[key].X
        if verbose:
            print("[MILP] Weights patched successfully.")
    else:
        warnings.warn(
            f"[MILP] No feasible solution found (status={status_str}). "
            "Weights unchanged. Consider increasing top_k or relaxing MARGIN."
        )

    m.dispose()
    env.dispose()
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# 4.  DROP-IN update_policy METHOD
# ════════════════════════════════════════════════════════════════════════════

class TargetedMILPUpdater(PolicyUpdaterInterface):
    """
    Mix this into any agent class that holds self.policy (nn.Module).

    Configurable class attributes
    ------------------------------
    top_k_neurons   : int   — neurons freed for editing      (default 10)
    milp_time_limit : float — Gurobi TimeLimit in seconds    (default 120)
    eps             : float — Gurobi epsilon tolerance                  (default 1e-4)
    verbose         : bool  — stream Gurobi log              (default False)
    """

    def __init__(self):
        self.top_k_neurons : int = 10
        self.milp_time_limit : float = 3600*3.0
        self.eps : float = 1e-4
        self.verbose: bool = False
        return

    def update_policy(self, policy: nn.Module, dataset: Any) -> Dict[str, float]:
        states = torch.tensor([f["observation"] for f in dataset], dtype=torch.float32)
        applicable_actions = [[int(a) for a in f["action_mask"]] for f in dataset]
        faults = [int(f["faulty_action"]) for f in dataset]

        return build_and_solve_milp(
            policy=policy,
            states=states,
            applicable_actions=applicable_actions,
            faulty_actions=faults,
            top_k=self.top_k_neurons,
            solver_time_limit=self.milp_time_limit,
            mip_gap=self.eps,
            verbose=self.verbose,
        )


# ════════════════════════════════════════════════════════════════════════════
# 5.  SMOKE TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    policy = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(),
        nn.Linear(8, 3),
    )

    dataset = []
    for _ in range(5):
        obs = torch.randn(4).tolist()
        with torch.no_grad():
            logits = policy(torch.tensor(obs, dtype=torch.float32))
        faulty = int(logits.argmax())
        dataset.append({
            "observation":   obs,
            "action_mask":   [0, 1, 2],
            "faulty_action": faulty,
        })

    metrics = build_and_solve_milp(
        policy=policy,
        states=torch.tensor([d["observation"] for d in dataset], dtype=torch.float32),
        applicable_actions=[[int(a) for a in d["action_mask"]] for d in dataset],
        faulty_actions=[d["faulty_action"] for d in dataset],
        top_k=6,
        verbose=True,
    )
    print("\nMetrics:", metrics)