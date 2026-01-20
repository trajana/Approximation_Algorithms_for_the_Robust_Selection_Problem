# primal_dual_rounding.py

# Using Primal-Dual Rounding to approximate the Robust Selection Problem with discrete uncertainty and the min-max
# criterion.

# Description: There are n items with costs c[s,i]. The goal is to select exactly p items such that the worst-case cost
# across k scenarios is minimized. The algorithm raises a dual variable until constraints become tight and selects items
# accordingly, maintaining dual feasibility. It achieves an approximation guarantee of ≤ 1/β_min (k for uniform weights)

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_primal_dual(costs, n, p, k, feas_tol=1e-12, select_tol=1e-9, debug=False):
    # --- Build cost matrix C[s, i] ---
    C = np.zeros((k, n), dtype=np.float64)
    for (s, i), val in costs.items():
        C[s - 1, i - 1] = float(val)

    # --- Init primal/dual ---
    x = np.zeros(n, dtype=int)
    S = set()
    b = np.ones(k, dtype=np.float64) / k  # uniform weights
    a = 0.0
    w = C.T @ b  # weighted costs
    gamma = np.zeros(n, dtype=np.float64)

    # --- Main loop: select p items ---
    while len(S) < p:
        # Slacks BEFORE raising a (store for defining Q)
        old_sigma = gamma - (a - w)

        # Numerics: clamp tiny negatives to zero
        neg = old_sigma < 0.0
        if np.any(neg):
            old_sigma[neg] = np.where(old_sigma[neg] > -feas_tol, 0.0, old_sigma[neg])

        # Mask for not-yet-selected items
        notS = np.ones(n, dtype=bool)
        if S:
            notS[list(S)] = False

        # Δ = min slack over remaining items
        if not np.any(notS):
            break  # safety
        delta = float(np.min(old_sigma[notS]))
        if -feas_tol < delta < 0.0:
            delta = 0.0
        if delta < -feas_tol:
            raise RuntimeError("Dual slack significantly negative; infeasibility suspected.")

        # Raise a (dual objective increases by (p - |S|)·Δ)
        a += delta

        # Keep selected constraints tight (increase gamma only for S; never decrease)
        if S:
            idx = np.fromiter(S, dtype=int)
            gamma[idx] = np.maximum(gamma[idx], a - w[idx])

        # Newly tight constraints: those with old slack == delta (within tolerance)
        Q = [i for i in range(n) if notS[i] and abs(old_sigma[i] - delta) <= select_tol]
        if not Q:
            # numeric fallback: pick the argmin of old_sigma over remaining
            d = int(np.argmin(np.where(notS, old_sigma, np.inf)))
            Q = [d]

        # --- Debug: check weighted costs of newly tight constraints ---
        if debug:
            vals = w[Q]  # weighted costs of the newly tight indices
            spread = float(np.ptp(vals)) if len(vals) > 1 else 0.0
            dev_from_alpha = float(np.max(np.abs(vals - a)))  # should be ~0 (γ_i=0 for i∉S)

            # new slacks after the α-update
            new_sigma = gamma - (a - w)
            dev_slack = float(np.max(np.abs(new_sigma[Q])))

            print(f"\nIter {len(S)} -> {len(S) + 1}")
            print(f"  |Q|={len(Q)}, a={a:.6g}")
            print(f"  w[Q] = {vals}")
            print(f"  spread(w[Q]) = {spread:.3e}")
            print(f"  max|w[Q] - a| = {dev_from_alpha:.3e}")
            print(f"  max|new_sigma[Q]| = {dev_slack:.3e}")

            if spread <= max(select_tol, 10 * feas_tol) and dev_from_alpha <= 10 * select_tol:
                print("  OK: all newly tight constraints have (numerically) equal w_i ≈ a.")
            else:
                print("  WARN: deviations detected")

        # Choose from Q the item with the smallest weighted value w_i (tie-break by index)
        chosen = min(Q, key=lambda j: (w[j], j))
        x[chosen] = 1
        S.add(chosen)

    # --- Evaluate primal and dual values ---
    scenario_costs = C @ x
    obj_val = float(np.max(scenario_costs))

    # Dual objective: p*a - sum_i gamma_i  (gamma kept minimal & feasible)
    obj_dual = float(p * a - np.sum(gamma))

    return obj_val, x.tolist(), obj_dual


def solve_primal(costs, n, p, k):
    try:

        # Create optimization model
        m = gp.Model("robust_selection_lp")

        # Create variables
        x = m.addVars(range(1, n + 1), vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")  # Continuous variable for selection
        z = m.addVar(name="z")  # Continuous variable for the worst-case cost

        # Set objective
        m.setObjective(z, GRB.MINIMIZE)

        # Add constraints
        m.addConstr(gp.quicksum(x[i] for i in range(1, n + 1)) == p, name="select_p_items")  # Select exactly p
        m.addConstrs(
            (gp.quicksum(costs[s, i] * x[i] for i in range(1, n + 1)) <= z for s in range(1, k + 1)),
            name="worst_case_cost")

        # Optimize model
        m.optimize()
        obj_val_primal_lp = m.ObjVal

        return obj_val_primal_lp

    # Error handling
    except gp.GurobiError as e:
        raise RuntimeError(
            f"Gurobi failed while solving the model (error code {e.errno}): {e}") from e
    except AttributeError as e:
        raise RuntimeError(
            "Failed to access solution attributes. "
            "This usually means the model was not solved to optimality.") from e


def solve_primal_dual_with_lp(costs, n, p, k, debug=False):
    obj_val, x_vec, obj_dual = solve_primal_dual(costs, n, p, k, debug=debug)
    obj_val_primal_lp = solve_primal(costs, n, p, k)  # OPT_LP = OPT_dual
    return obj_val, x_vec, obj_dual, obj_val_primal_lp
