# dual_approach.py

# Using Dual Approach to approximate the Robust Selection Problem with discrete uncertainty and the min-max
# criterion.

# Description: There are n items with costs c[s,i]. The goal is to select exactly p items such that the worst-case cost
# across k scenarios is minimized. The algorithm solves the dual problem and thereby obtains some weight vecor (ĉ). It
# then solves the nominal problem with respect to this represenative cost scenario.

import gurobipy as gp
from gurobipy import GRB

def solve_dual_approach(costs, n, p, k, debug=False):
    try:

        # Create model
        m = gp.Model("dual_approach")

        # Create dual variables
        alpha = m.addVar(name="alpha")
        beta = m.addVars(range(1, k + 1),  vtype=GRB.CONTINUOUS, name="beta_s")
        gamma = m.addVars(range(1, n + 1), vtype=GRB.CONTINUOUS, name="gamma")

        # Set objective
        m.setObjective(p * alpha - gp.quicksum(gamma[i] for i in range(1, n + 1)), GRB.MAXIMIZE)

        # Add constraints
        m.addConstrs((alpha - gp.quicksum(costs[s, i] * beta[s] for s in range(1, k + 1)) <= gamma[i] for i in range(1, n + 1)), name="item constraints")
        m.addConstr(gp.quicksum(beta[s] for s in range(1, k + 1)) == 1)

        # Optimize model
        m.optimize()
        obj_val_dual_lp = m.ObjVal

        beta_vals = m.getAttr("x", beta)  # optimale β_s

        c_hat = {
            i: sum(costs[s, i] * beta_vals[s] for s in range(1, k + 1))
            for i in range(1, n + 1)
        }

        # --- Select p smallest items by c_hat (nominal problem) ---
        selected_items = sorted(
            c_hat.items(),  # (item_id, c_hat)
            key=lambda kv: (kv[1], kv[0])  # (c_hat, id)
        )[:p]

        selected_indices = [i for i, _ in selected_items]  # 1-based indices

        # --- Build binary selection vector x ---
        x_opt = [1 if (i + 1) in selected_indices else 0 for i in range(n)]

        # --- Evaluate robust objective value ---
        scenario_costs = [
            sum(float(costs[(s, i)]) for i in selected_indices)
            for s in range(1, k + 1)
        ]
        obj_val_dual_nom = max(scenario_costs)

        return c_hat, x_opt, obj_val_dual_lp, obj_val_dual_nom

    # Error handling
    except gp.GurobiError as e:
        raise RuntimeError(
            f"Gurobi failed while solving the model (error code {e.errno}): {e}") from e
    except AttributeError as e:
        raise RuntimeError(
            "Failed to access solution attributes. "
            "This usually means the model was not solved to optimality.") from e








