# primal_rounding.py

# Using Primal Rounding to approximate the Robust Selection Problem with discrete uncertainty and the min-max criterion.

# Description: There are n items with cost c[s,i]. The goal is to pick exactly p items such that the worst-case cost is
# minimized. The problem is formulated as a Mixed Integer Linear Program using the epigraph-reformulation. The
# decision variable x is relaxed to a continuous variable and the solution is rounded to a feasible solution.

import gurobipy as gp
from gurobipy import GRB


def solve_primal_rounding(costs, n, p, k, debug=False):
    try:

        # Create optimization model
        m = gp.Model("primal_rounding")

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

        # Relaxed x-values
        x_val_primal_frac = [x[i].X for i in range(1, n + 1)]
        selected_items_primal = sorted(
            [(i, val) for i, val in enumerate(x_val_primal_frac)],
            key=lambda item: item[1],
            reverse=True
        )[:p]  # Select top p items
        selected_indices_primal = [i for i, _ in selected_items_primal]  # Get indices of selected
        tau = min(x_val_primal_frac[i] for i in selected_indices_primal)
        x_vector_primal_rounded = [1 if i in selected_indices_primal else 0 for i in range(n)]  # Binary vector

        # Post-solution checks and debug prints
        if debug:
            print("\n---Relaxed x-values (fractional):---")
            for i in range(n):
                print(f"x[{i + 1}] = {x_val_primal_frac[i]:.4f}")
            print("\n---Relaxed x-values (binary):---")
            for i in range(n):
                print(f"x[{i + 1}] = {x_vector_primal_rounded[i]}")

        # Compute worst-case cost of rounded solution (results)
        obj_val_primal = max(
            sum(costs[s, i + 1] for i in range(n) if x_vector_primal_rounded[i] == 1)
            for s in range(1, k + 1))  # Computes the maximum cost across all scenarios for the rounded solution

        # Debugging worst case cost
        if debug:
            print("\n--- Scenario costs (rounded solution): ---")
            scenario_costs = []
            for s in range(1, k + 1):
                cost_s = sum(costs[s, i + 1] for i in range(n) if x_vector_primal_rounded[i] == 1)
                scenario_costs.append(cost_s)
                print(f"Scenario {s}: total cost = {cost_s}")
            print(f"\nMax scenario cost (should match obj_val_primal): {max(scenario_costs)}")
            print(f"Returned objective value (obj_val_primal): {obj_val_primal}")

        return obj_val_primal, x_val_primal_frac, x_vector_primal_rounded, obj_val_primal_lp, tau

    # Error handling
    except gp.GurobiError as e:
        raise RuntimeError(
            f"Gurobi failed while solving the model (error code {e.errno}): {e}") from e
    except AttributeError as e:
        raise RuntimeError(
            "Failed to access solution attributes. "
            "This usually means the model was not solved to optimality.") from e
