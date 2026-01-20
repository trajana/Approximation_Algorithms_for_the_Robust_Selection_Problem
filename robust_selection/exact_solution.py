# exact_solution.py

# Exact solution for the Robust Selection Problem with discrete uncertainty and using the min-max criterion.

# Description: There are n items with cost c[s,i]. The goal is to pick exactly p items such that the wost-case cost is
# minimized. The problem is formulated as a Mixed Integer Linear Program using the epigraph-reformulation.

import gurobipy as gp
from gurobipy import GRB


def solve_exact_robust_selection(costs, n, p, k, debug=False):
    try:

        # Create optimization model
        m = gp.Model("exact_robust_selection")

        # Create variables
        x = m.addVars(range(1, n + 1), vtype=GRB.BINARY, name="x")  # Binary variable for selection
        z = m.addVar(name="z")  # Continuous variable for the worst-case cost

        # Set objective
        m.setObjective(z, GRB.MINIMIZE)

        # Add constraints
        m.addConstr(gp.quicksum(x[i] for i in range(1, n + 1)) == p, name="select_p_items")  # Select exactly

        m.addConstrs(
            (gp.quicksum(costs[s, i] * x[i] for i in range(1, n + 1)) <= z for s in range(1, k + 1)),
            name="worst_case_cost")  # Worst-case cost constraints

        # Optimize model
        m.optimize()

        # Post-solution checks and debug prints
        if debug:
            print("\n--- Debug: Selected item count ---")
            print(sum(x[i].X for i in range(1, n + 1)))  # should be equal to p
            print("\n--- Debug: Scenario costs ---")
            for s in range(1, k + 1):
                cost_s = sum(costs[s, i] * x[i].X for i in range(1, n + 1))
                print(f"Scenario {s}: total cost = {cost_s}")  # All cost_s ≤ z
            print(f"Max scenario cost (z) = {m.ObjVal}")  # z = max cost_s

        # Extract and return results
        x_val_exact = [x[i].X for i in range(1, n + 1)]
        obj_val_exact = m.ObjVal  # Value of z
        return obj_val_exact, x_val_exact

    # Error handling
    except gp.GurobiError as e:
        raise RuntimeError(
            f"Gurobi failed while solving the model (error code {e.errno}): {e}") from e
    except AttributeError as e:
        raise RuntimeError(
            "Failed to access solution attributes. "
            "This usually means the model was not solved to optimality.") from e
