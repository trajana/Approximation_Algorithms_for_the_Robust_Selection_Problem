# utils.py

# Utility functions for the robust selection problem.

# Includes generation of fixed or random cost matrices, conversion to dictionary format, debugging prints, helper
# functions for primal rounding.
# To use fixed costs, define scenario-specific cost vectors in get_fixed_costs().

import random
import pandas as pd
import pickle


# Fixed costs
def get_fixed_costs(n=None, k=None):
    fixed_costs = [
        [1, 1, 5, 5],  # Scenario 1
        [5, 5, 1, 1],  # Scenario 2
        [1, 1, 5, 5],
        [5, 5, 1, 1]
    ]

    if k is not None and k != len(fixed_costs):
        raise ValueError(f"k = {k}, but fixed costs have {len(fixed_costs)} rows")
    if n is not None and n != len(fixed_costs[0]):
        raise ValueError(f"n = {n}, but fixed costs have {len(fixed_costs[0])} columns")

    return fixed_costs


# Random costs
def get_random_costs(n, k, c_range=100):
    return [[random.randint(1, c_range) for _ in range(n)] for _ in range(k)]


# Print costs in a readable format
def dprint_costs(c, debug=False):
    if not debug:
        return
    k = len(c)
    n = len(c[0])
    for s in range(1, k + 1):
        print(f"Scenario c[{s}]:")
        for i in range(1, n + 1):
            print(f" c[{s}][{i}] = {c[s - 1][i - 1]}")


# Transform cost matrix into dictionary format: (s, i): c
def cost_matrix_to_dict(c):
    return {(s + 1, i + 1): c[s][i] for s in range(len(c)) for i in range(len(c[0]))}


# View all results from a .pkl in a table
def dprint_all_results_from_pkl(pkl_path, debug=False):
    if not debug:
        return
    with open(pkl_path, "rb") as f:
        all_results = pickle.load(f)
    # Convert the list of dictionaries into a pandas DataFrame for tabular display
    df = pd.DataFrame(all_results)
    # Configure pandas to show all columns and unlimited width in the console output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    # Print the DataFrame without the row index
    print(df.to_string(index=False))

def solve_opt_w(costs, n, p, k, feas_tol, select_tol, debug=False):
    # --- Build cost matrix C[s, i] ---
    C = np.zeros((k, n), dtype=np.float64)
    for (s, i), val in costs.items():
        C[s - 1, i - 1] = float(val)

    try:

        # Create optimization model
        m = gp.Model("LP_opt_w")

        # Create variables
        t = m.addVar(lb=0.0, name="t")  # Continuous variable to be minimized
        w_i = m.addVars(range(1, n + 1), name="w_i")  # Continuous variable for weighted cost vector
        beta = m.addVars(range(1, k + 1), lb=0.0, name="beta")  # Continuous variable for scenario weights
        alpha = m.addVars(range(1, k + 1), name="alpha")  # Dual vaiable for linearization
        lambda_ = m.addVars(range (1, k + 1), range(1, n + 1), lb=0.0, name="lambda")  # Dual variable for linearization

        # Set objective
        m.setObjective(t, GRB.MAXIMIZE)

        # Add constraints
        m.addConstrs(
            (w_i[i] == gp.quicksum(beta[s] * C[s-1, i-1] for s in range(1, k + 1)))
            for i in range(1, n + 1)
        )  # lp:compact-w

        m.addConstr(
            gp.quicksum(beta[s] for s in range(1, k+1)) == 1
        )  # lp:compact-simplex

        m.addConstrs(
            (p * alpha[s] - gp.quicksum(lambda_[s, i] for i in range(1, n + 1)) >= 0)
            for s in range(1, k + 1)
        )  # lp:compact-dual1

        m.addConstrs(
            (lambda_[s, i] >= alpha[s] - w_i[i] + t * C[s - 1, i - 1])
            for s in range(1, k + 1) for i in range(1, n + 1)
        )  # lp:compact-dual2

        # Optimize model
        m.optimize()

        # Optimal value
        if m.status != GRB.OPTIMAL:
            raise RuntimeError(f"LP_opt_w not optimal, status={m.status}")

        beta_star = np.array([beta[s].X for s in range(1, k + 1)], dtype=float)
        w_star = np.array([w_i[i].X for i in range(1, n + 1)], dtype=float)
        t_star = float(t.X)
        alpha_star = np.array([alpha[s].X for s in range(1, k + 1)], dtype=float)
        lambda_star = np.array([[lambda_[s, i].X for i in range(1, n + 1)] for s in range(1, k + 1)], dtype=float)

        # --- Check solution ---
        if debug:
            print("\nOptimal t* =", t_star)

            print("\nOptimal beta* (scenario weights):")
            for s in range(1, k + 1):
                print(f"  beta[{s}] = {beta[s].X}")

            print("\nOptimal w* (weighted costs):")
            for i in range(1, n + 1):
                print(f"  w[{i}] = {w_i[i].X}")

            print("\nOptimal alpha* (dual p-subset values per scenario):")
            for s in range(1, k + 1):
                print(f"  alpha[{s}] = {alpha[s].X}")

            if n * k <= 200 and debug:  # optional: nicht alles zuspammen
                print("\nOptimal lambda* (dual selectors):")
                for s in range(1, k + 1):
                    for i in range(1, n + 1):
                        print(f"  lambda[{s},{i}] = {lambda_[s, i].X}")

        return t_star, beta_star, w_star, alpha_star, lambda_star

    # Error handling
    except gp.GurobiError as e:
        raise RuntimeError(
            f"Gurobi failed while solving the model (error code {e.errno}): {e}") from e
    except AttributeError as e:
        raise RuntimeError(
            "Failed to access solution attributes. "
            "This usually means the model was not solved to optimality.") from e
