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

        # --- NEW: allow n to be int OR list of item ids ---
        if isinstance(n, int):
            items = list(range(1, n + 1))
            n_total = n
        else:
            items = list(n)
            n_total = len(items)

        # Create variables
        x = m.addVars(items, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")  # Continuous variable for selection
        z = m.addVar(name="z")  # Continuous variable for the worst-case cost

        # Set objective
        m.setObjective(z, GRB.MINIMIZE)

        # Add constraints
        m.addConstr(gp.quicksum(x[i] for i in items) == p, name="select_p_items")  # Select exactly p
        m.addConstrs(
            (gp.quicksum(costs[s, i] * x[i] for i in items) <= z for s in range(1, k + 1)),
            name="worst_case_cost")

        # Optimize model
        m.optimize()

        obj_val_primal_lp = m.ObjVal

        # Relaxed x-values
        x_frac_dict = {i: x[i].X for i in items}
        x_val_primal_frac = [x_frac_dict[i] for i in items]

        selected_items_primal = sorted(
            [(i, x_frac_dict[i]) for i in items],
            key=lambda item: item[1],
            reverse=True
        )[:p]  # Select top p items

        selected_indices_primal = [i for i, _ in selected_items_primal]  # Get indices of selected
        tau = min(x_frac_dict[i] for i in selected_indices_primal) if selected_indices_primal else None

        # --- keep old behavior: return a length-n binary vector only if items are 1..n ---
        if isinstance(n, int):
            x_vector_primal_rounded = [1 if (i + 1) in selected_indices_primal else 0 for i in range(n_total)]  # Binary vector
        else:
            # if called with a pool, return binary vector over that pool (same style as before: 0..len(pool)-1)
            x_vector_primal_rounded = [1 if items[j] in selected_indices_primal else 0 for j in range(n_total)]

        # Post-solution checks and debug prints
        if debug:
            print("\n---Relaxed x-values (fractional):---")
            for j in range(n_total):
                print(f"x[{items[j]}] = {x_val_primal_frac[j]:.4f}")

            print("\n---Relaxed x-values (binary):---")
            for j in range(n_total):
                print(f"x[{items[j]}] = {x_vector_primal_rounded[j]}")

        # Compute worst-case cost of rounded solution (results)
        obj_val_primal = max(
            sum(costs[s, i] for i in selected_indices_primal)
            for s in range(1, k + 1))  # Computes the maximum cost across all scenarios for the rounded solution

        # Debugging worst case cost
        if debug:
            print("\n--- Scenario costs (rounded solution): ---")
            scenario_costs = []
            for s in range(1, k + 1):
                cost_s = sum(costs[s, i] for i in selected_indices_primal)
                scenario_costs.append(cost_s)
                print(f"Scenario {s}: total cost = {cost_s}")
            print(f"\nMax scenario cost (should match obj_val_primal): {max(scenario_costs)}")
            print(f"Returned objective value (obj_val_primal): {obj_val_primal}")

        # --- NEW: add x_frac_dict as extra return at the end (minimal disruption) ---
        return obj_val_primal, x_val_primal_frac, x_vector_primal_rounded, obj_val_primal_lp, tau, x_frac_dict

    # Error handling
    except gp.GurobiError as e:
        raise RuntimeError(
            f"Gurobi failed while solving the model (error code {e.errno}): {e}") from e
    except AttributeError as e:
        raise RuntimeError(
            "Failed to access solution attributes. "
            "This usually means the model was not solved to optimality.") from e


def round_top_p(x_frac, p_rem):
    selected = [i for i, _ in sorted(x_frac.items(), key=lambda kv: (-kv[1], kv[0]))[:p_rem]]
    tau = min(x_frac[i] for i in selected) if selected else None
    return selected, tau


def robust_obj_of_set(costs_original, item_list, k):
    return max(
        sum(float(costs_original[(s, i)]) for i in item_list)
        for s in range(1, k + 1)
    )


def build_shifted_costs(costs, fixed, pool, p_rem, k):
    """Baue neue Kostentabelle costs_shift[(s, i)] mit
       c'_i^s = c_i^s + (sum_{i in fixed} c_i^s) / p_rem.
    """
    if p_rem <= 0:
        raise ValueError("p_rem must be >= 1")

    # Szenarioweise Fixkosten
    fixed_sum = {
        s: sum(float(costs[(s, i)]) for i in fixed) if fixed else 0.0
        for s in range(1, k + 1)
    }

    costs_shift = {}
    for s in range(1, k + 1):
        shift = fixed_sum[s] / p_rem  # gleiche Konstante pro Szenario für alle Pool-Items
        for i in pool:
            costs_shift[(s, i)] = float(costs[(s, i)]) + shift

    return costs_shift


def solve_two_branches_biggest_xi(costs, n, p, k, debug=False):
    """
    One-level branching für PRIMAL ROUNDING.
    Branching item: größtes x_i aus baseline LP.
    IN: i* fix, shift-trick, solve LP auf Rest, runde top (p-1).
    OUT: i* verboten, solve LP auf Rest (original costs), runde top p.
    Wähle Branch nach robuster Zielfunktion (original costs).
    """
    costs_original = costs
    items_all = list(range(1, n+1))

    # --- Baseline LP ---
    _, _, _, z0, _, x_frac0 = solve_primal_rounding(costs_original, items_all, p, k, debug=debug)

    # i_star wählen: "größtes fraktionales"
    eps = 1e-9
    frac_candidates = [(i, xi) for i, xi in x_frac0.items() if eps < xi < 1-eps]
    if frac_candidates:
        i_star, x_i_star = max(frac_candidates, key=lambda kv: (kv[1], -kv[0]))
    else:
        i_star, x_i_star = max(x_frac0.items(), key=lambda kv: (kv[1], -kv[0]))

    # ---------- Branch IN ----------
    pool_in = [i for i in items_all if i != i_star]
    p_rem_in = p - 1
    if p_rem_in < 0:
        raise ValueError("p must be >= 1")

    if p_rem_in == 0:
        S_in = [i_star]
        tau_in = 1.0
    else:
        costs_shift_in = build_shifted_costs(
            costs=costs_original,
            fixed=[i_star],
            pool=pool_in,
            p_rem=p_rem_in,
            k=k
        )
        _, _, _, z_in_lp, _, x_in_frac = solve_primal_rounding(costs_shift_in, pool_in, p_rem_in, k, debug=debug)
        picked_rest_in, tau_in = round_top_p(x_in_frac, p_rem_in)
        S_in = [i_star] + picked_rest_in

    obj_in = robust_obj_of_set(costs_original, S_in, k)

    # ---------- Branch OUT ----------
    pool_out = [i for i in items_all if i != i_star]
    if p > len(pool_out):
        raise ValueError("Cannot select p items after excluding i_star (p > n-1).")

    _, _, _, z_out_lp, _, x_out_frac = solve_primal_rounding(costs_original, pool_out, p, k, debug=debug)
    S_out, tau_out = round_top_p(x_out_frac, p)
    obj_out = robust_obj_of_set(costs_original, S_out, k)

    # ---------- Choose solution by objective ----------
    if (obj_in < obj_out) or (obj_in == obj_out and (tau_in is not None and tau_out is not None and tau_in >= tau_out)):
        chosen = S_in
        obj = obj_in
        branch = "IN"
        tau_solution = tau_in
    else:
        chosen = S_out
        obj = obj_out
        branch = "OUT"
        tau_solution = tau_out

    # "schlechtere Garantie" analog: min(tau_in, tau_out)
    # Achtung: tau ist bei primal rounding ein a-posteriori Faktor (ALG <= (1/tau)*LP)
    tau_guarantee = min(t for t in [tau_in, tau_out] if t is not None)

    x_vec = [1 if i in chosen else 0 for i in range(1, n+1)]

    if debug:
        print("\n=== Two-Branch Primal Rounding (biggest fractional x_i) ===")
        print(f"Baseline z_lp={z0}")
        print(f"i*={i_star} with x*={x_i_star}")
        print(f"IN : S_in={S_in} obj_in={obj_in} tau_in={tau_in}")
        print(f"OUT: S_out={S_out} obj_out={obj_out} tau_out={tau_out}")
        print(f"Chosen branch={branch} chosen={chosen} obj={obj}")
        print(f"tau_solution={tau_solution} tau_guarantee(min)={tau_guarantee}")

    info = {
        "i_star": i_star,
        "x_i_star": x_i_star,
        "z0_lp": z0,
        "tau_in": tau_in,
        "tau_out": tau_out,
        "tau_guarantee": tau_guarantee,
        "branch": branch,
        "S_in": S_in,
        "S_out": S_out,
        "obj_in": obj_in,
        "obj_out": obj_out
    }
    return chosen, x_vec, obj, info
