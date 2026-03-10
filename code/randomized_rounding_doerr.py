# randomized_rounding_doerr.py

# Using Dependent Randomized Rounding to approximate the Robust Selection Problem with discrete uncertainty and the
# min-max criterion.

# Description: There are n items with cost c[s,i]. The goal is to pick exactly p items such that the worst-case cost is
# minimized. The problem is formulated using some C>=0, where C is determined using binary search. The
# decision variable x is relaxed to a continuous variable and the solution is rounded to a feasible solution.


import math
import random
import gurobipy as gp
from gurobipy import GRB


def solve_lp_C(costs, n, p, k, C, debug=False):
    """
    LP for fixed threshold C.

    I_C := { i in [n] : max_s costs[(s,i)] <= C }.
    Variables x_i in [0,1] only for i in I_C.
    Constraints:
      sum_{i in I_C} x_i = p
      sum_{i in I_C} c_{s,i} x_i <= C  for all scenarios s
    Returns:
      (feasible: bool, x_dict: dict[int,float] | None)
    """
    # Build I_C
    I_C = []
    for i in range(1, n + 1):
        max_i = max(float(costs[(s, i)]) for s in range(1, k + 1))
        if max_i <= C:
            I_C.append(i)

    # Feasibility check: not enough items to pick p
    if len(I_C) < p:
        return False, None

    m = gp.Model("LP_C")
    if not debug:
        m.setParam("OutputFlag", 0)

    x = m.addVars(I_C, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")

    # Objective irrelevant (feasibility); use 0
    m.setObjective(0.0, GRB.MINIMIZE)

    m.addConstr(gp.quicksum(x[i] for i in I_C) == p, name="cardinality")

    m.addConstrs(
        (gp.quicksum(float(costs[(s, i)]) * x[i] for i in I_C) <= C for s in range(1, k + 1)),
        name="scenario_budget",
    )

    m.optimize()
    if m.Status == GRB.OPTIMAL:
        x_dict = {i: float(x[i].X) for i in I_C}
        return True, x_dict
    return False, None


def find_min_feasible_C(costs, n, p, k, debug=False):
    # LB: mindestens 0
    lb = 0.0

    # UB: simple feasible upper bound (pick p smallest by max_s c[s,i])
    items = list(range(1, n + 1))
    items_sorted = sorted(
        items,
        key=lambda i: max(float(costs[(s, i)]) for s in range(1, k + 1))
    )
    chosen_ub = items_sorted[:p]
    ub = max(
        sum(float(costs[(s, i)]) for i in chosen_ub)
        for s in range(1, k + 1)
    )

    # safety: falls ub==lb (komische mini-instanzen), nudgen
    if ub < lb:
        ub = lb

    best_C = None
    best_x = None # LP Lösung

    lo, hi = lb, ub
    # feste iterationen
    for _ in range(60):
        mid = 0.5 * (lo + hi)  # Start bei der Mitte
        feasible, x_dict = solve_lp_C(costs, n, p, k, mid, debug=debug)
        if feasible:
            best_C = mid
            best_x = x_dict
            hi = mid
        else:
            lo = mid

    if best_C is None or best_x is None:
        # als letzte chance: teste UB direkt (sollte feasible sein)
        feasible, x_dict = solve_lp_C(costs, n, p, k, ub, debug=debug)
        if feasible:
            return float(ub), x_dict
        raise RuntimeError("No feasible C found for LP_C. Check instance feasibility / costs encoding.")

    return float(best_C), best_x


def _pairwise_dependent_rounding(x, rng, eps=1e-12):
    """
    In-place pairwise dependent rounding.
    Maintains:
      - each step makes at least one variable integral
    """
    # Work on list of fractional indices
    frac = [i for i, v in x.items() if eps < v < 1.0 - eps]

    while frac:
        # find partner a
        a = frac.pop()
        if not (eps < x[a] < 1.0 - eps):
            continue

        # find partner b
        if not frac:
            # numerics: we ended up with one "fractional" left due to eps
            # force it to make sum integral (sum x should be p)
            ones = sum(1 for v in x.values() if v >= 1.0 - eps)
            need = int(round(sum(x.values()))) - ones  # should be 0 or 1 here
            x[a] = 1.0 if need >= 1 else 0.0
            break

        b = frac.pop()
        if not (eps < x[b] < 1.0 - eps):
            # put a back and continue
            frac.append(a)
            continue

        xa, xb = x[a], x[b] # aktuelle fraktionale werte

        # Compute feasible moves that keep sum:
        # (xa, xb) -> (xa + d, xb - d) --> mimics paper cases
        alpha = min(1.0 - xa, xb)      # max increase of a
        beta = min(xa, 1.0 - xb)       # max decrease of a (i.e., increase of b)

        # With prob beta/(alpha+beta): move +alpha (push a up, b down)
        # else move -beta (push a down, b up)
        # This preserves E[x] marginals and fixes at least one var.
        denom = alpha + beta
        if denom <= eps:
            # Already effectively integral; skip
            continue

        if rng.random() < (beta / denom): # Entscheidung 1 mit Wk beta/(alpha+beta) (Entscheidung 2 mit Wk alpha/(alpha+beta))
            # move +alpha
            x[a] = xa + alpha
            x[b] = xb - alpha
        else:
            # move -beta
            x[a] = xa - beta
            x[b] = xb + beta

        # Re-add if still fractional
        if x[a] <= eps:
            x[a] = 0.0
        elif x[a] >= 1.0 - eps:
            x[a] = 1.0
        else:
            frac.append(a)

        if x[b] <= eps:
            x[b] = 0.0
        elif x[b] >= 1.0 - eps:
            x[b] = 1.0
        else:
            frac.append(b)

    # Final cleanup
    for i, v in list(x.items()):
        if v <= eps:
            x[i] = 0.0
        elif v >= 1.0 - eps:
            x[i] = 1.0


def dependent_rounding(x_dict, p, rng, debug=False):
    """
    Input: fractional solution x_dict over some item subset (e.g. I_C) with sum x = p.
    Output: chosen_items list of exactly p item ids.
    """
    # copy
    x = {i: float(v) for i, v in x_dict.items()}

    # Numerical normalization of sum to p (evtl. remove?)
    s = sum(x.values())
    if s <= 0:
        return []

    # If solver returns tiny drift, rescale slightly (keeps within [0,1] in typical instances)
    if abs(s - p) > 1e-8:
        factor = p / s
        for i in x:
            x[i] = min(1.0, max(0.0, x[i] * factor))

    _pairwise_dependent_rounding(x, rng)

    eps2 = 1e-9
    chosen = [i for i, v in x.items() if v >= 1.0 - eps2]
    if len(chosen) != p:
        raise RuntimeError(f"dependent_rounding failed: |chosen|={len(chosen)} != p={p}")

    return chosen


def _robust_obj_of_set(costs, chosen_items, k):
    return max(
        sum(float(costs[(s, i)]) for i in chosen_items)
        for s in range(1, k + 1)
    )


def solve_randomized_rounding_doerr(costs, n, p, k, rr_trials=100, seed=None, debug=False):
    """
    Main entry point
    """
    rng = random.Random(seed)

    # 1) find smallest feasible C and a fractional solution over I_C
    C_star, x_dict = find_min_feasible_C(costs, n, p, k, debug=debug)

    # Build fractional list over full [1..n] (0 for items not in I_C)
    x_val_frac_list = [0.0] * n
    for i in range(1, n + 1):
        if i in x_dict:
            x_val_frac_list[i - 1] = float(x_dict[i])

    objs = [] # Zielfunktionswerte
    solutions = [] # zugehörige items

    # 2) randomized dependent rounding trials
    for t in range(rr_trials):
        chosen = dependent_rounding(x_dict, p, rng, debug=False)
        obj = _robust_obj_of_set(costs, chosen, k)

        objs.append(float(obj))
        solutions.append(list(chosen))

    if not objs:
        chosen = dependent_rounding(x_dict, p, rng, debug=False)
        obj = _robust_obj_of_set(costs, chosen, k)
        objs.append(float(obj))
        solutions.append(list(chosen))

    avg_obj = sum(objs) / len(objs)

    info = {
        "C_star": float(C_star),
        "rr_trials": int(rr_trials),
        "avg_obj": float(avg_obj),
        "all_objs": objs,
    }

    if debug:
        print("\n--- Doerr RR ---")
        print(f"C_star (LP_C threshold) = {C_star}")
        print(f"avg_obj = {avg_obj}")

    return float(avg_obj), x_val_frac_list, float(C_star), info

    # best_obj = math.inf
    # best_chosen = None
    #
    # # 2) randomized dependent rounding trials
    # for t in range(rr_trials):
    #     chosen = dependent_rounding(x_dict, p, rng, debug=False)
    #     obj = _robust_obj_of_set(costs, chosen, k)
    #     if obj < best_obj:
    #         best_obj = obj
    #         best_chosen = chosen
    #
    # if best_chosen is None:
    #     best_chosen = dependent_rounding(x_dict, p, rng, debug=False)
    #     best_obj = _robust_obj_of_set(costs, best_chosen, k)
    #
    # x_vector_rounded = [1 if (i + 1) in best_chosen else 0 for i in range(n)]
    #
    # info = {
    #     "C_star": float(C_star),
    #     "rr_trials": int(rr_trials),
    #     "chosen_items": list(best_chosen),
    # }
    # if debug:
    #     print("\n--- Doerr RR ---")
    #     print(f"C_star (LP_C threshold) = {C_star}")
    #     print(f"best_obj = {best_obj}")
    #     print(f"chosen_items = {best_chosen}")
    #
    # return best_obj, x_val_frac_list, x_vector_rounded, float(C_star), info
