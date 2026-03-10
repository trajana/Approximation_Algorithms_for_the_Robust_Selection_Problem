# main.py

# Main script to run the robust selection problem with the min-max criterion and discrete uncertainty.

# Calls functions from other modules to compute the exact and heuristic solutions.
# Adjustable parameters: n (total items), p (items to select), k (scenarios), c_range (random cost range), num_runs
# (number of runs in loop), n_values (for multiple n values), PLOT.
# Choose between fixed or random cost vectors. If using fixed costs, define them in utils.py (get_fixed_costs).

import pickle
import os
import math
from datetime import datetime
from exact_solution import solve_exact_robust_selection
from midpoint import solve_midpoint
from worst_case_p_item import solve_worst_case_p_item
from primal_rounding import (solve_primal_rounding, solve_two_branches_biggest_xi)
from primal_dual_rounding import solve_primal_dual_with_lp
from dual_approach import solve_dual_approach
from primal_dual_rounding_opt_w import (solve_opt_w_then_select_once,
                                        solve_opt_w_then_select_adaptive_remember, solve_opt_w_ntimes_then_select_adaptive_remember_rob_obj,
                                        solve_two_branches_smallest_wi,
                                        solve_two_branches_biggest_wi)
from randomized_rounding_doerr import solve_randomized_rounding_doerr
from utils import (get_fixed_costs, get_random_costs, dprint_costs, cost_matrix_to_dict,
                   dprint_all_results_from_pkl)

ALGORITHM_DISPATCH = {
    "primal": {
        "algorithm": "Primal Rounding",
        "function": solve_primal_rounding
    },
    "primal_branching": {
        "algorithm": "Primal Branching",
        "function": solve_two_branches_biggest_xi
    },
    "primal_dual": {
        "algorithm": "Primal-Dual Rounding",
        "function": solve_primal_dual_with_lp
    },
    "dual": {
        "algorithm": "Dual Approach",
        "function": solve_dual_approach
    },
    "midpoint": {
        "algorithm": "Midpoint Method",
        "function": solve_midpoint
    },
    "worst_case_p_item": {
        "algorithm": "Worst-Case per Item",
        "function": solve_worst_case_p_item
    },
    "opt_w": {
        "algorithm": "Optimize w",
        "function": solve_opt_w_then_select_once
    },
    "opt_w_remember": {
        "algorithm": "Optimize iteratively (remember)",
        "function": solve_opt_w_then_select_adaptive_remember
    },
    "opt_w_n_remember_rob_obj": {
        "algorithm": "Optimize n times iteratively (remember, min robust obj)",
        "function": solve_opt_w_ntimes_then_select_adaptive_remember_rob_obj
    },
    # "opt_w_n_remember": {
    #     "algorithm": "Optimize n times iteratively (remember)",
    #     "function": solve_opt_w_ntimes_then_select_adaptive_remember
    # },
    "solve_two_branches_smallest_wi": {
        "algorithm": "Branch by smallest wi",
        "function": solve_two_branches_smallest_wi
    },
    "solve_two_branches_biggest_wi": {
        "algorithm": "Branch by biggest wi",
        "function": solve_two_branches_biggest_wi
    },
    "doerr_rr": {
        "algorithm": "Doerr Randomized Rounding",
        "function": solve_randomized_rounding_doerr
    },
}

# Pre-initialize
var_values: list[int] = []
fixed_n: int | None = None
fixed_p: int | None = None
fixed_k: int | None = None
n: int | None = None
p: int | None = None
k: int | None = None

# Base data
ALGORITHMS = [
    #"doerr_rr",
    #"primal", "primal_dual", "dual", "midpoint", "worst_case_p_item",
    "opt_w",
    #"opt_w_remember",
    #"opt_w_n_remember_rob_obj",
    ##"opt_w_n_remember",
    #"solve_two_branches_smallest_wi",
    #"solve_two_branches_biggest_wi",
    #"primal_branching",
]
# Choose algorithms that should be run.
# Available: "primal_minmax", "primal_dual_minmax", "midpoint", "worst_case_p_item"
var_param = "p"  # x-axis for the plot, can be "n" or "k" or "p"
if var_param == "n":
    var_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]
    fixed_k = 5
    fixed_p = None  # p = n//2, so no need to set it explicitly
elif var_param == "k":
    fixed_n = 20
    var_values = [1, 2, 5, 10, 20, 30, 50, 70, 100]
    fixed_p = fixed_n // 2
elif var_param == "p":
    fixed_n = 70
    fixed_k = 5
    var_values = [1] + list(range(2, fixed_n, 2))  # p in steps of 2 from 2 to n-2 (plus 1)
num_runs = 100  # Number of runs for the loop
COST_MODE = "reproduce"    # Options: "random", "fixed", "reproduce"
c_range = 100  # Range for random costs [0, c_range]
PLOT = True  # Set True to enable plotting
DEBUG = False  # Set True to enable debug prints


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


if __name__ == "__main__":
    # Create unique results subfolder based on algorithm, k, and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RESULT_DIR = f"results/{var_param}_{timestamp}"
    os.makedirs(RESULT_DIR, exist_ok=True)
    if COST_MODE not in {"random", "fixed", "reproduce"}:
        raise ValueError(f"Unknown COST_MODE: {COST_MODE}")
    VAR_DIR_MAP = {
        "n": "n_var",
        "k": "k_var",
        "p": "p_var",
    }
    if var_param not in VAR_DIR_MAP:
        raise ValueError(f"Invalid var_param: {var_param}.")

    COSTS_SOURCE_DIR = os.path.join("repro_costs", VAR_DIR_MAP[var_param])

    results_by_alg = {}

    for algorithm in ALGORITHMS:
        algo_info = ALGORITHM_DISPATCH.get(algorithm)
        if not algo_info:
            print(f"Unknown algorithm '{algorithm}', skipping.")
            continue

        solve_function = algo_info["function"]

        algo_result_dir = os.path.join(RESULT_DIR, algorithm)
        os.makedirs(algo_result_dir, exist_ok=True)

        all_results = []

        for a in var_values:
            p_label = ""
            if var_param == "n":
                n = a
                if fixed_p is None:
                    p = n // 2
                    p_label = "n/2"
                else:
                    p = fixed_p
                    p_label = str(fixed_p)
                k = fixed_k
            elif var_param == "k":
                n = fixed_n
                p = fixed_p
                p_label = str(fixed_p)
                k = a
            elif var_param == "p":
                n = fixed_n
                p = a
                p_label = str(a)
                k = fixed_k

            # Ensure these exist for all branches before we solve the exact problem
            obj_val_exact: float = 0.0
            x_vector_exact: list[int] = [0] * int(n)

            print(f"\n=== Running experiments for n = {n}, p = {p}, k = {k} ===")

            for run in range(num_runs):
                print(f"\n=== Running {algorithm} for run {run + 1} ===")

                # Choose cost type
                if COST_MODE == "fixed":
                    c = get_fixed_costs(n, k)
                elif COST_MODE == "reproduce":
                    cost_file = os.path.join(
                        COSTS_SOURCE_DIR,
                        f"costs_n{n}_p{p}_k{k}_a{a}_run{run + 1}.pkl"
                    )
                    if not os.path.exists(cost_file):
                        raise FileNotFoundError(
                            f"Repro file not found: {cost_file}. "
                        )
                    with open(cost_file, "rb") as f:
                        c = pickle.load(f)
                    print(f"[Loaded costs] {cost_file}")
                elif COST_MODE == "random":
                    c = get_random_costs(n, k, c_range)
                else:
                    raise ValueError(f"Unknown COST_MODE: {COST_MODE}")

                # Print costs
                dprint("--- Cost matrix ---")
                dprint_costs(c, debug=DEBUG)
                costs = cost_matrix_to_dict(c)  # Convert costs to a dictionary with keys (s, i)
                dprint("--- Cost dictionary ---")
                dprint(costs)
                flat_costs = [costs[(s + 1, i + 1)] for s in range(k) for i in range(n)]  # Flattened cost list for .pkl

                # Exact problem
                print("\n--- Exact robust solution min-max ---")
                obj_val_exact, x_val_exact = (solve_exact_robust_selection
                                                            (costs, n, p, k, debug=DEBUG))
                obj_val_exact = obj_val_exact
                x_vector_exact = [1 if val > 0.5 else 0 for val in x_val_exact]  # For rounding discrepancy
                dprint(f"Selected items (exact): {x_vector_exact}")
                dprint(f"Objective value: {obj_val_exact:.2f}")

                result = solve_function(costs, n, p, k, debug=DEBUG)

                if algorithm == "primal":
                    print("\n--- Primal Rounding min-max ---")
                    obj_val_primal, x_val_primal_frac, x_vector_primal_rounded, obj_val_primal_lp, tau, _ = result
                    x_vector_primal_frac = [round(val, 2) for val in x_val_primal_frac]
                    fractional_count = sum(1 for val in x_val_primal_frac if 0.0001 < val < 0.9999)
                    fractional_ratio = fractional_count / n
                    dprint(f"Fractional values: {x_vector_primal_frac}")
                    dprint(f"Fractional variables: {fractional_count} out of {n} ({fractional_ratio:.2%})")
                    dprint(f"Selected items (rounded): {x_vector_primal_rounded}")
                    dprint(f"Objective value: {obj_val_primal:.2f}")

                    # Metrics calculations
                    ratio_primal_opt = obj_val_primal / obj_val_exact if obj_val_exact != 0 else math.nan
                    integrality_gap = obj_val_exact / obj_val_primal_lp if obj_val_primal_lp != 0 else math.nan
                    approximation_guarantee = min(k, n - p + 1)
                    a_posteriori_bound = 1 / tau if tau != 0 else math.nan
                    alg_div_opt_lp = obj_val_primal / obj_val_primal_lp if obj_val_primal_lp != 0 else math.nan
                    dprint(f"Approximation ratio: {ratio_primal_opt:.2f}")
                    dprint(f"Integrality gap: {integrality_gap:.2f}")
                    dprint(f"Approximation guarantee: min(k, n - p + 1) = {approximation_guarantee}")
                    dprint(f"A-posteriori bound: ALG ≤ (1/τ) · OPT → 1/τ = {a_posteriori_bound:.2f}")

                    # Store results
                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n,
                        "p": p,
                        "k": k,
                        "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_primal_lp": obj_val_primal_lp,
                        "obj_primal": obj_val_primal,
                        "ratio_alg_opt": ratio_primal_opt,
                        "tau": tau,
                        "a_posteriori_bound": a_posteriori_bound,
                        "alg_div_opt_lp": alg_div_opt_lp,
                        "approximation_guarantee": approximation_guarantee,
                        "fractional_count": fractional_count,
                        "fractional_ratio": fractional_ratio,
                        "x_vector_exact": x_vector_exact,
                        "x_vector_primal_frac": x_vector_primal_frac,
                        "x_vector_primal_rounded": x_vector_primal_rounded,
                        "flat_costs": flat_costs
                    })

                elif algorithm == "primal_dual":
                    print("\n--- Primal-Dual Rounding min-max ---")
                    obj_val_primaldual, x_vector_primaldual_rounded, obj_dual, obj_val_primal_lp = result
                    dprint(f"Selected items (rounded): {x_vector_primaldual_rounded}")
                    dprint(f"Objective value: {obj_val_primaldual:.2f}")

                    # Metrics calculations
                    ratio_primaldual_opt = obj_val_primaldual / obj_val_exact if obj_val_exact != 0 else math.nan
                    dprint(f"Approximation ratio: {ratio_primaldual_opt:.2f}")
                    approximation_guarantee = k
                    a_posteriori_bound = (obj_val_primaldual / obj_dual) if obj_dual != 0 else math.nan
                    alg_div_opt_lp = obj_val_primaldual / obj_val_primal_lp if obj_val_primal_lp != 0 else math.nan
                    dprint(f"a-posteriori (ALG/LB_dual): {a_posteriori_bound:.2f}")
                    dprint(f"a-posteriori (ALG/OPT_LP): {alg_div_opt_lp:.2f}")

                    # Store results
                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n,
                        "p": p,
                        "k": k,
                        "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_dual": obj_dual,
                        "obj_val_primaldual": obj_val_primaldual,
                        "a_posteriori_bound": a_posteriori_bound,
                        "alg_div_opt_lp": alg_div_opt_lp,
                        "approximation_guarantee": approximation_guarantee,
                        "ratio_alg_opt": ratio_primaldual_opt,
                        "x_vector_exact": x_vector_exact,
                        "x_vector_primaldual_rounded": x_vector_primaldual_rounded,
                        "flat_costs": flat_costs,
                    })

                elif algorithm == "dual":
                    print("\n--- Dual Approach ---")
                    c_hat, x_opt, obj_val_dual_lp, obj_val_dual_nom = result
                    dprint(f"Selected items: {x_opt}")
                    dprint(f"Objective value: {obj_val_dual_nom:.2f}")

                    # Metrics calculations
                    ratio_dual_opt = obj_val_dual_nom / obj_val_exact if obj_val_exact != 0 else math.nan
                    dprint(f"Approximation ratio: {ratio_dual_opt:.2f}")
                    approximation_guarantee = k

                    # Store results
                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n,
                        "p": p,
                        "k": k,
                        "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_val_dual_nom": obj_val_dual_nom,
                        "approximation_guarantee": approximation_guarantee,
                        "ratio_alg_opt": ratio_dual_opt,
                        "x_opt": x_opt,
                        "flat_costs": flat_costs,
                    })

                elif algorithm == "midpoint":
                    print("\n--- Midpoint Method ---")
                    costs_av, x_av, obj_val_av = result
                    dprint(f"Selected items (x_av): {x_av}")
                    dprint(f"Objective value: {obj_val_av:.2f}")

                    # Metrics calculations
                    ratio_av_opt = obj_val_av / obj_val_exact if obj_val_exact != 0 else math.nan
                    approximation_guarantee = k
                    dprint(f"Approximation ratio (ALG/OPT): {ratio_av_opt:.2f}")

                    # Store results
                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n,
                        "p": p,
                        "k": k,
                        "run": run + 1,
                        "obj_val_av": obj_val_av,
                        "approximation_guarantee": approximation_guarantee,
                        "ratio_alg_opt": ratio_av_opt,
                        "x_vector_av": x_av,
                        "flat_costs": flat_costs,
                    })

                elif algorithm == "worst_case_p_item":
                    print("\n--- Worst-Case-per-Item Method ---")
                    costs_wc, x_wc, obj_val_wc = result
                    dprint(f"Selected items (x_wc): {x_wc}")
                    dprint(f"Objective value: {obj_val_wc:.2f}")

                    # Metrics calculations
                    ratio_wc_opt = obj_val_wc / obj_val_exact if obj_val_exact != 0 else math.nan
                    approximation_guarantee = k
                    dprint(f"Approximation ratio (ALG/OPT): {ratio_wc_opt:.2f}")

                    # Store results
                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n,
                        "p": p,
                        "k": k,
                        "run": run + 1,
                        "obj_val_wc": obj_val_wc,
                        "approximation_guarantee": approximation_guarantee,
                        "ratio_alg_opt": ratio_wc_opt,
                        "x_vector_wc": x_wc,
                        "flat_costs": flat_costs,
                    })

                elif algorithm == "opt_w":
                    print("\n--- LP-opt-w (solve LP, then select p smallest w*) ---")
                    t_star, beta_star, w_star, x_opt_w, obj_val_opt_w = result

                    # Metrics calculations
                    dprint(f"Objective value: {obj_val_opt_w:.2f}")
                    ratio_optw_opt = obj_val_opt_w / obj_val_exact if obj_val_exact != 0 else math.nan
                    approximation_guarantee = (1.0 / t_star) if t_star > 0 else math.nan
                    dprint(f"Approximation ratio (ALG/OPT): {ratio_optw_opt:.2f}")

                    # Store results
                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n,
                        "p": p,
                        "k": k,
                        "run": run + 1,
                        "obj_val_opt_w": obj_val_opt_w,
                        "approximation_guarantee": approximation_guarantee,
                        "ratio_alg_opt": ratio_optw_opt,
                        "t_star": float(t_star),
                        "beta_star": [float(x) for x in beta_star],
                        "x_vector_opt_w": x_opt_w,
                        "flat_costs": flat_costs,
                    })

                elif algorithm == "opt_w_remember":
                    chosen, x_opt_w, obj_val_opt_w, t_hist, beta_hist = result
                    ratio = obj_val_opt_w / obj_val_exact if obj_val_exact != 0 else math.nan
                    # guarantee: use t from the FIRST LP only
                    t0 = float(t_hist[0]) if (t_hist and t_hist[0] is not None) else math.nan
                    if math.isfinite(t0) and t0 > 0.0:
                        approximation_guarantee = 1.0 / t0
                    else:
                        approximation_guarantee = math.nan

                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n, "p": p, "k": k, "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_val_opt_w": obj_val_opt_w,
                        "ratio_alg_opt": ratio,
                        "approximation_guarantee": approximation_guarantee,
                        "chosen": chosen,
                        "x_vector_opt_w": x_opt_w,
                        "t_hist": [float(x) for x in t_hist],
                        "beta_hist": [[float(y) for y in b] for b in beta_hist],
                        "flat_costs": flat_costs,
                    })

                elif algorithm == "opt_w_n_remember_rob_obj":
                    print("\n--- N-times (remember): choose by min robust objective ---")
                    chosen, x_opt_w, obj_val_opt_w, hist = result
                    ratio = obj_val_opt_w / obj_val_exact if obj_val_exact != 0 else math.nan

                    # guarantee: use t0 from the BASELINE LP only
                    t0 = math.nan
                    if hist and hist[0].get("baseline", False):
                        t0 = float(hist[0].get("t0", math.nan))

                    if math.isfinite(t0) and t0 > 0.0:
                        approximation_guarantee = 1.0 / t0
                    else:
                        approximation_guarantee = math.nan

                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n, "p": p, "k": k, "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_val_opt_w": obj_val_opt_w,
                        "ratio_alg_opt": ratio,
                        "approximation_guarantee": approximation_guarantee,
                        "chosen": chosen,
                        "x_vector_opt_w": x_opt_w,
                        "hist": hist,
                        "flat_costs": flat_costs,
                    })

                elif algorithm == "opt_w_n_remember":
                    chosen, x_opt_w, obj_val_opt_w, hist = result
                    ratio = obj_val_opt_w / obj_val_exact if obj_val_exact != 0 else math.nan
                    # guarantee: use t from the BASELINE LP only (first entry)
                    t0 = math.nan
                    if hist:
                        h0 = hist[0]
                        # support different keys depending on how you store it
                        if isinstance(h0, dict):
                            if h0.get("baseline") and h0.get("t0") is not None:
                                t0 = float(h0["t0"])
                            elif h0.get("t_star") is not None:
                                t0 = float(h0["t_star"])

                    if math.isfinite(t0) and t0 > 0.0:
                        approximation_guarantee = 1.0 / t0
                    else:
                        approximation_guarantee = math.nan

                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n, "p": p, "k": k, "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_val_opt_w": obj_val_opt_w,
                        "ratio_alg_opt": ratio,
                        "approximation_guarantee": approximation_guarantee,
                        "chosen": chosen,
                        "x_vector_opt_w": x_opt_w,
                        "hist": hist,  # ist schon eine Liste von dicts
                        "flat_costs": flat_costs,
                    })


                elif algorithm == "solve_two_branches_smallest_wi":
                    print("\n--- Two-Branch: smallest w_i ---")
                    chosen, x_vec, obj_val, info = result
                    ratio = obj_val / obj_val_exact if obj_val_exact != 0 else math.nan
                    t_chosen = float(info["t_chosen"])
                    approx_guarantee = (1.0 / t_chosen) if t_chosen > 0 else math.inf

                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n, "p": p, "k": k, "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_val_branch": obj_val,
                        "ratio_alg_opt": ratio,
                        "approximation_guarantee": approx_guarantee,
                        "t0": float(info["t0"]),
                        "t_in": float(info["t_in"]),
                        "t_out": float(info["t_out"]),
                        "t_chosen": float(info["t_chosen"]),
                        "chosen_branch": info["chosen_branch"],
                        "i_star": int(info["i_star"]),
                        "w_i_star": float(info["w_i_star"]),
                        "x_vector_branch": x_vec,
                        "chosen": chosen,
                        "flat_costs": flat_costs,
                    })


                elif algorithm == "solve_two_branches_biggest_wi":
                    print("\n--- Two-Branch: biggest w_i ---")
                    chosen, x_vec, obj_val, info = result
                    ratio = obj_val / obj_val_exact if obj_val_exact != 0 else math.nan
                    t_chosen = float(info["t_chosen"])
                    approx_guarantee = (1.0 / t_chosen) if t_chosen > 0 else math.inf

                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n, "p": p, "k": k, "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_val_branch": obj_val,
                        "ratio_alg_opt": ratio,
                        "approximation_guarantee": approx_guarantee,
                        "t0": float(info["t0"]),
                        "t_in": float(info["t_in"]),
                        "t_out": float(info["t_out"]),
                        "t_chosen": float(info["t_chosen"]),
                        "chosen_branch": info["chosen_branch"],
                        "i_star": int(info["i_star"]),
                        "w_i_star": float(info["w_i_star"]),
                        "x_vector_branch": x_vec,
                        "chosen": chosen,
                        "flat_costs": flat_costs,
                    })

                elif algorithm == "primal_branching":
                    print("\n--- Primal Branching (two-branch biggest fractional x_i) ---")
                    chosen, x_vec, obj_val_branch, info = result
                    ratio = obj_val_branch / obj_val_exact if obj_val_exact != 0 else math.nan

                    # a-posteriori: use tau_guarantee from info
                    tau_guarantee = info.get("tau_guarantee", math.nan)
                    a_posteriori_bound = (1.0 / tau_guarantee) if (
                                tau_guarantee not in [0, None] and math.isfinite(tau_guarantee)) else math.nan

                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n, "p": p, "k": k, "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_val_branch": obj_val_branch,
                        "ratio_alg_opt": ratio,
                        "tau_guarantee": tau_guarantee,
                        "approximation_guarantee": a_posteriori_bound,
                        "chosen_branch": info.get("branch"),
                        "i_star": info.get("i_star"),
                        "x_i_star": info.get("x_i_star"),
                        "S_in": info.get("S_in"),
                        "S_out": info.get("S_out"),
                        "obj_in": info.get("obj_in"),
                        "obj_out": info.get("obj_out"),
                        "x_vector_branch": x_vec,
                        "chosen": chosen,
                        "flat_costs": flat_costs,
                    })

                elif algorithm == "doerr_rr":
                    print("\n--- Doerr Randomized Rounding (LP + Dependent RR) ---")
                    obj_val_rr, x_val_rr_frac, C_star, info = result

                    dprint(f"C_star (LP feasibility threshold): {C_star}")
                    dprint(f"Average objective over {info.get('rr_trials')} RR trials: {obj_val_rr:.2f}")

                    # Metrics calculations
                    ratio_rr_opt = obj_val_rr / obj_val_exact if obj_val_exact != 0 else math.nan
                    # Theoretical guarantee (Doerr): O(log k / log log k)
                    if k is not None and k >= 3:
                        approximation_guarantee = math.log(k) / math.log(math.log(k))
                    else:
                        approximation_guarantee = math.nan  # loglog not defined / meaningless

                    # a-posteriori via LP threshold
                    alg_div_Cstar = obj_val_rr / C_star if C_star not in [0, None] else math.nan

                    dprint(f"Approximation ratio (ALG/OPT): {ratio_rr_opt:.2f}")
                    dprint(f"a-posteriori (ALG/C_star): {alg_div_Cstar:.2f}")

                    # Store results
                    all_results.append({
                        "algorithm": algorithm,
                        "varying_param": a,
                        "p_label": p_label,
                        "n": n,
                        "p": p,
                        "k": k,
                        "run": run + 1,
                        "obj_exact": obj_val_exact,
                        "obj_val_rr": obj_val_rr,
                        "C_star": C_star,
                        "ratio_alg_opt": ratio_rr_opt,
                        "alg_div_Cstar": alg_div_Cstar,
                        "approximation_guarantee": approximation_guarantee,
                        "rr_trials": info.get("rr_trials"),
                        "best_trial": info.get("best_trial"),
                        "chosen_items": info.get("chosen_items"),
                        "x_vector_rr_frac": [round(val, 4) for val in x_val_rr_frac],
                        "flat_costs": flat_costs,
                    })

        # Save results as pickle file
        with open(os.path.join(algo_result_dir, f"all_results.pkl"), "wb") as f:
            pickle.dump(all_results, f)
        print(f"Results for {algorithm} saved in {algo_result_dir} ")

        # View all results from a .pkl file
        dprint_all_results_from_pkl(os.path.join(algo_result_dir, f"all_results.pkl"), debug=DEBUG)

        # Plot results
        if PLOT:
            from plot import (plot_approx_ratio_only) #(plot_approx_ratio_only, plot_approximation_ratios_primal, plot_approximation_ratios_primaldual, plot_fractional_variable_count)

            if algorithm == "primal":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "doerr_rr":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "primal_dual":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "dual":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "midpoint":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "worst_case_p_item":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "opt_w":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "opt_w_remember":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "opt_w_n_remember":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "opt_w_n_remember_rob_obj":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "solve_two_branches_smallest_wi":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )
                # NEU:
                from plot import plot_branch_guarantees

                plot_branch_guarantees(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

            elif algorithm == "solve_two_branches_biggest_wi":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )
                # NEU:
                from plot import plot_branch_guarantees

                plot_branch_guarantees(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )

        results_by_alg[algorithm] = all_results

    if PLOT and len(results_by_alg) >= 2:
        from plot import plot_ratio_comp, plot_guarantee_comp, plot_branch_perf_vs_guarantee

        plot_ratio_comp(
            results_by_alg,
            num_runs, var_param,
            fixed_n=fixed_n if var_param != "n" else None,
            fixed_k=fixed_k if var_param != "k" else None,
            c_range=c_range,
            output_dir=RESULT_DIR
        )

        plot_guarantee_comp(
            results_by_alg,
            num_runs, var_param,
            fixed_n=fixed_n if var_param != "n" else None,
            fixed_k=fixed_k if var_param != "k" else None,
            c_range=c_range,
            output_dir=RESULT_DIR
        )

        plot_branch_perf_vs_guarantee(
            results_by_alg,
            num_runs,
            var_param,
            fixed_n=fixed_n if var_param != "n" else None,
            fixed_k=fixed_k if var_param != "k" else None,
            c_range=c_range,
            output_dir=RESULT_DIR,
        )



