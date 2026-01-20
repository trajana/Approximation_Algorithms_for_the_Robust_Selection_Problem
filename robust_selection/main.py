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
from primal_rounding import solve_primal_rounding
from primal_dual_rounding import solve_primal_dual_with_lp
from utils import (get_fixed_costs, get_random_costs, dprint_costs, cost_matrix_to_dict,
                   dprint_all_results_from_pkl)

ALGORITHM_DISPATCH = {
    "primal": {
        "algorithm": "Primal Rounding",
        "function": solve_primal_rounding
    },
    "primal_dual": {
        "algorithm": "Primal-Dual Rounding",
        "function": solve_primal_dual_with_lp
    },
    "midpoint": {
        "algorithm": "Midpoint Method",
        "function": solve_midpoint
    },
    "worst_case_p_item": {
        "algorithm": "Worst-Case per Item",
        "function": solve_worst_case_p_item
    }
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
ALGORITHMS = ["primal", "primal_dual", "midpoint", "worst_case_p_item"]  # Choose algorithms that should
# be run. Available: "primal_minmax", "primal_dual_minmax", "midpoint", "worst_case_p_item"
var_param = "p"  # x-axis for the plot, can be "n" or "k" or "p"
if var_param == "n":
    var_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
                  54, 56, 58, 60, 62, 64, 66, 68, 70]
    fixed_k = 5
    fixed_p = None  # p = n//2, so no need to set it explicitly
elif var_param == "k":
    fixed_n = 20
    var_values = [1, 2, 5, 10, 20, 30, 50, 70, 100]
    fixed_p = fixed_n // 2
elif var_param == "p":
    fixed_n = 70
    fixed_k = 5
    var_values = list(range(2, fixed_n, 2))  # p in steps of 2 from 2 to n-2
num_runs = 100  # Number of runs for the loop
COST_MODE = "random"    # Options: "random", "fixed", "reproduce"
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
                    obj_val_primal, x_val_primal_frac, x_vector_primal_rounded, obj_val_primal_lp, tau = result
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
                # plot_approximation_ratios_primal(
                #     all_results, num_runs, var_param,
                #     fixed_n=n, fixed_k=k, c_range=c_range,
                #     output_dir=algo_result_dir
                # )
                # plot_fractional_variable_count(
                #     all_results, num_runs, var_param,
                #     fixed_n=fixed_n if var_param != "n" else None,
                #     fixed_k=fixed_k if var_param != "k" else None,
                #     c_range=c_range,
                #     output_dir=algo_result_dir
                # )

            elif algorithm == "primal_dual":
                plot_approx_ratio_only(
                    all_results, num_runs, var_param,
                    fixed_n=n, fixed_k=k, c_range=c_range,
                    output_dir=algo_result_dir
                )
                # plot_approximation_ratios_primaldual(
                #     all_results, num_runs, var_param,
                #     fixed_n=n, fixed_k=k, c_range=c_range,
                #     output_dir=algo_result_dir
                # )

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

        results_by_alg[algorithm] = all_results

    if PLOT and len(results_by_alg) >= 2:
        from plot import plot_ratio_comp

        plot_ratio_comp(
            results_by_alg,
            num_runs, var_param,
            fixed_n=fixed_n if var_param != "n" else None,
            fixed_k=fixed_k if var_param != "k" else None,
            c_range=c_range,
            output_dir=RESULT_DIR
        )
