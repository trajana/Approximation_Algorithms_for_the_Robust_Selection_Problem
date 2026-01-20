# costs_random.py

# Script to generate and save random cost instances for the robust selection problem. This was used to create the cost
# matrices for the experiments in the thesis. The results are saved in the file repro_costs.

# Not intended to be run as part of the main program, just included for completeness and transparency.

import os
import pickle
from utils import get_random_costs

# Pre-initialize
var_values: list[int] = []
fixed_n: int | None = None
fixed_p: int | None = None
fixed_k: int | None = None
n: int | None = None
p: int | None = None
k: int | None = None

# Generate and save random cost instances for various (n, p, k) configurations
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
num_runs = 100
c_range = 100

OUTPUT_DIR = f"repro_costs/{var_param}_var"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    for run in range(1, num_runs + 1):
        cost_file = os.path.join(OUTPUT_DIR, f"costs_n{n}_p{p}_k{k}_a{a}_run{run}.pkl")
        if os.path.exists(cost_file):
            continue
        c = get_random_costs(n, k, c_range)
        with open(cost_file, "wb") as f:
            pickle.dump(c, f)
        print(f"Saved {cost_file}")
