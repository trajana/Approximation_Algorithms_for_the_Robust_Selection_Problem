# README.md

# Approximation Algorithms for the Robust Selection Problem

This repository provides a modular Python implementation for solving the **Robust Selection Problem** under discrete 
uncertainty using a variety of **approximation algorithms** or a heuristic. It is part of a Master's thesis.

## Problem Description

ROBUST SELECTION PROBLEM: The goal is to select exactly `p` items out of `n`, such that performance is robust across `k`
scenarios. Each item has a cost in each scenario c_{s,i}, and the objective is to minimize the worst-case cost of the 
selected items. We use discrete uncertainty, meaning there are `k` explicitly defined cost scenarios. To model 
robustness, we consider two criteria: min-max and max-min.

This problem is NP-hard, and approximation algorithms are studied as efficient alternatives to exact optimization.

Note that the robust selection problem under the max-min criterion with discrete uncertainty is inapproximable, 
so we focus on heuristic approaches for this case.

## Features

The repository provides both **exact formulations** (via Mixed Integer Linear Programming using Gurobi) 
and several **approximation algorithms** / **heuristics**:

The current implementation supports:

### Exact Algorithms
- **Exact min–max formulation** (`exact_solution_minmax.py`)
- **Exact max–min formulation** (`exact_solution_maxmin.py`)

### Approximation Algorithms
- **Primal Rounding**  
  - min–max variant (`primal_rounding_minmax.py`)  
  - max–min variant (`primal_rounding_maxmin.py`) --> heuristic only
- **Primal–Dual Rounding**
  - min–max variant (`primal_dual_rounding_minmax.py`)

---

## Theoretical Guarantees

| Algorithm                   | Criterion | File                             | Approximation Guarantee              |
|-----------------------------|-----------|----------------------------------|--------------------------------------|
| Exact (MILP)                | min–max   | `exact_solution_minmax.py`       | Optimal                              |
| Exact (MILP)                | max–min   | `exact_solution_maxmin.py`       | Optimal                              |
| Primal Rounding             | min–max   | `primal_rounding_minmax.py`      | ≤ min(k, n − p + 1)                  |
| Primal–Dual Rounding        | min–max   | `primal_dual_rounding_minmax.py` | ≤ 1/β_min ( = k for uniform weights) |
| Primal Rounding (heuristic) | max–min   | `primal_rounding_maxmin.py`      | No guarantee                         |

---

## Project Structure

<pre>selection_problem/
├── main.py                             # Change settings and run experiments here
├── exact_solution_minmax.py        
├── exact_solution_maxmin.py        
├── primal_rounding_minmax.py       
├── primal_rounding_maxmin.py       
├── primal_dual_rounding_minmax.py  
├── utils.py                            # Change fixed cost scenarios here
├── plot.py 
├── repro_costs/                        # Ensure this directory exists if using COST_MODE = "reproduce"
│   ├── n_var/                          # Randomly generated cost matrices used in thesis experiments
│   ├── k_var/                          # Randomly generated cost matrices used in thesis experiments
│   └── p_var/                          # Randomly generated cost matrices used in thesis experiments
├── costs_random.py                     # Was used to generate random costs for thesis experiments 
                                        (not needed for running - just included for transparency)
├── results/                        
├── requirements.txt                
└── README.md</pre>

## Running Experiments

1. **Adjust parameters** in `main.py`:

    \
    | Parameter    | Description                                                                                                | Values / Options                                                                                          | Location in main.py  | Default Value   |
    |--------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|------------|-----------------|
    | `ALGORITHMS` | Choose which approximation algorithms to run                                                               | `"primal_minmax"`, `"primal_maxmin"`, `"primal_dual_minmax"`                                              | line 50    | `["primal_dual_minmax", "primal_minmax", "primal_maxmin"]` |
    | `var_param`  | Choose which variable should vary                                                                          | `"n"` = number of items, `"k"` = number of scenarios, `"p"` = number of items to select                   | line 52    | `"n"`           |
    | `var_values` | List of values for the chosen `var_param`                                                                  | Example: `[2,4,6,…,70]` for `n`; `[1,2,5,…,100]` for `k`; `[2,4,…,68]` for `p`                        | lines 54–65 | `[2,4,...,70]` (if `n`); `[1,2,5,…,100]` (if `k`); `[2,4,…,68]` (if `p`) |
    | `num_runs`   | Number of repetitions per setting                                                                          | Integer (e.g. `100`)                                                                                      | line 66    | `100`           |
    | `COST_MODE`  | Toggle between fixed, random, or reproduced costs                                                          | `"fixed"` = use `utils.py` <br> `"random"` = generate random costs <br> `"reproduce"` = load from `repro_costs/{n_var\|k_var\|p_var}/` | line 67    | `"random"`   |
    | `c_range`    | Range for random costs                                                                                     | Integer (e.g. `100`)                                                                                      | line 68    | `100`           |
    | `PLOT`       | Enable/disable automatic plot generation                                                                   | `True` / `False`                                                                                          | line 69    | `True`          |
    | `DEBUG`      | Enable/disable detailed debug prints (not recommended for large instances due to excessive console output) | `True` / `False`                                                                                          | line 70    | `False`          |

    Note: If you want to **replicate** the experiments from the thesis, you need to set `COST_MODE="reproduce"` and run 
    the code three times, once for each varying parameter (`n`, `k`, and `p`), using the pre-specified default values 
    for all other parameters. 
    The input data used in the thesis experiments is not stored directly in this repository to keep the code base 
    lightweight. If you want to reproduce the results with `COST_MODE="reproduce"`, please download the cost archive 
    from the [latest Release](https://github.com/trajana/selection_problem/releases/tag/data-v1.0) and extract 
    it into the project root. Ensure the folder structure remains as shown above.

2. **Run the script**:
   ```bash
   python main.py
   ```
   
3. **Outputs**:
   - Pickle files (.pkl) with raw results (stored in results/)
   - Plots of approximation ratios, fractional variable count etc. (stored in results/)

# Dependencies

Install required packages via:

pip install -r requirements.txt

Also install Gurobi Optimizer separately and ensure you have an active Gurobi license.

To generate plots with LaTeX-rendered labels (text.usetex=True), you need a TeX distribution installed on your system 
(e.g., TeX Live, MiKTeX) along with dvipng and ghostscript. Without TeX, Matplotlib will fall back to mathtext if you 
set text.usetex=False in plot.py.

# Notes

- Developed and tested with Python 3.12
- Gurobi version 12.0.3
- Plots are generated using matplotlib with LaTeX support for consistent formatting

# Contact

Author: Trajana Erlenberg
