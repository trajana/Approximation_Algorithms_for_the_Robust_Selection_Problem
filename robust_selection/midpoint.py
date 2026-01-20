# midpoint.py

# Using the midpoint method to approximate the Robust Selection Problem with discrete uncertainty and the min-max
# criterion.

# Description: The nominal problem is solved with a representative cost vector built with the average costs each item
# over all scenarios.


def solve_midpoint(costs, n, p, k, debug=True):
    costs_av = [
        sum(costs[(s, i)] for s in range(1, k + 1)) / k
        for i in range(1, n + 1)
    ]

    selected_items = sorted(
        [(i, costs_av[i - 1]) for i in range(1, n + 1)],
        key=lambda item: item[1]
    )[:p]

    selected_indices = [i for i, _ in selected_items]

    x_av = [1 if (i + 1) in selected_indices else 0 for i in range(n)]

    obj_val_av = max(
        sum(costs[(s, i + 1)] for i in range(n) if x_av[i] == 1)
        for s in range(1, k + 1)
    )

    if debug:
        print("\n=== Midpoint Debug ===")
        print("\n--- Average costs per item (costs_av) ---")
        for i, val in enumerate(costs_av, start=1):
            print(f"Item {i}: average cost = {val}")
        print(f"\nSelected item indices (p={p} smallest): {selected_indices}")
        print("\nx_av (binary selection vector):")
        print(x_av)
        print(f"\nReturned objective value (obj_val_av): {obj_val_av}")

    return costs_av, x_av, obj_val_av