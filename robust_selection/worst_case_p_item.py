# worst_case_p_item

# Using the worst-case-peritem method to approximate the Robust Selection Problem with discrete uncertainty and the
# min-max criterion.

# Description: The nominal problem is solved with a representative cost vector built with the worst case for each item
# over all scenarios.

def solve_worst_case_p_item(costs, n, p, k, debug=True):
    costs_wc = [
        max(costs[(s, i)] for s in range(1, k + 1))
        for i in range(1, n + 1)
    ]

    selected_items = sorted(
        [(i, costs_wc[i - 1]) for i in range(1, n + 1)],
        key=lambda item: item[1]
    )[:p]

    selected_indices = [i for i, _ in selected_items]

    x_wc = [1 if (i + 1) in selected_indices else 0 for i in range(n)]

    scenario_costs = [
        sum(costs[(s, i + 1)] for i in range(n) if x_wc[i] == 1)
        for s in range(1, k + 1)
    ]
    obj_val_wc = max(scenario_costs)

    if debug:
        print("\n=== Worst-Case-Per-Item Debug ===")
        print("\n--- Worst-case costs per item (costs_wc) ---")
        for i, val in enumerate(costs_wc, start=1):
            print(f"Item {i}: worst-case cost = {val}")

        print("\n--- Sorted worst-case costs (ascending) ---")
        sorted_costs = sorted([(i, costs_wc[i - 1]) for i in range(1, n + 1)], key=lambda t: t[1])
        for rank, (i, val) in enumerate(sorted_costs, start=1):
            print(f"Rank {rank}: item {i}, wc_cost = {val}")

        print(f"\nSelected item indices (p={p} smallest): {selected_indices}")
        print("\nx_wc (binary selection vector):")
        print(x_wc)

        print("\n--- Scenario costs (worst-case-per-item solution) ---")
        for s, cost_s in enumerate(scenario_costs, start=1):
            print(f"Scenario {s}: total cost = {cost_s}")

        print(f"\nReturned objective value (obj_val_wc): {obj_val_wc}")

    return costs_wc, x_wc, obj_val_wc
