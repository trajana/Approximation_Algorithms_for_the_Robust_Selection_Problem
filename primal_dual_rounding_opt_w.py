# primal_dual_rounding_opt_w.py

# Using Primal-Dual Rounding to approximate the Robust Selection Problem with discrete uncertainty and the min-max
# criterion and first solving an LP to find the optimal weighted cost vector w.

# Description: There are n items with costs c[s,i]. The goal is to select exactly p items such that the worst-case cost
# across k scenarios is minimized. The algorithm raises a dual variable until constraints become tight and selects items
# accordingly, maintaining dual feasibility. It achieves an approximation guarantee of ≤ 1/β_min (k for uniform weights)

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_opt_w(costs, items, p_rem, k, debug=False):
    if p_rem <= 0:
        raise ValueError("p_rem must be >= 1 in solve_opt_w")

    n_rem = len(items)
    # --- Build cost matrix C[s, i] ---
    C = np.zeros((k, n_rem), dtype=np.float64)
    for j, i in enumerate(items):  # j = 0..n_rem-1, i = original item id
        for s in range(1, k + 1):
            C[s - 1, j] = float(costs[(s, i)])

    try:

        # Create optimization model
        m = gp.Model("LP_opt_w")

        # Create variables
        t = m.addVar(lb=0.0, name="t")  # Continuous variable to be maximized
        w = m.addVars(range(n_rem), lb=0.0, name="w_i")  # Continuous variable for weighted cost vector
        beta = m.addVars(range(1, k + 1), lb=0.0, name="beta")  # Continuous variable for scenario weights
        alpha = m.addVars(range(1, k + 1), name="alpha")  # Dual vaiable for linearization
        lambda_ = m.addVars(range (1, k + 1), range(n_rem), lb=0.0, name="lambda")  # Dual variable for linearization

        # Objective
        m.setObjective(t, GRB.MAXIMIZE)

        # Constraints
        m.addConstrs(
            (w[j] == gp.quicksum(beta[s] * C[s - 1, j] for s in range(1, k + 1)))
            for j in range(n_rem)
        )  # lp:compact-w

        m.addConstr(
            gp.quicksum(beta[s] for s in range(1, k+1)) == 1
        )  # lp:compact-simplex

        m.addConstrs(
            (p_rem * alpha[s] - gp.quicksum(lambda_[s, j] for j in range(n_rem)) >= 0)
            for s in range(1, k + 1)
        )  # lp:compact-dual1

        m.addConstrs(
            (lambda_[s, j] >= alpha[s] - w[j] + t * C[s - 1, j])
            for s in range(1, k + 1) for j in range(n_rem)
        )  # lp:compact-dual2

        # Optimize model
        m.optimize()

        # Optimal value
        if m.status != GRB.OPTIMAL:
            raise RuntimeError(f"LP_opt_w not optimal, status={m.status}")

        beta_star = np.array([beta[s].X for s in range(1, k + 1)], dtype=float)
        w_star = {items[j]: float(w[j].X) for j in range(n_rem)}
        t_star = float(t.X)
        alpha_star = np.array([alpha[s].X for s in range(1, k + 1)], dtype=float)
        lambda_star = np.array([[lambda_[s, j].X for j in range(n_rem)] for s in range(1, k + 1)], dtype=float)

        return t_star, beta_star, w_star, alpha_star, lambda_star

    # Error handling
    except gp.GurobiError as e:
        raise RuntimeError(
            f"Gurobi failed while solving the model (error code {e.errno}): {e}") from e
    except AttributeError as e:
        raise RuntimeError(
            "Failed to access solution attributes. "
            "This usually means the model was not solved to optimality.") from e


def select_p_smallest_w(w_star, costs, n, p, k, debug=True):

    # --- Select p smallest items by w_star ---
    selected_items = sorted(
        w_star.items(),  # (item_id, w_value)
        key=lambda kv: (kv[1], kv[0])  # (w, id)
    )[:p]

    selected_indices = [i for i, _ in selected_items]  # 1-based indices

    # --- Build binary selection vector x (length ) ---
    x_opt_w = [1 if (i + 1) in selected_indices else 0 for i in range(n)]

    # --- Evaluate robust objective value ---
    scenario_costs = [
        sum(float(costs[(s, i)]) for i in selected_indices)
        for s in range(1, k + 1)
    ]
    obj_val_opt_w = max(scenario_costs)

    return w_star, x_opt_w, obj_val_opt_w

def solve_opt_w_then_select_once(costs, n, p, k, debug=False):
    # Step 1: solve LP => get t*, beta*, w*, alpha*, lambda*
    items = list(range(1, n + 1))
    t_star, beta_star, w_star, alpha_star, lambda_star = solve_opt_w(
        costs=costs, items=items, p_rem=p, k=k, debug=debug
    )

    # Step 2: select p smallest items wrt w*
    # returns (w_star, x_opt_w, obj_val_opt_w)
    w_star, x_opt_w, obj_val_opt_w = select_p_smallest_w(
        w_star=w_star,
        costs=costs,
        n=n,
        p=p,
        k=k,
        debug=debug
    )

    return t_star, beta_star, w_star, x_opt_w, obj_val_opt_w


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


def solve_opt_w_then_select_adaptive_remember(costs, n, p, k, debug=False):
    """
    Adaptive remember variant
      - costs: ORIGINAL-Kosten costs[(s, i)]
      - repeat until |chosen| = p:
          * baue geshiftete Kosten für die verbleibenden Items (pool),
            die die Fixkosten von 'chosen' einpreisen
          * löse LP mit solve_opt_w auf diesen geshifteten Kosten
          * wähle EIN Item mit kleinstem w*
    """
    costs_original = costs  # zur Klarheit, das sind die echten Kosten

    pool = list(range(1, n + 1))
    chosen = []
    t_hist = []
    beta_hist = []

    while len(chosen) < p:
        p_rem = p - len(chosen)

        # 1) Geshiftete Kosten für die aktuelle Situation bauen
        costs_shift = build_shifted_costs(
            costs=costs_original,
            fixed=chosen,
            pool=pool,
            p_rem=p_rem,
            k=k
        )

        # 2) "Normales" LP auf den geshifteten Kosten und den verbleibenden Items
        t_star, beta_star, w_star, _, _ = solve_opt_w(
            costs=costs_shift,
            items=pool,
            p_rem=p_rem,
            k=k,
            debug=debug
        )

        # 3) Item mit kleinstem w* wählen
        next_item, next_w = min(w_star.items(), key=lambda kv: (kv[1], kv[0]))
        chosen.append(next_item)
        pool.remove(next_item)

        t_hist.append(t_star)
        beta_hist.append(beta_star)

        if debug:
            approx = (1.0 / t_star) if t_star > 0 else float("inf")
            print(f"\nIter {len(chosen)}/{p}: p_rem={p_rem}")
            print(f"  picked item {next_item} with w*={next_w}")
            print(f"  t*={t_star}  => approx={approx}")
            print(f"  beta*={beta_star}")

    # Binärer Vektor über alle n Items (1 wenn gewählt)
    x_opt_w = [1 if (i in chosen) else 0 for i in range(1, n + 1)]

    # Bewertung mit ORIGINAL-Kosten, nicht mit geshifteten
    scenario_costs = [
        sum(float(costs_original[(s, i)]) for i in chosen)
        for s in range(1, k + 1)
    ]
    obj_val_opt_w = max(scenario_costs)

    if debug:
        print("\n=== Adaptive-remember result (Shift-Trick) ===")
        print("Chosen items:", chosen)
        print("Objective (worst-case, original costs):", obj_val_opt_w)

    return chosen, x_opt_w, obj_val_opt_w, t_hist, beta_hist

#
# def solve_opt_w_ntimes_then_select_adaptive_remember(costs, n, p, k, debug=False):
#     """
#     Adaptive n-times Variante mit Remember und Shift.
#
#     Idee:
#       - costs: ORIGINAL-Kosten costs[(s, i)]
#       - initial: löse Baseline-LP auf allen Items (keine Fixen) mit p_rem = p
#       - in jeder Iteration:
#           * p_rem = p - |chosen|
#           * wenn p_rem == 1:
#                 - prüfe alle Kandidaten i0 in pool
#                 - wähle das i0 mit minimaler robuster Zielfunktion (keine LPs mehr)
#           * wenn p_rem >= 2:
#                 - für jeden Kandidaten i0 in pool:
#                     + fixed_now = chosen + [i0]
#                     + pool_wo = pool \ {i0}
#                     + baue geshiftete Kosten für (fixed_now, pool_wo, p_rem_after)
#                     + löse solve_opt_w(...) auf diesen geshifteten Kosten
#                     + wähle den Kandidaten mit größtem t_star
#     """
#
#     costs_original = costs  # zur Klarheit: das sind die echten Kosten
#
#     def robust_obj_of_set(item_list):
#         scenario_costs = [
#             sum(float(costs_original[(s, i)]) for i in item_list)
#             for s in range(1, k + 1)
#         ]
#         return max(scenario_costs)
#
#     pool = list(range(1, n + 1))
#     chosen = []
#     hist = []
#
#     # --- BASELINE LP (ohne Fix-Items, p_rem = p) ---
#     # hier brauchen wir kein shifting, da fixed = []
#     t0, beta0, w0, _, _ = solve_opt_w(
#         costs=costs_original,
#         items=pool,   # alle Items
#         p_rem=p,      # p Items wählen
#         k=k,
#         debug=False
#     )
#
#     hist.append({
#         "baseline": True,
#         "t0": float(t0),
#         "t_star": float(t0),
#         "beta0": [float(x) for x in beta0],
#         "beta_star": [float(x) for x in beta0],
#     })
#
#     # --- Iteratives Auswählen ---
#     while len(chosen) < p:
#         p_rem = p - len(chosen)
#
#         # SPEZIALFALL: Letzte Iteration -> nur noch 1 Item zu wählen
#         # Hier macht es wenig Sinn, p_rem_after = 0 in ein LP zu packen.
#         # Stattdessen: wähle das Item mit bester robuster Zielfunktion.
#         if p_rem == 1:
#             best_candidate = None
#             best_obj = float("inf")
#
#             for i0 in pool:
#                 completion = chosen + [i0]
#                 obj = robust_obj_of_set(completion)
#                 if (obj < best_obj) or (obj == best_obj and (best_candidate is None or i0 < best_candidate)):
#                     best_candidate = i0
#                     best_obj = obj
#
#             chosen.append(best_candidate)
#             pool.remove(best_candidate)
#
#             hist.append({
#                 "picked": best_candidate,
#                 "t_star": None,          # hier kein LP mehr
#                 "beta_star": None,
#                 "completion_used_for_eval": chosen[:],
#                 "obj_if_completed_now": best_obj,
#                 "pool_size_after": len(pool),
#                 "last_step_rule": "robust_obj_min",
#             })
#             break  # fertig, |chosen| = p
#
#         # NORMALFALL: p_rem >= 2 -> n-mal LP probing
#         best_candidate = None
#         best_t = -float("inf")   # wir maximieren t
#         best_beta = None
#         best_completion = None
#         best_obj = None          # nur Logging
#
#         for i0 in pool:
#             p_rem_after = p_rem - 1
#             fixed_now = chosen + [i0]
#             pool_wo = [j for j in pool if j != i0]
#
#             # Shift-Kosten für diese Konstellation (fixed_now, pool_wo, p_rem_after)
#             costs_shift = build_shifted_costs(
#                 costs=costs_original,
#                 fixed=fixed_now,
#                 pool=pool_wo,
#                 p_rem=p_rem_after,
#                 k=k
#             )
#
#             # LP auf geshifteten Kosten, p_rem_after Items aus pool_wo wählen
#             t_star, beta_star, w_star, _, _ = solve_opt_w(
#                 costs=costs_shift,
#                 items=pool_wo,
#                 p_rem=p_rem_after,
#                 k=k,
#                 debug=False
#             )
#
#             # Rest auffüllen (nur für completion)
#             if p_rem_after > 0:
#                 picked_rest = [
#                     item_id
#                     for item_id, _ in sorted(
#                         w_star.items(),
#                         key=lambda kv: (kv[1], kv[0])
#                     )[:p_rem_after]
#                 ]
#             else:
#                 picked_rest = []
#
#             completion = fixed_now + picked_rest
#             obj = robust_obj_of_set(completion)
#
#             # Auswahl: größtes t_star, bei Gleichstand kleinere Item-ID
#             if (t_star > best_t) or (t_star == best_t and (best_candidate is None or i0 < best_candidate)):
#                 best_candidate = i0
#                 best_t = t_star
#                 best_beta = beta_star
#                 best_completion = completion
#                 best_obj = obj
#
#         # Besten Kandidaten der Runde endgültig wählen
#         chosen.append(best_candidate)
#         pool.remove(best_candidate)
#
#         hist.append({
#             "picked": best_candidate,
#             "t_star": best_t,
#             "beta_star": best_beta,
#             "completion_used_for_eval": best_completion,
#             "obj_if_completed_now": best_obj,
#             "pool_size_after": len(pool),
#             "last_step_rule": "max_t_star",
#         })
#
#         if debug:
#             approx = (1.0 / best_t) if (best_t is not None and best_t > 0) else float("inf")
#             print(f"\nIter {len(chosen)}/{p}: p_rem={p_rem}")
#             print(f"  picked item {best_candidate}")
#             print(f"  t*={best_t}  => approx={approx}")
#             print(f"  beta*={best_beta}")
#             print(f"  completion robust obj (for info) = {best_obj}")
#             print(f"  chosen so far: {chosen}")
#
#     # Binärer Vektor
#     x_opt_w = [1 if i in chosen else 0 for i in range(1, n + 1)]
#     obj_val_opt_w = robust_obj_of_set(chosen)
#
#     if debug:
#         print("\n=== N-times adaptive-remember FINAL (Shift-Trick) ===")
#         print("Chosen items:", chosen)
#         print("Final worst-case objective (original costs):", obj_val_opt_w)
#
#     return chosen, x_opt_w, obj_val_opt_w, hist


def solve_two_branches_smallest_wi(costs, n, p, k, debug=False):
    """
    One-level branching on the item with smallest w_i from the baseline LP.

    Schritte:
      1) solve_opt_w auf allen Items -> (t0, beta0, w0)
      2) i_star = Item mit kleinstem w0[i]
      3) Branch IN:
           - i_star wird fix gewählt
           - verwende Cost-Shift auf pool_in = all \ {i_star}, p_rem_in = p-1
           - solve_opt_w mit geshifteten Kosten => (t_in, beta_in, w_in)
           - S_in = {i_star} ∪ (p-1 Items mit kleinstem w_in)
      4) Branch OUT:
           - i_star verboten, pool_out = all \ {i_star}
           - solve_opt_w auf Originalkosten, p_rem_out = p
           - S_out = p Items mit kleinstem w_out
      5)    - Wir wählen den Branch mit dem besseren Zielfunktionswert (kleineres robust_obj).
            - Die Garantie (Approx.-Konstante) ist trotzdem die schlechtere der beiden,
            also min(t_in, t_out).
      6) Rückgabe: chosen set, x-Vektor, robuster Wert, Info-Dict
    """

    costs_original = costs  # zur Klarheit

    # --- Baseline LP auf allen Items ---
    items_all = list(range(1, n + 1))
    t0, beta0, w0, _, _ = solve_opt_w(
        costs=costs_original,
        items=items_all,
        p_rem=p,
        k=k,
        debug=False
    )

    # --- Item für Branching wählen: kleinstes (nicht--0) w_i ---
    eps = 1e-9

    # Kandidaten nur mit w_i > eps
    positive_candidates = [(i, wi) for i, wi in w0.items() if wi > eps]

    if positive_candidates:
        # kleinstes strictly positives w_i (tie-break: kleinste ID)
        i_star, w_i_star = min(positive_candidates, key=lambda kv: (kv[1], kv[0]))
    else:
        # Fallback: alle w_i sind (nahezu) 0 → nimm das mit kleinster ID
        i_star, w_i_star = min(w0.items(), key=lambda kv: kv[0])

    # Helper: robuster Zielfunktionswert (mit ORIGINAL-Kosten)
    def robust_obj_of_set(item_list):
        scenario_costs = [
            sum(float(costs_original[(s, i)]) for i in item_list)
            for s in range(1, k + 1)
        ]
        return max(scenario_costs)

    # ---------- Branch IN: i_star MUSS in die Lösung ----------

    pool_in = [i for i in items_all if i != i_star]
    fixed_in = [i_star]
    p_rem_in = p - 1

    if p_rem_in < 0:
        raise ValueError("p must be at least 1.")

    if p_rem_in == 0:
        # Fall p = 1: Lösung ist genau {i_star}.
        # Es gibt nichts mehr zu optimieren; wir übernehmen einfach baseline-Garantie.
        t_in = t0
        beta_in = beta0
        w_in = {}
        S_in = [i_star]
    else:
        # Shift-Kosten für (fixed_in, pool_in, p_rem_in) bauen
        costs_shift_in = build_shifted_costs(
            costs=costs_original,
            fixed=fixed_in,
            pool=pool_in,
            p_rem=p_rem_in,
            k=k
        )

        # solve_opt_w auf dem verbleibenden Pool mit geshifteten Kosten
        t_in, beta_in, w_in, _, _ = solve_opt_w(
            costs=costs_shift_in,
            items=pool_in,
            p_rem=p_rem_in,
            k=k,
            debug=debug
        )

        # p_rem_in Items mit kleinsten w_in wählen
        picked_rest_in = [
            item_id
            for item_id, _ in sorted(
                w_in.items(),
                key=lambda kv: (kv[1], kv[0])
            )[:p_rem_in]
        ]
        S_in = [i_star] + picked_rest_in

    # ---------- Branch OUT: i_star VERBOTEN ----------

    pool_out = [i for i in items_all if i != i_star]
    p_rem_out = p

    if p_rem_out > len(pool_out):
        raise ValueError("Cannot select p items after excluding i_star (p > n-1).")

    # Hier gibt es keine Fix-Items, also auch kein Shifting notwendig
    t_out, beta_out, w_out, _, _ = solve_opt_w(
        costs=costs_original,
        items=pool_out,
        p_rem=p_rem_out,
        k=k,
        debug=debug
    )

    picked_out = [
        item_id
        for item_id, _ in sorted(
            w_out.items(),
            key=lambda kv: (kv[1], kv[0])
        )[:p_rem_out]
    ]
    S_out = picked_out


    # ---------- Neu: Zielfunktionswerte vergleichen ----------
    obj_in = robust_obj_of_set(S_in)
    obj_out = robust_obj_of_set(S_out)

    # Branch mit BESSEREM Zielfunktionswert wählen (kleineres obj)
    if (obj_in < obj_out) or (obj_in == obj_out and t_in >= t_out):
        chosen = S_in
        obj = obj_in
        branch = "IN"
        t_solution = t_in      # t des Branches, dessen Lösung wir zurückgeben
        beta_solution = beta_in
    else:
        chosen = S_out
        obj = obj_out
        branch = "OUT"
        t_solution = t_out
        beta_solution = beta_out

    # ---------- Schlechtere Garantie bestimmen ----------
    # schlechtere Garantie = kleineres t
    if t_in <= t_out:
        t_chosen = t_in
        beta_chosen = beta_in
    else:
        t_chosen = t_out
        beta_chosen = beta_out

    # x-Vektor (0/1 der Länge n)
    x = [1 if i in chosen else 0 for i in range(1, n + 1)]

    if debug:
        approx0 = (1.0 / t0) if t0 > 0 else float("inf")
        approx_in = (1.0 / t_in) if t_in > 0 else float("inf")
        approx_out = (1.0 / t_out) if t_out > 0 else float("inf")
        approx_solution = (1.0 / t_solution) if t_solution > 0 else float("inf")
        approx_guarantee = (1.0 / t_chosen) if t_chosen > 0 else float("inf")

        print("\n=== Two-Branch Smallest-w_i (Shift-Trick) ===")
        print(f"Baseline: t0={t0}  approx={approx0}")
        print(f"i* = {i_star} with w0[i*] = {w_i_star}")
        print(f"Branch IN : t_in={t_in}  approx={approx_in}  |S_in|={len(S_in)}  S_in={S_in}  obj_in={obj_in}")
        print(f"Branch OUT: t_out={t_out} approx={approx_out} |S_out|={len(S_out)} S_out={S_out} obj_out={obj_out}")
        print(f"Chosen branch (by obj) = {branch}")
        print(f"  -> solution t (this branch) = {t_solution} (approx={approx_solution})")
        print(f"  -> guarantee uses t_chosen = {t_chosen} (approx={approx_guarantee})")
        print(f"Chosen set: {chosen}")
        print(f"Robust objective (worst-case, original costs): {obj}")

    info = {
        "i_star": i_star,
        "w_i_star": w_i_star,
        "t0": t0,
        "beta0": beta0,
        "t_in": t_in,
        "beta_in": beta_in,
        "t_out": t_out,
        "beta_out": beta_out,
        "chosen_branch": branch,     # Branch der endgültigen Lösung (nach obj)
        "t_chosen": t_chosen,        # t der SCHLECHTEREN Garantie (min(t_in, t_out))
        "beta_chosen": beta_chosen,  # zugehöriges beta der schlechteren Garantie
        "S_in": S_in,
        "S_out": S_out,
    }

    return chosen, x, obj, info


def solve_two_branches_biggest_wi(costs, n, p, k, debug=False):
    """
    One-level branching on the item with biggest *fractional* w_i from the baseline LP,
    MIT Remember-Logik via Cost-Shift.

    Schritte:
      1) Baseline: solve_opt_w auf allen Items -> (t0, beta0, w0)
      2) i_star = Item mit größtem (nicht fast-0) w0[i]
      3) Branch IN:
           - i_star wird fix gewählt
           - verwende Cost-Shift auf pool_in = all \ {i_star}, p_rem_in = p-1
           - solve_opt_w mit geshifteten Kosten => (t_in, beta_in, w_in)
           - S_in = {i_star} ∪ (p-1 Items mit kleinstem w_in)
      4) Branch OUT:
           - i_star verboten, pool_out = all \ {i_star}
           - solve_opt_w auf Originalkosten, p_rem_out = p
           - S_out = p Items mit kleinstem w_out
      5) Wähle den Branch mit schlechterer Garantie (kleinerem t)
      6) Rückgabe: chosen set, x-Vektor, robuster Wert, Info-Dict
    """
    costs_original = costs  # zur Klarheit

    # --- Baseline LP auf allen Items ---
    items_all = list(range(1, n + 1))
    t0, beta0, w0, _, _ = solve_opt_w(
        costs=costs_original,
        items=items_all,
        p_rem=p,
        k=k,
        debug=False
    )

    # --- Item für Branching wählen: größtes (nicht-0) w_i ---
    eps = 1e-9
    positive_candidates = [(i, wi) for i, wi in w0.items() if wi > eps]
    if positive_candidates:
        # größtes strictly positives w_i (tie-break: kleinste ID)
        i_star, w_i_star = max(positive_candidates, key=lambda kv: (kv[1], -kv[0]))
    else:
        # Fallback: alle w_i sind (nahezu) 0 → nimm die kleinste ID
        i_star = min(w0.keys())
        w_i_star = w0[i_star]

    # Helper: robuster Zielfunktionswert einer Menge (mit ORIGINAL-Kosten)
    def robust_obj_of_set(item_list):
        scenario_costs = [
            sum(float(costs_original[(s, i)]) for i in item_list)
            for s in range(1, k + 1)
        ]
        return max(scenario_costs)

    # ---------- Branch IN: i_star MUSS in die Lösung ----------
    pool_in = [i for i in items_all if i != i_star]
    fixed_in = [i_star]
    p_rem_in = p - 1
    if p_rem_in < 0:
        raise ValueError("p must be at least 1.")

    if p_rem_in == 0:
        # Fall p = 1: Lösung ist genau {i_star}.
        # Es gibt nichts mehr zu optimieren; wir übernehmen einfach baseline-Garantie.
        t_in = t0
        beta_in = beta0
        w_in = {}
        S_in = [i_star]
    else:
        # Shift-Kosten für (fixed_in, pool_in, p_rem_in) bauen
        costs_shift_in = build_shifted_costs(
            costs=costs_original,
            fixed=fixed_in,
            pool=pool_in,
            p_rem=p_rem_in,
            k=k
        )
        # solve_opt_w auf dem verbleibenden Pool mit geshifteten Kosten
        t_in, beta_in, w_in, _, _ = solve_opt_w(
            costs=costs_shift_in,
            items=pool_in,
            p_rem=p_rem_in,
            k=k,
            debug=debug
        )
        # p_rem_in Items mit kleinsten w_in wählen
        picked_rest_in = [
            item_id
            for item_id, _ in sorted(
                w_in.items(),
                key=lambda kv: (kv[1], kv[0])
            )[:p_rem_in]
        ]
        S_in = [i_star] + picked_rest_in

    # ---------- Branch OUT: i_star VERBOTEN ----------
    pool_out = [i for i in items_all if i != i_star]
    p_rem_out = p
    if p_rem_out > len(pool_out):
        raise ValueError("Cannot select p items after excluding i_star (p > n-1).")

    # Keine Fix-Items -> kein Shifting nötig
    t_out, beta_out, w_out, _, _ = solve_opt_w(
        costs=costs_original,
        items=pool_out,
        p_rem=p_rem_out,
        k=k,
        debug=debug
    )
    picked_out = [
        item_id
        for item_id, _ in sorted(
            w_out.items(),
            key=lambda kv: (kv[1], kv[0])
        )[:p_rem_out]
    ]
    S_out = picked_out


    # ---------- Neu: Zielfunktionswerte vergleichen ----------
    obj_in = robust_obj_of_set(S_in)
    obj_out = robust_obj_of_set(S_out)

    # Branch mit BESSEREM Zielfunktionswert wählen
    if (obj_in < obj_out) or (obj_in == obj_out and t_in >= t_out):
        chosen = S_in
        obj = obj_in
        branch = "IN"
        t_solution = t_in
        beta_solution = beta_in
    else:
        chosen = S_out
        obj = obj_out
        branch = "OUT"
        t_solution = t_out
        beta_solution = beta_out

    # Schlechtere Garantie
    if t_in <= t_out:
        t_chosen = t_in
        beta_chosen = beta_in
    else:
        t_chosen = t_out
        beta_chosen = beta_out

    # x-Vektor und robuster Wert (obj)
    x = [1 if i in chosen else 0 for i in range(1, n + 1)]

    if debug:
        approx0 = (1.0 / t0) if t0 > 0 else float("inf")
        approx_in = (1.0 / t_in) if t_in > 0 else float("inf")
        approx_out = (1.0 / t_out) if t_out > 0 else float("inf")
        approx_solution = (1.0 / t_solution) if t_solution > 0 else float("inf")
        approx_guarantee = (1.0 / t_chosen) if t_chosen > 0 else float("inf")

        print("\n=== Two-Branch Biggest-w_i (Shift-Trick) ===")
        print(f"Baseline: t0={t0}  approx={approx0}")
        print(f"i* = {i_star} with w0[i*] = {w_i_star}")
        print(f"Branch IN : t_in={t_in}  approx={approx_in}  |S_in|={len(S_in)}  S_in={S_in}  obj_in={obj_in}")
        print(f"Branch OUT: t_out={t_out} approx={approx_out} |S_out|={len(S_out)} S_out={S_out} obj_out={obj_out}")
        print(f"Chosen branch (by obj) = {branch}")
        print(f"  -> solution t (this branch) = {t_solution} (approx={approx_solution})")
        print(f"  -> guarantee uses t_chosen = {t_chosen} (approx={approx_guarantee})")
        print(f"Chosen set: {chosen}")
        print(f"Robust objective (worst-case, original costs): {obj}")

    info = {
        "i_star": i_star,
        "w_i_star": w_i_star,
        "t0": t0,
        "beta0": beta0,
        "t_in": t_in,
        "beta_in": beta_in,
        "t_out": t_out,
        "beta_out": beta_out,
        "chosen_branch": branch,     # welcher Branch liefert die Lösung
        "t_chosen": t_chosen,        # schlechteste Garantie
        "beta_chosen": beta_chosen,  # zugehöriges beta
        "S_in": S_in,
        "S_out": S_out,
    }

    return chosen, x, obj, info



#
# def solve_opt_w_remember(costs, pool, fixed, p_rem, k, debug=False):
#     """
#     Solve the compact LP on the remaining pool, while accounting for already fixed items.
#     Returns: t_star, beta_star, w_star_dict (for pool items), alpha_star, lambda_star
#     """
#     n_rem = len(pool)
#
#     # Cost matrix for pool items
#     C = np.zeros((k, n_rem), dtype=np.float64)
#     for j, item_id in enumerate(pool):
#         for s in range(1, k + 1):
#             C[s - 1, j] = float(costs[(s, item_id)])
#
#     # Precompute scenario sums of fixed items: fixed_cost[s] = sum_{i in fixed} c_i^s
#     fixed_cost = np.zeros(k, dtype=np.float64)
#     for s in range(1, k + 1):
#         fixed_cost[s - 1] = sum(float(costs[(s, i)]) for i in fixed) if fixed else 0.0
#
#     try:
#         m = gp.Model("LP_opt_w_remember")
#         m.Params.OutputFlag = 1 if debug else 0
#
#         t = m.addVar(lb=0.0, name="t")
#         w = m.addVars(range(n_rem), lb=0.0, name="w")
#         beta = m.addVars(range(1, k + 1), lb=0.0, name="beta")
#         alpha = m.addVars(range(1, k + 1), name="alpha")
#         lambda_ = m.addVars(range(1, k + 1), range(n_rem), lb=0.0, name="lambda")
#
#         m.setObjective(t, GRB.MAXIMIZE)
#
#         # w_j = sum_s beta_s * c_{s,j}
#         m.addConstrs(
#             (w[j] == gp.quicksum(beta[s] * C[s - 1, j] for s in range(1, k + 1)))
#             for j in range(n_rem)
#         )
#         m.addConstr(gp.quicksum(beta[s] for s in range(1, k + 1)) == 1)
#
#         # sum_{i in fixed} w_i  =  sum_s beta_s * sum_{i in fixed} c_i^s
#         sum_w_fixed = gp.quicksum(beta[s] * fixed_cost[s - 1] for s in range(1, k + 1))
#
#         # REMEMBER-dual1:
#         # p_rem * alpha_s - sum_j lambda_{s,j} >= - sum_{i in fixed}(w_i - t c_i^s)
#         # <=> p_rem*alpha_s - sum_j lambda_{s,j} + sum_w_fixed - t*fixed_cost[s] >= 0
#         m.addConstrs(
#             (p_rem * alpha[s]
#              - gp.quicksum(lambda_[s, j] for j in range(n_rem))
#              + sum_w_fixed
#              - t * float(fixed_cost[s - 1]) >= 0)
#             for s in range(1, k + 1)
#         )
#
#         # dual2 unchanged (only pool items)
#         m.addConstrs(
#             (lambda_[s, j] >= alpha[s] - w[j] + t * C[s - 1, j])
#             for s in range(1, k + 1) for j in range(n_rem)
#         )
#
#         m.optimize()
#         if m.status != GRB.OPTIMAL:
#             raise RuntimeError(f"LP_opt_w_remember not optimal, status={m.status}")
#
#         t_star = float(t.X)
#         beta_star = np.array([beta[s].X for s in range(1, k + 1)], dtype=float)
#         w_star = {pool[j]: float(w[j].X) for j in range(n_rem)}
#         alpha_star = np.array([alpha[s].X for s in range(1, k + 1)], dtype=float)
#         lambda_star = np.array([[lambda_[s, j].X for j in range(n_rem)] for s in range(1, k + 1)], dtype=float)
#
#         return t_star, beta_star, w_star, alpha_star, lambda_star
#
#     except gp.GurobiError as e:
#         raise RuntimeError(f"Gurobi failed (error code {e.errno}): {e}") from e



def solve_opt_w_ntimes_then_select_adaptive_remember_rob_obj(costs, n, p, k, debug=False):
    """
    Adaptive variant with n-times probing each iteration, BUT with remembering.

    Iteration r:
      - already fixed items: chosen
      - remaining pool: pool
      - need to pick p_rem = p - |chosen| more items

      For every candidate item i0 in pool:
        - force i0 to be chosen NOW, while also remembering already chosen items
          => solve remember-LP with fixed = chosen + [i0]
             on pool_without_i0 and p_rem' = p_rem - 1
        - complete the set by taking the (p_rem-1) smallest w* from that LP
        - evaluate robust objective of the completed set (chosen + i0 + picked_rest)

      Pick the candidate i0 that yields the best (smallest) robust objective.
      Commit it: chosen += [i0], pool -= {i0}.
      Next iteration: remember chosen (fixed=chosen) in all candidate LPs.
    """

    def robust_obj_of_set(item_list):
        scenario_costs = [
            sum(float(costs[(s, i)]) for i in item_list)
            for s in range(1, k + 1)
        ]
        return max(scenario_costs)

    pool = list(range(1, n + 1))
    chosen = []

    hist = []  # list of dicts per iteration

    # --- BASELINE LP (the only guarantee-compatible one) ---
    t0, beta0, w0, _, _ = solve_opt_w_remember(
        costs=costs,
        pool=pool,  # all items
        fixed=[],  # nothing fixed
        p_rem=p,  # select p
        k=k,
        debug=False
    )
    hist.append({
        "baseline": True,
        "t0": float(t0),
        "beta0": [float(x) for x in beta0],
    })

    while len(chosen) < p:
        p_rem = p - len(chosen)

        best_candidate = None
        best_obj = float("inf")
        best_t = None
        best_beta = None
        best_completion = None

        # Try forcing each candidate i0 as the next packed item
        for i0 in pool:
            p_rem_after = p_rem - 1
            fixed_now = chosen + [i0]

            if p_rem_after == 0:
                completion = fixed_now
                obj = robust_obj_of_set(completion)
                # tie-break: smaller id wins
                if (obj < best_obj) or (obj == best_obj and (best_candidate is None or i0 < best_candidate)):
                    best_candidate = i0
                    best_obj = obj
                    best_t = None
                    best_beta = None
                    best_completion = completion
                continue

            pool_wo = [j for j in pool if j != i0]

            # REMEMBER: include already chosen items in fixed
            t_star, beta_star, w_star, _, _ = solve_opt_w_remember(
                costs=costs,
                pool=pool_wo,
                fixed=fixed_now,        # <-- remember chosen + force i0
                p_rem=p_rem_after,      # choose remaining items count
                k=k,
                debug=False
            )

            picked_rest = [
                item_id
                for item_id, _ in sorted(w_star.items(), key=lambda kv: (kv[1], kv[0]))[:p_rem_after]
            ]

            completion = fixed_now + picked_rest
            obj = robust_obj_of_set(completion)

            if (obj < best_obj) or (obj == best_obj and (best_candidate is None or i0 < best_candidate)):
                best_candidate = i0
                best_obj = obj
                best_t = t_star
                best_beta = beta_star
                best_completion = completion

        # Commit the best next item
        chosen.append(best_candidate)
        pool.remove(best_candidate)

        hist.append({
            "picked": best_candidate,
            "obj_if_completed_now": best_obj,
            "t_star": best_t,
            "beta_star": best_beta,
            "completion_used_for_eval": best_completion,
            "pool_size_after": len(pool),
        })

    x_opt_w = [1 if i in chosen else 0 for i in range(1, n + 1)]
    obj_val_opt_w = robust_obj_of_set(chosen)

    if debug:
        print("\n=== N-times adaptive-remember FINAL ===")
        print("Chosen items:", chosen)
        print("Final worst-case objective:", obj_val_opt_w)

    return chosen, x_opt_w, obj_val_opt_w, hist