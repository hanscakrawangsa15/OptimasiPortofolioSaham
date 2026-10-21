
"""Simulated Annealing for portfolio optimization."""

import time

import numpy as np

import config
from decoder import decode_batch, init_population
from fitness import Evaluator, deb_better


SA_PARAM = {
    "T0": 1.0,
    "T_end_ratio": 0.001,
    "p_swap": 0.35,
    "p_cash": 0.20,
    "sigma_mut": 0.10,
    "calibration_samples": 30,
    "target_acceptance": 0.80,
}


def generate_neighbor(x, P, rng, p_swap, p_cash, sigma_mut):
    """Generate one neighboring chromosome using one move."""
    y = x.copy()
    N = P.n

    eligible = np.flatnonzero(P.layak_beli)
    order = eligible[np.argsort(-y[eligible], kind="stable")]
    selected = order[:min(P.K, len(order))]
    unselected = order[P.K:]

    u = rng.random()

    if u < p_swap:
        # Move 1: exchange a selected stock with an unselected stock.
        if len(selected) and len(unselected):
            a = rng.choice(selected)
            b = rng.choice(unselected)
            y[a], y[b] = y[b], y[a]

    elif u < p_swap + p_cash:
        # Move 2: perturb the cash gene.
        y[-1] += rng.normal(0.0, sigma_mut)

    else:
        # Move 3: transfer raw weight between selected stocks.
        if len(selected) >= 2:
            i, j = rng.choice(selected, size=2, replace=False)

            wi = y[N + i]
            wj = y[N + j]
            delta = rng.normal(0.0, sigma_mut)

            lower = max(-wi, wj - 1.0)
            upper = min(1.0 - wi, wj)

            if lower <= upper:
                delta = np.clip(delta, lower, upper)
                y[N + i] = wi + delta
                y[N + j] = wj - delta

    return np.clip(y, 0.0, 1.0)


def accept_neighbor(R_new, v_new, R_cur, v_cur, T_R, T_v, rng):
    """Apply Deb's feasibility rules and Metropolis acceptance."""
    new = (R_new, v_new)
    current = (R_cur, v_cur)

    if deb_better(new, current):
        return True

    new_feasible = v_new <= 0.0
    cur_feasible = v_cur <= 0.0

    # Never leave a feasible solution for an infeasible one.
    if new_feasible != cur_feasible:
        return False

    if new_feasible:
        delta = R_new - R_cur
        temperature = max(T_R, 1e-12)
        probability = np.exp(np.clip(delta / temperature, -700, 0))
        return rng.random() < probability

    # When both solutions are infeasible, prioritize risk violation.
    delta_v = v_new - v_cur
    if not np.isclose(delta_v, 0.0, atol=1e-12):
        temperature = max(T_v, 1e-12)
        probability = np.exp(np.clip(-delta_v / temperature, -700, 0))
        return rng.random() < probability

    delta = R_new - R_cur
    temperature = max(T_R, 1e-12)
    probability = np.exp(np.clip(delta / temperature, -700, 0))
    return rng.random() < probability



def calibrate_temperature(
    x, R_cur, v_cur, P, ev, rng,
    n_samples, target_acceptance,
    p_swap, p_cash, sigma_mut,
    T_R, T_v, return_rate=False,
):
    """Calibrate initial temperature using sampled neighbor moves."""
    deltas = []

    # Sample neighboring solutions within the shared evaluation budget.
    for _ in range(n_samples):
        if ev.n_eval >= ev.budget:
            break

        y = generate_neighbor(
            x, P, rng,
            p_swap=p_swap,
            p_cash=p_cash,
            sigma_mut=sigma_mut,
        )

        R_new, _, v_new = ev.evaluate(y)
        R_new = float(R_new[0])
        v_new = float(v_new[0])

        # Moves accepted directly by Deb's rules need no temperature.
        if deb_better((R_new, v_new), (R_cur, v_cur)):
            continue

        new_feasible = v_new <= 0.0
        cur_feasible = v_cur <= 0.0

        # Do not calibrate transitions forbidden by feasibility rules.
        if new_feasible != cur_feasible:
            continue

        if new_feasible:
            delta = R_new - R_cur
            if delta < 0:
                deltas.append((delta, T_R))
        else:
            delta_v = v_new - v_cur

            if not np.isclose(delta_v, 0.0, atol=1e-12):
                if delta_v > 0:
                    deltas.append((-delta_v, T_v))
            else:
                delta = R_new - R_cur
                if delta < 0:
                    deltas.append((delta, T_R))

    # If no probabilistic worsening moves were sampled, use a fallback.
    if not deltas:
        temperature = float(SA_PARAM["T0"])
        if return_rate:
            return temperature, None
        return temperature

    def acceptance_rate(T):
        probabilities = [
            np.exp(np.clip(delta / max(scale * T, 1e-12), -700, 0))
            for delta, scale in deltas
        ]
        return float(np.mean(probabilities))

    # Find a temperature whose estimated mean acceptance
    # is as close as possible to the requested target.
    low, high = 1e-8, 1.0

    while acceptance_rate(high) < target_acceptance and high < 1e8:
        high *= 2.0

    for _ in range(60):
        mid = (low + high) / 2.0

        if acceptance_rate(mid) < target_acceptance:
            low = mid
        else:
            high = mid

    temperature = high
    rate = acceptance_rate(temperature)

    if return_rate:
        return temperature, rate
    return temperature


def run(P, budget=config.BUDGET_EVAL, seed=0, trace=False, **kw):
    """Run one Simulated Annealing optimization."""
    prm = {**SA_PARAM, **kw}

    if budget < 2:
        raise ValueError("budget must be at least 2")
    if prm["T0"] <= 0:
        raise ValueError("T0 must be positive")
    if not 0 < prm["T_end_ratio"] < 1:
        raise ValueError("T_end_ratio must be between 0 and 1")
    if not 0 <= prm["p_swap"] <= 1:
        raise ValueError("p_swap must be between 0 and 1")
    if not 0 <= prm["p_cash"] <= 1:
        raise ValueError("p_cash must be between 0 and 1")
    if prm["p_swap"] + prm["p_cash"] > 1:
        raise ValueError("p_swap + p_cash must not exceed 1")
    if prm["sigma_mut"] < 0:
        raise ValueError("sigma_mut must be nonnegative")
    if not 0 < prm["target_acceptance"] < 1:
        raise ValueError("target_acceptance must be between 0 and 1")
    if prm["calibration_samples"] < 1:
        raise ValueError("calibration_samples must be positive")

    rng = np.random.default_rng(seed)
    ev = Evaluator(P, budget)
    start_time = time.perf_counter()

    x = init_population(P, rng, 1, n_seed=1)[0]
    R, sigma, viol = ev.evaluate(x)

    R_cur = float(R[0])
    sigma_cur = float(sigma[0])
    viol_cur = float(viol[0])

    T_R = max(abs(R_cur) * 0.05, 1e-4)
    T_v = max(P.sigma_max * 0.05, 1e-5)

    trace_data = {
        key: [] for key in (
            "R", "sigma", "viol", "g0", "kas",
            "n_unik", "lots_terbaik", "temperature",
        )
    } if trace else None

    def record_trace(temperature):
        if trace_data is None:
            return

        lots, _, cash = decode_batch(x, P)
        best = ev.best

        trace_data["R"].append(np.array([R_cur]))
        trace_data["sigma"].append(np.array([sigma_cur]))
        trace_data["viol"].append(np.array([viol_cur]))
        trace_data["g0"].append(np.array([np.clip(x[-1], 0, 1)]))
        trace_data["kas"].append(cash.copy())
        trace_data["n_unik"].append(
            len({tuple(np.flatnonzero(row > 0)) for row in lots})
        )
        trace_data["lots_terbaik"].append(best["lots"].copy())
        trace_data["temperature"].append(temperature)

    
    # Calibrate temperature while counting every evaluation in the same budget.
    n_cal = min(
        int(prm["calibration_samples"]),
        max(0, budget - ev.n_eval - 1),
    )

    T0 = calibrate_temperature(
        x, R_cur, viol_cur, P, ev, rng,
        n_samples=n_cal,
        target_acceptance=prm["target_acceptance"],
        p_swap=prm["p_swap"],
        p_cash=prm["p_cash"],
        sigma_mut=prm["sigma_mut"],
        T_R=T_R,
        T_v=T_v,
    )

    # Reserve one initial evaluation and all calibration evaluations.
    
    T_end = T0 * prm["T_end_ratio"]
    remaining_steps = budget - ev.n_eval

    if remaining_steps <= 0:
        result = ev.hasil(seed, time.perf_counter() - start_time)
        return result

    # Ada satu langkah pencarian untuk setiap evaluasi tetangga.
    alpha = (T_end / T0) ** (1.0 / remaining_steps)
    T = T0


    record_trace(T)

    while ev.n_eval < budget:
        y = generate_neighbor(
            x, P, rng,
            p_swap=prm["p_swap"],
            p_cash=prm["p_cash"],
            sigma_mut=prm["sigma_mut"],
        )

        R_new, sigma_new, v_new = ev.evaluate(y)
        R_new = float(R_new[0])
        sigma_new = float(sigma_new[0])
        v_new = float(v_new[0])

        if accept_neighbor(
            R_new, v_new, R_cur, viol_cur,
            T_R * T, T_v * T, rng,
        ):
            x = y
            R_cur = R_new
            sigma_cur = sigma_new
            viol_cur = v_new

        T = max(T_end, T * alpha)
        record_trace(T)

    result = ev.hasil(seed, time.perf_counter() - start_time)

    if trace:
        result.trace = {
            key: np.asarray(values)
            for key, values in trace_data.items()
        }

    return result
