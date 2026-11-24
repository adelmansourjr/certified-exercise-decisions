"""Bound and evaluate exercise for a two-future-date arithmetic-average put."""

import argparse
import itertools
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import ndtr, ndtri, roots_hermitenorm, roots_legendre


BLOCKS = 12
CASES = 1024
BUDGET = 19
GUARD = 1e-11
MAX_LEAVES = 36
SQRT_2PI = np.sqrt(2 * np.pi)
TERMS = [
    term for degree in (1, 2, 3)
    for term in itertools.combinations_with_replacement(range(8), degree)
]
THRESHOLDS = np.linspace(-0.01, 0.01, 81)


def sample(rng, size, shifted=False):
    return np.column_stack((
        rng.uniform(0.65, 1.31, size),
        rng.uniform(0.72, 1.08, size),
        rng.choice([7, 15] if shifted else [3, 7, 15], size=size),
        rng.uniform(0.018, 0.13, size),
        rng.uniform(0.10, 0.55, size),
        rng.uniform(*( (0.49, 0.84) if shifted else (0.09, 0.48) ), size=size),
        rng.uniform(*( (0.26, 0.61) if shifted else (0.12, 0.43) ), size=size),
        rng.uniform(*( (0.44, 0.69) if shifted else (0.12, 0.43) ), size=size),
    ))


def payoff(p):
    return np.maximum(0, 1 - p[:, 1])


def dynamics(p):
    s, _, _, r, t1, _, vol1, _ = p.T
    scale = vol1 * np.sqrt(t1)
    log_location = np.log(s) + r * t1 - 0.5 * scale**2
    return log_location, scale


def intermediate(p, x):
    """Compute the intermediate value at a batch of intermediate spots."""
    _, average, number, rate, _, t2, _, vol2 = p.T[:, :, None]
    accrued = number * average
    exercise = np.maximum(0, 1 - (accrued + x) / (number + 1))
    effective_strike = np.maximum(0, number + 2 - accrued - x)
    sigma = vol2 * np.sqrt(t2)
    discount = np.exp(-rate * t2)
    log_moneyness = np.log(np.maximum(1e-200, x)) - np.log(np.maximum(1e-200, effective_strike))
    d1 = (log_moneyness + rate * t2) / sigma + sigma / 2
    final = (discount * effective_strike * ndtr(sigma - d1) - x * ndtr(-d1)) / (number + 2)
    return np.maximum(exercise, final)


def quadrature(p, nodes, weights):
    loc, scale = dynamics(p)
    x = np.exp(loc[:, None] + scale[:, None] * nodes)
    return np.exp(-p[:, 3] * p[:, 4]) * (intermediate(p, x) @ weights)


def hermite(p, nodes):
    x, weights = roots_hermitenorm(nodes)
    return quadrature(p, x, weights / SQRT_2PI)


def composite(p, pieces):
    x, w = roots_legendre(12)
    grid = np.linspace(-9, 9, pieces + 1)
    half_widths = np.diff(grid)[:, None] / 2
    z = (grid[:-1, None] + half_widths * (1 + x)).ravel()
    weights = (half_widths * w).ravel() * np.exp(-z*z / 2) / SQRT_2PI
    return np.concatenate([quadrature(p[i:i + 128], z, weights) for i in range(0, len(p), 128)])


def direct_quad(state, diagnostics=None):
    p = state[None, :]
    spot, average, number, rate, t1, t2, v1, v2 = state
    loc, scale = dynamics(p)
    accrued = average * number
    middle_boundary = number + 1 - accrued
    final_boundary = middle_boundary + 1
    terminal_discount = np.exp(-rate * t2)
    sigma = v2 * np.sqrt(t2)

    def kernel(z):
        s = np.exp(loc[0] + scale[0] * z)
        exercise = max(0, (middle_boundary - s) / (number + 1))
        strike = final_boundary - s
        if strike <= 0:
            value = exercise
        else:
            d1 = (np.log(s / strike) + rate * t2) / sigma + sigma / 2
            continuation = (strike * terminal_discount * ndtr(sigma - d1) - s * ndtr(-d1)) / (number + 2)
            value = max(exercise, continuation)
        return value * np.exp(-z*z / 2) / SQRT_2PI

    breaks = []
    for threshold in (middle_boundary, final_boundary):
        if threshold > 0:
            z = (np.log(threshold) - loc[0]) / scale[0]
            if -9 < z < 9:
                breaks.append(z)
    info = quad(kernel, -9, 9, epsabs=2e-11, epsrel=2e-9, points=breaks, full_output=True)
    result, error = info[:2]
    assert error < 1e-8
    discounted = result * np.exp(-rate * t1)
    if len(info) > 3:
        difference = float(abs(discounted - composite(p, 384)[0]))
        assert difference < 2e-7, (info[3], difference)
        if diagnostics is not None:
            diagnostics.append(difference)
    return discounted


def audit_terminal(p):
    spot, average, number, rate, _, t2, _, v2 = p
    s1 = spot * 0.92
    sigma = v2 * np.sqrt(t2)
    cutoff = number + 2 - number * average - s1
    if cutoff <= 0:
        return 0.0

    def kernel(z):
        final_spot = s1 * np.exp(rate * t2 - sigma * sigma / 2 + sigma * z)
        return max(0, cutoff - final_spot) / (number + 2) * np.exp(-z*z / 2) / SQRT_2PI

    edge = (np.log(cutoff / s1) - rate * t2 + sigma*sigma/2) / sigma
    numeric = np.exp(-rate * t2) * quad(kernel, -9, 9, points=[edge] if -9 < edge < 9 else None, epsabs=1e-12)[0]
    g1 = max(0, 1 - (number * average + s1) / (number + 1))
    return float(abs(intermediate(p[None, :], np.array([[s1]]))[0, 0] - max(g1, numeric)))


def reference(p, audit=False):
    if audit:
        # Independently integrate every held-out state. The faster composite
        # implementation is used for training and audited on held-out states.
        roundoff_checks = []
        fine = np.array([direct_quad(state, roundoff_checks) for state in p])
        sample_indices = np.unique(np.r_[np.arange(8), np.argsort(np.abs(fine - payoff(p)))[:8]])
        check = composite(p[sample_indices], 256)
        errors = abs(check - fine[sample_indices])
        assert max(errors, default=0) < 4e-6
        final_errors = [audit_terminal(p[i]) for i in sample_indices[:8]]
        assert max(final_errors, default=0) < 1e-10
        return fine, {
            "adaptive_quad_calls": len(fine),
            "adaptive_quad_diagnostic_count": len(roundoff_checks),
            "adaptive_quad_diagnostic_max_abs_error": max(roundoff_checks, default=0),
            "composite_audit_count": len(sample_indices),
            "reference_max_audit_abs_error": float(max(errors, default=0)),
            "terminal_independent_audit_count": len(final_errors),
            "terminal_max_audit_abs_error": max(final_errors, default=0),
        }
    fine = composite(p, 64)
    coarse = composite(p, 32)
    replace = (np.abs(fine - coarse) > 4e-6) | (np.abs(fine - payoff(p)) <= 1e-5)
    for i in np.flatnonzero(replace):
        fine[i] = direct_quad(p[i])
    return fine, {}


def split_distribution(p):
    loc, sigma = dynamics(p)
    support = p[:, 2] + 2 - p[:, 2] * p[:, 1]
    forward = np.exp(loc + 0.5 * sigma**2)
    return loc, sigma, support, forward, np.exp(-p[:, 3] * p[:, 4])


def get_spots(loc, sigma, u):
    return np.exp(loc[:, None] + sigma[:, None] * ndtri(u))


def intervals(p, loc, sigma, forward, disc, quantiles, endpoints, values):
    a, b = endpoints[:, :-1], endpoints[:, 1:]
    u, v = quantiles[:, :-1], quantiles[:, 1:]
    mass = v - u
    truncated_first = forward[:, None] * (
        ndtr(ndtri(v) - sigma[:, None]) - ndtr(ndtri(u) - sigma[:, None])
    )
    conditional_mean = np.divide(truncated_first, mass, out=(a + b) / 2, where=mass > 0)
    conditional_mean = np.clip(conditional_mean, a, b)
    ratio = np.divide(conditional_mean - a, b - a, out=np.zeros_like(a), where=b > a)
    lower = intermediate(p, conditional_mean) * mass * disc[:, None]
    chord = values[:, :-1] + ratio * (values[:, 1:] - values[:, :-1])
    upper = np.maximum(lower, chord * mass * disc[:, None])
    return lower, upper


def initial(p, bins):
    loc, sigma, support, forward, disc = split_distribution(p)
    last_cdf = ndtr((np.log(support) - loc) / sigma)
    quantiles = last_cdf[:, None] * np.linspace(0, 1, bins + 1)
    endpoints = get_spots(loc, sigma, quantiles)
    endpoints[:, 0] = 0
    endpoints[:, -1] = support
    evaluations = intermediate(p, endpoints)
    evaluation_slice = intervals(p, loc, sigma, forward, disc, quantiles, endpoints, evaluations)
    return (loc, sigma, forward, disc), (quantiles, endpoints, evaluations, *evaluation_slice)


def certified(lower, upper, strike_payoff):
    return (lower > strike_payoff + GUARD) | (upper < strike_payoff - GUARD) | (strike_payoff <= 0)


def adaptive(p, budget):
    n = len(p)
    bins = 4
    exercise = payoff(p)
    distribution, init = initial(p, bins)
    quantiles, endpoints, values, a, b = init

    def leaf_buffer(initial_values):
        return np.pad(initial_values, ((0, 0), (0, MAX_LEAVES - bins)))

    q0, q1 = leaf_buffer(quantiles[:, :-1]), leaf_buffer(quantiles[:, 1:])
    x0, x1 = leaf_buffer(endpoints[:, :-1]), leaf_buffer(endpoints[:, 1:])
    f0, f1 = leaf_buffer(values[:, :-1]), leaf_buffer(values[:, 1:])
    lower, upper = leaf_buffer(a), leaf_buffer(b)
    low, high = lower.sum(axis=1), upper.sum(axis=1)
    counts = np.full(n, bins, dtype=int)
    calls = np.full(n, 2 * bins + 1, dtype=int)
    remaining = budget * n - int(calls.sum())
    while remaining >= 3:
        active = np.flatnonzero(~certified(low, high, exercise) & (counts < MAX_LEAVES))
        if not len(active):
            break
        available = remaining // 3
        if len(active) > available:
            worst_case_regret = np.minimum(high[active] - exercise[active], exercise[active] - low[active])
            active = active[np.argsort(worst_case_regret, kind="stable")[-available:]]
        chosen = np.argmax(upper[active] - lower[active], axis=1)
        destination = counts[active]
        selected = (active, chosen)
        distribution_sub = [item[active] for item in distribution]
        loc, sigma, _, _ = distribution_sub
        u = (q0[selected] + q1[selected]) / 2
        x = get_spots(loc, sigma, u[:, None])[:, 0]
        x = np.clip(x, x0[selected], x1[selected])
        f = intermediate(p[active], x[:, None])[:, 0]
        next_low, next_high = intervals(
            p[active], *distribution_sub,
            np.column_stack((q0[selected], u, q1[selected])),
            np.column_stack((x0[selected], x, x1[selected])),
            np.column_stack((f0[selected], f, f1[selected])),
        )
        low[active] += next_low.sum(axis=1) - lower[selected]
        high[active] += next_high.sum(axis=1) - upper[selected]
        new = (active, destination)
        q0[new], q1[new] = u, q1[selected]
        x0[new], x1[new] = x, x1[selected]
        f0[new], f1[new] = f, f1[selected]
        q1[selected], x1[selected], f1[selected] = u, x, f
        lower[selected], lower[new] = next_low.T
        upper[selected], upper[new] = next_high.T
        counts[active] += 1
        calls[active] += 3
        remaining -= 3 * len(active)
    # Sum the final leaves directly to avoid drift in cached updates.
    return lower.sum(axis=1), upper.sum(axis=1), calls


def regress(train, label, cal, cal_cont, test):
    centre, spread = train.mean(axis=0), train.std(axis=0)

    def features(states):
        x = (states - centre) / spread
        return np.column_stack((np.ones(len(x)), *[x[:, term].prod(axis=1) for term in TERMS]))

    x = features(train)
    targets = label - payoff(train)
    y = np.column_stack((np.maximum(0, targets), targets))
    coefficients = np.linalg.solve(x.T @ x + 0.02 * np.eye(x.shape[1]), x.T @ y)
    cal_gap = cal_cont - payoff(cal)
    cal_itm = payoff(cal) > 0
    calibrated = features(cal) @ coefficients
    scores = {}
    for index, method in enumerate(("price_target_ridge", "signed_target_ridge")):
        errors = (calibrated[:, index, None] > THRESHOLDS) != (cal_gap[:, None] > 0)
        regret = (errors * (np.abs(cal_gap) * cal_itm)[:, None]).mean(axis=0)
        threshold = THRESHOLDS[regret.argmin()]
        score = features(test) @ coefficients[:, index]
        scores[method] = score + payoff(test), score > threshold
    return scores


def metrics(value, decide, cert, calls, exercise, truth):
    itm = exercise > 0
    wrong = itm & (decide != (truth > exercise))
    regret = wrong * abs(truth - exercise)
    near = itm & (abs(truth - exercise) < 0.003)
    return {
        "regret": float(regret.mean()),
        "wrong_action_rate": float(wrong.mean()),
        "near_boundary_regret": float(regret[near].mean()) if near.any() else None,
        "near_boundary_count": int(near.sum()),
        "value_rmse": float(np.sqrt(np.mean((np.maximum(exercise, value) - np.maximum(exercise, truth))**2))),
        "certificate_fraction": float(cert.mean()),
        "strict_itm_certified": int((cert & itm).sum()),
        "certified_wrong": int((cert & wrong).sum()),
        "evaluations_per_case": float(calls.mean()),
    }


def panel(rng, shifted, train, train_cont, cal, cal_cont):
    random_states = sample(rng, CASES // 2, shifted)
    search = sample(rng, 4096, shifted)
    search = search[payoff(search) > 0]
    selected = np.argsort(np.abs(hermite(search, 7) - payoff(search)))[:CASES // 2]
    p = np.r_[random_states, search[selected]]
    exercise = payoff(p)
    truth, audit = reference(p, audit=True)
    methods, times = {}, {}

    def bounds_method(key, lower, upper, counts, start):
        times[key] = time.perf_counter() - start
        inside = (truth >= lower - 2e-7) & (truth <= upper + 2e-7)
        assert inside.all(), (key, np.flatnonzero(~inside).tolist())
        centre = (lower + upper) / 2
        proof = certified(lower, upper, exercise)
        assert not np.any(proof & (exercise > 0) & ((centre > exercise) != (truth > exercise)))
        methods[key] = centre, centre > exercise, proof, counts

    start = time.perf_counter()
    lower, upper, counts = adaptive(p, BUDGET)
    assert counts.sum() <= CASES * BUDGET
    bounds_method("adaptive_convex", lower, upper, counts, start)
    start = time.perf_counter()
    *_, lower, upper = initial(p, 9)[1]
    bounds_method("fixed_convex_19", lower.sum(axis=1), upper.sum(axis=1), np.full(CASES, 19), start)
    start = time.perf_counter()
    value = hermite(p, BUDGET)
    times["hermite_19"] = time.perf_counter() - start
    methods["hermite_19"] = value, value > exercise, np.zeros(CASES, bool), np.full(CASES, BUDGET)
    start = time.perf_counter()
    low = hermite(p, 7)
    budget_cases = ((BUDGET - 7) * CASES) // 32
    candidates = np.flatnonzero(exercise > 0)
    chosen = candidates[np.argsort(np.abs(low[candidates] - exercise[candidates]))[:budget_cases]]
    low[chosen] = hermite(p[chosen], 32)
    staged_calls = np.full(CASES, 7)
    staged_calls[chosen] += 32
    assert staged_calls.sum() <= CASES * BUDGET
    times["hermite_adaptive_7_32"] = time.perf_counter() - start
    methods["hermite_adaptive_7_32"] = low, low > exercise, np.zeros(CASES, bool), staged_calls
    for name, (value, decide) in regress(train, train_cont, cal, cal_cont, p).items():
        methods[name] = value, decide, np.zeros(CASES, bool), np.zeros(CASES)
    slices = {
        "all": slice(None), "random": slice(0, CASES // 2), "boundary_enriched": slice(CASES // 2, None)
    }
    return {
        "audit": audit,
        "results": {
            label: {
                name: metrics(value[rows], act[rows], cert[rows], calls[rows], exercise[rows], truth[rows])
                for name, (value, act, cert, calls) in methods.items()
            }
            for label, rows in slices.items()
        },
    }, times


def paired(differences, rng):
    d = np.array(differences)
    boot = d[rng.integers(len(d), size=(5000, len(d)))].mean(axis=1)
    return [float(d.mean()), *np.quantile(boot, (0.025, 0.975)).tolist()]


def summarize(blocks, rng):
    first = blocks[0]["results"]
    comparison_pairs = (
        ("adaptive_convex", "hermite_19"),
        ("adaptive_convex", "hermite_adaptive_7_32"),
        ("adaptive_convex", "fixed_convex_19"),
        ("signed_target_ridge", "price_target_ridge"),
    )
    result = {
        label: {
            "mean": {
                method: {
                    name: float(np.mean([value for b in blocks if (value := b["results"][label][method][name]) is not None]))
                    for name in first[label][method]
                }
                for method in first[label]
            },
            "regret_improvement": {
                f"{candidate}_vs_{baseline}": paired([
                    b["results"][label][baseline]["regret"] - b["results"][label][candidate]["regret"]
                    for b in blocks
                ], rng)
                for candidate, baseline in comparison_pairs
            },
        }
        for label in first
    }
    result["audit"] = {
        name: (float(max(b["audit"][name] for b in blocks)) if "_max_" in name
               else int(sum(b["audit"][name] for b in blocks)))
        for name in blocks[0]["audit"]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {"interpolation": [], "shifted": []}
    timing = []
    for i, seed in enumerate(np.random.SeedSequence(7451962).spawn(BLOCKS)):
        rng = np.random.default_rng(seed)
        train, cal = sample(rng, 1100), sample(rng, 400)
        train_values, cal_values = reference(train)[0], reference(cal)[0]
        for label, shifted in (("interpolation", False), ("shifted", True)):
            result, times = panel(rng, shifted, train, train_values, cal, cal_values)
            output[label].append(result)
            timing.append(times)
        print(f"Finished block {i + 1}/{BLOCKS}", file=sys.stderr, flush=True)
    rng = np.random.default_rng(447721)
    data = {
        "design": {"blocks": BLOCKS, "cases_per_block_and_panel": CASES, "evaluations_per_case_budget": BUDGET,
                   "train_labels_per_block": 1100, "calibration_labels_per_block": 400},
        "panels": {label: summarize(blocks, rng) for label, blocks in output.items()},
        "blocks": output,
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    timings = {
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024),
        "median_solver_seconds_per_1024": {key: float(np.median([r[key] for r in timing])) for key in timing[0]},
    }
    print(json.dumps(timings), file=sys.stderr)


if __name__ == "__main__":
    main()
