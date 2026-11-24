"""Compare training targets for one-step American put exercise decisions."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


TRIALS = 16
DEGREES = [term for d in (1, 2, 3) for term in itertools.combinations_with_replacement(range(4), d)]
THRESHOLDS = np.linspace(-0.02, 0.02, 81)
METRICS = ("regret", "pricing_rmse", "wrong_itm_rate", "near_boundary_regret", "boundary_count")


def sample(rng, n, shifted=False):
    return np.column_stack((
        rng.uniform(0.65, 1.30, n),
        rng.uniform(*( (0.49, 0.68) if shifted else (0.10, 0.48) ), n),
        rng.uniform(0.01, 0.12, n),
        rng.uniform(0.04, 1.25, n),
    ))


def values(contracts, steps):
    spot, volatility, rate, maturity = contracts.T
    dt = maturity / steps
    log_up = volatility * np.sqrt(dt)
    up = np.exp(log_up)
    discount = np.exp(-rate * dt)
    probability = (1 / discount - 1 / up) / (up - 1 / up)
    assert np.all((probability > 0) & (probability < 1))
    log_spot = np.log(spot)
    values = np.maximum(0, 1 - np.exp(log_spot[:, None] + log_up[:, None] * (2 * np.arange(steps + 1) - steps)))
    for period in range(steps - 1, 0, -1):
        continuation = discount[:, None] * ((1 - probability[:, None]) * values[:, :-1] + probability[:, None] * values[:, 1:])
        intrinsic = 1 - np.exp(log_spot[:, None] + log_up[:, None] * (2 * np.arange(period + 1) - period))
        values = np.maximum(intrinsic, continuation)
    continuation = discount * ((1 - probability) * values[:, 0] + probability * values[:, 1])
    payoff = np.maximum(0, 1 - spot)
    return payoff, continuation - payoff


def features(contracts):
    centre = np.array([0.975, 0.29, 0.065, 0.645])
    scale = np.array([0.325, 0.19, 0.055, 0.605])
    x = (contracts - centre) / scale
    return np.column_stack((np.ones(len(x)), *[x[:, term].prod(axis=1) for term in DEGREES]))


def fit(x, targets):
    return np.linalg.solve(x.T @ x + 0.001 * np.eye(x.shape[1]), x.T @ targets)


def select_threshold(score, payoff, delta):
    mistakes = (score[:, None] > THRESHOLDS) != (delta[:, None] > 0)
    regret = mistakes * (np.abs(delta) * (payoff > 0))[:, None]
    best = regret.mean(axis=0).argmin()
    return THRESHOLDS[best]


def evaluate(score, payoff, truth, threshold):
    itm = payoff > 0
    choose_continue = (~itm) | (score > threshold)
    wrong = choose_continue != ((~itm) | (truth > 0))
    near = itm & (np.abs(truth) <= 0.003)
    regret = wrong * np.abs(truth)
    price_error = np.maximum(0, score) - np.maximum(0, truth)
    assert near.any() and (regret >= 0).all()
    return np.array([regret.mean(), np.sqrt((price_error ** 2).mean()), wrong[itm].mean(), regret[near].mean(), near.sum()])


def trial(seed):
    rng = np.random.default_rng(seed)
    train, cal = sample(rng, 900), sample(rng, 500)
    _, train_delta = values(train, 96)
    coefficients = fit(features(train), np.column_stack((np.maximum(0, train_delta), train_delta)))
    cal_payoff, cal_delta = values(cal, 96)
    thresholds = [
        select_threshold(score, cal_payoff, cal_delta)
        for score in (features(cal) @ coefficients).T
    ]
    panels = {}
    for label, shifted in (("matched_range", False), ("higher_volatility", True)):
        test = sample(rng, 1300, shifted)
        payoff, delta = values(test, 96)
        price_score, signed_score = (features(test) @ coefficients).T
        _, coarse = values(test, 16)
        panels[label] = {
            "price_target": evaluate(price_score, payoff, delta, thresholds[0]),
            "continuation_target": evaluate(signed_score, payoff, delta, thresholds[1]),
            "coarse_tree": evaluate(coarse, payoff, delta, 0),
        }
    return panels


def summarize(panels, rng):
    estimates = {method: np.stack([p[method] for p in panels]) for method in panels[0]}
    indices = rng.integers(len(panels), size=(5000, len(panels)))
    changes = {}
    for method in ("price_target", "coarse_tree"):
        d = estimates[method][:, 0] - estimates["continuation_target"][:, 0]
        low, high = np.quantile(d[indices].mean(axis=1), (0.025, 0.975))
        changes[method] = [float(d.mean()), float(low), float(high)]
    return {
        "means": {k: dict(zip(METRICS, v.mean(axis=0).tolist())) for k, v in estimates.items()},
        "continuation_regret_reduction_against": changes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    panels = [trial(s) for s in np.random.SeedSequence(587135).generate_state(TRIALS)]
    rng = np.random.default_rng(89335)
    result = {name: summarize([p[name] for p in panels], rng) for name in panels[0]}
    args.output.write_text(json.dumps({"independent_runs": TRIALS, "panels": result}, indent=2) + "\n")


if __name__ == "__main__":
    main()
