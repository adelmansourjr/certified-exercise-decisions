"""Generate locked synthetic exercise cases, audit and time the compiled solvers."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import ndtr, roots_hermitenorm, roots_legendre


HERE = Path(__file__).resolve().parent
BLOCKS = 8
BATCH = 1024
REPEATS = 32
NORM = (2 * np.pi) ** -0.5
PANELS = ("interpolation", "volatility_shift", "jump_mixture")
METHODS = ("reuse_convex", "hermite19", "staged7_32", "screen_gh18")


def sample(rng, size, panel):
    shift = panel > 0
    jump = panel == 2
    def unif(a, b):
        return rng.uniform(a, b, size)
    return np.column_stack((
        unif(0.65, 1.31), unif(0.72, 1.08),
        rng.choice([7, 15] if shift else [3, 7, 15], size), unif(0.018, 0.13),
        unif(0.10, 0.55), unif(0.49, 0.84) if shift else unif(0.09, 0.48),
        unif(0.26, 0.61) if shift else unif(0.12, 0.43), unif(0.44, 0.69) if shift else unif(0.12, 0.43),
        unif(0.06, 0.18) if jump else np.zeros(size),
        unif(0.62, 0.84) if jump else np.ones(size),
        unif(0.44, 0.69) if jump else np.full(size, 0.5),
    ))


def component(p, stressed):
    spot, _, _, rate, t1, _, vol1, _, q, multiplier, jump_vol = p.T
    if stressed:
        return np.log(spot * multiplier) + (rate - 0.5 * jump_vol**2) * t1, jump_vol * np.sqrt(t1), q
    regular = (1 - q * multiplier) / (1 - q)
    return np.log(spot * regular) + (rate - 0.5 * vol1**2) * t1, vol1 * np.sqrt(t1), 1 - q


def outer_nodes(p, xs, weights):
    _, mean, n, r, t1, t2, _, v2, q, _, _ = p.T
    k = n + 2 - n * mean
    result = np.zeros(len(p))
    for stressed in range(1 + bool(q.max())):
        loc, sigma1, prob = component(p, stressed)
        s1 = np.exp(loc[:, None] + sigma1[:, None] * xs)
        residual = np.maximum(k[:, None] - s1, 0)
        root_sigma = (v2 * np.sqrt(t2))[:, None]
        d = (np.log(s1 / np.maximum(residual, 1e-200)) + (r*t2)[:, None]) / root_sigma + root_sigma / 2
        x1 = np.maximum(0, 1 - (s1 + (n * mean)[:, None]) / (n + 1)[:, None])
        x2 = (np.exp(-r*t2)[:, None] * residual * ndtr(root_sigma-d) - s1*ndtr(-d)) / (n+2)[:, None]
        result += prob * (np.maximum(x1, x2) @ weights)
    return np.exp(-r*t1) * result


def adaptive_reference(p):
    _, mean, n, r, t1, t2, _, v2, q, _, _ = p
    strike = n + 2 - n * mean
    vol = v2 * np.sqrt(t2)
    disc2 = np.exp(-r * t2)
    estimate = 0.0
    diagnostics = 0
    for stressed in range(1 + bool(q)):
        loc, sigma, weight = component(p[None, :], stressed)
        loc, sigma, weight = loc[0], sigma[0], weight[0]

        def integrand(z):
            s1 = np.exp(loc + sigma*z)
            k = strike - s1
            if k <= 0:
                return 0
            intrinsic = max(0, (k - 1) / (n+1))
            d1 = (np.log(s1/k) + r*t2) / vol + vol / 2
            terminal = (disc2 * k * ndtr(vol - d1) - s1 * ndtr(-d1)) / (n+2)
            return max(intrinsic, terminal) * np.exp(-z*z/2)*NORM

        points = [(np.log(k)-loc)/sigma for k in (strike - 1, strike) if k > 0]
        val, err, _, *warning = quad(
            integrand, -9, 9, epsabs=1e-11, epsrel=2e-9,
            points=[z for z in points if -9 < z < 9], full_output=True,
        )
        if err > 2e-8:
            raise RuntimeError(f"Independent integral did not converge: error estimate {err}")
        diagnostics += bool(warning)
        estimate += weight * val
    return estimate * np.exp(-r*t1), diagnostics


def terminal_reference_check(p):
    s0, mean, n, r, _, t2, _, v2, *_ = p
    s1 = 0.92 * s0
    k = n + 2 - n * mean - s1
    if k <= 0:
        return 0.0
    vol = v2 * np.sqrt(t2)
    d1 = (np.log(s1/k) + r*t2) / vol + vol/2
    analytic = (np.exp(-r*t2)*k*ndtr(vol-d1) - s1*ndtr(-d1))/(n+2)
    edge = (np.log(k/s1) - r*t2 + vol*vol/2)/vol
    value = quad(lambda z: max(0, k-s1*np.exp(r*t2-vol*vol/2+vol*z))
                 * np.exp(-z*z/2)*NORM/(n+2), -9, 9, epsabs=1e-12,
                 points=[edge] if -9 < edge < 9 else None)[0] * np.exp(-r*t2)
    return float(abs(value - analytic))


def fixtures():
    hermite_z, hermite_w = roots_hermitenorm(7)
    leg_z, leg_w = roots_legendre(12)
    edges = np.linspace(-9, 9, 257)
    normal_z = (edges[:-1, None] + 0.5*np.diff(edges)[:, None]*(leg_z+1)).ravel()
    normal_w = (0.5*np.diff(edges)[:, None]*leg_w).ravel() * np.exp(-normal_z**2/2)*NORM
    rows, truth, audits = [], [], []
    for i, seed in enumerate(np.random.SeedSequence(66170119).spawn(BLOCKS)):
        rng = np.random.default_rng(seed)
        for p in range(len(PANELS)):
            random = sample(rng, BATCH // 2, p)
            candidates = sample(rng, 3072, p)
            candidates = candidates[candidates[:, 1] < 1]
            proxy = outer_nodes(candidates, hermite_z, hermite_w*NORM)
            chosen = np.argsort(np.abs(proxy - (1 - candidates[:, 1])), kind="stable")[:BATCH // 2]
            states = np.r_[random, candidates[chosen]]
            assert len(states) == BATCH and (states[:, 8] > 0).all() == (p == 2)
            results = np.array([adaptive_reference(s) for s in states])
            references, warnings = results.T
            easy = np.arange(8)
            edge = np.argsort(np.abs(references - (1 - states[:, 1])))[:8]
            subjects = np.unique(np.r_[easy, edge, np.flatnonzero(warnings)])
            composite = outer_nodes(states[subjects], normal_z, normal_w)
            discrepancies = np.abs(references[subjects] - composite)
            terminal_discrepancy = max(terminal_reference_check(s) for s in states[easy])
            assert discrepancies.max() < 3e-7 and terminal_discrepancy < 1e-10
            rows.append(states)
            truth.append(references)
            audits.append({"outer_oracle_checks": len(subjects), "terminal_oracle_checks": 8,
                           "outer_max_error": float(discrepancies.max()), "terminal_max_error": terminal_discrepancy,
                           "quadrature_diagnostic_components": int(warnings.sum())})
        print(f"Built and audited scenario block {i + 1}/{BLOCKS}", flush=True)
    return np.array(rows), np.array(truth), audits


def summarize(states, estimates, decisions):
    payoff = np.maximum(0, 1-states[:, :, 1])
    acts, certified, calls, lo, hi = decisions.transpose(3, 0, 1, 2)
    assert ((acts == 0) | (acts == 1)).all()
    assert ((certified == 0) | (certified == 1)).all()
    limit = 19 * (((states[:, :, 8] > 0).astype(int)+1)*(payoff > 0)).sum(axis=1)
    assert (calls.sum(axis=2) <= limit[:, None]).all()
    mistake = (acts.astype(bool) != (estimates > payoff)[:, None, :]) & (payoff > 0)[:, None, :]
    assert not (mistake & (certified > 0)).any()
    assert (((lo <= estimates[:, None, :]+2e-7) & (hi >= estimates[:, None, :]-2e-7)) | (lo < 0)).all()
    regret = mistake * np.abs(estimates-payoff)[:, None, :]
    near = (payoff > 0) & (np.abs(estimates-payoff) < 0.003)

    result = {}
    for name, ix in (("all", slice(None)), ("random", slice(0, BATCH//2)), ("boundary_enriched", slice(BATCH//2, None))):
        n = near[:, ix]
        result[name] = [
            [dict(
                mean_regret=float(regret[b, m, ix].mean()),
                errors=int(mistake[b, m, ix].sum()),
                near_boundary_count=int(n[b].sum()),
                near_boundary_regret=float(regret[b, m, ix][n[b]].mean()),
                certified=int(certified[b, m, ix].sum()),
                strict_positive_intrinsic_certified=int((certified[b, m, ix]*(payoff[b, ix] > 0)).sum()),
                positive_intrinsic=int((payoff[b, ix] > 0).sum()),
                evaluations=int(calls[b, m, ix].sum()),
            ) for m in range(len(METHODS))]
            for b in range(len(states))
        ]
    return result


def bootstrap(vals, rng):
    estimates = np.mean(vals[rng.integers(len(vals), size=(5000, len(vals)))], axis=1)
    return [float(vals.mean()), *np.quantile(estimates, [0.025, 0.975]).tolist()]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cxx", default="clang++")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    states, estimates, audits = fixtures()
    with tempfile.TemporaryDirectory(prefix="wei_exercise_") as temp:
        temp = Path(temp)
        binary = temp / "exercise"
        data, out, times = (temp / name for name in ("states.txt", "actions.txt", "times.txt"))
        flags = ["-std=c++17", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror", "-pedantic"]
        subprocess.run([args.cxx, *flags, str(HERE / "solver.cpp"), "-o", str(binary)], check=True)
        with data.open("w") as f:
            print(len(states), BATCH, file=f)
            np.savetxt(f, states.reshape(-1, states.shape[-1]), fmt="%.17g")
        subprocess.run([str(binary), str(data), str(out), str(times), str(REPEATS)], check=True)
        decisions = np.loadtxt(out).reshape(len(states), len(METHODS), BATCH, 5)
        wall = np.loadtxt(times).reshape(len(states), len(METHODS), REPEATS, 4)[..., -1]
        fixture_hash = file_hash(data)
        decision_hash = file_hash(out)
        oracle_hash = hashlib.sha256(estimates.tobytes()).hexdigest()

    records = summarize(states, estimates, decisions)
    quality = {}
    timing = {}
    rng_quality, rng_timing = np.random.default_rng(1894), np.random.default_rng(1895)
    for p, label in enumerate(PANELS):
        ix = slice(p, len(states), len(PANELS))
        timing[label] = {
            "milliseconds_per_1024": dict(zip(METHODS, np.median(wall[ix], axis=(0, 2)).tolist())),
            "mean_block_median_ratio_ci_95": {
                f"reuse_convex_vs_{name}": bootstrap(
                    np.median(wall[ix][:, 0, :], axis=1) / np.median(wall[ix][:, m, :], axis=1), rng_timing)
                for m, name in enumerate(METHODS) if m > 0
            },
            "block_times_ms": wall[ix].tolist(),
        }
        quality[label] = {"audits": audits[ix]}
        for scope, panel_records in records.items():
            blocks = panel_records[ix]
            combined = np.array([[rec["mean_regret"] for rec in block] for block in blocks])
            quality[label][scope] = {
                "block_outcomes": blocks,
                "outcomes": {
                    name: {
                        key: float(np.mean([b[m][key] for b in blocks])) if "regret" in key
                        else int(sum(b[m][key] for b in blocks))
                        for key in blocks[0][m]
                    } for m, name in enumerate(METHODS)
                },
                "excess_regret_ci_95": {
                    f"reuse_convex_vs_{name}": bootstrap(combined[:, 0] - combined[:, m], rng_quality)
                    for m, name in enumerate(METHODS) if m > 0
                },
            }
    rss_divisor = 2**20 if platform.system() == "Darwin" else 1024
    rss = {family: resource.getrusage(field).ru_maxrss / rss_divisor
           for family, field in (("python", resource.RUSAGE_SELF), ("children_max", resource.RUSAGE_CHILDREN))}
    chip = platform.machine()
    if platform.system() == "Darwin":
        hardware = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True)
        for line in hardware.stdout.splitlines():
            if line.strip().startswith("Chip:"):
                chip = line.strip().split(":", 1)[1].strip()
                break
    study = {
        "block_count": BLOCKS,
        "cases_per_panel_per_block": BATCH,
        "methods_in_block_order": METHODS,
        "fixture_sha256": fixture_hash,
        "decision_sha256": decision_hash,
        "oracle_sha256": oracle_hash,
        "panels": quality,
    }
    runtime = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "chip": chip,
        "compiler": subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0],
        "compiler_flags": flags,
        "cpu_threads_per_solver": 1,
        "repeats": REPEATS,
        "peak_rss_mib": rss,
        "timings": timing,
    }
    for filename, content in (("results.json", study), ("timing.json", runtime)):
        (args.output / filename).write_text(json.dumps(content, indent=2) + "\n")


if __name__ == "__main__":
    main()
