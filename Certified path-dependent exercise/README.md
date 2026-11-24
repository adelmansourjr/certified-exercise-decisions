# Certified path-dependent exercise

**Decision question.** Can a pricing system spend its continuation evaluations on difficult decisions and establish when the answer is already clear? This experiment uses a path-dependent Bermudan arithmetic-average put with two future exercise dates. The current stock price and the average of previous fixings are separate state variables.

A convexity argument gives a lower continuation bound by conditional Jensen and an upper bound by the chord over each bin of intermediate spot. Refine the bins only where the bounds still straddle the exercise payoff. Strong comparators use high-order Gauss–Hermite rules and margin-based escalation. They receive the same maximum allowance of **19 intermediate-value evaluations per test case**, counted across each batch. Each held-out value and decision was checked with independent adaptive numerical integration.

## Results

Each panel contained 12,288 states from twelve independent blocks. Half were unselected random states and half were deliberately selected near an inexpensive approximation of the decision boundary. Shifted states use longer final intervals and higher volatilities. The model, seeds, numerical policies and metrics were fixed before aggregate outcomes.

| Method | Interpolation errors | Shift errors | Evaluations per case, interpolation / shift | Certified fraction, interpolation / shift |
|---|---:|---:|---:|---:|
| Adaptive convex bounds | 0/12,288 | 0/12,288 | 9.44 / 9.88 | 99.992% / 99.992% |
| Fixed convex bounds | 102/12,288 | 179/12,288 | 19 / 19 | 97.990% / 96.541% |
| Fixed 19-node Gauss–Hermite | 24/12,288 | 34/12,288 | 19 / 19 | 0% / 0% |
| Staged 7+32-node Gauss–Hermite | 17/12,288 | 24/12,288 | 19 / 19 | 0% / 0% |

The adaptive policy used **50.3% and 48.0% fewer** kernel evaluations than the two-stage baseline. It certified **24,574 of 24,576** cases; the remaining two used its bracket midpoint decision. Zero-intrinsic states receive the structural continue certificate. Among states with positive intrinsic payoff it certified 10,896/10,897 interpolation and 10,934/10,935 shifted states.

Mean regret against the adaptive rule was higher for the strong two-stage baseline by **5.74e-8** strike units on interpolation (paired block-bootstrap 95% interval **[2.02e-8, 1.05e-7]**) and **7.21e-8** on shift (**[3.03e-8, 1.21e-7]**). These absolute improvements are small because quadrature generally makes accurate decisions and its mistakes tend to be close to indifference. On the unselected interpolation slice fixed Gauss–Hermite and adaptive convex bounds both made zero errors. The boundary-enriched slices account for most of the gap. Report both when discussing deployment.

**Decision quality differs from pricing quality.** Under the same split and 1,100 reference training labels per block, cubic ridge regression on signed continuation reduced mean regret by 81.7% and 56.5% versus regression on value above exercise, with paired differences of 0.00102 and 0.00175 strike units. Both learned surrogates remained much weaker than the numerical controls. The adaptive bracket midpoint had higher pricing RMSE than either Gauss–Hermite control. Its purpose is determining the exercise side of a bracket. Adaptive bounds took about 2.7 ms versus 0.8 ms for the staged baseline per 1,024 cases in this CPU implementation. Kernel savings alone do not imply a speed gain when the kernel is inexpensive.

## Scope and verification

The certificate follows from convexity and the known lognormal conditional first moments in this specified model. It acts on the one-step exercise choice with optimal decisions thereafter. The model is synthetic and keeps piecewise deterministic volatility. This demonstrates a decision-specific allocation rule designed for a benchmark with independent integration, with no inference about real markets, general exotic-option profitability, or arbitrary stochastic volatility. A worthwhile next study couples certified stopping to a more expensive kernel and measures wall-clock crossover.

All 24,576 test states used adaptive quadrature. The separate composite method agreed within 5.8e-8 on 382 deterministic audit states. The last-date analytic kernel agreed with a separate numerical integral on all 192 audited terminal states. The protocol records an early reference correction. QUADPACK issued a diagnostic on two held-out states; a higher-resolution composite integral agreed within 1.82e-8 on both. [Protocol](Protocol.md) · [Block and aggregate results](results.json) · [Runtime record](runtime.json)

## Reproduce

Use Python with the versions in [requirements.txt](requirements.txt). The stored run used Python 3.12.8 and a single BLAS thread. Peak process RSS was about 120 MiB.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python run.py --output replay.json
cmp results.json replay.json
```

[Previous vanilla exercise experiment](../Exercise%20decisions/README.md) · [BPS-cGAN paper record](https://research-information.bris.ac.uk/en/publications/bps-cgan-generating-american-option-data-at-a-prescribed-exercise/)
