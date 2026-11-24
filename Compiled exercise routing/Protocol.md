# Wall-clock test of certified exercise routing

## Question and change

The prior vectorized Python study saved continuation evaluations and ran slower. Does a compiled implementation that reuses a parent interval's conditional-mean value at the split point improve actual decision latency? Reuse reduces each additional split to two new kernel evaluations. One initial support bin per return component gives simpler routing than the prior equal-probability initialization. Evaluate binary exercise across a batch. All policies may return continuation without integration when immediate exercise has zero payoff. All other outputs are binary decisions.

## Model and fresh panels

Use the earlier two-future-date arithmetic-average Bermudan put. Rates are positive and each interval is risk neutral. Include two lognormal panels under the earlier data ranges, one interpolated and one shifted in volatility and timing. Add a separate first-interval jump-mixture panel on the shifted parameter ranges. The mixture weights are independent of spot. With probability q, apply multiplier j in [0.62, 0.84] and draw a lognormal return with volatility 0.44–0.69; otherwise apply multiplier (1−qj)/(1−q) and use the state's first-interval volatility. The mixture-adjusted forward is the no-arbitrage forward. This is a discrete, known scenario mixture with deterministic second-interval volatility.

Use eight independent blocks from SeedSequence(66170119), 1,024 states per panel in each. Half each panel are unselected random states and half are the closest-to-boundary positive-intrinsic states in an independent 3,072-state candidate pool, ranked by the same low-fidelity seven-node Gauss–Hermite score for all methods. Generate the panels before timing and log SciPy’s direct adaptive-quadrature result for every test contract. Audit a deterministic 16-case subgroup per block and panel using independent dense composite Gauss–Legendre. Audit the conditional final-date analytic kernel with a separate numerical integral on eight cases per block and panel. Evaluate conditional expectation separately per mixture component. Reference-tolerance and known approximate-enrichment caveats apply. All data are synthetic and no historical exercise, bid, ask or price data are used.

## Compiled policies and budgets

All solvers use the same C++17 kernel, compiler flags, numerical primitive and original states. Charge one unit for each requested intermediate-kernel evaluation, even when a node falls outside the payoff support, and treat the analytic values at zero and the known upper support as shared elementary boundary operations. The common cap is 19 units per return component per positive-intrinsic state, pooled within each 1,024-state batch. Report actual units. Compare four solvers:

1. Fixed 19-node Gauss–Hermite per component.
2. Staged Gauss–Hermite, seven nodes per component followed by 32 nodes for the smallest absolute first-pass margins fitting the remaining cap.
3. One-bin analytical convexity screen followed, where unresolved, by 18-node Gauss–Hermite per component. The screen is the same one used by the candidate.
4. Candidate convexity screen and reusable refinement. Use conditional Jensen lower bounds and chord upper bounds as before. Split the highest-gap interval using its already evaluated conditional mean as the new cut; spend two new evaluations. Give each state at most 63 units per component and let unused units flow to later rows; reserve one initial evaluation per remaining component. An unresolved state uses the bracket midpoint. This is a deterministic streaming allocation fixed before running, with no test oracle or tuning.

A mathematically valid aggregate bracket is a weighted sum over mixture components, because each component retains the same convex function of intermediate spot. Apply a 1e-10 floating-point guard at the root. Validate every claimed certificate against the independent oracle and check the complete candidate and screen bounds with tolerance 2e-7.

## Timing and decision rule

Benchmark fixed batches with two warmups per solver, then 32 repeats in a single process. Randomize both batch order and solver order independently for each repeat with a fixed C++ timing seed. Elapsed time is from the dense batch input in memory to written outputs, including preprocessing, queue sorting, allocation, and decisions. Exclude generation, reference computation, compilation, text parsing and process startup. Keep timings and raw per-block metrics. Record host model, compiler, RSS, and replay conditions.

Report action errors, mean and near-boundary regret, strict coverage, evaluation counts, median milliseconds, and paired intervals. Bootstrap paired differences across independent blocks, not repeat trials, with 5,000 draws. Numerical speed of the candidate passes only if each regime's upper interval for candidate/baseline timing ratio is below one and the upper interval for excess mean regret is below 1e-7 strike units, compared to every stated control. Retain and report partial comparisons and small absolute effects when that criterion fails. Include both the half random and half boundary-enriched slices for quality scope.

[Earlier path-dependent experiment](../Certified%20path-dependent%20exercise/README.md) · [Wei and Zheng's exercise-boundary paper record](https://research-information.bris.ac.uk/en/publications/bps-cgan-generating-american-option-data-at-a-prescribed-exercise/)

The engineering retries repaired an axis-order mismatch in the summary parser and replaced a sandbox-blocked hardware query with a system profiler query. Test fixtures, solver policies, numerical tolerances, timing order and decision thresholds stayed fixed.
