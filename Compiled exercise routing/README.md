# Faster certified exercise with fewer continuation calls

The Python prototype cut continuation calls and took longer than quadrature. This study times a compiled solver that starts with a one-bin screen and reuses a previously evaluated conditional mean as the split point. Choosing where to refine then costs two new continuation evaluations. The original and new studies differ in both implementation and routing, so the benchmark measures the combined change.

## Held-out results

The experiment uses a path-dependent arithmetic-average Bermudan put, with averaged past fixings and two future exercise dates. Each panel has 8,192 fresh states in eight independent blocks. Half are sampled directly and half are selected with a separate seven-node proxy to concentrate on the exercise boundary. The third panel is a risk-neutral two-component jump mixture for the first interval. Each solver receives an equal 19-evaluation cap per return component, shared within each batch, and shares the same compiled numerical kernel and zero-payoff shortcut.

| Panel | Adaptive wall time | Fastest control wall time | Speedup | Candidate mistakes / staged GH mistakes | Candidate certificates |
|---|---:|---:|---:|---:|---:|
| Interpolation | 0.313 ms | 0.708 ms | 2.26× | 0 / 12 | 8,191 |
| Higher and shifted volatility | 0.328 ms | 0.647 ms | 1.97× | 0 / 5 | 8,190 |
| Jump mixture | 0.520 ms | 1.219 ms | 2.34× | 0 / 14 | 8,192 |

Wall times are medians per batch of 1,024 states after warmup, using 32 repeats with randomized batch and solver order on one Apple M4 CPU thread. The fastest control in all panels uses the candidate's one-bin convexity screen and runs 18-node Gauss–Hermite on unresolved cases. The staged seven-plus-32-node control gives the lowest overall control error count. Both controls and fixed 19-node Gauss–Hermite use the same C++ kernel. Paired block-bootstrap candidate/fastest-control time ratios were **0.439 [0.432, 0.446]**, **0.505 [0.495, 0.514]**, and **0.428 [0.423, 0.433]**.

All **24,576** candidate decisions agreed with independently integrated reference decisions. The candidate certified **24,573** states, including the structural continuation cases with zero exercise payoff. Relative to the fastest control it used 73.6–78.5% fewer charged continuation evaluations. This count measures requested kernel calls. Clock time also includes analytic endpoints, probability calculations, queueing and output administration. The candidate used the final bracket midpoint for the three unresolved states. The bounds and decisions were audited independently.

The decision regret gains were small in absolute strike units. Against staged Gauss–Hermite the three mean regret advantages were approximately 3.64e-8, 3.02e-8 and 5.04e-8, with paired 95% intervals above zero on the full panels. The prespecified speed and action-quality criterion passed in each panel against all three controls. The separate end-to-end replay preserved the complete outcome JSON. Relative to the fastest control, the measured speedups on that replay were 2.22×, 1.95× and 2.34×. [Protocol](Protocol.md) · [Results and slices](results.json) · [Timing traces and hardware](timing.json) · [Replay timing traces](timing_replay.json)

## Scope and independent checks

The benchmark measures **batch exercise-decision latency** from in-memory states through written in-memory decisions on this host. It excludes compiler startup, fixture generation, reference pricing and text I/O. It does not establish price accuracy or latency for other hardware. The options and jump scenarios are synthetic. This study neither evaluates historical quotes nor observes trading or exercise decisions. The mixture changes first-interval distributional tails while preserving the risk-neutral forward and the same second-interval conditional kernel. Extending the certificate to additional exercise dates or path-dependent volatility is a further mathematical question.

The reference solver integrates every root continuation with independent adaptive quadrature. Dense composite integration and separate terminal-fixing integration audit 385 and 192 deterministic cases, respectively. The largest discrepancies were 1.03e-7 for the outer reference and 1.12e-16 for the terminal kernel. All reported C++ certificates contained the reference within the protocol tolerance. The Python driver peaked at 104.25 MiB and the largest child process, including compilation, at 138.05 MiB. [Earlier Python study](../Certified%20path-dependent%20exercise/README.md)

## Reproduce

Use Python 3.12, [pinned Python packages](requirements.txt), and a C++17 compiler. Running the experiment regenerates all fixtures and then compiles and times the policies. The deterministic action data and aggregate outcome JSON reproduce byte for byte on the recorded environment. Timings are a new measurement each time.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python run.py --output replay
cmp results.json replay/results.json
```

[Wei and Zheng's BPS-cGAN paper record](https://research-information.bris.ac.uk/en/publications/bps-cgan-generating-american-option-data-at-a-prescribed-exercise/)
