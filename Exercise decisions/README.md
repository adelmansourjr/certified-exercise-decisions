# Pricing accuracy and exercise decisions

**Research problem.** Can targeted learning near the exercise boundary reduce the loss from a wrong exercise decision, especially when volatility shifts? Wei and Zheng's BPS-cGAN research gives a direct starting point for controlling where synthetic training examples lie relative to that boundary. [BPS-cGAN](https://research-information.bris.ac.uk/en/publications/bps-cgan-generating-american-option-data-at-a-prescribed-exercise/)

Sixteen independent runs tested two equal-capacity regressions and a numerical control against a 96-step American-put reference. Each model was trained on 900 simulator calls and used 500 validation cases for its decision threshold. Regret is the value lost by forcing the first action, scaled by strike.

| Held-out range | Price target mean regret | Signed continuation target mean regret | 16-step solver mean regret |
|---|---:|---:|---:|
| In-range volatility | 0.0000715 | 0.0000679 | 0.00000376 |
| Higher volatility | 0.000468 | 0.000456 | 0.000000312 |

The paired regret reductions from training on signed continuation were **0.00000364 [0.00000039, 0.00000813]** and **0.0000119 [0.00000011, 0.0000271]**. That is a small training-objective effect and the 16-step numeric control remained far stronger. The shifted panel contains far fewer near-boundary contracts, averaging about 37 of 1,300. Present the two panels and the numerical baseline in any discussion.

The decision problem becomes more interesting when cheap classical solvers are no longer accurate enough, for example with path dependence, multiple state variables or scarce near-boundary labels. Fix one such setting and compare least-squares Monte Carlo, a coarse solver where applicable, and decision-aware models. Test whether allocating fixed simulator calls by expected decision regret beats uniform and boundary-proximity sampling. Hold out volatility processes or market periods by entire family, and assess both pricing error and decision regret with simulator-call and inference-cost accounting. The pilot uses synthetic contracts and a one-step outcome under a shared reference model.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python run.py --output replay.json
cmp results.json replay.json
```

[Protocol](Protocol.md) · [All outcomes](results.json)
