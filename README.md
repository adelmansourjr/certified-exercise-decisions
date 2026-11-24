# Certified Path-Dependent Exercise Decisions

## Latest experiment

A compiled convexity-based solver agreed with independent adaptive quadrature on all 24,576 synthetic exercise states across three fresh panels. It ran 1.95–2.34× as fast as the strongest compiled control on the repeated Apple M4 in-memory benchmarks.

[Read the report and replay the experiment](Compiled%20exercise%20routing/README.md)

## Earlier experiment

Tests adaptive convexity certificates for synthetic path-dependent Bermudan put exercise.

The convexity method certified 24,574 of 24,576 synthetic exercise decisions and agreed with the independent quadrature reference on every certified action. It used 48–50% fewer continuation-kernel evaluations than staged Gauss–Hermite. Python runtime and pricing RMSE were higher.

[Full report, code and reproduction](Certified%20path-dependent%20exercise/README.md)

[Earlier pilot](Exercise%20decisions/README.md)
