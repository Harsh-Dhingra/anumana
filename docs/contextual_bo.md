# Contextual Bayesian optimization — design notes

This is the design doc for phase 1.2: turning today's offline non-contextual
BO into the actual research contribution. **Do not start implementation
until the multi-scenario grid (phase 1.1) confirms BO has signal across the
scenario space.**

## What it is

A single GP fit across many scenarios, with **context features** appended to
the input. The GP learns

    f(theta, c) -> tracking quality

where `theta` is the tracker parameter vector and `c` is the scene context
(target density, measurement rate, dispersion, ...).

At inference time, given a new scenario with context `c*`, the GP predicts
`f(theta, c*)` for any candidate `theta` without retraining — the BO can
suggest a good `theta` immediately, based on what it learned in similar
scenes.

This is the difference between

    "tune the tracker for this one scenario from scratch"  (vanilla BO)

and

    "given a scene's characteristics, propose parameters that have worked
     on similar scenes before"                              (contextual BO)

The second is what an autotuning system *for deployment* would do.

## Two training regimes

### (a) Batch / transfer regime

Phase 1.2 starts here.

1. Sweep the scenario grid with vanilla BO. Each (cell, seed) produces a
   training set of `(theta, c, y)` triples.
2. Pool all triples into one dataset.
3. Fit a single GP `f(theta, c) -> y` on the pool.
4. Evaluate on held-out scenarios: given `c*`, ask the GP for argmax over
   `theta`, run the tracker once with that `theta`, compare to:
   - vanilla offline BO (full budget per held-out scenario)
   - best-of-pool (the `theta` that worked best in training, scenario-agnostic)
   - default Stone Soup parameters
   - expert-tuned per scenario class

The headline result is whether one-shot context-conditioned proposals match
multi-trial scenario-specific BO. If yes, the system saves a budget of
N tracker runs per scenario at inference time — which is the operational
value.

### (b) Online regime

Phase 1.2 (optional) or phase 2 (paper extension).

Same GP, but updated incrementally as new scenarios arrive. The "training
set" is the entire history of `(theta, c, y)` from every scenario the
system has ever seen. The GP is fit incrementally (or refit on a sliding
window).

We probably skip (b) for the workshop paper and mention it as future work.
Doing it well requires GP scaling tricks (sparse GP, deep kernels) that
balloon the scope.

## Implementation plan

### `anumana.optimizers.ContextualBayesOpt`

API parity with `BayesOpt`, plus:

```python
class ContextualBayesOpt(BayesOpt):
    def __init__(self, context_dim: int, ...):
        ...

    def suggest(self, context: np.ndarray, num_points: int = 1) -> np.ndarray:
        """Suggest params conditioned on the current scene context."""
        ...

    def observe(self, x: np.ndarray, context: np.ndarray, y: np.ndarray) -> None:
        """Record an observation with its context."""
        ...

    def fit_on_pool(self, X: np.ndarray, C: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP on a pre-collected pool of (param, context, score) triples."""
        ...
```

Internal: same SingleTaskGP, but the input dim is `param_dim + context_dim`.
At suggest-time, the acquisition function is optimized over `theta` with
`c` held fixed (we pass `c` as a constant tensor into the acquisition).

### Acquisition with fixed context

BoTorch supports this via `FixedFeatureAcquisitionFunction`. Pattern:

```python
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction

acqf = UpperConfidenceBound(model=gp, beta=2.0)
fixed_acqf = FixedFeatureAcquisitionFunction(
    acq_function=acqf,
    d=param_dim + context_dim,
    columns=list(range(param_dim, param_dim + context_dim)),
    values=context_vec.tolist(),
)
optimize_acqf(fixed_acqf, bounds=param_bounds, q=1, ...)
```

### Context normalisation

`SceneFeatures` returns features in very different units (density is ~1e-7,
measurement rate is ~50, arena size is 5000). The GP kernel will struggle.
We standardise the context during training and apply the same
standardisation at inference.

## Evaluation

### Splits

- **Train scenarios:** ~80% of the grid cells.
- **Held-out scenarios:** ~20%, chosen to include scenes *outside* the
  training distribution along at least one axis (e.g., higher target counts
  than in training). This stress-tests generalisation.

### Headline plot

X-axis: held-out scenario index, sorted by difficulty.
Y-axis: composite score.
Lines:
- vanilla BO with full budget (lower bound on what's achievable)
- contextual BO with one-shot suggestion (our method)
- best-of-pool (no context, scenario-agnostic)
- default parameters (upper bound on what we beat)

If contextual BO sits between "vanilla BO with full budget" and "best-of-pool",
that's the win.

### Statistical reporting

For each scenario, run multiple seeds (n=5 minimum). Report mean and
bootstrap 95% CI on the composite score. Use a non-parametric test
(Wilcoxon signed-rank) for pairwise comparisons.

## Risks / things to watch

1. **GP scaling.** Exact GP is O(N^3) in training points. The grid produces
   N = cells * seeds * trials = potentially 1000+ points. At 1000 points the
   GP fits in <1 second. At 10,000 we need sparse GPs.

2. **Context collinearity.** Density and measurement_rate are correlated by
   construction (rate scales with target count). Drop one or apply PCA on
   the context features.

3. **Kernel choice.** Default RBF/Matern may not capture the right
   structure. Worth trying an additive kernel that separates param and
   context contributions, or a learned deep kernel for the context.

4. **Train/test contamination.** If train and test cells share parameter
   regions, the GP "succeeds" trivially. The test split must include
   genuinely novel context vectors.

## What we are NOT doing in phase 1.2

- Streaming / online context updates inside a single tracker run.
- Active context-aware acquisition (acquisition functions that account
  for context uncertainty).
- Sparse / scalable GPs beyond what BoTorch ships with.

These are FUSION 2027 fodder.
