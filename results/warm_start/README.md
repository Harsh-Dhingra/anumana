# Week 1: warm-start BO gate — FAILED (as pre-registered)

The hypothesis the PPO negative result motivated: the contextual GP is a
weak one-shot predictor but a strong BO *initializer*. Pre-registered
gate (`docs/strategy.md`): does warm-started BO reach vanilla-BO-8
converged quality (61.53) in ≤4 per-scenario trials?

## Config

- Contextual pool: 10 train cells × 3 seeds × 8 BO trials = 240 rollouts.
- Held-out: 4 v3 cells × 3 seeds = 12 (cell, seed) points.
- 8 trials/scenario for both vanilla BO and warm-start BO.
- `n_bootstrap = 2` (trial 0 = contextual proposal, trial 1 = random,
  trials 2+ = per-scenario GP-UCB).

## Result — mean best-so-far across held-out

| trial | vanilla BO | warm-start BO |
|---|---|---|
| 1 | 93.24 | **72.94** |
| 2 | 75.41 | 69.20 |
| 3 | 75.41 | 65.19 |
| 4 | 70.43 | 64.76 |
| 5 | 70.12 | 64.65 |
| 6 | 64.66 | 62.29 |
| 7 | 62.79 | 62.29 |
| 8 | 61.53 | 61.18 |

**GATE VERDICT: FAIL.** Warm-start reaches vanilla-BO-8 converged
quality only at trial 8, not ≤4.

## Honest interpretation

- Warm-start Pareto-dominates vanilla on the **mean** at every trial
  count (strictly lower 1–8).
- But the gap collapses to ~0 by trial 8 — warm-start does **not**
  improve the converged answer, it only reaches it somewhat sooner
  (~1–2 trials in the regime that matters).
- The acceleration is driven by cells where the contextual prior is
  good (e.g. N=8 c=3 m=1.0 s=2: ws@1=56.3 vs van@8=84.7). On the
  clutter-extrapolation cell the prior actively hurts (N=10 c=8 s=1:
  ws@1=126.9 vs van@8=74.1) — the same failure mode documented in v3.
- n=12, σ≈20: the acceleration is **almost certainly not statistically
  significant.**

## Consequence

Honoring the pre-registered gate. The warm-start hybrid is **not** a
strong enough positive result to anchor the paper. Per `docs/strategy.md`
FAIL branch: the paper is now a **pure honest-benchmark + negative-results
study**. Warm-start is reported as a documented, honestly-caveated
attempt (this curve + these caveats), not the headline.

This is the second failed primary hypothesis (contextual-BO-wins →
failed; warm-start gate → failed). The realistic deliverable is the
open reproducible benchmark + honest findings, not a method
contribution. Target venue: NeurIPS ICBINB / ICML AutoML workshop.

## Reproduce

```bash
python scripts/run_warm_start_gate.py --trials 8 --pool-trials 8 \
    --train-seeds 0 1 2 --eval-seeds 0 1 2 --n-bootstrap 2 \
    --out outputs/warm_start/gate_v1.json
```
