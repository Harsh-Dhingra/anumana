# Phase 1.4: PPO vs Contextual BO — the negative result

The load-bearing experiment for the originally-intended paper. It did not
hold. Documented honestly here; this is the result that triggered the
strategy pivot (see `docs/strategy.md`).

## Config

- 10 training cells × 3 seeds × 8 BO trials → 240-rollout contextual-BO pool
- PPO: 20,000 training rollouts, 4 parallel envs (single-step contextual
  bandit env, VecNormalize over 4-D context)
- 4 held-out cells × 3 eval seeds = 12 evaluation points (same held-out
  set as the v3 contextual-BO experiment)
- PPO training wall-clock: 4300 s. BO pool: 198 s.

## Result (mean ± std, n=12)

| Method | Score (lower=better) | Training rollouts |
|---|---|---|
| Vanilla BO (8 trials/scenario) | **61.53 ± 20.31** | per-scenario |
| PPO one-shot | 63.72 ± 17.40 | 20,000 |
| Contextual BO one-shot | 69.36 ± 20.89 | 240 |
| Random search (8 trials) | 70.12 ± 18.66 | per-scenario |
| Default params | 90.87 ± 18.60 | — |

- Contextual BO one-shot is **8.9% worse** than PPO one-shot, despite
  PPO using **83× more** training rollouts.
- Contextual BO one-shot ≈ random-search-8 (the damaging fact).
- Vanilla per-scenario BO with 8 trials beats every one-shot method.
- Everything beats default by ~25%.

## Honest interpretation

With n=12 and σ≈20, **none of the learned-method differences are
statistically significant.** The honest statement: all tuning methods
are roughly indistinguishable, and all beat default by ~25%. The
intended "contextual BO is the sample-efficient winner" claim is dead.

Per-cell inspection: contextual BO loses worst on the clutter-
extrapolation cells (N=10, clutter=8) — consistent with the v3 finding
that the contextual GP fails outside its training support.

## Consequence

Triggered the strategy pivot to "honest benchmark + warm-start hybrid"
(`docs/strategy.md`, 2026-05-16). The negative result is reported as a
finding, not hidden. The warm-start hybrid is the positive contribution
this result motivates: the contextual prior is weak alone but may be a
strong BO initializer.

## Reproduce

```bash
python scripts/run_ppo_vs_bo.py --ppo-steps 20000 --ppo-envs 4 \
    --bo-pool-trials 8 --train-seeds 0 1 2 --eval-seeds 0 1 2 \
    --out outputs/ppo_vs_bo/results_v1.json
python scripts/plot_ppo_vs_bo.py --json results/ppo_vs_bo/results_v1.json
```
