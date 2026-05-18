# The benchmark — main result (lean_v1)

The paper's core table. 6 train + 4 held-out cells, 3 train / 3 eval
seeds (n=12 held-out points), 8 trials/scenario, 20k PPO steps,
bootstrap 95% CIs (5000 resamples). Scenarios fully seeded.

## Table (mean composite, lower = better)

| Method | mean | 95% CI | std |
|---|---|---|---|
| Default params | 90.87 | [80.81, 102.13] | 18.60 |
| Random search (K=8) | 70.12 | [58.47, 79.66] | 18.66 |
| Vanilla BO (K=8) | 61.53 | [49.32, 71.84] | 20.31 |
| **Contextual BO (one-shot)** | **75.01** | **[58.30, 94.93]** | **32.25** |
| Warm-start BO (K=8) | 59.85 | [48.10, 70.26] | 19.63 |
| PPO (one-shot) | 61.76 | [49.18, 72.83] | 20.75 |

## Honest findings

**F1 — Tuning beats no tuning (statistically clean).** Default (90.87,
CI lower bound 80.81) does not overlap the best methods' CIs (upper
bounds ~70–72). Any reasonable tuning improves ~32–34% over untuned
defaults.

**F2 — One-shot contextual BO is the *worst* learned method and is
actively harmful under distribution shift (the headline negative).**
Contextual BO one-shot (75.01) is worse on the mean than even random
search (70.12), with ~1.6× the variance of every other method
(std 32.25 vs ~19–20). Cause is visible per-cell: on the
clutter-extrapolation cells (N=10, clutter=8) it returns 154.7 and
109.7 — *worse than untuned default* (135.8, 108.4) — because the GP
posterior mean confidently extrapolates outside its training support.
This confirms the v3 finding at benchmark scale with statistics.

**F3 — Among per-scenario / hybrid methods, nothing beats anything
(honest null).** Vanilla BO (61.53), warm-start BO (59.85), and PPO
one-shot (61.76) are statistically indistinguishable (means within ~2,
CIs heavily overlapping). Warm-start has the best mean but it is noise —
consistent with the Week-1 gate FAIL.

**F4 — Context-conditioned RL degrades gracefully where context-
conditioned GP does not.** PPO one-shot is also context-conditioned and
one-shot, yet on the clutter-extrapolation cells it returns 60.4 / 90.2
/ 62.9 — reasonable — vs contextual BO's catastrophic 154.7 / 109.7.
The RL policy (bounded, VecNormalize-clipped observations) extrapolates
more safely than the GP posterior mean. A genuinely interesting
methodological observation for the discussion.

## Why n=12 (3 seeds) is adequate here

F1 (tuning > default) is significant at n=12 (non-overlapping CIs).
F2 (contextual one-shot worst + high variance) is a variance/qualitative
story, not a tight-CI story — more seeds only sharpen it. F3 is a null;
more seeds would only further confirm "indistinguishable." Expanding to
5 seeds is optional robustness, not required for any stated claim. PPO
is cached (`outputs/ppo_cache_lean`) so expansion is cheap if a reviewer
asks.

## Figures

- `benchmark_forest.png` — method means + 95% CI (paper Figure 1).
- `benchmark_by_kind.png` — per-cell-kind breakdown (interpolation vs
  extrap-clutter vs extrap-maneuver); shows contextual BO's
  extrapolation blow-up.

## Reproduce

```bash
python scripts/run_benchmark.py --train-seeds 0 1 2 --eval-seeds 0 1 2 \
    --trials 8 --pool-trials 8 --ppo-steps 20000 --ppo-envs 4 \
    --pool-jobs 6 --ppo-cache outputs/ppo_cache_lean \
    --out outputs/benchmark/lean_v1.json
python scripts/plot_benchmark.py --json outputs/benchmark/lean_v1.json
```

PPO train ~3950s (cached after first run); pool 19s (parallel);
held-out eval serial.
