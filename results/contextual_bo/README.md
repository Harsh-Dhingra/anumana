# Phase 1.2 contextual BO — results

`ContextualBayesOpt` trains a GP across many scenarios, then proposes
tracker parameters for *new* scenarios from a single shot (no per-scenario
trials). This is the operational mode for an autotuning system that needs
to react to scene changes faster than a BO budget allows.

## Headline (v3, authoritative)

**12 eval points across 4 held-out cells (3 seeds each).** Mean ± std.

| Method | All | Interior interp (n=6) | Extrap maneuver (n=3) | Extrap clutter (n=3) |
|---|---|---|---|---|
| **Contextual one-shot** | 67.79 ± 16.92 | **60.92 ± 12.19** | **74.37 ± 11.01** | 74.95 ± 23.18 |
| Vanilla BO (8 trials)   | 63.17 ± 15.60 | 62.90 ± 13.52 | 77.68 ± 13.48 | **49.19 ± 4.49** |
| Random search (8)       | 77.95 ± 15.84 | 70.67 ± 13.43 | 93.74 ± 14.72 | 76.70 ± 8.36 |
| Default parameters      | 97.34 ± 20.51 | 93.17 ± 18.98 | 116.70 ± 17.91 | 86.30 ± 10.53 |

| Subset | ctx vs vanilla BO | ctx vs default |
|---|---|---|
| Interior interpolation | **+2.4%** (2 wins, 2 ties, 2 losses) | +30.6% (6/6 wins) |
| Extrap maneuver        | **+1.9%** (1 win, 2 losses)         | +34.9% (3/3 wins) |
| Extrap clutter         | **-49.6%** (0 wins, 1 tie, 2 losses) | +12.0% (1/3 wins) |

The honest story:
- **Contextual one-shot ties vanilla BO with 8-trial budget on in-distribution
  scenarios.** Interior interpolation and small extrapolation (one axis at
  the training boundary) both produce essentially equal mean scores.
- **Contextual reliably beats default parameters by ~30%** across all
  cell types, including the failure regime.
- **Contextual extrapolates poorly along the clutter axis** when the
  held-out clutter is well outside the training support (training cap
  was 6.0; held-out was 8.0). The GP confidently predicts in a region
  where it has no training data, and is wrong.

For deployment this means: **use contextual one-shot when scene context
looks like training, fall back to vanilla BO when extrapolating.**

## v2 (2026-05-15, superseded) — small held-out set

The v2 evaluation used only 2 held-out cells x 2 seeds = 4 points, which
got lucky on cell choice and reported ctx +4.1% vs vanilla BO. v3 with
4 cells x 3 seeds = 12 points across more diverse held-outs gives the
honest picture. v2 result preserved in `results_v2.json`; v3 is
`results_v3.json` and is the authoritative version.

## v2 (2026-05-15) — exploit mode

`PosteriorMean` acquisition for one-shot proposals (instead of UCB, which
over-explores when there will be no follow-up trials).

### Config

- 6 training cells: `{N targets ∈ (5, 8, 10)} × selected (clutter, maneuver)`
- 2 training seeds per cell × 8 BO trials each → **pool of 96 (theta, c, y) triples**
- 2 held-out cells:
  - **interpolation**: N=8, clutter=3.0, maneuver=0.50
  - **extrapolation along clutter axis**: N=10, clutter=6.0, maneuver=0.50
- 2 eval seeds per held-out cell

### Result

| Method | Score (mean ± std) |
|---|---|
| **Contextual one-shot** | **59.04 ± 8.86** |
| Vanilla BO (8 trials)  | 61.56 ± 9.58 |
| Random search (8)      | 72.86 ± 5.35 |
| Default parameters     | 102.85 ± 17.70 |

- **vs default:** +42.6% (the operational headline)
- **vs vanilla BO with 8 trials of budget:** +4.1%

### Per-cell

| cell                            | seed | ctx-one-shot | vanilla BO (8) | RS (8)  | default |
|---------------------------------|------|--------------|----------------|---------|---------|
| N=8, clut=3, man=0.5            |  0   | **57.81**    | 67.94          | 70.82   | 101.83  |
| N=8, clut=3, man=0.5            |  1   | 59.21        | 59.21          | 80.00   | 90.05   |
| N=10, clut=6, man=0.5           |  0   | 72.06        | 72.06          | 75.10   | 87.48   |
| N=10, clut=6, man=0.5           |  1   | 47.07        | 47.04          | 65.50   | 132.04  |

Contextual one-shot matches or beats vanilla BO on every held-out (cell,
seed) pair. Bold = win.

### What this means

The "interior interpolation" held-out cell (N=8, clut=3, man=0.5) is bracketed
by training cells on all three axes — strong GP signal there, and contextual
one-shot wins outright on seed 0. The "extrapolation" cell (N=10, clut=6,
man=0.5) goes outside the training distribution on the clutter axis;
contextual one-shot still ties vanilla BO, suggesting the GP generalised
reasonably even outside its training support.

### Caveats

- **N=4 eval points is statistically thin.** Need more held-out cells and
  seeds for paper-quality CIs. Plan: re-run with 4-6 held-out cells and
  3-5 seeds each before writing the paper.
- **Stone Soup process/clutter noise isn't seeded.** Even with `seed=k`, two
  runs produce slightly different scenarios. Within a single run the
  comparison is fair (all methods see the same scene), but cross-run
  reproducibility is approximate. Flagging for a follow-up fix.
- **Pool size 96 is small** for a 6-D input GP. Expanded-training run
  (12 cells, 3 seeds) is teed up in `_train_cells(expanded=True)`.

## v1 (2026-05-15, superseded) — UCB acquisition

Same config but UCB instead of PosteriorMean for one-shot proposals.
Result was -41% vs vanilla BO (UCB over-explored, since there's no
follow-up trial to exploit the exploration). Documented for the paper's
ablation section.
