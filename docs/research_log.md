# Research log

Daily lab notebook. Append a new section per session — what was tried, what
worked, what surprised, what's next. This is the source material for the
paper's experiments section, so be specific about parameters, seeds, and
observed numbers.

Format per entry:

```
## YYYY-MM-DD — short title

**Goal:** what I set out to do today.
**Did:** bullets of what happened.
**Results:** numbers, with seed and config.
**Surprises:** anything that broke or behaved unexpectedly.
**Next:** the very next action.
```

---

## 2026-05-14 — Repo scaffold + end-to-end pipeline

**Goal:** stand up the project from zero — repo, deps, swarm scenario, JPDA
tracker, metrics, BO loop. Validate end-to-end on a single scenario.

**Did:**
- Created [github.com/Harsh-Dhingra/anumana](https://github.com/Harsh-Dhingra/anumana),
  BSD-3-Clause, hatchling-based pyproject.
- Python 3.11 venv, Stone Soup 1.8, BoTorch 0.17, Torch 2.12.
- `anumana.scenarios.SwarmScenario` — materialised swarm sim
  (N targets entering from one edge, ConstantVelocity + process noise,
  uniform clutter, configurable detection probability).
- `anumana.trackers.run_jpda` — JPDA tracker built from a `JPDAParams`
  config; one-to-one `GNNWith2DAssignment` for the initiator,
  `JPDA` (probabilistic) for the main tracker.
- `anumana.metrics.compute_track_quality` — Stone Soup OSPA + GOSPA wrappers,
  greedy nearest-neighbour ID-switch and fragmentation counters,
  composite reward.
- `anumana.optimizers.RandomSearch` and `anumana.optimizers.BayesOpt`
  (GP-UCB on BoTorch's SingleTaskGP, beta=2.0).
- `anumana.AutoTuner` ties it together with a clean `optimize(num_trials)`
  API.

**Results:** (seed=42, 10 targets, 40 steps, clutter rate 3.0)
- Parameter sweep across `gate_size` and `process_noise_scale` showed
  composite reward varies measurably across the search space — BO has
  signal to work with.
- *(End-to-end BO vs RandomSearch comparison logged once demo completes.)*

**Surprises:**
- Stone Soup's `MultiMeasurementInitiator` requires a one-to-one
  `DataAssociator`; passing `JPDA` blows up with
  `'MultipleHypothesis' object has no attribute 'prediction'`.
  Fixed by building a parallel `GNNWith2DAssignment` for the initiator.
- `SimpleManager.generate_metrics()` returns a `dict[str, TimeRangeMetric]`,
  not a list. OSPA values are scalars; GOSPA values are dicts with
  `distance / localisation / missed / false / switching` components.
- Ground-truth paths are only accessible after the simulator has been
  iterated. Refactored `SwarmScenario` to materialise into a cached list
  of frames on construction, so the same scenario can be replayed
  deterministically across BO trials.

**Next:**
- Compare BO vs RandomSearch best-so-far across multiple scenario seeds.
- Add contextual BO: condition the GP on `SceneFeatures`, evaluate transfer
  across scenario types.
- Wire `scripts/run_experiment.py` (Hydra entry point) to drive scenario
  grid sweeps and log to W&B.
- Begin lit review (see [docs/litreview.md](litreview.md)) — week-1 reading
  goal is papers 1–5.

---

## 2026-05-15 — Phase 1.1 grid pilot

**Goal:** validate that the BO win on a single scenario (commit 689a202)
generalises across the scenario space, so the result isn't anecdotal.

**Did:**
- Built `anumana.experiments.grid` (cells, seeds, optimizer sweep, CSV out).
- Ran pilot: 8 cells = {N targets in (5, 15)} × {clutter in (1.0, 5.0)} ×
  {maneuver in (0.10, 1.00)}, detection probability 0.9, duration 20
  steps, 2 seeds, 12 trials per optimizer per (cell, seed) pair.
  Total: 384 tracker runs, 32 (cell, seed, optimizer) triples.
- Result archived to [results/grid_pilot/pilot.csv](../results/grid_pilot/pilot.csv)
  with mean-improvement heatmap.

**Results:**
- **BO wins 14/16 (cell, seed) pairs**, 1 tie, 1 RS win.
- Mean improvement (BO over RS): **+13.4%**.
- Median improvement: +6.4%.
- p25 to p75 spread: +0.0% to +21.3%.
- Max win: 65% (N=5, clutter=5.0, maneuver=0.10, seed=0).
- Only RS win: -5.8% (N=5, clutter=1.0, maneuver=0.10, seed=0).
- Per-cell aggregate (mean across 2 seeds):

| N | clutter | maneuver | RS    | BO    | Δ%    |
|---|---------|----------|-------|-------|-------|
| 5 | 1.0     | 0.10     | 56.96 | 58.14 | -2.3  |
| 5 | 1.0     | 1.00     | 64.42 | 61.18 | +7.2  |
| 5 | 5.0     | 0.10     | 53.65 | 25.34 | +51.1 |
| 5 | 5.0     | 1.00     | 52.87 | 48.66 | +9.2  |
| 15| 1.0     | 0.10     |107.88 |104.73 | +2.8  |
| 15| 1.0     | 1.00     | 71.01 | 48.55 |+31.3  |
| 15| 5.0     | 0.10     |104.25 |104.25 | 0.0   |
| 15| 5.0     | 1.00     | 85.72 | 78.77 | +8.1  |

**Surprises:**
- BO's biggest wins are at **high clutter + low maneuver** (51% at N=5,
  clut=5) and **larger swarms + high maneuver** (31% at N=15, man=1.0).
  These are the regimes where parameter mismatch matters most operationally:
  high clutter means gate size is critical, high maneuver means process
  noise scaling is critical. Encouraging from a counter-UAS-framing
  perspective.
- BO ties or loses on **low-clutter + low-maneuver + low-target** cases.
  Read: when the scene is easy, default parameters are nearly optimal and
  there's nothing for BO to find. Honest in the paper, not damning.
- **Compute scaling is alarming.** N=15 cells took 200-500s each (vs
  2-5s for N=5). Per-trial cost grew roughly quadratically with target
  count (PDAHypothesiser + JPDA build measurement-track gate matrices
  whose size scales with both). The full grid as currently designed
  (up to N=50) is infeasible serially; need either parallelism across
  cells or smaller per-cell scenarios.

**Next:**
- Phase 1.2: implement `ContextualBayesOpt`, fit on the pooled pilot
  history, evaluate one-shot proposals on held-out scenarios.
- Compute fix: add `joblib` parallel cell execution to `run_grid` so the
  full sweep is feasible.
- Begin lit review while phase 1.2 develops (Frazier BO tutorial first).

---

## 2026-05-15 (afternoon) — Phase 1.2 contextual BO first results

**Goal:** demonstrate that one-shot context-conditioned parameter proposals
can match or beat vanilla BO with a full trial budget on held-out scenarios.
This is the paper's actual contribution.

**Did:**
- Implemented `anumana.optimizers.ContextualBayesOpt`: SingleTaskGP over the
  joint (theta, c) space, BoTorch `FixedFeatureAcquisitionFunction` to fix
  context at inference, internal context standardisation.
- Patched `GridResult` to save per-trial parameter vectors (`trial_xs`)
  so the contextual GP can train on every BO trial, not just the
  per-cell best.
- Wrote `scripts/run_contextual_bo.py`: collects a training pool from
  vanilla BO across multiple cells, fits the contextual GP, evaluates
  one-shot proposals on held-out cells against four baselines (vanilla BO
  full budget, random search full budget, default parameters, contextual
  exploit / explore).
- Wrote `tests/test_contextual_bo.py` (3 tests, all pass).

**Results:**
- **v1 (UCB acquisition):** contextual one-shot was +6.9% vs default,
  -41% vs vanilla BO (8 trials). Bad: UCB explores, but for one-shot
  there's no follow-up trial to exploit the exploration.
- **v2 (PosteriorMean acquisition):** contextual one-shot is +42.6% vs
  default, +4.1% vs vanilla BO (8 trials). One-shot matches or beats
  the full-budget vanilla BO on every held-out (cell, seed) pair.

Headline table (mean +/- std, 4 held-out points):

| method                  | score          |
|-------------------------|----------------|
| **contextual one-shot** | **59.04 ± 8.86** |
| vanilla BO (8 trials)   | 61.56 ± 9.58   |
| random search (8)       | 72.86 ± 5.35   |
| default parameters      | 102.85 ± 17.70 |

Archived to [results/contextual_bo/](../results/contextual_bo/).

**Surprises:**
- Acquisition-function choice mattered far more than expected. UCB's
  exploration bonus is harmful in a one-shot setting. This is obvious in
  retrospect (no future trial to exploit), but it's the kind of detail
  that kills a result silently if you copy a vanilla BO setup.
- **Stone Soup isn't fully seeded.** Process noise and clutter generation
  use Stone Soup's internal RNG, which isn't tied to `seed`. Two runs of
  the same `SwarmScenarioConfig(seed=k)` produce slightly different
  scenarios. Within-run comparison is fair; cross-run reproducibility is
  approximate. Filed as a follow-up.
- Interior interpolation (training surrounded the test cell on all three
  axes) gave the strongest contextual win (-15% on seed 0, exact tie on
  seed 1). Extrapolation cell (clutter outside training support) still
  matched vanilla BO. The GP generalises better than I expected.

**Caveats / things to fix:**
- N=4 held-out evaluation points is statistically thin. Need 4-6 held-out
  cells × 3-5 seeds for paper-quality CIs.
- Pool size 96 is small for 6-D GP input. The expanded-training option
  (`--expanded-train`, 12 cells × 3 seeds = ~300 points) is teed up.
- Need to ablate: pool size, kernel choice, beta when UCB is used in
  online (not one-shot) mode.

**Next:**
- Either (a) expand the contextual-BO eval (more cells, more seeds) to
  firm up the headline result, OR (b) move to phase 1.3 (IMM model
  weights in the search space). Decision pending.
- Begin lit review in parallel (Frazier BO tutorial, Krause-Ong contextual
  BO, Stone Soup paper).
- Fix Stone Soup seeding for reproducibility.

---

## 2026-05-15 (evening) — Phase 1.2 v3 expanded, more honest result

**Goal:** firm up the v2 contextual BO result (n=4) with a properly sized
held-out set covering interior interpolation and extrapolation along both
clutter and maneuver axes.

**Did:**
- Stone Soup non-determinism fix: `SwarmScenario._materialize` now
  snapshots numpy's global RNG state, seeds it from `cfg.seed + 10000`,
  iterates, and restores. Two `SwarmScenario(cfg)` instances now produce
  byte-identical detection streams (verified by
  `tests/test_pipeline.py::test_scenario_deterministic`).
- Expanded the held-out set in `scripts/run_contextual_bo.py
  --expanded-train` to 4 cells covering both interpolation
  (N=7 clut=2 man=0.3; N=9 clut=4 man=0.3) and extrapolation along
  clutter (N=10 clut=8 man=0.5) and maneuver (N=8 clut=3 man=1.0).
- Wrote `scripts/analyse_contextual_bo.py` with bootstrap CIs and a
  cell-kind split (interior_interpolation / extrapolation_clutter /
  extrapolation_maneuver).
- Ran the expanded experiment: 12 training cells x 3 seeds x 8 BO trials
  -> 288 (theta, c, y) triples; 4 held-out cells x 3 eval seeds = 12
  evaluation points. Wall-clock 10-12 min total.

**Results (v3, authoritative):**

| Method                  | All (n=12)  | Interior (n=6) | Extrap-maneuver (n=3) | Extrap-clutter (n=3) |
|-------------------------|-------------|----------------|-----------------------|----------------------|
| **Contextual one-shot** | 67.79 ± 16.92 | **60.92 ± 12.19** | **74.37 ± 11.01** | 74.95 ± 23.18 |
| Vanilla BO (8 trials)   | 63.17 ± 15.60 | 62.90 ± 13.52  | 77.68 ± 13.48         | **49.19 ± 4.49**     |
| Random search (8)       | 77.95 ± 15.84 | 70.67 ± 13.43  | 93.74 ± 14.72         | 76.70 ± 8.36         |
| Default parameters      | 97.34 ± 20.51 | 93.17 ± 18.98  | 116.70 ± 17.91        | 86.30 ± 10.53        |

Contextual vs vanilla BO improvement %:
- interior interpolation: +2.4% (2 wins, 2 ties, 2 losses) — effectively tied
- extrap maneuver: +1.9% (1 win, 2 losses) — effectively tied
- extrap clutter: -49.6% (0 wins, 1 tie, 2 losses) — clear failure

Contextual vs default improvement %:
- interior interpolation: +30.6% (6/6 wins)
- extrap maneuver: +34.9% (3/3 wins)
- extrap clutter: +12.0% (1/3 wins; the other 2 perform worse than default)

**Surprises:**
- The v2 result of +4.1% vs vanilla BO was an artifact of small n
  (4 eval points) landing on easy cells. The honest result with 12
  points across 4 distinct cell types is that **contextual one-shot
  ties full-budget vanilla BO** in-distribution rather than beating it.
- Extrapolation failure is dramatic. On N=10 clut=8 (training capped at
  clutter=6), the GP confidently predicts and is badly wrong. Two of
  three seeds produced scores that were *worse than default parameters*.
- The extrap_maneuver cell (man=1.0, training capped at man=0.5) was
  surprisingly fine — same pattern as in-distribution. Maneuver
  extrapolation appears easier than clutter extrapolation, possibly
  because process noise scales monotonically with maneuver in a way the
  GP captures.

**What this means for the paper:**
The story is "contextual one-shot replaces N-trial BO for in-distribution
scenarios; falls back to vanilla BO for out-of-distribution clutter; both
beat defaults by ~30%." That's a stronger and more honest contribution
than "ctx beats everything everywhere" — and aligns with what reviewers
expect of GP-based methods (smooth in-distribution, poor extrapolation).

**Next:**
- Mention v3 as authoritative result; commit + push.
- Decide between (a) push contextual harder (sparse GPs, better kernels,
  ensemble methods) for the paper's "method extensions" section, vs
  (b) move on to phase 1.3 (IMM weights) or 1.5 (lit review).
- Compute scaling is finally biting: N=12 cells took up to 170s each.
  Need joblib parallelism before any further scaling.
