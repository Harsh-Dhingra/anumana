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
