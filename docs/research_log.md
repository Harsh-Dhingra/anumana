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
