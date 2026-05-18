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

---

## 2026-05-15 (late evening) — Phase 1.5 lit review, first pass

**Goal:** confirm or kill the contextual-BO-for-tracker-autotuning
contribution before spending more engineering. Specifically the
~30%-prior-art risk that was flagged at the start of the project.

**Did:**
- Ran eight targeted searches across arxiv, IEEE Xplore, ScienceDirect,
  Google Scholar via WebSearch/WebFetch.
- Triaged hits by threat level (RED/ORANGE/YELLOW/GREEN), wrote each up
  in [docs/litreview.md](litreview.md).
- Updated [docs/design.md](design.md) with the new positioning.

**Findings:**
- **No RED threats.** Nobody has published "contextual BO for
  tracker autotuning on radar MTT" specifically.
- **Two ORANGE threats** (same problem, different method) — both from
  the same TUM/FAU group:
  1. **Stephan et al. 2022, *"Scene-adaptive radar tracking with deep
     reinforcement learning"*** (Machine Learning with Applications).
     Deep RL with reward formulations tied to an Unscented Kalman
     Filter. Owns the "scene-adaptive radar tracker parameter tuning"
     framing from the RL angle.
  2. **Ott et al. 2022, *"Uncertainty-based Meta-RL for Robust Radar
     Tracking"*** (arXiv:2210.14532). Meta-RL + OOD detection for
     cross-scenario tracking. 16% over Meta-RL baselines, 72% F1 OOD
     detection.
- **One YELLOW threat:** *Tuning Multi Object Tracking Systems using
  Bayesian Optimization* (FUSION 2021, IEEE 9626895). TPE on GM-PHD,
  no context, no transfer.

**Verdict: GO with adjusted positioning.**

The novelty framing shifts from "first scene-adaptive tracker tuner"
(Stephan 2022 has that) to **"first contextual Bayesian optimization
approach to scene-adaptive tracker autotuning, sample-efficient
alternative to RL methods, demonstrated on JPDA in counter-UAS swarm
scenarios."**

This is *better* for the paper because:
- Clear positioning against named prior work, easy for reviewers to
  place us.
- Sample-efficiency story is concrete: BO ~10^2 training points, RL
  ~10^4-10^6.
- v3's "fails on clutter extrapolation" result becomes a planted
  future-work hook ("extending contextual BO with OOD detection a la
  Ott 2022").
- Open-source `anumana` library is real differentiation — none of the
  prior work shipped code.

**What this changes in the project plan:**
1. **Phase 1.4 (PPO baseline) is now critical, not nice-to-have.** We
   must compare contextual BO to an RL baseline on the same task or
   reviewers will ask "why didn't you compare to Stephan 2022?"
2. **Working title:** *"Contextual Bayesian Optimization for
   Sample-Efficient Scene-Adaptive Multi-Target Tracker Autotuning."*
3. **Must-read first:** Stephan 2022 full paper, then Ott 2022, then
   Krause-Ong 2011 (the theoretical foundation), then FUSION 2021
   "Tuning MOT with BO."
4. **Open issues:** full FUSION 2023/2024/2025 proceedings haven't
   been combed paper-by-paper; defense-specific venues (SPIE, IET,
   AESS) also not deeply searched. ~10-20% residual prior-art risk.

**Next:**
- READ the two ORANGE-threat papers in full (this is on the user, not
  the assistant — agent can summarise abstracts, not digest method
  sections rigorously).
- Phase 1.4: build PPO baseline so the sample-efficiency comparison is
  on solid ground. The RL baseline is now load-bearing for the paper.
- Either before or in parallel: comb FUSION 2024 / 2025 proceedings by
  hand for any closer prior art that wasn't surfaced by web search.

---

## 2026-05-15 (late evening, deep pass) — Phase 1.5 deep lit review

**Goal:** convert the ~30% residual prior-art risk into hard data.
Search aggressively, read closest prior art in full, update the novelty
call.

**Did:**
- Eight additional WebSearch queries plus targeted WebFetch on
  candidate papers and conference proceedings.
- Pulled FUSION 2023, 2024, 2025 dblp paper-title lists: **no direct
  hits** at title level.
- arXiv 2024–2026 sweeps on "contextual Bayesian optimization sensor
  parameter," "cognitive radar BO," "scene-adaptive tracking 2024":
  **no direct hits.**
- IEEE TAES + IET Radar Sonar & Nav recent listings: **no direct
  hits.**
- Checked Stone Soup itself for built-in BO-tuned components: the
  sensor managers ship Random / BruteForce / Greedy / OptimizeBrute /
  OptimizeBasinHopping / MCTS variants. None are BO-tuned, none are
  context-conditioned, none target tracker hyperparameter selection.
- GitHub code search ("Stone Soup Bayesian optimization tuning tracker
  open source"): **no equivalent library found.**
- **Read Ott et al. 2022 end-to-end** (PDF saved via WebFetch then
  parsed). Findings tightened the novelty call:
  - Their **context prior** is a 2-D Gaussian on RAI mean and std.
    Same structural idea as our `SceneFeatures` (we use 3-D).
  - They tune **14-dim UKF hyperparameter** action space (gating
    threshold + Q + R covariance entries).
  - They train **3 rooms → test 2 unseen rooms.** 4M training steps.
  - Beat fixed-param baseline by 35%; MAML/Reptile by 16%; OOD F1=72%.
  - Code not public.

**Findings:**
- **No RED threats** found even with deep search.
- **Ott 2022 elevated to ORANGE-RED.** The "context prior + cross-
  scenario tracker tuning" framing is theirs. Reading the full paper
  narrows our novelty more than the first-pass abstract triage
  suggested.

**Verdict (updated):** Still GO, with **further narrowed positioning.**
The contribution is:
- First **contextual Bayesian optimization** approach to this problem
  (vs Ott/Stephan's meta-RL+SAC).
- First on **JPDA + Stone Soup** (vs UKF + custom Infineon pipeline).
- First with **counter-UAS swarm framing**.
- First with **open-source implementation**.
- Sample-efficiency story: ~10² BO observations vs ~4×10⁶ RL steps
  to convergence.

Suggested paper framing (replaces my earlier "first scene-adaptive
tracker tuner"):

> "We adapt the context-prior framework for scene-adaptive tracker
> parameter tuning (Ott et al., 2022; Stephan et al., 2022) to a
> Bayesian optimization setting. Where prior work uses meta-RL+SAC
> requiring O(10^6) training steps, our contextual GP-UCB achieves
> comparable cross-scenario generalization from O(10^2) training
> points. We demonstrate on JPDA trackers in counter-UAS swarm
> scenarios using the open-source Stone Soup library, the first
> publicly released implementation in this line of work."

**Surprises:**
- Reading Ott 2022 in full showed the conceptual overlap is *closer*
  than abstract-level triage suggested. The "context prior" lexicon
  itself is theirs.
- But: their method (meta-RL+SAC), tracker (UKF), domain (indoor
  person tracking), and code (not public) leave four meaningful axes
  of differentiation for us.
- The honest positioning is "BO alternative + open source +
  counter-UAS framing," not "novel problem framing."

**Implications for project plan:**
- **Phase 1.4 (PPO baseline) is now load-bearing.** Without a
  head-to-head sample-efficiency comparison, the paper has no story.
  This was previously "nice-to-have"; it's now the whole paper.
- The eventual benchmark comparison ideally re-implements Ott 2022 or
  at minimum an SB3 PPO on the same hyperparameter space. The
  Infineon dataset isn't public so exact reproduction is impossible;
  a clean PPO on the same Stone Soup task is the realistic comparison.

**Next:**
- User: read Stephan 2022 (paywalled, needs institutional access) and
  Ott 2022 (arXiv, free — PDF already on disk) in full.
- Phase 1.4: start PPO baseline via Stable-Baselines3 on the same
  search space.
- Phase 1.5 follow-ups (deferred): SPIE Defense + Commercial Sensing
  proceedings, IEEE Radar Conference 2024/2025, Chinese radar venues.
  ~10-15% residual prior-art risk.

---

## 2026-05-16 — Phase 1.4 PPO baseline: negative result + strategy pivot

**Goal:** the load-bearing sample-efficiency comparison — contextual BO
vs PPO on the same task — to back the paper's headline.

**Did:**
- Built `anumana.optimizers.ppo_tuner` (single-step contextual-bandit
  gym env + SB3 PPO + VecNormalize). Fixed a circular import
  (`grid.py` → `AutoTuner`).
- Ran the full head-to-head: 20k PPO rollouts (4 envs) vs a 240-rollout
  contextual-BO pool, both evaluated one-shot on the v3 held-out set
  (4 cells × 3 seeds = 12 points). PPO train: 4300 s.

**Results (mean ± std, n=12):**
- Vanilla BO (8 trials): **61.53 ± 20.31** (best)
- PPO one-shot: 63.72 ± 17.40 (20k rollouts)
- Contextual BO one-shot: 69.36 ± 20.89 (240 rollouts)
- Random search (8): 70.12 ± 18.66
- Default: 90.87 ± 18.60

**The intended headline is dead.** Contextual BO one-shot is 8.9% worse
than PPO despite 83× fewer rollouts; ≈ random-search-8; beaten by
vanilla-BO-8. With n=12, σ≈20 none of the learned-method differences
are significant — honest read: all tuning methods indistinguishable,
all beat default by ~25%. Archived to `results/ppo_vs_bo/`.

**Surprises:**
- Even an under-trained PPO (20k vs Ott's 4M steps) beat contextual BO
  one-shot. A better-trained PPO would only widen the gap. The "BO is
  more sample-efficient than RL" narrative cannot be defended on this
  data.
- Contextual BO loses worst exactly where v3 predicted (clutter
  extrapolation, N=10/clut=8). Consistent failure mode, not noise.

**Decision: strategy pivot (see `docs/strategy.md`).**
Paper becomes "honest benchmark + warm-start hybrid":
1. Negative finding reported straight (de-risks novelty critique).
2. Positive contribution = warm-start BO (contextual prior as a BO
   initializer, not a one-shot predictor) — the hypothesis this
   negative result directly motivates.
3. Durable artifact = `anumana`, first open testbed in this niche.

**Next:**
- Week 1: implement `WarmStartBayesOpt`, gate experiment
  (does warm-started BO reach vanilla-BO-8 quality in ≤4 trials?).
- Week 2: joblib parallelism + full grid.
- Week 3: write. Week 4: arXiv + workshop.

---

## 2026-05-17 — Week 1 warm-start gate: FAILED (pre-registered)

**Goal:** test whether the contextual GP is a weak one-shot predictor
but a strong BO initializer (the hypothesis the PPO negative result
motivated).

**Did:**
- Implemented `anumana.optimizers.WarmStartBayesOpt` (+ 3 tests).
- Ran the gate: 240-rollout contextual pool, 4 held-out cells × 3
  seeds, vanilla BO vs warm-start BO 8 trials each.

**Result — mean best-so-far across held-out:**

| trial | vanilla | warm-start |
|-------|---------|------------|
| 1     | 93.24   | 72.94      |
| 4     | 70.43   | 64.76      |
| 8     | 61.53   | 61.18      |

**GATE: FAIL.** Warm-start reaches vanilla-BO-8 converged quality only
at trial 8, not ≤4.

**Honest read:** warm-start Pareto-dominates vanilla on the mean at
every trial count, but the gap collapses to ~0 by trial 8 (no
improvement to the converged answer, just ~1–2 trials faster in the
regime that matters), is driven by good-prior cells, hurts on the
clutter-extrapolation cell (v3 failure mode again), and is not
significant at n=12 / σ≈20.

**Surprises:** the strict gate hid a real-but-modest Pareto effect.
Resisted the temptation to re-frame around it — that's the same
overclaiming trap as the contextual-BO story. Honored the gate.

**Decision:** Strategy FAIL branch. Two failed primary hypotheses now.
Technical-novelty well is dry. Paper = **pure honest-benchmark +
negative-results study**; warm-start reported as a caveated documented
attempt, not the headline. No more method swings. Execute the
benchmark paper. Archived `results/warm_start/`.

**Next:**
- Week 2: joblib parallel cell execution in `run_grid` (now justified —
  benchmark credibility needs full grid coverage), then the full grid
  with all 5 optimizers + proper seeds + bootstrap CIs.
- Week 3: write the benchmark paper (NeurIPS ICBINB / ICML AutoML).

---

## 2026-05-17 (later) — Week 2 benchmark complete; the paper's core result

**Did:** Week 2a joblib parallelism (906b29c), Week 2b benchmark
harness + PPO caching (3047ac7, smoke incl cache-load), then the full
lean benchmark: 6 train + 4 held-out cells, 3 train / 3 eval seeds
(n=12), 8 trials, 20k PPO steps. Plots generated, archived to
`results/benchmark/`.

**Result (mean composite, 95% bootstrap CI, n=12):**

| method | mean | CI | std |
|---|---|---|---|
| default | 90.87 | [80.81,102.13] | 18.60 |
| random search (K=8) | 70.12 | [58.47,79.66] | 18.66 |
| vanilla BO (K=8) | 61.53 | [49.32,71.84] | 20.31 |
| contextual BO (one-shot) | 75.01 | [58.30,94.93] | 32.25 |
| warm-start BO (K=8) | 59.85 | [48.10,70.26] | 19.63 |
| PPO (one-shot) | 61.76 | [49.18,72.83] | 20.75 |

**Findings (sharper than "all noise"):**
- **F1** tuning beats no-tuning, significant (default CI lower 80.81 >
  best CI uppers ~70–72); ≈32–34% improvement.
- **F2 (headline)** one-shot contextual BO is the *worst* learned
  method (75.0 > random 70.1) with ~1.6× the variance; on
  clutter-extrapolation cells it returns 154.7 / 109.7 — *worse than
  untuned default* — the GP posterior mean confidently extrapolates
  out of support. v3 failure confirmed at scale with statistics.
- **F3** vanilla BO / warm-start / PPO statistically
  indistinguishable; warm-start's best mean is noise (consistent with
  the Week-1 gate FAIL).
- **F4** PPO (also one-shot, context-conditioned) does NOT collapse on
  the extrapolation cells (60.4 / 90.2 / 62.9 vs ctx 154.7 / 109.7).
  Bounded clipped RL policy extrapolates more safely than the GP
  posterior mean — a real methodological observation for the paper.

**Decision:** n=12 (3 seeds) is adequate for every stated claim — F1 is
significant, F2 is a variance/qualitative story, F3 is a null. 5-seed
expansion is optional robustness (PPO cached so cheap), not required.

**Surprise:** the honest result is *better* than the feared "everything
is noise." There is a clean, statistically-supported negative finding
(F2) plus a genuine methodological nuance (F4). This is a coherent
corrective to Stephan/Ott's "context-conditioned RL wins big" with no
public baseline — exactly the kind of contribution a benchmark /
ICBINB paper exists for.

**Next:** paper draft results+discussion filled with these numbers
(done); full tightening pass; Week 4 venue-template port + arXiv.
