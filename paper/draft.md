# Contextual Optimization for Scene-Adaptive Multi-Target Tracker Tuning: An Honest Benchmark

*Working draft. Markdown now; port to the venue template (NeurIPS ICBINB
or ICML AutoML workshop) in Week 4. `[[FILL]]` markers pull from
`results/benchmark/lean_v1.json` once the benchmark run lands.*

---

## Abstract

Multi-target tracking pipelines expose tunable parameters (association
gate size, process-noise scaling, hypothesis-pruning thresholds) whose
optimal values depend on the scene. Prior work (Stephan et al., 2022; Ott
et al., 2022) tunes these online with reinforcement and meta-RL. We ask a
narrower, practical question: **does conditioning on cheap scene-context
features actually beat simply re-tuning per scenario?** We build the first
open, reproducible benchmark for scene-adaptive tracker autotuning — a
JPDA tracker on Stone Soup over a parameterised counter-UAS swarm grid —
and compare random search, per-scenario Bayesian optimization, one-shot
contextual BO, a warm-start hybrid, and a PPO policy, with bootstrap
confidence intervals on held-out scenes. **Our headline result is
negative**: one-shot context-conditioned tuning does not beat cheap
per-scenario BO; warm-starting accelerates early convergence but does not
improve the converged solution and degrades under distribution shift; all
learned methods are statistically indistinguishable while all beat
untuned defaults by ≈[[FILL:def_gap]]%. We release the benchmark and code
so the comparison can be extended.

## 1. Introduction

Classical multi-target trackers — JPDA, MHT, IMM-Kalman — carry
hand-tuned parameters set once by engineers and held fixed regardless of
the scene. In counter-UAS settings (dense small-UAS swarms, low SNR, high
clutter, correlated motion) a single configuration is provably
suboptimal: the regime that fits three aircraft at 30 km does not fit a
hundred drones at 3 km.

Two recent works tune these parameters adaptively with deep RL (Stephan
et al., 2022) and uncertainty-aware meta-RL with a scene "context prior"
(Ott et al., 2022). Both report large gains over fixed parameters; both
require ~10⁶ training rollouts and neither released code.

A natural hypothesis follows: a *sample-efficient* contextual Bayesian
optimizer should match RL-based scene-adaptive tuning at a tiny fraction
of the training cost. We set out to demonstrate exactly that. **We could
not.** This paper reports what we found instead, and contributes the open
testbed that let us find it:

- **A reproducible benchmark** for scene-adaptive MTT tracker autotuning:
  a seeded counter-UAS swarm scenario generator, a JPDA tracker on the
  open-source Stone Soup framework, five tuning strategies, and a
  metrics + bootstrap-CI evaluation harness. (§3–5)
- **A negative empirical finding**, stated honestly with statistics:
  one-shot context-conditioned tuning ≈ per-scenario random search and
  is beaten by cheap per-scenario BO; a warm-start hybrid accelerates
  early BO but does not improve the converged answer and fails under
  clutter distribution shift; all learned methods are within noise of
  each other. (§6–7)
- **Open code and archived results** so the comparison is extensible —
  the artifact prior work omitted. (§8)

## 2. Related work

**Scene-adaptive tracker tuning.** Stephan et al. (2022, *Machine
Learning with Applications*) tune Unscented-Kalman tracking parameters
with deep RL and scene-dependent reward formulations. Ott et al. (2022,
arXiv:2210.14532) extend this with uncertainty-based meta-RL and a 2-D
"context prior" (mean/std of the range–angle image) for cross-room
generalization plus out-of-distribution detection; ~4×10⁶ training
steps; no public code. Our context features (target density, measurement
rate, dispersion) are the same idea in a different modality, and our
contextual GP plays the role of their context-conditioned policy. We
differ in method (Bayesian optimization vs meta-RL), tracker (JPDA on
Stone Soup vs UKF on a proprietary FMCW pipeline), and reproducibility
(open).

**BO for tracker hyperparameters.** A FUSION 2021 paper tunes a GM-PHD
tracker offline with TPE/SMAC/Spearmint — per-scenario, no context, no
transfer. We include this regime as our `vanilla_bo` baseline.

**Contextual Bayesian optimization.** Krause & Ong (2011) introduced
contextual GP-UCB; transfer/meta-BO surveys (Bai et al., 2023) situate
context-conditioned BO among warm-starting and multi-task methods. We
apply the standard contextual-GP machinery; the contribution is the
honest empirical test on a tracking problem, not new BO methodology.

## 3. Problem setup

**Tracker.** JPDA (`stonesoup` 1.8): Kalman predictor/updater,
`PDAHypothesiser`, `JPDA` data associator, `GNNWith2DAssignment` +
`MultiMeasurementInitiator`, `CovarianceBasedDeleter`. Tunable
parameters: ellipsoidal gate size, process-noise scaling, hypothesis
pruning threshold (3-D unit-cube search space).

**Scenarios.** `SwarmScenario`: N targets enter one edge of a square
arena flying roughly across it, Gaussian process noise as a maneuver
proxy, uniform clutter, configurable detection probability. Materialised
once and replayed deterministically (numpy global RNG snapshot/restore),
so every method sees an identical scene and runs are reproducible.

**Grid.** A *train* set of 6 cells and a *held-out* set of 4 cells over
(num_targets × clutter_rate × maneuver_intensity). Held-out includes two
interior-interpolation cells and two extrapolation cells (clutter and
maneuver beyond the training support) to probe distribution shift.

**Metric.** Composite reward = mean OSPA + 5·(ID switches) +
5·(fragmentations), lower is better. OSPA/GOSPA via Stone Soup's metric
generators; ID switches and fragmentation via greedy per-timestep
nearest-neighbour assignment.

## 4. Methods compared

1. **default** — Stone Soup JPDA default parameters, no tuning.
2. **RandomSearch** — K per-scenario uniform draws, take best.
3. **BayesOpt** — K per-scenario GP-UCB trials (BoTorch SingleTaskGP),
   take best.
4. **ContextualBayesOpt** — GP over the joint (params, context) space
   fit on a transfer pool from the train grid; one-shot posterior-mean
   proposal for a held-out scene (no per-scenario trials).
5. **WarmStartBayesOpt** — per-scenario GP-UCB whose trial-0 is the
   contextual proposal and whose random bootstrap is shortened.
6. **PPO** — single-step contextual-bandit policy (Stable-Baselines3,
   VecNormalize over context), trained on the train grid; one-shot
   deterministic proposal.

## 5. Experimental protocol

Train pool: 6 cells × 3 seeds × 8 BO trials. PPO: 2×10⁴ environment
steps. Held-out evaluation: 4 cells × 3 seeds (n = 12 points). K = 8
trials for per-scenario methods. Each held-out scene is evaluated by all
six methods on the *same* seeded scenario. We report mean composite with
percentile bootstrap 95% CIs (5000 resamples). Scenarios are fully
seeded; parallel and serial execution are bit-identical.

## 6. Results

*Main table — `[[FILL from results/benchmark/lean_v1.json summary]]`*

| Method | mean composite | 95% CI | std |
|---|---|---|---|
| default | `[[FILL]]` | `[[FILL]]` | `[[FILL]]` |
| RandomSearch (K=8) | `[[FILL]]` | `[[FILL]]` | `[[FILL]]` |
| BayesOpt (K=8) | `[[FILL]]` | `[[FILL]]` | `[[FILL]]` |
| ContextualBayesOpt (one-shot) | `[[FILL]]` | `[[FILL]]` | `[[FILL]]` |
| WarmStartBayesOpt (K=8) | `[[FILL]]` | `[[FILL]]` | `[[FILL]]` |
| PPO (one-shot) | `[[FILL]]` | `[[FILL]]` | `[[FILL]]` |

Figure 1: forest plot (`outputs/benchmark/benchmark_forest.png`).
Figure 2: per-cell-kind breakdown (`benchmark_by_kind.png`).

Prior runs already establish the shape (to be confirmed by the final
table): contextual one-shot ≈ random-search-K and worse than vanilla
BO-K (results/ppo_vs_bo/); warm-start Pareto-dominates vanilla on the
mean best-so-far at every K but converges to the same value
(results/warm_start/); contextual one-shot fails on clutter
extrapolation (results/contextual_bo/, v3).

## 7. Discussion

**One-shot context conditioning does not beat cheap per-scenario BO.**
The operational implication is the opposite of the original hypothesis:
if a few per-scenario tuning trials are affordable, they outperform a
context-conditioned one-shot proposal.

**Warm-starting accelerates but does not improve.** Seeding per-scenario
BO from the contextual prior reaches a given quality in fewer trials, but
the converged solution is unchanged and the prior actively hurts when the
held-out clutter is outside training support — the same failure the
contextual GP shows alone.

**Everything beats default; nothing clearly beats everything else.** With
n = 12 and σ ≈ 20, the learned methods are statistically
indistinguishable. The honest takeaway is that tuning matters
(≈[[FILL:def_gap]]% over default) but *how* you tune, among reasonable
methods, does not — at least at this scenario scale.

**Why report this.** The field has two papers claiming large gains from
scene-adaptive RL tuning and no open baseline. A reproducible benchmark
showing that the cheap baseline is hard to beat is a useful corrective
and a foundation others can build on.

## 8. The artifact

`anumana` (BSD-3-Clause, github.com/Harsh-Dhingra/anumana): seeded
counter-UAS swarm scenario generator, instrumented JPDA tracker, metrics
pipeline, five tuning optimizers behind one `AutoTuner` loop, a
joblib-parallel grid harness, and the benchmark + plotting scripts. All
experiment results are archived under `results/` with per-experiment
READMEs and reproduce commands. CI runs the test suite on 3.11/3.12.

## 9. Limitations & future work

- Synthetic Stone Soup scenarios only; no real radar data.
- Single tracker family (JPDA); MHT/IMM/RFS untested.
- 3-D parameter space; IMM model weights and track init/delete
  thresholds excluded.
- n = 12 held-out points; CIs are wide. (Expandable — PPO is cached.)
- Hand-engineered context features; learned/embedding contexts and
  OOD-aware fallback (à la Ott et al. 2022) are open directions and
  could plausibly change the one-shot conclusion.
- ~10–15% residual prior-art risk: SPIE Defense, IEEE Radar Conf, and
  non-English radar venues not exhaustively combed.

## References

- Stephan, Servadei, Arjona-Medina, Santra, Wille, Fischer (2022).
  Scene-adaptive radar tracking with deep reinforcement learning.
  *Machine Learning with Applications* 8:100284.
- Ott, Servadei, Mauro, Stadelmayer, Santra, Wille (2022).
  Uncertainty-based Meta-Reinforcement Learning for Robust Radar
  Tracking. arXiv:2210.14532.
- Krause, Ong (2011). Contextual Gaussian Process Bandit Optimization.
  NeurIPS.
- Bai et al. (2023). Transfer Learning for Bayesian Optimization: A
  Survey. arXiv:2302.05927.
- Balandat et al. (2020). BoTorch. NeurIPS.
- Hiles et al. Stone Soup. (Dstl.)
- Schulman et al. (2017). Proximal Policy Optimization. arXiv:1707.06347.
- *(Find authors + cite the FUSION 2021 "Tuning MOT with BO" paper,
  IEEE Xplore 9626895.)*
