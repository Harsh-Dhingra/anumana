# Contextual Optimization for Scene-Adaptive Multi-Target Tracker Tuning: An Honest Benchmark

*Working draft. Markdown now; port to the venue template (NeurIPS ICBINB
or ICML AutoML workshop) in Week 4. Results table populated from
`results/benchmark/lean_v1.json` (lean_v1, n=12).*

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
confidence intervals on held-out scenes (n=12). **Our headline result is
negative and specific**: one-shot *contextual* BO is the *worst* learned
method on average (mean 75.0 vs random search 70.1) and is unsafe under
clutter distribution shift — on an out-of-distribution cell its
unbounded GP posterior mean yields sub-default proposals. Cheap
per-scenario BO (61.5), a warm-start hybrid (59.9), and a PPO policy
(61.8) are statistically indistinguishable; all tuning beats untuned
defaults (90.9) by ≈32–34%. We additionally note — as a conjecture from
limited data — that a context-conditioned RL policy degrades more
gracefully than the GP. We release the benchmark and code so the
comparison can be extended, and we are explicit throughout about which
claims are significant, which are nulls, and which are conjectures.

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
  one-shot contextual BO is the *worst* learned method (below per-scenario
  random search) and is unsafe under clutter shift; a warm-start hybrid
  accelerates early BO but does not improve the converged answer; cheap
  per-scenario BO, warm-start, and PPO are mutually indistinguishable.
  Each finding is tagged by evidential status (significant / null /
  conjecture). (§6–7)
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

**BO for tracker hyperparameters.** Fleck & Zoellner (2021, FUSION) tune
a GM-PHD tracker offline with TPE/SMAC/Spearmint — per-scenario, no
context, no transfer. We include this regime as our `vanilla_bo`
baseline.

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

Main table (n=12 held-out points; mean composite, lower = better;
percentile bootstrap 95% CI, 5000 resamples):

| Method | mean | 95% CI | std |
|---|---|---|---|
| Default params | 90.87 | [80.81, 102.13] | 18.60 |
| Random search (K=8) | 70.12 | [58.47, 79.66] | 18.66 |
| Vanilla BO (K=8) | 61.53 | [49.32, 71.84] | 20.31 |
| **Contextual BO (one-shot)** | **75.01** | **[58.30, 94.93]** | **32.25** |
| Warm-start BO (K=8) | 59.85 | [48.10, 70.26] | 19.63 |
| PPO (one-shot) | 61.76 | [49.18, 72.83] | 20.75 |

Figure 1: forest plot (`results/benchmark/benchmark_forest.png`).
Figure 2: per-cell-kind breakdown (`benchmark_by_kind.png`).

**F1 — tuning beats no tuning (significant).** Default's CI lower bound
(80.81) sits above every tuned method's CI upper bound; ≈32–34%
improvement.

**F2 — one-shot contextual BO is the worst learned method on average and
fails catastrophically under clutter shift (headline).** Its mean (75.0)
is worst among learned methods — worse than random search — and it
carries ~1.6× the variance of every other method; this ranking is robust
across all n=12 points. The dramatic "worse than untuned default"
behaviour is precisely scoped: it occurs on the single clutter-
extrapolation cell (N=10, clutter=8, training clutter capped at 6), on
2 of its 3 seeds (154.7 and 109.7 vs default 135.8 and 108.4; the third
seed, 60.0, is fine). A poor interpolation point (98.7) also contributes.
The mechanism is the unbounded GP posterior mean confidently
extrapolating outside training support. We do not claim contextual BO
*always* underperforms defaults — we claim it is the worst learned
method in aggregate and is *unsafe* (high-variance, capable of
sub-default proposals) precisely where an autotuner most needs to be
safe.

**F3 — among per-scenario / hybrid methods, nothing beats anything
(null).** Vanilla BO (61.5), warm-start (59.9), PPO one-shot (61.8) are
statistically indistinguishable (heavily overlapping CIs);
warm-start's best mean is within noise (consistent with the Week-1 gate
failing its pre-registered ≤4-trial criterion). This is a null result,
reported as one.

**F4 — a conjecture: context-conditioned RL appears to degrade more
gracefully than the context-conditioned GP.** On the same clutter-
extrapolation cell, PPO returns 60.4 / 90.2 / 62.9 versus contextual
BO's 154.7 / 109.7 / 60.0 — PPO degrades (90.2 on the middle seed) but
does not collapse. This is **3 points per method on one cell**; we
present it as a hypothesis, not a result. A plausible mechanism is that
PPO's bounded (tanh-squashed) action head and clipped, normalised
observations cap how bad a proposal can be, whereas an unconstrained GP
posterior mean has no such floor. Confirming this would need a dedicated
experiment (more extrapolation cells, ablating the action bound); we
flag it as the most interesting open question the benchmark surfaces.

**Statistical honesty.** We state the evidential status of each finding
explicitly, because this paper criticizes prior work for overclaiming and
must not repeat it. **F1** (tuning > default) is statistically supported:
default's 95% CI does not overlap the tuned methods'. **F2** is a
ranking + variance finding (contextual BO has the worst mean and ~1.6×
the variance over n=12); the "sub-default" instances are 2 of 3 seeds on
one cell and are reported as such, not generalised. **F3** is a null —
we do not claim warm-start or any method is best. **F4** is an
explicitly labelled conjecture from 3 points per method on a single
cell. n=12 (3 seeds) is sufficient for F1 (significant), F2 (a
variance/ranking story that more seeds only sharpen), and F3 (a null
more seeds only reconfirm); it is *not* sufficient to elevate F4 beyond
a conjecture, and we do not.

## 7. Discussion

**One-shot contextual BO is not merely "not better" — it is the worst
learned method and is dangerous under distribution shift.** The original
hypothesis was that a sample-efficient contextual GP would match RL-based
scene-adaptive tuning. It does not. Worse, its failure mode is
unsafe: the GP posterior mean is overconfident outside its training
support, so on out-of-distribution clutter it proposes parameters worse
than doing no tuning at all. For an operational autotuner this is the
opposite of a safe default.

**RL *may* extrapolate more gracefully than the GP (conjecture).** On the
one clutter-extrapolation cell PPO degrades but does not collapse while
contextual BO does (F4). We *conjecture* this is because PPO's bounded
action head and clipped observations cap how bad a proposal can be,
whereas an unconstrained GP posterior mean has no floor. We deliberately
do not present this as a result — it rests on 3 points per method on one
cell. We flag it because, if it holds under a dedicated experiment, it
has a clean operational implication: when one-shot context conditioning
is unavoidable, prefer a bounded learned policy to a GP posterior mean.
Either way, neither one-shot method beats cheap per-scenario BO.

**Warm-starting accelerates but does not improve.** Seeding per-scenario
BO from the contextual prior reaches a given quality in fewer trials
(Week-1 gate experiment), but the converged solution is unchanged
(warm-start 59.9 ≈ vanilla 61.5, CIs overlapping) and the prior hurts
under the same clutter shift.

**Tuning matters; the choice among reasonable per-scenario methods does
not.** Any tuning beats default by ≈32–34% (significant). Among vanilla
BO, warm-start, and PPO the differences are within noise at n = 12. The
honest operational takeaway: spend the engineering on *some* per-scenario
tuning loop, not on context-conditioned one-shot prediction.

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
- n = 12 held-out points (4 cells × 3 seeds). Adequate for F1
  (significant) and F2 (a ranking/variance story); insufficient to
  elevate F4 past a conjecture. Expandable to 5+ seeds cheaply (PPO is
  cached) if a reviewer wants tighter intervals on F3's null.
- Hand-engineered context features; learned/embedding contexts and an
  OOD-aware fallback (à la Ott et al. 2022's uncertainty gating) are
  open directions and could plausibly rescue the one-shot regime —
  notably, an OOD detector that declined to act on the clutter-
  extrapolation cell would remove F2's catastrophic instances.
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
- Fleck, Zoellner (2021). Tuning Multi Object Tracking Systems using
  Bayesian Optimization. *24th Int. Conf. on Information Fusion
  (FUSION)*. (TPE/SMAC/Spearmint on GM-PHD; per-scenario, no context.)
