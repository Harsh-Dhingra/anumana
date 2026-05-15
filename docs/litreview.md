# Literature review

Active triage of prior art for the contextual-BO-for-tracker-autotuning
contribution. Each entry has: short summary, relevance, and an explicit
*threat level* (RED = direct prior art, ORANGE = same problem different
method, YELLOW = adjacent or supporting, GREEN = background/foundational).

## Threat triage (2026-05-15)

### ORANGE — closest prior art, must cite and position against

#### Stephan, Servadei, Arjona-Medina, Santra, Wille, Fischer (2022)
**"Scene-adaptive radar tracking with deep reinforcement learning"**
*Machine Learning with Applications*, vol. 8, p. 100284.
[DOI](https://doi.org/10.1016/j.mlwa.2022.100284) ·
[FAU CRIS](https://cris.fau.de/publications/280896216/)

- **Problem:** identical to ours — tracker parameters are usually set by
  engineers and "independent of the scene tracked, often resulting in
  non-optimal and poorly performing tracking." Proposes scene-adaptive
  parameter choice.
- **Method:** Deep Reinforcement Learning, with reward formulations
  adapted to the dynamics of an Unscented Kalman Filter tracker.
- **Differences from us:**
  - RL not contextual BO (less sample-efficient by design)
  - UKF, not JPDA (different tracker family)
  - Different scenarios (not counter-UAS / swarm specifically)
- **Threat level:** **ORANGE**. They own the "scene-adaptive radar
  tracker parameter tuning" framing from the RL angle. We must
  position our work as "the sample-efficient BO alternative" rather
  than "first to do this."
- **Action:** READ FULL PAPER. Identify their parameter set, reward
  formulation, evaluation protocol. Plan to cite as the primary prior
  work in our introduction and to use as a comparison baseline if
  possible.

#### Ott, Servadei, et al. (2022)
**"Uncertainty-based Meta-Reinforcement Learning for Robust Radar Tracking"**
[arXiv:2210.14532](https://arxiv.org/abs/2210.14532)

- **Problem:** generalize tracker behavior to unseen scenarios via
  meta-RL, with OOD detection on top.
- **Method:** Meta-RL + uncertainty-based out-of-distribution detection.
- **Result:** 16% over related Meta-RL approaches, 35% over baselines,
  72% F1 on OOD detection.
- **Differences from us:**
  - Meta-RL not BO
  - Same TUM/FAU group; same problem framing
  - Does NOT compare to BO methods
  - No swarm/counter-UAS framing
- **Threat level:** **ORANGE**. Same problem space, different method.
  Important because they explicitly address cross-scenario generalization
  — which is also our v3 evaluation axis. They claim OOD detection,
  which aligns with our finding that contextual GP fails on extrapolation
  (their meta-RL detects OOD; ours fails silently).
- **Action:** READ FULL PAPER. Their OOD-detection mechanism may inspire
  a fallback for our contextual GP when the test context is far from
  training support.

### YELLOW — BO for tracker tuning, no context / no transfer

#### Anonymous (FUSION 2021), "Tuning Multi Object Tracking Systems using Bayesian Optimization"
[IEEE Xplore 9626895](https://ieeexplore.ieee.org/document/9626895/)

- **Problem:** offline hyperparameter optimization of tracking systems.
- **Method:** TPE (Tree-structured Parzen Estimator) with EI
  acquisition, compared against MCMC, SMAC, Spearmint.
- **Tracker:** GM-PHD (Gaussian Mixture Probability Hypothesis Density).
- **Differences from us:**
  - Per-scenario BO, no context conditioning, no transfer evaluation
  - GM-PHD tracker family, not JPDA/MHT
  - No swarm/counter-UAS framing
- **Threat level:** **YELLOW**. Establishes that "BO can tune trackers"
  — we already knew that. Does *not* take our specific cut.
- **Action:** Find authors via IEEE Xplore, cite as the prior offline-BO
  work. Not a baseline; we cover their case as our "vanilla BO" baseline.

#### García-Fernández et al. (2018)
**"Hyper-parameter optimization tools comparison for multiple object tracking applications"**
*Machine Vision and Applications*.
[Springer link](https://link.springer.com/article/10.1007/s00138-018-0984-1)

- Older HPO-tools comparison for MOT (likely vision-MOT context).
- **Threat level:** **YELLOW**. Establishes the broader "tune trackers
  with HPO" line of work. Cite briefly.

### GREEN — foundations / background

#### Krause & Ong (2011)
**"Contextual Gaussian Process Bandit Optimization"** NeurIPS.
- The theoretical basis for what we're doing. Must cite as foundation.
- Action: READ. Algorithm 1 is essentially our `ContextualBayesOpt`.

#### Frazier (2018)
**"A Tutorial on Bayesian Optimization"** arXiv:1807.02811.
- Standard BO reference. Read for self-education and citation.

#### Snoek, Larochelle, Adams (2012)
**"Practical Bayesian Optimization of Machine Learning Algorithms"** NeurIPS.
- The "BO for hyperparameters" classic.

#### Bai, Tian, Lai, Wei (2023)
**"Transfer Learning for Bayesian Optimization: A Survey"** arXiv:2302.05927.
- Survey of methods to transfer BO across tasks. Our contextual BO is
  one approach in this space. Cite for context.

#### Balandat et al. (2020)
**"BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization"** NeurIPS.
- The library we use. Cite.

#### Hiles, Thomas, Pinto et al. — Stone Soup
- The simulator we use. Cite the Stone Soup paper.

#### Multi-target tracking classics (Blackman 2004; Bar-Shalom & Li)
- Need to cite for MHT / JPDA / Kalman foundations.

### Other / less direct

- **"Scene-adaptive radar tracking with deep RL"** is published in
  *Machine Learning with Applications* — that's where we know the AutoML
  community publishes scene-adaptive work. Suggests AutoML workshop is
  the right venue for us.
- Counter-UAS / swarm tracking is an active operational topic, but
  our search did not surface a research paper combining (BO autotuning +
  counter-UAS + radar MTT). Combining these is part of our novelty.

## Novelty call (tentative, 2026-05-15)

**GO**, with adjusted positioning.

Our cut — **contextual Bayesian optimization for online multi-target
tracker parameter autotuning** — does not appear in the literature based
on the searches conducted. The closest priors are:

1. RL/meta-RL approaches to the *same problem* (Stephan 2022, Ott 2022).
2. Per-scenario BO for *different trackers* (FUSION 2021 paper, GM-PHD).

The honest positioning shifts from "first scene-adaptive tracker tuner"
to **"first contextual Bayesian optimization approach to scene-adaptive
tracker autotuning, sample-efficient alternative to RL methods,
demonstrated on JPDA trackers in counter-UAS swarm scenarios."**

The shift is *good* for the paper:
- Clear positioning against named prior work (Stephan 2022 is the
  comparison point).
- Sample-efficiency story is concrete: BO needs ~10² training points,
  RL needs ~10⁴–10⁶.
- RL baseline (planned phase 1.4) is now essential — we must
  compare contextual BO to an RL baseline on the same task. Without it,
  the sample-efficiency claim is hand-wavy.

### What this changes in the project plan

1. **RL baseline is critical, not nice-to-have.** Phase 1.4 moves up
   in priority. Without it, reviewers will ask "why didn't you compare
   to Stephan 2022?"
2. **Counter-UAS / swarm framing is real differentiation.** Stephan
   2022 doesn't focus on this; we should explicitly evaluate on
   high-target-count low-SNR scenarios.
3. **OOD failure is honest.** Ott 2022 has OOD detection — we don't,
   yet, and our v3 result shows the GP fails on extrapolation. This
   is a planted future-work hook: "extending contextual BO with OOD
   detection a la Ott 2022 is left to future work."
4. **Title and abstract must explicitly compare to RL.** Suggested
   title: *"Contextual Bayesian Optimization for Sample-Efficient
   Scene-Adaptive Multi-Target Tracker Autotuning."*
5. **Open-source repository is differentiation.** None of the prior
   work has shipped code. Our public `anumana` library is a real
   contribution beyond the paper.

## Must-reads, in order

1. **Stephan et al. 2022** — full paper. Get the parameter set, reward
   formulation, scenarios.
2. **Ott et al. 2022** — full paper. Understand the meta-RL setup and
   OOD detection mechanism.
3. **Krause & Ong 2011** — foundational, light read.
4. **FUSION 2021 "Tuning MOT with BO"** — for completeness and citation.
5. **Frazier 2018** — for self-education.

## Search queries used (for reproducibility)

- "contextual Bayesian optimization multi-target tracker parameter tuning JPDA MHT"
- "Bayesian optimization tracker hyperparameters tuning radar multi-target 2024 2025"
- "autotuning multi-target tracking algorithm parameters Bayesian"
- "transfer learning Bayesian optimization tracker parameters scene context radar"
- "Stephan Servadei radar tracking reinforcement learning parameters arxiv preprint"
- "FUSION conference Bayesian optimization tracker tuning 2023 2024"
- "counter-UAS swarm drone tracking algorithm autotuning adaptive parameters 2024 2025 2026"
- "contextual Gaussian process UAS drone tracking parameter optimization sensor"

## Open issues / what I haven't checked

- **Full FUSION 2023, 2024, 2025 proceedings**, paper-by-paper. Web
  search surfaces popular papers; conference proceedings can hide
  closer prior art. Should be done by hand against dblp's FUSION lists.
- **Defense-specific venues (SPIE, IET Radar Sonar & Nav, AESS)** —
  some BO/tracker tuning work may sit there and not surface in standard
  searches.
- **Industry / classified work** (Anduril, Shield AI, Lockheed
  publications) — likely exists but hard to find. Probably not papered.
- **Krause-Ong contextual GP-UCB applied to non-tracking domains** — if
  another group has applied contextual BO to a closely related sensor
  parameter problem, it'd weaken our "first" claim within MTT but
  strengthen the methodological framing.

## Status legend (used elsewhere)
- `[ ]` unread
- `[~]` skimmed
- `[x]` read and notes taken below
- `[!]` directly cited in the paper
