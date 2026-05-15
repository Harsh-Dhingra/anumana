# Literature review

Active triage of prior art for the contextual-BO-for-tracker-autotuning
contribution. Each entry has: short summary, relevance, and an explicit
*threat level* (RED = direct prior art, ORANGE = same problem different
method, YELLOW = adjacent or supporting, GREEN = background/foundational).

## Updated threat triage (2026-05-15, deep-pass)

After extensive search: full FUSION 2023/2024/2025 dblp listings, arXiv
2024–2026 targeted queries, IEEE TAES + IET Radar Sonar & Nav, defense
venues, GitHub code search, Stone Soup itself (sensor managers, not
BO-tuned), and the full text of Ott et al. 2022. Findings tightened
materially.

**Key update:** reading Ott 2022 in full revealed that the
**"context prior for cross-scenario tracker parameter tuning"** framing
is theirs (2022). They use a 2-D context vector (RAI mean and std);
we use a 3-D one (target density, measurement rate, dispersion). The
*conceptual* contribution narrows to **method (contextual BO vs
meta-RL) + tracker (JPDA on Stone Soup vs UKF on Infineon pipeline) +
scenarios (counter-UAS swarm vs indoor person tracking) + open source**.

**Net verdict: still GO, with further narrowed positioning.** No RED
threat surfaced after the deep pass. The contribution is real but
narrower than my first-pass write-up suggested.

### ORANGE-RED — closest prior art, must cite and position against

#### Ott, Servadei, Mauro, Stadelmayer, Santra, Wille (2022) *(full text read)*
**"Uncertainty-based Meta-Reinforcement Learning for Robust Radar Tracking"**
[arXiv:2210.14532](https://arxiv.org/abs/2210.14532), Infineon Technologies + TUM.

- **Problem:** scene-adaptive tracker parameter tuning with
  cross-scenario generalization. Identical framing to ours.
- **Method:** Meta-RL with SAC + bootstrap critic with random priors.
- **Context prior:** 2-D Gaussian context, **mean and std of the
  Range-Angle Image (RAI) intensity** — proxies for scene difficulty.
  Sampled and added to actor and critic hidden layers.
- **Tracker:** Unscented Kalman Filter (UKF). 14-dimensional action
  space (gating threshold + Q/R covariance entries).
- **Reward:** `−R = ρ(N̂, N) + (1/M) Σ (1 − pₖ(Pₖ))` — relative target-
  count error + likelihood of missing ground-truth positions under the
  UKF variance.
- **Eval split:** train on 3 rooms, test on 2 unseen rooms.
  4M training steps to convergence.
- **Result:** beats fixed-parameter baseline by **35%**, beats
  MAML/Reptile by **16%**, OOD detection **F1 = 72%**.
- **Code:** not public (Infineon dataset, indoor person tracking).
- **Differences from us:**
  - Meta-RL+SAC vs contextual GP-UCB
  - UKF vs JPDA + Stone Soup
  - 2-D RAI-intensity context vs 3-D scene-feature context
  - Indoor person tracking vs counter-UAS swarm scenarios
  - 4M training steps vs ~10² BO observations
  - Has OOD detection; we don't (our v3 fails silently on extrapolation)
- **Threat level:** **ORANGE-RED**. The "context prior for cross-
  scenario tracker tuning" idea is theirs. Our differentiation is
  method + tracker + scenarios + open source + sample efficiency.

#### Stephan, Servadei, Arjona-Medina, Santra, Wille, Fischer (2022)
**"Scene-adaptive radar tracking with deep reinforcement learning"**
*Machine Learning with Applications*, vol. 8, p. 100284.
[DOI](https://doi.org/10.1016/j.mlwa.2022.100284) ·
[FAU CRIS](https://cris.fau.de/publications/280896216/)
(PDF paywalled at Elsevier; abstract + Ott 2022's reference [16] confirm setup.)

- **Problem:** scene-adaptive radar tracker parameter tuning. Original
  paper that Ott 2022 cites as the basis.
- **Method:** Deep RL with PPO (from Ott's ref [16]); two reward
  formulations.
- **Tracker:** Unscented Kalman Filter.
- **Differences from us:** same as Ott 2022 (RL not BO, UKF not JPDA,
  no swarm).
- **Threat level:** **ORANGE**. They established the framing.
- **Action:** READ FULL PAPER (paywalled — try institutional access
  or contact authors). My web triage couldn't get past the abstract.

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

## Novelty call (deep-pass, 2026-05-15)

**GO**, with **further narrowed positioning** after the deep pass.

Our cut — **contextual Bayesian optimization for online multi-target
tracker parameter autotuning** — does not appear in the literature even
after extensive searches across FUSION 2023/2024/2025, arXiv 2024–2026,
IEEE TAES, IET Radar Sonar & Nav, defense venues, and GitHub.

The closest prior art is:
1. Ott 2022 (meta-RL + 2-D RAI context, UKF). **Reading the full text
   tightened this to ORANGE-RED.** The "context prior" framing is theirs.
2. Stephan 2022 (deep RL, UKF). Original; abstract-only access.
3. FUSION 2021 BO-for-MOT paper (per-scenario offline TPE, GM-PHD,
   no context).

Honest framing for the paper, after reading Ott 2022:

> "We adapt the *context-prior framework for scene-adaptive tracker
> parameter tuning* (Ott et al., 2022; Stephan et al., 2022) to a
> Bayesian optimization setting. Where prior work uses meta-RL +
> SAC requiring O(10^6) training steps, our contextual GP-UCB
> achieves comparable cross-scenario generalization from O(10^2)
> training points. We demonstrate on JPDA trackers in counter-UAS
> swarm scenarios using the open-source Stone Soup library, the first
> publicly released implementation in this line of work."

This narrower framing is more honest. It also defines the **must-have
empirical bar:** a head-to-head sample-efficiency comparison against a
re-implementation of Ott 2022 (or at least an SB3 PPO baseline aimed at
the same hyperparameter space).

The shift is *good* for the paper:
- Clear positioning against named prior work (Stephan 2022 is the
  comparison point).
- Sample-efficiency story is concrete: BO needs ~10² training points,
  RL needs ~10⁴–10⁶.
- RL baseline (planned phase 1.4) is now essential — we must
  compare contextual BO to an RL baseline on the same task. Without it,
  the sample-efficiency claim is hand-wavy.

### What this changes in the project plan

1. **RL baseline is the load-bearing claim.** Phase 1.4 moves up in
   priority. Sample-efficiency vs Ott 2022's meta-RL is the whole
   paper. Without a head-to-head we have no story.
2. **Counter-UAS / swarm framing is real differentiation.** None of
   the prior work emphasises this; we should explicitly evaluate on
   high-target-count low-SNR scenarios. JPDA + Stone Soup matches
   the operational counter-UAS setting better than Ott 2022's UKF +
   indoor person tracking.
3. **OOD failure is honest.** Ott 2022 has OOD detection (their key
   contribution beyond Stephan 2022) — we don't, yet, and our v3
   result shows the contextual GP fails on extrapolation. This is a
   planted future-work hook: "extending contextual BO with OOD
   detection a la Ott 2022 is left to future work."
4. **Title and abstract must explicitly position vs RL.** Suggested
   title: *"Contextual Bayesian Optimization for Sample-Efficient
   Scene-Adaptive Multi-Target Tracker Autotuning."*
5. **Open-source repository is differentiation.** None of the prior
   work shipped code. Our public `anumana` library is a real
   contribution beyond the paper.
6. **Cite Ott 2022 in introduction, not just related work.** Their
   context-prior framework is the direct lineage we're building on;
   honest framing acknowledges this upfront and contrasts the method.

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

## Deep-pass searches completed (2026-05-15)

- FUSION 2023, 2024, 2025 dblp paper-title lists: **no direct hits.**
- arXiv 2024–2026 targeted queries (contextual BO / sensor parameter /
  adaptive tuning / cognitive radar): **no direct hits.**
- IEEE TAES + IET Radar Sonar & Nav recent issues: **no direct hits.**
- Stone Soup itself (sensor managers, BO-related modules): **no
  BO-tuned tracker functionality.**
- GitHub code search ("Stone Soup Bayesian optimization tuning
  tracker"): **no equivalent open-source library.**
- Adjacent: cognitive radar BO, sensor management BO, GP-as-motion-
  model (GaPP-Class 2025): all **GREEN** — different problems.
- Forward-citation pull of Stephan 2022 / Ott 2022 via Semantic
  Scholar: API restricted; manual web search of follow-up titles
  surfaced no Bayesian-optimization variants in this lineage.
- Ott 2022 full text **read end-to-end** (PDF saved locally). The
  context-prior + cross-scenario tracking framing belongs to them.

## Residual open issues (~10–15% prior-art risk)

- **Stephan 2022 full text** still paywalled; user should access via
  institution. The abstract + Ott's citation make the method clear,
  but I haven't seen the exact reward formulation, parameter list, or
  scenarios in detail.
- **SPIE Defense + Commercial Sensing proceedings** (annual conference,
  many short defense-applied papers) not paper-by-paper combed.
- **IEEE Radar Conference 2024 / 2025 proceedings** not combed.
- **Chinese / European defense venues** (Journal of Radars, Chinese
  Journal of Aeronautics, Defence Technology, etc.) not searched.
- **Industry / classified work** (Anduril, Shield AI, Lockheed,
  DRDO LRDE) — likely exists internally but unlikely to be in public
  literature. Not a publication threat, but means our library may be
  less novel than it looks to people who've seen internal versions.
- **Krause-Ong contextual GP-UCB applied to other sensor-parameter
  problems** — if another group has applied contextual BO to a closely
  related problem (sensor scheduling, waveform design), it'd weaken
  our "novel application" claim. Quick check found nothing definitive
  but the search wasn't exhaustive.

## Status legend (used elsewhere)
- `[ ]` unread
- `[~]` skimmed
- `[x]` read and notes taken below
- `[!]` directly cited in the paper
