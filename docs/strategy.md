# Strategy (decision record, 2026-05-16)

## Situation

The originally intended contribution — "contextual Bayesian optimization
is the sample-efficient winner for scene-adaptive tracker tuning" — does
**not** hold. Empirically (see `results/` and `docs/research_log.md`):

- Contextual BO one-shot (69.4 ± 20.9) is **worse** than PPO one-shot
  (63.7 ± 17.4) despite 83× fewer training rollouts.
- Contextual BO one-shot ≈ random-search-8 (70.1).
- Vanilla per-scenario BO with 8 trials (61.5) beats every one-shot
  method.
- All learned methods beat default params (~90.9) by ~25%.
- With n=12, σ≈20, **none of the learned-method differences are
  statistically significant.**

This is a real limitation of one-shot context-conditioned tuning, not a
bug. Engineering the contextual GP harder is the grad-student trap and is
explicitly rejected.

## Decision

Pivot the paper to **"honest benchmark + warm-start hybrid."**

The paper reports three honest things:

1. **Negative finding.** One-shot scene-adaptive tracker tuning
   (Stephan/Ott 2022's RL approach, and our contextual-BO analogue)
   does not beat cheap per-scenario BO. Reported straight. This
   de-risks the novelty critique because we are not overclaiming.
2. **Positive contribution: warm-start hybrid.** The contextual prior
   is weak *alone* but should be a strong *initializer*. Warm-starting
   per-scenario BO from the contextual GP proposal should reach
   vanilla-BO quality in fewer trials. This is the hypothesis the
   negative result directly motivates.
3. **Durable artifact.** `anumana` is the first open, reproducible
   testbed for scene-adaptive MTT tracker autotuning. Contribution
   survives regardless of whether the hybrid lands.

## Why this over alternatives

- **vs. engineer-until-it-wins:** v3 extrapolation failure +
  barely-beating-random indicate a genuine concept limitation. A 5%
  "win" is inside the noise band — p-hacking. Rejected.
- **vs. pure negative-results paper:** publishable (NeurIPS ICBINB)
  but thin. The hybrid gives it a spine.
- **vs. library-only, no paper:** weaker credential for defense-tech
  positioning than library + honest paper.

For the project's actual goals (independent build, defense-tech
credibility, open-source asset, counter-UAS / India framing), "built
the testbed everyone in this niche will use and reported honestly what
works" is a stronger signal than a marginal method win.

## Plan with decision gates

**Week 1 — warm-start hybrid, time-boxed (5 days).**
Implement `WarmStartBayesOpt`: contextual GP proposal seeds a short
per-scenario BO. Produce best-so-far-vs-trials curves on the v3 held-out
cells.
- **GATE:** does warm-started BO reach vanilla-BO-8 quality in ≤4
  per-scenario trials?
  - YES → strong positive result, proceed full.
  - NO → drop the hybrid; paper is pure honest-benchmark. Still
    publishable.

**Week 2 — lock experiments.**
Implement the deferred joblib parallelism (now justified: a benchmark
paper's credibility requires full grid coverage). Run the complete grid
with proper seeds and bootstrap CIs.

**Week 3 — write.**
Honest-benchmark framing + hybrid result if it landed. Target: ICML
2026 AutoML workshop or NeurIPS ICBINB.

**Week 4 — ship.**
arXiv preprint + workshop submission + LinkedIn rollout.

## Expectation setting

Workshop-tier, honest, narrow paper + a real open-source library,
finishable solo in ~4 weeks. Not famous-making. Credibly demonstrates
end-to-end rigor and honest reporting of a failed primary hypothesis —
the signal that matters for defense-tech and independent-builder
positioning.

## Status

- [x] Strategy accepted (2026-05-16)
- [x] Week 1: `WarmStartBayesOpt` + gate experiment — **GATE FAILED**
      (2026-05-17). Warm-start Pareto-dominates vanilla on the mean but
      only reaches vanilla-BO-8 converged quality at trial 8, not ≤4;
      effect collapses at convergence, driven by good-prior cells, not
      significant at n=12. Per the FAIL branch below, the paper is now
      a **pure honest-benchmark + negative-results study**. Warm-start
      is reported as a caveated documented attempt, not the headline.
- [ ] Week 2: joblib parallelism + full grid (benchmark core)
- [ ] Week 3: paper draft (honest-benchmark framing, NeurIPS ICBINB /
      ICML AutoML)
- [ ] Week 4: arXiv + submission

## Post-gate decision (2026-05-17)

Two failed primary hypotheses (contextual-BO-wins; warm-start gate).
The technical-novelty well is dry. The deliverable is the **open
reproducible benchmark + honest findings**, not a method contribution.
No further method swings — execute the benchmark paper:

1. **Negative findings (the substance):** one-shot context-conditioned
   tuning ≈ random search; warm-start accelerates BO early but doesn't
   improve the converged answer and fails on clutter extrapolation;
   all learned methods statistically indistinguishable; all beat
   default ~25%.
2. **The artifact:** `anumana`, first open reproducible testbed for
   scene-adaptive MTT tracker autotuning, with 5 optimizers
   (random, vanilla BO, contextual BO, warm-start BO, PPO) on a
   parameterised counter-UAS swarm scenario grid.
3. **The honest contribution to the field:** a rigorous comparison
   nobody has published, plus the tooling to extend it.
