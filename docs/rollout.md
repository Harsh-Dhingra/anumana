# Rollout text (DRAFT — do not post until you decide)

Prepared so it's ready on go. Nothing here is posted. All of it is
public-under-your-name and permanent once it goes out — review carefully.

The honest-negative angle is the *strength*, not something to soften. A
defense-tech engineer can smell a hype post; an honest "I tried X, data
said no, here is the open benchmark" is the rarer, more credible signal.
Keep it honest.

---

## 1. arXiv submission

**Title:** Contextual Optimization for Scene-Adaptive Multi-Target
Tracker Tuning: An Honest Benchmark

**Authors:** Harsh Dhingra

**Primary category:** cs.LG
**Cross-list:** eess.SP (radar/signal processing), cs.RO (optional)

**Comments field:** Code, reproducible benchmark, and per-experiment
logs: https://github.com/Harsh-Dhingra/anumana

**Abstract (plain text, ~210 words):**

Multi-target tracking pipelines expose tunable parameters (association
gate size, process-noise scaling, hypothesis-pruning thresholds) whose
optimal values depend on the scene. Prior work tunes these online with
reinforcement and meta-reinforcement learning. We ask a narrower,
practical question: does conditioning on cheap scene-context features
actually beat simply re-tuning per scenario? We build, to our knowledge,
the first open and reproducible benchmark for scene-adaptive tracker
autotuning -- a JPDA tracker on Stone Soup over a parameterised
counter-UAS swarm grid -- and compare random search, per-scenario
Bayesian optimization, one-shot contextual BO, a warm-start hybrid, and
a PPO policy, with bootstrap confidence intervals on held-out scenes.
Our headline result is negative and specific: one-shot contextual BO is
the worst learned method on average and is unsafe under clutter
distribution shift, where its unbounded GP posterior mean yields
sub-default proposals. Cheap per-scenario BO, a warm-start hybrid, and a
PPO policy are statistically indistinguishable; all tuning beats untuned
defaults by roughly a third. We additionally note, as a conjecture from
limited data, that a context-conditioned RL policy degrades more
gracefully than the GP. We release the benchmark and code, and are
explicit throughout about which claims are significant, which are nulls,
and which are conjectures.

**Logistics note:** first-time arXiv submitters to cs.LG may need an
endorsement. Check arXiv's endorsement page before submitting; budget a
few days. Submit the PDF compiled from `paper/main.tex` (after the
venue-style swap if submitting to ICBINB in parallel — arXiv preprint
can use the plain article build).

---

## 2. LinkedIn post (primary)

> A few weeks ago I set out to show that contextual Bayesian
> optimization could match RL-based scene-adaptive tuning of radar
> multi-target trackers at a fraction of the compute.
>
> The data said no. I'm releasing it anyway.
>
> The field has two papers claiming large gains from context-conditioned
> RL tuning of radar trackers — and zero public baselines. So I built
> one: **anumana**, an open, reproducible benchmark (BSD-3).
>
> What's in it: a seeded counter-UAS swarm scenario generator on Stone
> Soup, a JPDA tracker + metrics pipeline, and five tuning methods
> compared head-to-head with bootstrap confidence intervals.
>
> What I found:
> • Tuning beats untuned defaults by ~33% (statistically significant).
> • One-shot contextual BO is the *worst* learned method, and is unsafe
>   under distribution shift — its GP confidently extrapolates to
>   proposals worse than no tuning at all.
> • Cheap per-scenario BO, a warm-start hybrid, and a PPO policy are
>   statistically indistinguishable.
> • A conjecture worth chasing: a bounded RL policy seems to extrapolate
>   more safely than a GP posterior mean.
>
> Two hypotheses died honestly along the way. I pre-registered a
> success gate; it failed; I reported it. Every decision is logged in
> the repo.
>
> Negative results with open tooling are undervalued. If you work on
> tracking, autotuning, or counter-UAS, the benchmark is there to
> extend and to argue with.
>
> Repo: github.com/Harsh-Dhingra/anumana
> Preprint: [arXiv link once live]

**Tags to consider:** #BayesianOptimization #MultiTargetTracking
#CounterUAS #ReproducibleResearch #RadarTracking. Tag Stone Soup / Dstl
only if accurate and appropriate.

---

## 3. Short version (X / threads opener, optional)

> I tried to show contextual Bayesian optimization beats RL for
> scene-adaptive radar tracker tuning. It doesn't — it's the *worst*
> learned method and unsafe under distribution shift. Releasing the
> open benchmark + the honest negative result anyway. Most of this
> field has no public baseline. github.com/Harsh-Dhingra/anumana

---

## 4. Pre-post honesty checklist

- [ ] PDF compiles and renders correctly (you, on Overleaf/TeX).
- [ ] Every number in the post matches `results/benchmark/lean_v1.json`.
- [ ] "To our knowledge / first open one I could find" hedge kept
      everywhere — the lit review has ~10–15% residual prior-art risk
      (SPIE, IEEE Radar Conf, non-English venues uncombed). Do NOT post
      an unqualified "first benchmark" claim.
- [ ] No "defense system" / no overstated India-impact framing. This is
      a benchmark + negative result, not a deployed capability.
- [ ] arXiv link present only once the preprint is actually live.
- [ ] You are comfortable this is permanent and public under your name.
