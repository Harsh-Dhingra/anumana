# anumana

**An open, reproducible benchmark for scene-adaptive multi-target tracker autotuning.**

`anumana` (Sanskrit *अनुमान*, "inference," from the Nyaya school of Indian
logic) is an open-source Python testbed for comparing methods that tune
multi-target tracker parameters to the scene being tracked. It wraps a
JPDA tracker on [Stone Soup](https://stonesoup.readthedocs.io/),
generates parameterised counter-UAS-style swarm scenarios, and pits
several tuning strategies against each other on held-out scenes with
proper statistics.

It exists because, to our knowledge, no open reproducible comparison of
these methods on radar-style multi-target tracking existed. Prior work in
this line (Stephan et al. 2022; Ott et al. 2022, both RL/meta-RL) shipped
no code.

## Honest status

This project set out to show that *contextual Bayesian optimization* is a
sample-efficient winner for scene-adaptive tracker tuning. **The data did
not support that.** Across held-out counter-UAS swarm scenarios:

- One-shot context-conditioned tuning (contextual BO; PPO) is **not**
  better than cheap per-scenario Bayesian optimization.
- Warm-starting per-scenario BO from a contextual prior accelerates
  early convergence but does **not** improve the converged answer and
  hurts on out-of-distribution clutter.
- All learned tuning methods are roughly statistically indistinguishable;
  all beat untuned defaults by ~25%.

So `anumana` is published as what it honestly is: **the benchmark and the
negative findings**, not a method that wins. Full reasoning in
[`docs/strategy.md`](docs/strategy.md) and
[`docs/research_log.md`](docs/research_log.md).

Pre-alpha; APIs may change. First release accompanies a workshop paper
(target: NeurIPS ICBINB / ICML AutoML).

## What's in the box

Five tuning strategies + an untuned baseline, on one scenario grid:

| Method | Per-scenario trials? | Pre-trained? |
|---|---|---|
| `default` (Stone Soup JPDA defaults) | no | no |
| `RandomSearch` | yes | no |
| `BayesOpt` (GP-UCB, BoTorch) | yes | no |
| `ContextualBayesOpt` (one-shot, GP over scene context) | no | yes (transfer pool) |
| `WarmStartBayesOpt` (contextual prior seeds per-scenario BO) | yes | yes |
| PPO (one-shot, Stable-Baselines3) | no | yes (RL training) |

Metrics: OSPA, GOSPA, identity switches, track fragmentation, and a
composite reward. Scenarios are fully seeded and reproducible.

## Usage

```python
from anumana import AutoTuner
from anumana.scenarios import SwarmScenario, SwarmScenarioConfig
from anumana.optimizers import BayesOpt

scn = SwarmScenario(SwarmScenarioConfig(num_targets=20, clutter_rate=3.0))
result = AutoTuner(scn, BayesOpt(seed=0)).optimize(num_trials=15)
print(result.best_score, result.best_params)
```

## Project layout

```
src/anumana/
  scenarios/    # SwarmScenario — Stone Soup-backed, fully-seeded swarm sim
  trackers/     # JPDA tracker wrapped with telemetry instrumentation
  metrics/      # OSPA, GOSPA, ID switches, fragmentation, composite reward
  optimizers/   # RandomSearch, BayesOpt, ContextualBayesOpt,
                #   WarmStartBayesOpt, PPO tuner (optional [rl] extra)
  context/      # scene-context feature extractor
  experiments/  # multi-scenario grid harness (joblib-parallel)
  tuner.py      # AutoTuner: scenario -> tracker -> metric -> optimizer loop

scripts/        # benchmark + experiment entry points and analysis/plots
results/        # archived experiment results (CSV/JSON + READMEs)
docs/           # strategy.md, research_log.md, litreview.md, design.md
tests/          # pytest suite (smoke + integration + grid + optimizers)
```

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl]"

# the benchmark (5 methods + default, bootstrap CIs)
python scripts/run_benchmark.py --eval-seeds 0 1 2 --out outputs/benchmark/run.json
python scripts/plot_benchmark.py --json outputs/benchmark/run.json

# multi-scenario grid sweep (joblib-parallel)
python scripts/run_grid.py --kind pilot --num-trials 12 --csv outputs/grid/pilot.csv

pytest tests/
```

## License

BSD-3-Clause, matching Stone Soup.

## Acknowledgements

Built on [Stone Soup](https://github.com/dstl/Stone-Soup) (Dstl, UK),
[BoTorch](https://botorch.org/) (Meta AI), and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). All
BSD/MIT-licensed.
