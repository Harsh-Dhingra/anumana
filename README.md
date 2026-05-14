# anumana

**Online autotuning of multi-target trackers for saturated swarm scenarios.**

`anumana` (Sanskrit *अनुमान*, "inference," from the Nyaya school of Indian logic) is an open-source Python library that wraps multi-target tracking algorithms from [Stone Soup](https://stonesoup.readthedocs.io/) and adds an online autotuning layer based on contextual Bayesian optimization. The target regime is where classical trackers struggle most — high target density, low individual SNR, high clutter, and correlated motion — i.e., the conditions of small-UAS swarm scenarios.

## Status

**Pre-alpha. Under active development.** No stable API yet. First release will accompany an upcoming workshop paper submission.

## Planned usage

```python
from anumana import AutoTuner
from anumana.context import SceneFeatures
from anumana.optimizers import ContextualBO
from stonesoup.tracker.simple import MultiTargetTracker

tracker = MultiTargetTracker(...)  # any Stone Soup tracker
tuner = AutoTuner(
    tracker,
    context_extractor=SceneFeatures(),
    optimizer=ContextualBO(),
)

for measurements in stream:
    tracks = tuner.step(measurements)
```

## Method (one paragraph)

The tuner observes scene-context features on a sliding window — estimated target density, maneuver intensity (from residual statistics), clutter rate (from miss-detection rates), and an SNR proxy. It feeds these into a contextual Gaussian Process and proposes updated tracker parameters (gate sizes, process noise scaling, hypothesis pruning thresholds, IMM model weights) to optimize a composite tracking-quality metric (OSPA + identity switches + fragmentation). Across many scenarios the tuner learns a context → parameter mapping that transfers across scene types. A reinforcement learning baseline (PPO) is included for comparison.

## Project layout

```
src/anumana/        # library code
configs/            # Hydra config tree (scenario, tracker, optimizer, experiment)
experiments/        # experiment entry points
notebooks/          # exploration only
tests/              # pytest suite
scripts/            # one-off utilities
docs/               # documentation
```

## License

BSD-3-Clause, matching Stone Soup.

## Acknowledgements

Built on top of [Stone Soup](https://github.com/dstl/Stone-Soup) (Defence Science and Technology Laboratory, UK) and [BoTorch](https://botorch.org/) (Meta AI). Both BSD-licensed.
