"""Top-level AutoTuner: optimizer-agnostic loop over a scenario.

Offline BO for v0. The optimizer proposes a parameter vector in unit-cube
coordinates, the tuner decodes it into JPDAParams, runs the tracker on the
scenario, scores the resulting tracks against ground truth with the composite
metric, and feeds (x, y) back to the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from anumana.metrics import TrackQualityReport, compute_track_quality
from anumana.optimizers.search_space import (
    DEFAULT_SEARCH_SPACE,
    ParamSpec,
    params_from_unit_cube,
)
from anumana.trackers import JPDAParams, run_jpda


class Optimizer(Protocol):
    def suggest(self, num_points: int = 1) -> np.ndarray: ...
    def observe(self, x: np.ndarray, y: np.ndarray) -> None: ...


@dataclass
class Trial:
    iteration: int
    x: np.ndarray
    params: JPDAParams
    report: TrackQualityReport
    score: float


@dataclass
class TuningResult:
    best_score: float
    best_params: JPDAParams
    best_iteration: int
    trials: list[Trial] = field(default_factory=list)

    @property
    def history(self) -> np.ndarray:
        return np.array([t.score for t in self.trials])

    @property
    def best_so_far(self) -> np.ndarray:
        return np.minimum.accumulate(self.history)


class AutoTuner:
    def __init__(
        self,
        scenario,
        optimizer: Optimizer,
        space: list[ParamSpec] | None = None,
    ) -> None:
        self.scenario = scenario
        self.optimizer = optimizer
        self.space = space or DEFAULT_SEARCH_SPACE

    def optimize(self, num_trials: int, verbose: bool = False) -> TuningResult:
        truths = self.scenario.ground_truth_paths

        best_score = float("inf")
        best_params: JPDAParams | None = None
        best_iter = -1
        trials: list[Trial] = []

        for i in range(num_trials):
            x = self.optimizer.suggest(num_points=1)[0]
            params = params_from_unit_cube(x, self.space)
            tracks, _ = run_jpda(self.scenario, params)
            report = compute_track_quality(tracks, truths)
            score = report.composite

            self.optimizer.observe(np.atleast_2d(x), np.atleast_1d(score))
            trials.append(Trial(i, x, params, report, score))

            if score < best_score:
                best_score = score
                best_params = params
                best_iter = i

            if verbose:
                print(
                    f"trial {i:3d}  score={score:7.2f}  best={best_score:7.2f}  "
                    f"gate={params.gate_size:5.2f}  pn={params.process_noise_scale:5.2f}  "
                    f"prune={params.pruning_threshold:.2e}"
                )

        assert best_params is not None
        return TuningResult(
            best_score=best_score,
            best_params=best_params,
            best_iteration=best_iter,
            trials=trials,
        )
