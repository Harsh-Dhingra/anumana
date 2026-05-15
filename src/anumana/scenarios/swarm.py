"""Stone Soup-backed swarm scenario for counter-UAS tracker evaluation.

Targets enter from one edge of a square arena flying roughly toward the
opposite edge, with Gaussian process noise as a maneuver proxy. Measurements
are 2D position with additive Gaussian noise plus uniform clutter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.simulator.simple import (
    MultiTargetGroundTruthSimulator,
    SimpleDetectionSimulator,
)
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.state import GaussianState


@dataclass
class SwarmScenarioConfig:
    num_targets: int = 20
    duration_steps: int = 200
    timestep_s: float = 0.1
    arena_size_m: float = 5000.0
    target_initial_velocity_mps: float = 15.0
    maneuver_intensity: float = 0.5
    measurement_noise_m: float = 5.0
    detection_probability: float = 0.9
    clutter_rate: float = 5.0
    seed: int = 0


class SwarmScenario:
    """Materialised swarm scenario.

    On construction the underlying Stone Soup simulators are run once and
    the resulting (timestamp, detections) tuples plus ground-truth paths
    are cached. Iterating the scenario then replays the cached data
    deterministically — required for BO sweeps that re-run trackers on the
    same scenario many times.
    """

    def __init__(
        self,
        cfg: SwarmScenarioConfig,
        start_time: datetime | None = None,
    ) -> None:
        self.cfg = cfg
        self.start_time = start_time or datetime(2026, 1, 1, 0, 0, 0)
        self._build()
        self._materialize()

    def _build(self) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        half = self.cfg.arena_size_m / 2.0

        transition = CombinedLinearGaussianTransitionModel(
            [
                ConstantVelocity(self.cfg.maneuver_intensity),
                ConstantVelocity(self.cfg.maneuver_intensity),
            ]
        )

        preexisting = []
        for _ in range(self.cfg.num_targets):
            x = -half + rng.uniform(0.0, half * 0.1)
            y = rng.uniform(-half * 0.8, half * 0.8)
            angle = rng.uniform(-np.pi / 8, np.pi / 8)
            vx = self.cfg.target_initial_velocity_mps * np.cos(angle)
            vy = self.cfg.target_initial_velocity_mps * np.sin(angle)
            preexisting.append([x, vx, y, vy])

        initial_state = GaussianState(
            StateVector([[0.0], [0.0], [0.0], [0.0]]),
            CovarianceMatrix(np.diag([10.0, 1.0, 10.0, 1.0])),
            timestamp=self.start_time,
        )

        self._gt_sim = MultiTargetGroundTruthSimulator(
            transition_model=transition,
            initial_state=initial_state,
            timestep=timedelta(seconds=self.cfg.timestep_s),
            number_steps=self.cfg.duration_steps,
            birth_rate=0.0,
            death_probability=0.0,
            preexisting_states=preexisting,
        )

        self._measurement_model = LinearGaussian(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.diag([self.cfg.measurement_noise_m**2] * 2),
        )

        self._det_sim = SimpleDetectionSimulator(
            groundtruth=self._gt_sim,
            measurement_model=self._measurement_model,
            meas_range=np.array([[-half, half], [-half, half]]),
            detection_probability=self.cfg.detection_probability,
            clutter_rate=self.cfg.clutter_rate,
        )

    def _materialize(self) -> None:
        """Iterate the underlying Stone Soup simulators and cache results.

        Stone Soup's `ConstantVelocity` (process noise) and
        `SimpleDetectionSimulator` (clutter generation, detection misses)
        both draw from numpy's global RNG. To make scenarios reproducible
        across processes and across calls, we snapshot the global RNG
        state, seed it deterministically from `cfg.seed`, materialise, and
        restore on exit so we don't leak side-effects into the caller.
        """
        self._frames: list = []
        truths: set = set()
        saved_state = np.random.get_state()
        # Offset so the simulator seed doesn't collide with the seed used
        # for initial-state placement in `_build`.
        np.random.seed(self.cfg.seed + 10_000)
        try:
            for timestamp, detections in self._det_sim:
                self._frames.append((timestamp, set(detections)))
                truths.update(self._gt_sim.groundtruth_paths)
        finally:
            np.random.set_state(saved_state)
        self._truths = truths

    def __iter__(self) -> Iterator:
        return iter(self._frames)

    @property
    def ground_truth_paths(self):
        return self._truths

    @property
    def measurement_model(self) -> LinearGaussian:
        return self._measurement_model

    @property
    def transition_model(self) -> CombinedLinearGaussianTransitionModel:
        return self._gt_sim.transition_model
