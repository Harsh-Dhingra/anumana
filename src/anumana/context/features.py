"""Scene-context feature extractor.

Computes a small vector of statistics from a scenario's detections + tracks
that BO conditions on when learning a context → parameter mapping.

For the v0 offline-BO phase we compute one feature vector per scenario from
the cached detection stream. Online streaming features (computed on a sliding
window inside the tracker loop) come in a later iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SceneFeatures:
    estimated_target_density: float
    measurement_rate: float
    measurement_dispersion: float
    arena_size_m: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.estimated_target_density,
                self.measurement_rate,
                self.measurement_dispersion,
                self.arena_size_m,
            ],
            dtype=float,
        )

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "estimated_target_density",
            "measurement_rate",
            "measurement_dispersion",
            "arena_size_m",
        ]


def extract_scene_features(scenario) -> SceneFeatures:
    """Compute a SceneFeatures vector from a materialised SwarmScenario."""
    cfg = scenario.cfg
    arena_area = cfg.arena_size_m**2

    positions: list[np.ndarray] = []
    per_step_counts: list[int] = []
    for _, detections in scenario:
        per_step_counts.append(len(detections))
        for d in detections:
            sv = np.asarray(d.state_vector).flatten()
            positions.append(sv)

    if not positions:
        return SceneFeatures(
            estimated_target_density=0.0,
            measurement_rate=0.0,
            measurement_dispersion=0.0,
            arena_size_m=cfg.arena_size_m,
        )

    pos_array = np.stack(positions)
    measurement_rate = float(np.mean(per_step_counts))
    estimated_density = measurement_rate / max(arena_area, 1.0)
    dispersion = float(np.mean(np.std(pos_array, axis=0)))

    return SceneFeatures(
        estimated_target_density=estimated_density,
        measurement_rate=measurement_rate,
        measurement_dispersion=dispersion,
        arena_size_m=cfg.arena_size_m,
    )
