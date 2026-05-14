"""Search-space definition shared between random search, BO and (later) RL.

Each tunable parameter is described once here; optimizers translate it into
their own native representation (BO tensor, RL action space, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anumana.trackers import JPDAParams


@dataclass
class ParamSpec:
    name: str
    low: float
    high: float
    log_scale: bool = False

    def encode(self, value: float) -> float:
        """Map a parameter value to the unit cube [0, 1]."""
        if self.log_scale:
            lo, hi = np.log(self.low), np.log(self.high)
            return float((np.log(value) - lo) / (hi - lo))
        return float((value - self.low) / (self.high - self.low))

    def decode(self, u: float) -> float:
        """Map a unit-cube coordinate to the parameter range."""
        u = float(np.clip(u, 0.0, 1.0))
        if self.log_scale:
            lo, hi = np.log(self.low), np.log(self.high)
            return float(np.exp(lo + u * (hi - lo)))
        return float(self.low + u * (self.high - self.low))


DEFAULT_SEARCH_SPACE: list[ParamSpec] = [
    ParamSpec(name="gate_size", low=1.0, high=20.0, log_scale=False),
    ParamSpec(name="process_noise_scale", low=0.1, high=10.0, log_scale=True),
    ParamSpec(name="pruning_threshold", low=1e-6, high=1e-1, log_scale=True),
]


def params_from_unit_cube(
    u: np.ndarray, space: list[ParamSpec] | None = None
) -> JPDAParams:
    """Build a JPDAParams from a unit-cube vector `u`.

    Untuned fields fall back to JPDAParams defaults.
    """
    space = space or DEFAULT_SEARCH_SPACE
    assert len(u) == len(space), f"expected {len(space)} dims, got {len(u)}"
    values = {spec.name: spec.decode(ui) for spec, ui in zip(space, u)}
    return JPDAParams(**values)
