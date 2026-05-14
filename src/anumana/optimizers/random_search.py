"""Random-search baseline. Sanity check and lower bound for BO comparisons."""

from __future__ import annotations

import numpy as np

from anumana.optimizers.search_space import DEFAULT_SEARCH_SPACE, ParamSpec


class RandomSearch:
    def __init__(
        self,
        space: list[ParamSpec] | None = None,
        seed: int = 0,
    ) -> None:
        self.space = space or DEFAULT_SEARCH_SPACE
        self.rng = np.random.default_rng(seed)

    def suggest(self, num_points: int = 1) -> np.ndarray:
        """Sample `num_points` unit-cube vectors uniformly."""
        return self.rng.uniform(0.0, 1.0, size=(num_points, len(self.space)))

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:  # noqa: ARG002
        """Random search has no state to update; kept for API parity with BO."""
        return None
