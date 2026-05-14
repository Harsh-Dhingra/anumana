"""Offline Bayesian optimization over tracker parameters (no context).

GP-UCB on a single-task GP fit to all observations so far. Works in the
unit cube [0, 1]^d so the search space is independent of the BO backend.
"""

from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from anumana.optimizers.search_space import DEFAULT_SEARCH_SPACE, ParamSpec


class BayesOpt:
    """Offline BO with GP-UCB. Minimises the observed metric.

    `observe()` accepts y as "lower is better" (the composite reward). We
    flip the sign internally because BoTorch maximises.
    """

    def __init__(
        self,
        space: list[ParamSpec] | None = None,
        seed: int = 0,
        beta: float = 2.0,
    ) -> None:
        self.space = space or DEFAULT_SEARCH_SPACE
        self.rng = np.random.default_rng(seed)
        self.beta = beta
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        torch.manual_seed(seed)

    @property
    def n_observations(self) -> int:
        return len(self.y)

    def suggest(self, num_points: int = 1) -> np.ndarray:
        """If we have fewer than 5 observations, sample randomly; else use GP-UCB."""
        if self.n_observations < 5:
            return self.rng.uniform(0.0, 1.0, size=(num_points, len(self.space)))

        train_x = torch.tensor(np.array(self.X), dtype=torch.double)
        train_y = -torch.tensor(self.y, dtype=torch.double).unsqueeze(-1)

        gp = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acqf = UpperConfidenceBound(model=gp, beta=self.beta)
        bounds = torch.stack(
            [torch.zeros(len(self.space), dtype=torch.double),
             torch.ones(len(self.space), dtype=torch.double)]
        )
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=num_points,
            num_restarts=10,
            raw_samples=256,
        )
        return candidate.detach().cpu().numpy()

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)
        for xi, yi in zip(x, y):
            self.X.append(np.asarray(xi, dtype=float))
            self.y.append(float(yi))
