"""Warm-start Bayesian optimization.

The negative result (results/ppo_vs_bo/) showed that one-shot
context-conditioned tuning does not beat cheap per-scenario BO. This
optimizer tests the hypothesis that result motivates: the contextual GP
is a weak *predictor* but a strong *initializer*.

`WarmStartBayesOpt` runs per-scenario GP-UCB exactly like `BayesOpt`,
except trial 0 is the contextual model's exploit proposal for the
scene's context (instead of a random bootstrap point), and the random
bootstrap is shortened (default 2 vs BayesOpt's 5) because we already
start from an informed point.

The headline question: does warm-started BO reach vanilla-BO-8 quality
in <= 4 per-scenario trials?
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

from anumana.optimizers.contextual_bo import ContextualBayesOpt
from anumana.optimizers.search_space import DEFAULT_SEARCH_SPACE, ParamSpec


class WarmStartBayesOpt:
    """Per-scenario GP-UCB seeded by a contextual-GP proposal."""

    def __init__(
        self,
        contextual_model: ContextualBayesOpt,
        context: np.ndarray,
        space: list[ParamSpec] | None = None,
        seed: int = 0,
        beta: float = 2.0,
        n_bootstrap: int = 2,
    ) -> None:
        self.contextual_model = contextual_model
        self.context = np.asarray(context, dtype=float)
        self.space = space or DEFAULT_SEARCH_SPACE
        self.rng = np.random.default_rng(seed)
        self.beta = beta
        # n_bootstrap counts trial 0 (the warm start) + random points.
        self.n_bootstrap = max(2, int(n_bootstrap))
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        torch.manual_seed(seed)

    @property
    def n_observations(self) -> int:
        return len(self.y)

    def suggest(self, num_points: int = 1) -> np.ndarray:
        n = self.n_observations
        dim = len(self.space)

        # Trial 0: the warm start — contextual GP exploit proposal.
        if n == 0:
            return self.contextual_model.suggest(
                self.context, num_points=num_points, exploit=True
            )

        # Trials 1 .. n_bootstrap-1: random exploration to condition the GP.
        if n < self.n_bootstrap:
            return self.rng.uniform(0.0, 1.0, size=(num_points, dim))

        # Trials n_bootstrap .. : per-scenario GP-UCB.
        train_x = torch.tensor(np.array(self.X), dtype=torch.double)
        train_y = -torch.tensor(self.y, dtype=torch.double).unsqueeze(-1)
        gp = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acqf = UpperConfidenceBound(model=gp, beta=self.beta)
        bounds = torch.stack(
            [
                torch.zeros(dim, dtype=torch.double),
                torch.ones(dim, dtype=torch.double),
            ]
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
