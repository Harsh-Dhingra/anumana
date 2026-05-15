"""Contextual Bayesian optimization over tracker parameters.

Extends `BayesOpt` to condition on a scene-context vector `c`. The GP
learns f(theta, c) -> y over the joint (param, context) space. At suggest
time we use BoTorch's `FixedFeatureAcquisitionFunction` to fix the context
columns to the current scene's `c*` and optimise the acquisition over
parameter columns only.

Two training modes:

1. **Batch / transfer** (`fit_on_pool`): pre-collected pool of
   (theta, c, y) triples from many scenarios; fit once, then call
   `suggest(context)` on a new scene for a one-shot proposal.
2. **Online** (`observe(...)` repeatedly): treats every new observation as
   training data, refits on demand.

Context features are standardised (zero mean, unit variance) using
statistics computed on the first batch of observations.
"""

from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition import PosteriorMean, UpperConfidenceBound
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from anumana.optimizers.search_space import DEFAULT_SEARCH_SPACE, ParamSpec


class ContextualBayesOpt:
    """Context-conditioned GP-UCB. Minimises the observed metric."""

    def __init__(
        self,
        context_dim: int,
        space: list[ParamSpec] | None = None,
        seed: int = 0,
        beta: float = 2.0,
        bootstrap_random: int = 5,
    ) -> None:
        self.space = space or DEFAULT_SEARCH_SPACE
        self.param_dim = len(self.space)
        self.context_dim = context_dim
        self.rng = np.random.default_rng(seed)
        self.beta = beta
        self.bootstrap_random = bootstrap_random
        torch.manual_seed(seed)

        self.X: list[np.ndarray] = []
        self.C: list[np.ndarray] = []
        self.y: list[float] = []

        self._ctx_mean: np.ndarray | None = None
        self._ctx_std: np.ndarray | None = None
        self._gp: SingleTaskGP | None = None

    @property
    def n_observations(self) -> int:
        return len(self.y)

    def _standardise_context(self, c: np.ndarray) -> np.ndarray:
        assert self._ctx_mean is not None and self._ctx_std is not None
        std = np.where(self._ctx_std > 1e-9, self._ctx_std, 1.0)
        return (c - self._ctx_mean) / std

    def _fit_gp(self) -> None:
        """Fit the GP on the current pool."""
        X = np.array(self.X, dtype=float)
        C = np.array(self.C, dtype=float)
        y = np.array(self.y, dtype=float)

        if self._ctx_mean is None:
            self._ctx_mean = C.mean(axis=0)
            self._ctx_std = C.std(axis=0)

        C_std = self._standardise_context(C)
        XC = np.concatenate([X, C_std], axis=1)

        train_x = torch.tensor(XC, dtype=torch.double)
        train_y = -torch.tensor(y, dtype=torch.double).unsqueeze(-1)

        gp = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self._gp = gp

    def fit_on_pool(
        self,
        X: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Replace the current pool with a pre-collected (theta, c, y) batch and fit."""
        self.X = [np.asarray(x, dtype=float) for x in X]
        self.C = [np.asarray(c, dtype=float) for c in C]
        self.y = [float(v) for v in y]
        self._ctx_mean = None
        self._ctx_std = None
        self._fit_gp()

    def suggest(
        self,
        context: np.ndarray,
        num_points: int = 1,
        exploit: bool = False,
    ) -> np.ndarray:
        """Return `num_points` unit-cube parameter vectors for this context.

        If `exploit=True`, optimise the posterior mean instead of UCB. Use
        this for one-shot proposals where there will be no follow-up trials
        to exploit BO exploration (i.e., operational deployment).
        """
        if self.n_observations < self.bootstrap_random or self._gp is None:
            return self.rng.uniform(0.0, 1.0, size=(num_points, self.param_dim))

        c_std = self._standardise_context(np.asarray(context, dtype=float))
        if exploit:
            acqf = PosteriorMean(model=self._gp)
        else:
            acqf = UpperConfidenceBound(model=self._gp, beta=self.beta)
        full_dim = self.param_dim + self.context_dim
        fixed = FixedFeatureAcquisitionFunction(
            acq_function=acqf,
            d=full_dim,
            columns=list(range(self.param_dim, full_dim)),
            values=c_std.tolist(),
        )
        bounds = torch.stack(
            [
                torch.zeros(self.param_dim, dtype=torch.double),
                torch.ones(self.param_dim, dtype=torch.double),
            ]
        )
        candidate, _ = optimize_acqf(
            acq_function=fixed,
            bounds=bounds,
            q=num_points,
            num_restarts=10,
            raw_samples=256,
        )
        return candidate.detach().cpu().numpy()

    def observe(
        self,
        x: np.ndarray,
        context: np.ndarray,
        y: np.ndarray,
        refit: bool = True,
    ) -> None:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        c = np.atleast_2d(np.asarray(context, dtype=float))
        if c.shape[0] == 1 and x.shape[0] > 1:
            c = np.repeat(c, x.shape[0], axis=0)
        ys = np.atleast_1d(np.asarray(y, dtype=float))
        for xi, ci, yi in zip(x, c, ys):
            self.X.append(xi)
            self.C.append(ci)
            self.y.append(float(yi))
        if refit and self.n_observations >= self.bootstrap_random:
            self._fit_gp()
