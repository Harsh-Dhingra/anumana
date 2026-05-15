"""Tests for ContextualBayesOpt."""

from __future__ import annotations

import numpy as np

from anumana.optimizers import ContextualBayesOpt


def test_fit_on_pool_and_suggest():
    """Synthetic check: GP-UCB on a context-dependent quadratic should
    suggest interior points (not boundary) for an in-distribution context."""
    rng = np.random.default_rng(0)
    n = 40
    theta = rng.uniform(0, 1, size=(n, 3))
    c = rng.uniform(-1, 1, size=(n, 2))
    y = np.sum((theta - 0.5) ** 2, axis=1) + 0.5 * c[:, 0]

    opt = ContextualBayesOpt(context_dim=2, seed=0)
    opt.fit_on_pool(theta, c, y)
    assert opt.n_observations == n
    assert opt._gp is not None

    # Interior context, GP should be confident about the interior optimum.
    proposal = opt.suggest(np.array([0.0, 0.0]), num_points=1)[0]
    assert proposal.shape == (3,)
    assert np.all((proposal >= 0.0) & (proposal <= 1.0))
    # Loose check that it's somewhere reasonable; UCB explores so we don't
    # require it to be exactly at 0.5.
    assert np.linalg.norm(proposal - 0.5) < 0.5


def test_bootstrap_random_before_fit():
    """Before bootstrap, suggest() falls back to random samples."""
    opt = ContextualBayesOpt(context_dim=2, seed=0, bootstrap_random=5)
    assert opt.n_observations == 0
    # No GP yet; suggest should still return a unit-cube point.
    proposal = opt.suggest(np.array([0.0, 0.0]), num_points=1)[0]
    assert np.all((proposal >= 0.0) & (proposal <= 1.0))


def test_observe_incremental_fits_when_bootstrapped():
    opt = ContextualBayesOpt(context_dim=2, seed=0, bootstrap_random=3)
    rng = np.random.default_rng(0)
    for _ in range(2):
        x = rng.uniform(0, 1, size=3)
        c = rng.uniform(-1, 1, size=2)
        opt.observe(x, c, np.array([rng.uniform()]), refit=True)
    # Below bootstrap threshold, no GP fit yet.
    assert opt._gp is None
    # Crossing the threshold triggers a fit.
    x = rng.uniform(0, 1, size=3)
    c = rng.uniform(-1, 1, size=2)
    opt.observe(x, c, np.array([rng.uniform()]), refit=True)
    assert opt.n_observations == 3
    assert opt._gp is not None
