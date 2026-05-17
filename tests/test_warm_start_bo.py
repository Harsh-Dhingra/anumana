"""Tests for WarmStartBayesOpt."""

from __future__ import annotations

import numpy as np

from anumana.optimizers import ContextualBayesOpt, WarmStartBayesOpt


def _fitted_contextual(seed: int = 0) -> ContextualBayesOpt:
    rng = np.random.default_rng(seed)
    n = 40
    theta = rng.uniform(0, 1, size=(n, 3))
    c = rng.uniform(-1, 1, size=(n, 2))
    y = np.sum((theta - 0.5) ** 2, axis=1) + 0.5 * c[:, 0]
    opt = ContextualBayesOpt(context_dim=2, seed=seed)
    opt.fit_on_pool(theta, c, y)
    return opt


def test_first_suggestion_is_contextual_warm_start():
    ctx_model = _fitted_contextual()
    context = np.array([0.0, 0.0])
    ws = WarmStartBayesOpt(ctx_model, context, seed=0)

    first = ws.suggest(num_points=1)[0]
    expected = ctx_model.suggest(context, num_points=1, exploit=True)[0]
    assert np.allclose(first, expected), "trial 0 must be the contextual proposal"
    assert first.shape == (3,)


def test_bootstrap_then_gp():
    ctx_model = _fitted_contextual()
    ws = WarmStartBayesOpt(ctx_model, np.array([0.0, 0.0]), seed=0, n_bootstrap=2)
    rng = np.random.default_rng(1)

    # Trial 0 (warm), trial 1 (random bootstrap), trials 2+ (GP-UCB).
    for i in range(5):
        x = ws.suggest(num_points=1)[0]
        assert x.shape == (3,)
        assert np.all((x >= 0.0) & (x <= 1.0))
        ws.observe(x, np.array([float(rng.uniform())]))
    assert ws.n_observations == 5


def test_observe_accumulates():
    ctx_model = _fitted_contextual()
    ws = WarmStartBayesOpt(ctx_model, np.array([0.0, 0.0]), seed=0)
    ws.observe(np.array([0.1, 0.2, 0.3]), np.array([5.0]))
    ws.observe(np.array([0.4, 0.5, 0.6]), np.array([3.0]))
    assert ws.n_observations == 2
    assert ws.y == [5.0, 3.0]
