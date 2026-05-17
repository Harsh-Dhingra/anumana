"""Tests for the PPO tuner env. PPO training itself is too slow for CI;
we only exercise the gym env contract here."""

from __future__ import annotations

import numpy as np
import pytest

from anumana.experiments import GridCell

sb3 = pytest.importorskip("stable_baselines3")
from anumana.optimizers.ppo_tuner import TrackerTuningEnv  # noqa: E402


@pytest.fixture(scope="module")
def env() -> TrackerTuningEnv:
    cells = [
        GridCell(
            num_targets=3,
            duration_steps=10,
            clutter_rate=1.0,
            maneuver_intensity=0.5,
            detection_probability=0.9,
        )
    ]
    return TrackerTuningEnv(cells, [0], seed=0)


def test_observation_and_action_spaces(env):
    assert env.observation_space.shape == (4,)
    assert env.action_space.shape == (3,)
    assert np.all(env.action_space.low == 0.0)
    assert np.all(env.action_space.high == 1.0)


def test_reset_returns_context(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,)
    assert np.isfinite(obs).all()
    assert isinstance(info, dict)


def test_step_is_single_step_bandit(env):
    env.reset(seed=0)
    action = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (4,)
    assert terminated is True  # single-step contextual bandit
    assert truncated is False
    assert reward <= 0.0  # reward = -composite, composite >= 0
    assert "composite" in info
    assert info["composite"] >= 0.0
