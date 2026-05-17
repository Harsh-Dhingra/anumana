"""PPO baseline for scene-adaptive tracker parameter tuning.

Frames the problem as a contextual bandit (single-step episodes): each
episode samples a scenario, the agent observes its scene context, picks a
parameter vector in the unit cube, and receives `-composite_score` as
reward. PPO learns a policy mapping context -> parameters, mirroring the
role of the contextual GP in `ContextualBayesOpt`.

This is the load-bearing comparison for the paper: contextual BO vs PPO
on the same task, with sample efficiency as the headline axis. Closest
prior is Ott et al. 2022 (meta-RL + SAC, 14-dim UKF hyperparameters,
4M training steps to convergence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from anumana.context import extract_scene_features
from anumana.experiments import GridCell
from anumana.metrics import compute_track_quality
from anumana.optimizers.search_space import (
    DEFAULT_SEARCH_SPACE,
    ParamSpec,
    params_from_unit_cube,
)
from anumana.scenarios import SwarmScenario, SwarmScenarioConfig
from anumana.trackers import run_jpda


@dataclass
class PPOTunerConfig:
    train_cells: Sequence[GridCell]
    train_seeds: Sequence[int]
    space: Sequence[ParamSpec] | None = None
    total_timesteps: int = 10_000
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    seed: int = 0


class TrackerTuningEnv(gym.Env):
    """Single-step contextual-bandit gym env for tracker hyperparameter tuning."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        train_cells: Sequence[GridCell],
        train_seeds: Sequence[int],
        space: Sequence[ParamSpec] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.train_cells = list(train_cells)
        self.train_seeds = list(train_seeds)
        self.space = list(space or DEFAULT_SEARCH_SPACE)
        self._rng = np.random.default_rng(seed)

        # 4-D context: (density, measurement_rate, dispersion, arena_size).
        # Use a wide observation box; we VecNormalize externally.
        self.observation_space = spaces.Box(
            low=np.array([-1e10] * 4, dtype=np.float32),
            high=np.array([1e10] * 4, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.space),),
            dtype=np.float32,
        )
        self._current_scenario: SwarmScenario | None = None
        self._current_context: np.ndarray | None = None

    def _sample_scenario(self) -> SwarmScenario:
        cell = self.train_cells[self._rng.integers(len(self.train_cells))]
        seed_val = int(self.train_seeds[self._rng.integers(len(self.train_seeds))])
        cfg = SwarmScenarioConfig(
            num_targets=cell.num_targets,
            duration_steps=cell.duration_steps,
            clutter_rate=cell.clutter_rate,
            maneuver_intensity=cell.maneuver_intensity,
            detection_probability=cell.detection_probability,
            seed=seed_val,
        )
        return SwarmScenario(cfg)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_scenario = self._sample_scenario()
        features = extract_scene_features(self._current_scenario)
        self._current_context = features.as_array().astype(np.float32)
        return self._current_context.copy(), {}

    def step(self, action: np.ndarray):
        assert self._current_scenario is not None
        action = np.clip(action.astype(np.float64), 0.0, 1.0)
        params = params_from_unit_cube(action, list(self.space))
        tracks, _ = run_jpda(self._current_scenario, params)
        report = compute_track_quality(
            tracks, self._current_scenario.ground_truth_paths
        )
        reward = float(-report.composite)
        terminated = True
        truncated = False
        info = {
            "composite": report.composite,
            "mean_ospa": report.mean_ospa,
            "id_switches": report.id_switches,
            "fragmentations": report.fragmentations,
        }
        # next observation is moot (terminated=True); return last context.
        return self._current_context.copy(), reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None


def _make_env(
    train_cells: Sequence[GridCell],
    train_seeds: Sequence[int],
    space: Sequence[ParamSpec] | None,
    seed: int,
):
    def _thunk():
        env = TrackerTuningEnv(train_cells, train_seeds, space, seed=seed)
        return env

    return _thunk


def train_ppo(cfg: PPOTunerConfig, *, verbose: int = 0) -> tuple[PPO, VecNormalize]:
    env_fns = [
        _make_env(cfg.train_cells, cfg.train_seeds, cfg.space, seed=cfg.seed + i)
        for i in range(cfg.n_envs)
    ]
    vec_cls = DummyVecEnv if cfg.n_envs == 1 else SubprocVecEnv
    vec_env = vec_cls(env_fns)
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        verbose=verbose,
        seed=cfg.seed,
    )
    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=False)
    return model, vec_env


def ppo_propose(
    model: PPO,
    vec_env: VecNormalize,
    context: np.ndarray,
    deterministic: bool = True,
) -> np.ndarray:
    """One-shot proposal: given a context vector, return a param vector in unit cube."""
    obs = np.asarray(context, dtype=np.float32)[None, :]
    norm_obs = vec_env.normalize_obs(obs)
    action, _ = model.predict(norm_obs, deterministic=deterministic)
    return np.clip(action[0], 0.0, 1.0)


def save_ppo(model: PPO, vec_env: VecNormalize, path_dir) -> None:
    """Persist a trained PPO policy + its observation-normalisation stats."""
    from pathlib import Path

    d = Path(path_dir)
    d.mkdir(parents=True, exist_ok=True)
    model.save(str(d / "ppo_model"))
    vec_env.save(str(d / "vecnormalize.pkl"))


def load_ppo(
    path_dir,
    train_cells: Sequence[GridCell],
    train_seeds: Sequence[int],
    space: Sequence[ParamSpec] | None = None,
) -> tuple[PPO, VecNormalize]:
    """Reload a cached PPO policy + VecNormalize stats for inference.

    VecNormalize.load needs a venv to wrap; we give it a 1-env dummy that
    is never stepped (we only call normalize_obs for one-shot proposals).
    """
    from pathlib import Path

    d = Path(path_dir)
    dummy = DummyVecEnv(
        [_make_env(list(train_cells), list(train_seeds), space, seed=0)]
    )
    vec_env = VecNormalize.load(str(d / "vecnormalize.pkl"), dummy)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(str(d / "ppo_model"))
    return model, vec_env


def cached_paths_exist(path_dir) -> bool:
    from pathlib import Path

    d = Path(path_dir)
    return (d / "ppo_model.zip").exists() and (d / "vecnormalize.pkl").exists()
