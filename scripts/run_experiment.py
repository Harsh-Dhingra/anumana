"""Hydra-driven experiment entry point.

Default config compares RandomSearch and BayesOpt on a single SwarmScenario.
Override via the CLI, e.g.:

    python scripts/run_experiment.py scenario.num_targets=20 \\
        experiment.num_trials_per_scenario=50

W&B logging is opt-in (wandb.mode=disabled by default in the local config).
"""

from __future__ import annotations

import time
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from anumana import AutoTuner
from anumana.context import extract_scene_features
from anumana.optimizers import BayesOpt, RandomSearch
from anumana.scenarios import SwarmScenario, SwarmScenarioConfig


def _build_scenario(cfg: DictConfig) -> SwarmScenario:
    scn_cfg = SwarmScenarioConfig(
        num_targets=int(cfg.scenario.num_targets),
        duration_steps=int(cfg.scenario.duration_steps),
        timestep_s=float(cfg.scenario.timestep_s),
        arena_size_m=float(cfg.scenario.arena_size_m),
        target_initial_velocity_mps=float(cfg.scenario.target_initial_velocity_mps),
        maneuver_intensity=float(cfg.scenario.maneuver_intensity),
        measurement_noise_m=float(cfg.scenario.get("measurement_noise_m", 5.0)),
        detection_probability=float(cfg.scenario.detection_probability),
        clutter_rate=float(cfg.scenario.clutter_rate),
        seed=int(cfg.seed),
    )
    return SwarmScenario(scn_cfg)


def _build_optimizer(name: str, seed: int) -> Any:
    if name == "random_search":
        return RandomSearch(seed=seed)
    if name in ("bayes_opt", "contextual_bo"):
        return BayesOpt(seed=seed)
    raise ValueError(f"unknown optimizer: {name}")


def _maybe_init_wandb(cfg: DictConfig) -> Any:
    if cfg.wandb.mode in (None, "disabled", "off"):
        return None
    import wandb

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run = _maybe_init_wandb(cfg)
    scn = _build_scenario(cfg)
    features = extract_scene_features(scn)
    print(f"scenario features: {features}")
    print(f"truths: {len(scn.ground_truth_paths)}")

    num_trials = int(cfg.experiment.num_trials_per_scenario)

    rs_optimizer = _build_optimizer("random_search", int(cfg.seed))
    bo_optimizer = _build_optimizer(cfg.optimizer.name, int(cfg.seed))

    print(f"\n=== RandomSearch baseline ({num_trials} trials) ===")
    t0 = time.time()
    rs_result = AutoTuner(scn, rs_optimizer).optimize(num_trials)
    rs_runtime = time.time() - t0
    print(f"  best: {rs_result.best_score:.2f} at iter {rs_result.best_iteration}  ({rs_runtime:.1f}s)")

    print(f"\n=== {cfg.optimizer.name} ({num_trials} trials) ===")
    t0 = time.time()
    bo_result = AutoTuner(scn, bo_optimizer).optimize(num_trials)
    bo_runtime = time.time() - t0
    print(f"  best: {bo_result.best_score:.2f} at iter {bo_result.best_iteration}  ({bo_runtime:.1f}s)")

    improvement = rs_result.best_score - bo_result.best_score
    pct = 100.0 * improvement / max(rs_result.best_score, 1e-9)
    print(f"\n=== Summary ===")
    print(f"  RandomSearch best: {rs_result.best_score:7.2f}")
    print(f"  {cfg.optimizer.name:<16s} best: {bo_result.best_score:7.2f}")
    print(f"  improvement: {improvement:+.2f}  ({pct:+.1f}%)")

    if run is not None:
        run.log(
            {
                "random_search/best_score": rs_result.best_score,
                "random_search/runtime_s": rs_runtime,
                f"{cfg.optimizer.name}/best_score": bo_result.best_score,
                f"{cfg.optimizer.name}/runtime_s": bo_runtime,
                "improvement_pct": pct,
            }
        )
        for i, trial in enumerate(rs_result.trials):
            run.log({"trial/iter": i, "trial/random_search_score": trial.score})
        for i, trial in enumerate(bo_result.trials):
            run.log({"trial/iter": i, f"trial/{cfg.optimizer.name}_score": trial.score})
        run.finish()

    print("\ndone.")


if __name__ == "__main__":
    main()
