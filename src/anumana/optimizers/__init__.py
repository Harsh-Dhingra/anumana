"""Bayesian optimization and RL backends for tracker parameter tuning."""

from anumana.optimizers.bayes_opt import BayesOpt
from anumana.optimizers.contextual_bo import ContextualBayesOpt
from anumana.optimizers.random_search import RandomSearch
from anumana.optimizers.search_space import (
    DEFAULT_SEARCH_SPACE,
    ParamSpec,
    params_from_unit_cube,
)
from anumana.optimizers.warm_start_bo import WarmStartBayesOpt

try:
    from anumana.optimizers.ppo_tuner import (
        PPOTunerConfig,
        TrackerTuningEnv,
        cached_paths_exist,
        load_ppo,
        ppo_propose,
        save_ppo,
        train_ppo,
    )

    _HAS_PPO = True
except Exception:  # stable_baselines3 + gymnasium are an optional [rl] extra
    _HAS_PPO = False
    import traceback

    _PPO_IMPORT_TB = traceback.format_exc()


__all__ = [
    "BayesOpt",
    "ContextualBayesOpt",
    "DEFAULT_SEARCH_SPACE",
    "ParamSpec",
    "RandomSearch",
    "WarmStartBayesOpt",
    "params_from_unit_cube",
]
if _HAS_PPO:
    __all__ += [
        "PPOTunerConfig",
        "TrackerTuningEnv",
        "cached_paths_exist",
        "load_ppo",
        "ppo_propose",
        "save_ppo",
        "train_ppo",
    ]
