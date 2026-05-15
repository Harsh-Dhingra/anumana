"""Bayesian optimization and RL backends for tracker parameter tuning."""

from anumana.optimizers.bayes_opt import BayesOpt
from anumana.optimizers.contextual_bo import ContextualBayesOpt
from anumana.optimizers.random_search import RandomSearch
from anumana.optimizers.search_space import (
    DEFAULT_SEARCH_SPACE,
    ParamSpec,
    params_from_unit_cube,
)

__all__ = [
    "BayesOpt",
    "ContextualBayesOpt",
    "DEFAULT_SEARCH_SPACE",
    "ParamSpec",
    "RandomSearch",
    "params_from_unit_cube",
]
