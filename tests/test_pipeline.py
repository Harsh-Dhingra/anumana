"""Integration tests: full scenario -> tracker -> metrics -> optimizer loop."""

from __future__ import annotations

import pytest

from anumana import AutoTuner
from anumana.context import extract_scene_features
from anumana.metrics import compute_track_quality
from anumana.optimizers import BayesOpt, RandomSearch
from anumana.scenarios import SwarmScenario, SwarmScenarioConfig
from anumana.trackers import JPDAParams, run_jpda


@pytest.fixture(scope="module")
def small_scenario() -> SwarmScenario:
    cfg = SwarmScenarioConfig(
        num_targets=3,
        duration_steps=15,
        clutter_rate=1.0,
        seed=123,
    )
    return SwarmScenario(cfg)


def test_scenario_materialises(small_scenario):
    frames = list(small_scenario)
    assert len(frames) == small_scenario.cfg.duration_steps
    assert len(small_scenario.ground_truth_paths) == small_scenario.cfg.num_targets


def test_jpda_runs(small_scenario):
    tracks, tele = run_jpda(small_scenario, JPDAParams())
    assert isinstance(tracks, set)
    assert len(tele.num_tracks) == small_scenario.cfg.duration_steps


def test_metrics_compute(small_scenario):
    tracks, _ = run_jpda(small_scenario, JPDAParams())
    report = compute_track_quality(tracks, small_scenario.ground_truth_paths)
    assert report.mean_ospa >= 0.0
    assert report.mean_gospa >= 0.0
    assert report.id_switches >= 0
    assert report.fragmentations >= 0
    assert report.composite >= 0.0


def test_scene_features(small_scenario):
    features = extract_scene_features(small_scenario)
    assert features.measurement_rate > 0.0
    assert features.estimated_target_density > 0.0
    assert features.arena_size_m == small_scenario.cfg.arena_size_m


def test_random_search_loop(small_scenario):
    result = AutoTuner(small_scenario, RandomSearch(seed=0)).optimize(3)
    assert len(result.trials) == 3
    assert result.best_score == min(t.score for t in result.trials)


def test_bayes_opt_loop(small_scenario):
    # 6 trials so BO actually fits a GP for the last call.
    result = AutoTuner(small_scenario, BayesOpt(seed=0)).optimize(6)
    assert len(result.trials) == 6
    assert result.best_score == min(t.score for t in result.trials)
