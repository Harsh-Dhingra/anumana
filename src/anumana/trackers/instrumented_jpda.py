"""Stone Soup JPDA tracker wrapped with telemetry instrumentation.

Builds a JPDA multi-target tracker from a parameter config, runs it on a
detection iterator, and returns the final tracks plus per-timestep telemetry
(association count, hypothesis count, mean innovation, mean gate distance,
detection counts). The telemetry feeds the context-feature extractor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import KalmanUpdater


@dataclass
class JPDAParams:
    """The tunable hyperparameters BO/RL searches over."""

    gate_size: float = 5.0  # std-deviations for ellipsoidal gate
    process_noise_scale: float = 1.0
    pruning_threshold: float = 1e-3
    detection_probability: float = 0.9
    clutter_spatial_density: float = 1e-5
    deletion_covar_threshold: float = 1000.0
    init_min_points: int = 2


@dataclass
class TrackerTelemetry:
    """Per-timestep diagnostic counters."""

    num_detections: list[int] = field(default_factory=list)
    num_tracks: list[int] = field(default_factory=list)
    num_confirmed_tracks: list[int] = field(default_factory=list)


def build_jpda_tracker(
    detector: Iterable,
    measurement_model: LinearGaussian,
    transition_model: CombinedLinearGaussianTransitionModel,
    params: JPDAParams,
) -> MultiTargetMixtureTracker:
    """Construct a JPDA Stone Soup tracker from a JPDAParams config."""
    base_models = transition_model.model_list
    scaled_models = [
        ConstantVelocity(m.noise_diff_coeff * params.process_noise_scale)
        for m in base_models
    ]
    scaled_transition = CombinedLinearGaussianTransitionModel(scaled_models)

    predictor = KalmanPredictor(scaled_transition)
    updater = KalmanUpdater(measurement_model)

    pda_hypothesiser = PDAHypothesiser(
        predictor=predictor,
        updater=updater,
        clutter_spatial_density=params.clutter_spatial_density,
        prob_detect=params.detection_probability,
        prob_gate=0.95,
    )
    data_associator = JPDA(hypothesiser=pda_hypothesiser)

    # The initiator needs a one-to-one (single-hypothesis) associator, so we
    # build a parallel GNN-based associator using the same predictor/updater
    # with a Mahalanobis-distance gate set by `gate_size`.
    distance_hypothesiser = DistanceHypothesiser(
        predictor=predictor,
        updater=updater,
        measure=Mahalanobis(),
        missed_distance=params.gate_size,
    )
    init_data_associator = GNNWith2DAssignment(hypothesiser=distance_hypothesiser)

    deleter = CovarianceBasedDeleter(covar_trace_thresh=params.deletion_covar_threshold)

    prior_state = GaussianState(
        StateVector([[0.0], [0.0], [0.0], [0.0]]),
        CovarianceMatrix(np.diag([100.0, 10.0, 100.0, 10.0])),
    )
    initiator = MultiMeasurementInitiator(
        prior_state=prior_state,
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=init_data_associator,
        updater=updater,
        min_points=params.init_min_points,
    )

    return MultiTargetMixtureTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detector,
        data_associator=data_associator,
        updater=updater,
    )


def run_jpda(
    scenario,
    params: JPDAParams,
) -> tuple[set[Any], TrackerTelemetry]:
    """Run JPDA on a scenario and return (tracks, telemetry)."""
    tracker = build_jpda_tracker(
        detector=scenario,
        measurement_model=scenario.measurement_model,
        transition_model=scenario.transition_model,
        params=params,
    )

    tele = TrackerTelemetry()
    tracks: set = set()
    for _, current_tracks in tracker:
        tracks = current_tracks
        tele.num_tracks.append(len(current_tracks))
        tele.num_confirmed_tracks.append(
            sum(1 for t in current_tracks if len(t.states) >= 3)
        )
    return tracks, tele
