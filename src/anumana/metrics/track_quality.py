"""Track-quality metrics for multi-target tracker evaluation.

Wraps Stone Soup's OSPA/GOSPA and adds simple identity-switch and
fragmentation counters. Returns a `TrackQualityReport` dataclass with
scalar summaries suitable as Bayesian-optimization rewards.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from stonesoup.dataassociator.tracktotrack import TrackToTruth
from stonesoup.measures import Euclidean
from stonesoup.metricgenerator.manager import SimpleManager
from stonesoup.metricgenerator.ospametric import GOSPAMetric, OSPAMetric


@dataclass
class TrackQualityReport:
    mean_ospa: float
    mean_gospa: float
    id_switches: int
    fragmentations: int
    num_tracks: int
    num_truths: int
    composite: float = 0.0
    per_timestep: dict[str, list[float]] = field(default_factory=dict)


def compute_track_quality(
    tracks: Iterable,
    truths: Iterable,
    *,
    c: float = 100.0,
    p: float = 2.0,
    association_threshold: float = 50.0,
    composite_weights: tuple[float, float, float] = (1.0, 5.0, 5.0),
) -> TrackQualityReport:
    """Compute OSPA, GOSPA, ID switches and fragmentation.

    Composite score (used as BO reward, lower is better):
        composite = w_ospa * mean_ospa + w_id * id_switches + w_frag * fragmentations
    """
    tracks_list = list(tracks)
    truths_list = list(truths)

    measure = Euclidean(mapping=(0, 2), mapping2=(0, 2))
    ospa = OSPAMetric(p=p, c=c, measure=measure)
    gospa = GOSPAMetric(p=p, c=c, measure=measure)

    manager = SimpleManager(generators=[ospa, gospa])
    manager.add_data(groundtruth_paths=truths_list, tracks=tracks_list)
    metrics = manager.generate_metrics()

    ospa_values: list[float] = []
    gospa_values: list[float] = []
    for key, m in metrics.items():
        if key.startswith("GOSPA"):
            gospa_values = [float(v.value["distance"]) for v in m.value]
        elif key.startswith("OSPA"):
            ospa_values = [float(v.value) for v in m.value]

    mean_ospa = float(np.mean(ospa_values)) if ospa_values else c
    mean_gospa = float(np.mean(gospa_values)) if gospa_values else c

    id_switches, fragmentations = _count_id_switches_and_fragmentations(
        tracks_list, truths_list, association_threshold=association_threshold, measure=measure
    )

    w_ospa, w_id, w_frag = composite_weights
    composite = w_ospa * mean_ospa + w_id * id_switches + w_frag * fragmentations

    return TrackQualityReport(
        mean_ospa=mean_ospa,
        mean_gospa=mean_gospa,
        id_switches=id_switches,
        fragmentations=fragmentations,
        num_tracks=len(tracks_list),
        num_truths=len(truths_list),
        composite=composite,
        per_timestep={"ospa": ospa_values, "gospa": gospa_values},
    )


def _count_id_switches_and_fragmentations(
    tracks,
    truths,
    *,
    association_threshold: float,
    measure: Euclidean,
) -> tuple[int, int]:
    """Greedy per-timestep nearest-neighbour assignment, then count:

    - id_switches: a truth's assigned track ID changes between consecutive
      timesteps where both timesteps had an assignment.
    - fragmentations: a truth has an assignment, loses it for >=1 timestep,
      then regains one (any assignment).
    """
    if not tracks or not truths:
        return 0, 0

    timestamps = sorted({s.timestamp for t in truths for s in t.states})

    truth_to_track_at_t: dict = defaultdict(dict)
    for ts in timestamps:
        active_tracks = [
            (t, _state_at(t, ts)) for t in tracks
        ]
        active_tracks = [(t, s) for t, s in active_tracks if s is not None]
        active_truths = [
            (g, _state_at(g, ts)) for g in truths
        ]
        active_truths = [(g, s) for g, s in active_truths if s is not None]

        if not active_tracks or not active_truths:
            continue

        cost = np.full((len(active_truths), len(active_tracks)), np.inf)
        for i, (_, gs) in enumerate(active_truths):
            for j, (_, ts_state) in enumerate(active_tracks):
                d = float(measure(gs, ts_state))
                if d <= association_threshold:
                    cost[i, j] = d

        used_tracks: set[int] = set()
        order = np.argsort(cost.min(axis=1))
        for i in order:
            row = cost[i].copy()
            row[list(used_tracks)] = np.inf
            j = int(np.argmin(row))
            if np.isfinite(row[j]):
                used_tracks.add(j)
                truth_obj = active_truths[i][0]
                track_obj = active_tracks[j][0]
                truth_to_track_at_t[id(truth_obj)][ts] = id(track_obj)

    id_switches = 0
    fragmentations = 0
    for assignments in truth_to_track_at_t.values():
        ordered_ts = sorted(assignments.keys())
        prev_track = None
        had_gap = False
        for i, ts in enumerate(ordered_ts):
            cur = assignments[ts]
            if prev_track is not None and cur != prev_track and not had_gap:
                id_switches += 1
            if i > 0 and (ts - ordered_ts[i - 1]).total_seconds() > 0:
                gap_len = (ts - ordered_ts[i - 1]).total_seconds()
                expected = (
                    (ordered_ts[i] - ordered_ts[i - 1]).total_seconds()
                )
                if gap_len > expected * 1.5:
                    fragmentations += 1
                    had_gap = True
                else:
                    had_gap = False
            prev_track = cur

    return id_switches, fragmentations


def _state_at(path, timestamp):
    """Return the State at `timestamp` if present in `path`, else None."""
    for state in path.states:
        if state.timestamp == timestamp:
            return state
    return None
