"""Multi-scenario grid evaluation for BO vs RandomSearch.

A `GridCell` is one point in the (num_targets x clutter_rate x
maneuver_intensity x ...) sweep. For each cell we run multiple seeds and,
for each seed, both RandomSearch and BayesOpt. Results are returned as a
list of `GridResult` rows that can be dumped to CSV or pandas.
"""

from __future__ import annotations

import csv
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from anumana import AutoTuner
from anumana.context import extract_scene_features
from anumana.optimizers import BayesOpt, RandomSearch
from anumana.scenarios import SwarmScenario, SwarmScenarioConfig


@dataclass(frozen=True)
class GridCell:
    num_targets: int
    duration_steps: int
    clutter_rate: float
    maneuver_intensity: float
    detection_probability: float


@dataclass
class GridResult:
    cell: GridCell
    seed: int
    optimizer: str
    num_trials: int
    best_score: float
    best_iteration: int
    runtime_s: float
    history: list[float] = field(default_factory=list)
    trial_xs: list[list[float]] = field(default_factory=list)
    scene_features: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self.cell)
        d.update(
            {
                "seed": self.seed,
                "optimizer": self.optimizer,
                "num_trials": self.num_trials,
                "best_score": self.best_score,
                "best_iteration": self.best_iteration,
                "runtime_s": self.runtime_s,
                "history": json.dumps(list(self.history)),
                "trial_xs": json.dumps(self.trial_xs),
            }
        )
        d.update({f"feat_{k}": v for k, v in self.scene_features.items()})
        return d


def _build_optimizer(name: str, seed: int):
    if name == "random_search":
        return RandomSearch(seed=seed)
    if name == "bayes_opt":
        return BayesOpt(seed=seed)
    raise ValueError(name)


def run_cell(
    cell: GridCell,
    *,
    seed: int,
    num_trials: int,
    optimizers: Iterable[str] = ("random_search", "bayes_opt"),
    on_progress: Callable[[GridResult], None] | None = None,
) -> list[GridResult]:
    """Run all optimizers on a single (cell, seed) and return per-optimizer rows."""
    scn_cfg = SwarmScenarioConfig(
        num_targets=cell.num_targets,
        duration_steps=cell.duration_steps,
        clutter_rate=cell.clutter_rate,
        maneuver_intensity=cell.maneuver_intensity,
        detection_probability=cell.detection_probability,
        seed=seed,
    )
    scn = SwarmScenario(scn_cfg)
    features = extract_scene_features(scn)
    feat_dict = {
        "target_density": features.estimated_target_density,
        "measurement_rate": features.measurement_rate,
        "dispersion": features.measurement_dispersion,
    }

    rows: list[GridResult] = []
    for opt_name in optimizers:
        optimizer = _build_optimizer(opt_name, seed=seed)
        t0 = time.time()
        tuning = AutoTuner(scn, optimizer).optimize(num_trials)
        runtime = time.time() - t0

        result = GridResult(
            cell=cell,
            seed=seed,
            optimizer=opt_name,
            num_trials=num_trials,
            best_score=tuning.best_score,
            best_iteration=tuning.best_iteration,
            runtime_s=runtime,
            history=[t.score for t in tuning.trials],
            trial_xs=[list(map(float, np.asarray(t.x).flatten())) for t in tuning.trials],
            scene_features=feat_dict,
        )
        rows.append(result)
        if on_progress is not None:
            on_progress(result)
    return rows


def run_grid(
    cells: Iterable[GridCell],
    seeds: Iterable[int],
    *,
    num_trials: int = 15,
    optimizers: Iterable[str] = ("random_search", "bayes_opt"),
    csv_path: str | Path | None = None,
    verbose: bool = True,
) -> list[GridResult]:
    """Run the cartesian product cells x seeds, return all results.

    If `csv_path` is given, rows are appended as they complete (so partial
    progress is preserved on crash / kill).
    """
    cells = list(cells)
    seeds = list(seeds)
    total = len(cells) * len(seeds) * len(list(optimizers))
    optimizers = list(optimizers)

    csv_path = Path(csv_path) if csv_path else None
    csv_file = None
    csv_writer = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")

    all_results: list[GridResult] = []
    done = 0

    def _on_progress(r: GridResult) -> None:
        nonlocal done, csv_writer
        done += 1
        if csv_writer is None and csv_file is not None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(r.to_row().keys()))
            csv_writer.writeheader()
        if csv_writer is not None:
            csv_writer.writerow(r.to_row())
            csv_file.flush()
        if verbose:
            print(
                f"  [{done:3d}/{total}] {r.optimizer:<13s} "
                f"N={r.cell.num_targets:3d} clut={r.cell.clutter_rate:4.1f} "
                f"man={r.cell.maneuver_intensity:4.2f} seed={r.seed} "
                f"best={r.best_score:7.2f} ({r.runtime_s:5.1f}s)"
            )

    try:
        for cell, seed in itertools.product(cells, seeds):
            rows = run_cell(
                cell,
                seed=seed,
                num_trials=num_trials,
                optimizers=optimizers,
                on_progress=_on_progress,
            )
            all_results.extend(rows)
    finally:
        if csv_file is not None:
            csv_file.close()

    return all_results


def summarise(results: list[GridResult]) -> dict:
    """Aggregate grid results into a quick win/loss/tie summary."""
    by_cell_seed: dict[tuple, dict[str, GridResult]] = {}
    for r in results:
        key = (r.cell, r.seed)
        by_cell_seed.setdefault(key, {})[r.optimizer] = r

    wins_bo = 0
    wins_rs = 0
    ties = 0
    improvements: list[float] = []
    for opts in by_cell_seed.values():
        if "bayes_opt" not in opts or "random_search" not in opts:
            continue
        rs = opts["random_search"].best_score
        bo = opts["bayes_opt"].best_score
        improvements.append((rs - bo) / max(rs, 1e-9))
        if bo < rs - 1e-9:
            wins_bo += 1
        elif rs < bo - 1e-9:
            wins_rs += 1
        else:
            ties += 1

    arr = np.array(improvements) if improvements else np.zeros(0)
    return {
        "num_cell_seed_pairs": len(by_cell_seed),
        "bo_wins": wins_bo,
        "rs_wins": wins_rs,
        "ties": ties,
        "mean_improvement_pct": float(100 * arr.mean()) if arr.size else 0.0,
        "median_improvement_pct": float(100 * np.median(arr)) if arr.size else 0.0,
        "min_improvement_pct": float(100 * arr.min()) if arr.size else 0.0,
        "max_improvement_pct": float(100 * arr.max()) if arr.size else 0.0,
    }
