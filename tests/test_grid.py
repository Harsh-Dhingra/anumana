"""Tests for the grid-experiment harness."""

from __future__ import annotations

from pathlib import Path

from anumana.experiments import GridCell, run_grid
from anumana.experiments.grid import summarise


def test_run_grid_one_cell(tmp_path: Path) -> None:
    cells = [
        GridCell(
            num_targets=3,
            duration_steps=10,
            clutter_rate=1.0,
            maneuver_intensity=0.5,
            detection_probability=0.9,
        )
    ]
    csv_path = tmp_path / "grid.csv"
    results = run_grid(
        cells=cells,
        seeds=[0],
        num_trials=4,
        csv_path=csv_path,
        verbose=False,
    )
    assert len(results) == 2  # RS + BO
    assert {r.optimizer for r in results} == {"random_search", "bayes_opt"}
    assert csv_path.exists()

    summary = summarise(results)
    assert summary["num_cell_seed_pairs"] == 1
    # With 4 trials BO falls back to random, so RS and BO should tie.
    assert summary["bo_wins"] + summary["rs_wins"] + summary["ties"] == 1


def test_run_grid_parallel_matches_serial(tmp_path: Path) -> None:
    """Parallel execution must produce the same set of results as serial."""
    cells = [
        GridCell(
            num_targets=3,
            duration_steps=8,
            clutter_rate=1.0,
            maneuver_intensity=0.5,
            detection_probability=0.9,
        ),
        GridCell(
            num_targets=4,
            duration_steps=8,
            clutter_rate=2.0,
            maneuver_intensity=0.3,
            detection_probability=0.9,
        ),
    ]
    serial = run_grid(cells, seeds=[0], num_trials=3, verbose=False, n_jobs=1)
    parallel = run_grid(
        cells, seeds=[0], num_trials=3, verbose=False, n_jobs=2,
        csv_path=tmp_path / "p.csv",
    )
    assert len(serial) == len(parallel) == 4  # 2 cells x 1 seed x 2 opt
    # Scenarios are fully seeded, so per-(cell,seed,opt) best scores match.
    key = lambda r: (r.cell.num_targets, r.optimizer)  # noqa: E731
    s = {key(r): round(r.best_score, 6) for r in serial}
    p = {key(r): round(r.best_score, 6) for r in parallel}
    assert s == p, f"parallel != serial: {s} vs {p}"
    assert (tmp_path / "p.csv").exists()
