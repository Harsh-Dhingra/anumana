"""Phase 1.2: train ContextualBayesOpt on a training pool of scenarios,
evaluate one-shot proposals on held-out scenarios.

Pipeline:

1. Define training cells and held-out cells. Held-out includes at least one
   interior interpolation point and one extrapolation point.
2. For each (cell, seed) in training cells, run vanilla BayesOpt for
   `train_trials` trials. Collect every (theta, c, y) triple.
3. Fit `ContextualBayesOpt` on the pooled training triples.
4. For each held-out cell:
   - Get one-shot proposal from the contextual GP given that cell's context.
   - Run the tracker once with the proposed theta -> score `s_ctx`.
   - Run vanilla BayesOpt with `eval_trials` trials -> best score `s_bo`.
   - Run RandomSearch with `eval_trials` trials -> best score `s_rs`.
   - Run tracker with default params -> score `s_default`.
5. Print + save a comparison table.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from anumana import AutoTuner
from anumana.context import extract_scene_features
from anumana.experiments import GridCell
from anumana.metrics import compute_track_quality
from anumana.optimizers import (
    BayesOpt,
    ContextualBayesOpt,
    RandomSearch,
    params_from_unit_cube,
)
from anumana.scenarios import SwarmScenario, SwarmScenarioConfig
from anumana.trackers import JPDAParams, run_jpda


def _scenario(cell: GridCell, seed: int) -> SwarmScenario:
    return SwarmScenario(
        SwarmScenarioConfig(
            num_targets=cell.num_targets,
            duration_steps=cell.duration_steps,
            clutter_rate=cell.clutter_rate,
            maneuver_intensity=cell.maneuver_intensity,
            detection_probability=cell.detection_probability,
            seed=seed,
        )
    )


def _context_vec(scn: SwarmScenario) -> np.ndarray:
    f = extract_scene_features(scn)
    return np.array(
        [f.estimated_target_density, f.measurement_rate, f.measurement_dispersion],
        dtype=float,
    )


def collect_training_pool(
    cells: list[GridCell],
    seeds: list[int],
    train_trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run vanilla BO on each (cell, seed) and pool every trial as a triple."""
    X_list: list[np.ndarray] = []
    C_list: list[np.ndarray] = []
    y_list: list[float] = []

    for cell in cells:
        for seed in seeds:
            scn = _scenario(cell, seed)
            ctx = _context_vec(scn)
            t0 = time.time()
            res = AutoTuner(scn, BayesOpt(seed=seed)).optimize(train_trials)
            dt = time.time() - t0
            print(
                f"  train: N={cell.num_targets:2d} clut={cell.clutter_rate:4.1f} "
                f"man={cell.maneuver_intensity:4.2f} seed={seed} "
                f"trials={train_trials} best={res.best_score:6.2f} ({dt:5.1f}s)"
            )
            for t in res.trials:
                X_list.append(np.asarray(t.x, dtype=float).flatten())
                C_list.append(ctx.copy())
                y_list.append(float(t.score))
    return np.array(X_list), np.array(C_list), np.array(y_list)


def evaluate_held_out(
    held_out: list[GridCell],
    seeds: list[int],
    ctx_bo: ContextualBayesOpt,
    eval_trials: int,
) -> list[dict]:
    rows: list[dict] = []
    for cell in held_out:
        for seed in seeds:
            scn = _scenario(cell, seed)
            ctx = _context_vec(scn)
            truths = scn.ground_truth_paths

            # 1. Contextual one-shot (exploit mode = posterior mean)
            proposal = ctx_bo.suggest(ctx, num_points=1, exploit=True)[0]
            params_ctx = params_from_unit_cube(proposal)
            t0 = time.time()
            tracks, _ = run_jpda(scn, params_ctx)
            ctx_score = compute_track_quality(tracks, truths).composite
            ctx_runtime = time.time() - t0

            # 2. Vanilla BO with eval_trials
            t0 = time.time()
            bo_res = AutoTuner(scn, BayesOpt(seed=seed)).optimize(eval_trials)
            bo_runtime = time.time() - t0

            # 3. RandomSearch with eval_trials
            t0 = time.time()
            rs_res = AutoTuner(scn, RandomSearch(seed=seed)).optimize(eval_trials)
            rs_runtime = time.time() - t0

            # 4. Default parameters
            t0 = time.time()
            default_tracks, _ = run_jpda(scn, JPDAParams())
            default_score = compute_track_quality(
                default_tracks, truths
            ).composite
            default_runtime = time.time() - t0

            row = {
                **asdict(cell),
                "seed": seed,
                "contextual_one_shot": ctx_score,
                "contextual_one_shot_runtime_s": ctx_runtime,
                "vanilla_bo_best": bo_res.best_score,
                "vanilla_bo_runtime_s": bo_runtime,
                "random_search_best": rs_res.best_score,
                "random_search_runtime_s": rs_runtime,
                "default_params": default_score,
                "default_runtime_s": default_runtime,
                "ctx_vs_default_pct": 100.0
                * (default_score - ctx_score)
                / max(default_score, 1e-9),
                "ctx_vs_vanilla_pct": 100.0
                * (bo_res.best_score - ctx_score)
                / max(bo_res.best_score, 1e-9),
            }
            rows.append(row)
            print(
                f"  eval:  N={cell.num_targets:2d} clut={cell.clutter_rate:4.1f} "
                f"man={cell.maneuver_intensity:4.2f} seed={seed}  "
                f"ctx={ctx_score:6.2f}  bo={bo_res.best_score:6.2f}  "
                f"rs={rs_res.best_score:6.2f}  def={default_score:6.2f}"
            )
    return rows


def _train_cells(expanded: bool = False) -> list[GridCell]:
    base = dict(duration_steps=20, detection_probability=0.9)
    cells = [
        GridCell(num_targets=5, clutter_rate=1.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=5, clutter_rate=3.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=1.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=6.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=10, clutter_rate=3.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=10, clutter_rate=1.0, maneuver_intensity=0.5, **base),
    ]
    if expanded:
        cells.extend(
            [
                GridCell(num_targets=5, clutter_rate=6.0, maneuver_intensity=0.5, **base),
                GridCell(num_targets=8, clutter_rate=3.0, maneuver_intensity=0.1, **base),
                GridCell(num_targets=10, clutter_rate=6.0, maneuver_intensity=0.1, **base),
                GridCell(num_targets=12, clutter_rate=3.0, maneuver_intensity=0.5, **base),
                GridCell(num_targets=12, clutter_rate=1.0, maneuver_intensity=0.1, **base),
                GridCell(num_targets=5, clutter_rate=3.0, maneuver_intensity=0.1, **base),
            ]
        )
    return cells


def _held_out_cells(expanded: bool = False) -> list[GridCell]:
    base = dict(duration_steps=20, detection_probability=0.9)
    if not expanded:
        return [
            # Interior interpolation
            GridCell(num_targets=8, clutter_rate=3.0, maneuver_intensity=0.5, **base),
            # Extrapolation along clutter axis
            GridCell(num_targets=10, clutter_rate=6.0, maneuver_intensity=0.5, **base),
        ]
    # Larger held-out for statistically meaningful CIs. None of these cells
    # appear in `_train_cells(expanded=True)`.
    return [
        # Interior interpolation (all three axes between training cells)
        GridCell(num_targets=7, clutter_rate=2.0, maneuver_intensity=0.3, **base),
        GridCell(num_targets=9, clutter_rate=4.0, maneuver_intensity=0.3, **base),
        # Extrapolation along clutter axis (above training max of 6.0)
        GridCell(num_targets=10, clutter_rate=8.0, maneuver_intensity=0.5, **base),
        # Extrapolation along maneuver axis (above training max of 0.5)
        GridCell(num_targets=8, clutter_rate=3.0, maneuver_intensity=1.0, **base),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-trials", type=int, default=8)
    ap.add_argument("--eval-trials", type=int, default=8)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--expanded-train", action="store_true",
                    help="use the 12-cell expanded training set")
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/contextual_bo/results.json")
    )
    args = ap.parse_args()

    train_cells = _train_cells(expanded=args.expanded_train)
    held_out = _held_out_cells(expanded=args.expanded_train)

    print("=== contextual BO ===")
    print(f"  train cells:    {len(train_cells)}")
    print(f"  held-out cells: {len(held_out)}")
    print(f"  train seeds:    {args.train_seeds}")
    print(f"  eval seeds:     {args.eval_seeds}")
    print(f"  train trials:   {args.train_trials}")
    print(f"  eval trials:    {args.eval_trials}")
    print()

    print("--- collecting training pool ---")
    t0 = time.time()
    X, C, y = collect_training_pool(train_cells, args.train_seeds, args.train_trials)
    print(f"  pool size: {len(y)}  (took {time.time()-t0:.1f}s)")

    print("\n--- fitting contextual GP ---")
    ctx_bo = ContextualBayesOpt(context_dim=C.shape[1], seed=0)
    ctx_bo.fit_on_pool(X, C, y)
    print(f"  fit on {ctx_bo.n_observations} points; context dim={C.shape[1]}")

    print("\n--- evaluating on held-out ---")
    rows = evaluate_held_out(held_out, args.eval_seeds, ctx_bo, args.eval_trials)

    print("\n=== summary ===")
    ctx_arr = np.array([r["contextual_one_shot"] for r in rows])
    bo_arr = np.array([r["vanilla_bo_best"] for r in rows])
    rs_arr = np.array([r["random_search_best"] for r in rows])
    def_arr = np.array([r["default_params"] for r in rows])

    print(f"  contextual one-shot:   {ctx_arr.mean():7.2f}  +/- {ctx_arr.std():5.2f}")
    print(f"  vanilla BO ({args.eval_trials} trials): {bo_arr.mean():7.2f}  +/- {bo_arr.std():5.2f}")
    print(f"  random search ({args.eval_trials}):    {rs_arr.mean():7.2f}  +/- {rs_arr.std():5.2f}")
    print(f"  default params:        {def_arr.mean():7.2f}  +/- {def_arr.std():5.2f}")
    print(
        f"  ctx vs default:        {100*(def_arr.mean()-ctx_arr.mean())/def_arr.mean():+.1f}%"
    )
    print(
        f"  ctx vs vanilla BO:     {100*(bo_arr.mean()-ctx_arr.mean())/bo_arr.mean():+.1f}%"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n  saved -> {args.out}")


if __name__ == "__main__":
    main()
