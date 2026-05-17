"""The benchmark: 5 tuning methods + default on a held-out scenario grid.

This produces the paper's main results table. Methods:

- default            : Stone Soup default JPDAParams (no tuning)
- random_search      : K per-scenario random trials, take best
- vanilla_bo         : K per-scenario GP-UCB trials, take best
- contextual_bo      : one-shot, GP trained on a transfer pool
- warm_start_bo      : K per-scenario trials seeded by the contextual GP
- ppo_oneshot        : one-shot, PPO policy trained on the train grid

Training phase:
- Contextual-BO pool collected over the train grid (joblib-parallel,
  embarrassingly parallel, no model to pickle).
- PPO trained once and cached to disk (--ppo-cache); reused thereafter.

Eval phase: serial over held-out (cell, seed) for correctness/simplicity
(the fitted GP / PPO policy are awkward to ship to joblib workers; the
tracker runs dominate anyway). Bootstrap 95% CIs on every method.
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
from anumana.experiments.grid import run_cell  # reused for parallel pool
from anumana.metrics import compute_track_quality
from anumana.optimizers import (
    BayesOpt,
    ContextualBayesOpt,
    PPOTunerConfig,
    RandomSearch,
    WarmStartBayesOpt,
    cached_paths_exist,
    load_ppo,
    params_from_unit_cube,
    ppo_propose,
    save_ppo,
    train_ppo,
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


def _ctx(scn: SwarmScenario) -> np.ndarray:
    return extract_scene_features(scn).as_array()


def _train_cells() -> list[GridCell]:
    base = dict(duration_steps=20, detection_probability=0.9)
    return [
        GridCell(num_targets=5, clutter_rate=1.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=5, clutter_rate=3.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=1.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=6.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=10, clutter_rate=3.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=10, clutter_rate=1.0, maneuver_intensity=0.5, **base),
    ]


def _held_out_cells() -> list[GridCell]:
    base = dict(duration_steps=20, detection_probability=0.9)
    return [
        GridCell(num_targets=7, clutter_rate=2.0, maneuver_intensity=0.3, **base),
        GridCell(num_targets=9, clutter_rate=4.0, maneuver_intensity=0.3, **base),
        GridCell(num_targets=10, clutter_rate=8.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=3.0, maneuver_intensity=1.0, **base),
    ]


def _bootstrap_ci(x, n_boot=5000, alpha=0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    if len(x) == 0:
        return float("nan"), float("nan")
    b = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(b, 100 * alpha / 2)), float(
        np.percentile(b, 100 * (1 - alpha / 2))
    )


def collect_pool(train_cells, seeds, pool_trials, n_jobs):
    """Parallel BO pool over the train grid (uses run_cell with bayes_opt)."""
    from joblib import Parallel, delayed

    tasks = [(c, s) for c in train_cells for s in seeds]
    nested = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_cell)(
            c, seed=s, num_trials=pool_trials,
            optimizers=("bayes_opt",), on_progress=None,
        )
        for c, s in tasks
    )
    X, C, y = [], [], []
    for (c, s), grid_rows in zip(tasks, nested):
        ctx = _ctx(_scenario(c, s))
        for r in grid_rows:  # one GridResult per optimizer (here: bayes_opt)
            # GridResult serialises per-trial params + scores as JSON.
            xs = json.loads(r.to_row()["trial_xs"])
            ys = json.loads(r.to_row()["history"])
            for xv, yv in zip(xs, ys):
                X.append(np.asarray(xv, float))
                C.append(ctx.copy())
                y.append(float(yv))
    return np.array(X), np.array(C), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--pool-trials", type=int, default=8)
    ap.add_argument("--ppo-steps", type=int, default=20000)
    ap.add_argument("--ppo-envs", type=int, default=4)
    ap.add_argument("--pool-jobs", type=int, default=6)
    ap.add_argument("--ppo-cache", type=Path, default=Path("outputs/ppo_cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/benchmark/results.json"))
    args = ap.parse_args()

    train_cells = _train_cells()
    held_out = _held_out_cells()

    print("=== anumana benchmark ===")
    print(f"  train cells {len(train_cells)} seeds {args.train_seeds}")
    print(f"  held-out cells {len(held_out)} seeds {args.eval_seeds}")
    print(f"  trials/scenario {args.trials}  pool-trials {args.pool_trials}")
    print(f"  ppo-steps {args.ppo_steps}  ppo-cache {args.ppo_cache}")
    print()

    # --- PPO: load from cache or train once ---
    if cached_paths_exist(args.ppo_cache):
        print(f"--- loading cached PPO from {args.ppo_cache} ---")
        ppo_model, ppo_venv = load_ppo(
            args.ppo_cache, train_cells, args.train_seeds
        )
        ppo_train_time = 0.0
    else:
        print("--- training PPO (no cache) ---")
        t0 = time.time()
        ppo_model, ppo_venv = train_ppo(
            PPOTunerConfig(
                train_cells=train_cells,
                train_seeds=args.train_seeds,
                total_timesteps=args.ppo_steps,
                n_envs=args.ppo_envs,
                seed=0,
            ),
            verbose=0,
        )
        ppo_train_time = time.time() - t0
        save_ppo(ppo_model, ppo_venv, args.ppo_cache)
        print(f"  PPO trained in {ppo_train_time:.1f}s, cached -> {args.ppo_cache}")

    # --- Contextual-BO pool (parallel) + fit ---
    print("\n--- collecting contextual-BO pool (parallel) ---")
    t0 = time.time()
    X, C, y = collect_pool(
        train_cells, args.train_seeds, args.pool_trials, args.pool_jobs
    )
    pool_rollouts = len(y)
    print(f"  pool {pool_rollouts} triples in {time.time()-t0:.1f}s")
    ctx_bo = ContextualBayesOpt(context_dim=C.shape[1], seed=0)
    ctx_bo.fit_on_pool(X, C, y)

    # --- Eval on held-out (serial) ---
    print("\n--- evaluating held-out ---")
    rows = []
    for cell in held_out:
        for seed in args.eval_seeds:
            scn = _scenario(cell, seed)
            ctx = _ctx(scn)
            truths = scn.ground_truth_paths

            default = compute_track_quality(
                run_jpda(scn, JPDAParams())[0], truths
            ).composite
            rs = AutoTuner(scn, RandomSearch(seed=seed)).optimize(args.trials)
            vbo = AutoTuner(scn, BayesOpt(seed=seed)).optimize(args.trials)
            ws = AutoTuner(
                scn, WarmStartBayesOpt(ctx_bo, ctx, seed=seed, n_bootstrap=2)
            ).optimize(args.trials)
            cprop = ctx_bo.suggest(ctx, num_points=1, exploit=True)[0]
            c_score = compute_track_quality(
                run_jpda(scn, params_from_unit_cube(cprop))[0], truths
            ).composite
            pprop = ppo_propose(ppo_model, ppo_venv, ctx)
            p_score = compute_track_quality(
                run_jpda(scn, params_from_unit_cube(pprop))[0], truths
            ).composite

            row = {
                **asdict(cell),
                "seed": seed,
                "default": default,
                "random_search": rs.best_score,
                "vanilla_bo": vbo.best_score,
                "contextual_bo_oneshot": c_score,
                "warm_start_bo": ws.best_score,
                "ppo_oneshot": p_score,
            }
            rows.append(row)
            print(
                f"  N={cell.num_targets:2d} c={cell.clutter_rate:4.1f} "
                f"m={cell.maneuver_intensity:4.2f} s={seed}  "
                f"def={default:6.2f} rs={rs.best_score:6.2f} "
                f"vbo={vbo.best_score:6.2f} ctx={c_score:6.2f} "
                f"ws={ws.best_score:6.2f} ppo={p_score:6.2f}"
            )

    methods = [
        "default", "random_search", "vanilla_bo",
        "contextual_bo_oneshot", "warm_start_bo", "ppo_oneshot",
    ]
    print(f"\n=== benchmark summary (mean [95% CI], n={len(rows)}) ===")
    summary = {}
    for m in methods:
        vals = np.array([r[m] for r in rows], float)
        lo, hi = _bootstrap_ci(vals)
        summary[m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "ci_lo": lo,
            "ci_hi": hi,
        }
        print(f"  {m:24s} {vals.mean():7.2f}  [{lo:6.2f}, {hi:6.2f}]  "
              f"std {vals.std():5.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {
            "config": {
                "train_cells": len(train_cells),
                "held_out_cells": len(held_out),
                "train_seeds": list(args.train_seeds),
                "eval_seeds": list(args.eval_seeds),
                "trials": args.trials,
                "pool_trials": args.pool_trials,
                "pool_rollouts": int(pool_rollouts),
                "ppo_steps": args.ppo_steps,
                "ppo_train_time_s": float(ppo_train_time),
                "n_eval_points": len(rows),
            },
            "summary": summary,
            "rows": rows,
        },
        indent=2, default=str,
    ))
    print(f"\n  saved -> {args.out}")


if __name__ == "__main__":
    main()
