"""Phase 1.4: head-to-head PPO vs ContextualBayesOpt on the same task.

This is the load-bearing experiment for the paper. We train both methods
on the same set of scenarios with the same parameter search space, then
evaluate one-shot proposals on held-out cells.

Pipeline:
1. Define train and held-out cells. Use the same cells as v3 expanded.
2. Train PPO for `--ppo-steps` steps on the training distribution.
3. Build a contextual BO training pool by running vanilla BO on training
   cells (8 trials per cell-seed pair) and pool all observations.
4. Fit ContextualBayesOpt on the pool.
5. For each held-out (cell, seed):
   - PPO one-shot proposal -> tracker score.
   - Contextual BO one-shot proposal (exploit) -> tracker score.
   - Vanilla BO with the same trial budget as contextual training.
   - Random search with the same trial budget.
   - Default parameters.
6. Save JSON, print summary, optionally generate sample-efficiency plot.

The headline number is "PPO's score after X tracker rollouts vs
contextual BO's score after Y tracker rollouts" — sample efficiency.
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
    PPOTunerConfig,
    RandomSearch,
    params_from_unit_cube,
    ppo_propose,
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


def _context_vec(scn: SwarmScenario) -> np.ndarray:
    return extract_scene_features(scn).as_array()


def _train_cells() -> list[GridCell]:
    base = dict(duration_steps=20, detection_probability=0.9)
    # Mirror scripts/run_contextual_bo.py::_train_cells(expanded=True).
    cells = [
        GridCell(num_targets=5, clutter_rate=1.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=5, clutter_rate=3.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=1.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=6.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=10, clutter_rate=3.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=10, clutter_rate=1.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=5, clutter_rate=6.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=3.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=10, clutter_rate=6.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=5, clutter_rate=3.0, maneuver_intensity=0.1, **base),
    ]
    return cells


def _held_out_cells() -> list[GridCell]:
    base = dict(duration_steps=20, detection_probability=0.9)
    return [
        GridCell(num_targets=7, clutter_rate=2.0, maneuver_intensity=0.3, **base),
        GridCell(num_targets=9, clutter_rate=4.0, maneuver_intensity=0.3, **base),
        GridCell(num_targets=10, clutter_rate=8.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=3.0, maneuver_intensity=1.0, **base),
    ]


def collect_bo_pool(cells, seeds, trials_per_cell):
    X_list, C_list, y_list = [], [], []
    pool_rollouts = 0
    for cell in cells:
        for seed in seeds:
            scn = _scenario(cell, seed)
            ctx = _context_vec(scn)
            t0 = time.time()
            res = AutoTuner(scn, BayesOpt(seed=seed)).optimize(trials_per_cell)
            dt = time.time() - t0
            print(
                f"  bo-train: N={cell.num_targets:2d} clut={cell.clutter_rate:4.1f} "
                f"man={cell.maneuver_intensity:4.2f} seed={seed} "
                f"trials={trials_per_cell} best={res.best_score:6.2f} ({dt:5.1f}s)"
            )
            for t in res.trials:
                X_list.append(np.asarray(t.x, dtype=float).flatten())
                C_list.append(ctx.copy())
                y_list.append(float(t.score))
                pool_rollouts += 1
    return np.array(X_list), np.array(C_list), np.array(y_list), pool_rollouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps", type=int, default=20_000)
    ap.add_argument("--ppo-envs", type=int, default=4)
    ap.add_argument("--bo-pool-trials", type=int, default=8,
                    help="trials per (cell, seed) when building the BO training pool")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/ppo_vs_bo/results.json"),
    )
    args = ap.parse_args()

    train_cells = _train_cells()
    held_out = _held_out_cells()

    print("=== PPO vs contextual BO ===")
    print(f"  train cells:    {len(train_cells)}")
    print(f"  held-out cells: {len(held_out)}")
    print(f"  train seeds:    {args.train_seeds}")
    print(f"  eval seeds:     {args.eval_seeds}")
    print(f"  PPO steps:      {args.ppo_steps}  (n_envs={args.ppo_envs})")
    print(f"  BO pool trials: {args.bo_pool_trials} per (cell, seed)")
    bo_pool_rollouts_est = len(train_cells) * len(args.train_seeds) * args.bo_pool_trials
    print(f"  BO pool budget: ~{bo_pool_rollouts_est} tracker rollouts")
    print()

    # --- 1) Train PPO ---
    print("--- training PPO ---")
    ppo_cfg = PPOTunerConfig(
        train_cells=train_cells,
        train_seeds=args.train_seeds,
        total_timesteps=args.ppo_steps,
        n_envs=args.ppo_envs,
        seed=0,
    )
    t0 = time.time()
    ppo_model, ppo_venv = train_ppo(ppo_cfg, verbose=0)
    ppo_train_time = time.time() - t0
    print(f"  PPO trained in {ppo_train_time:.1f}s "
          f"({args.ppo_steps} tracker rollouts)")

    # --- 2) Build BO training pool + fit contextual GP ---
    print("\n--- building BO training pool ---")
    t0 = time.time()
    X, C, y, bo_pool_rollouts = collect_bo_pool(
        train_cells, args.train_seeds, args.bo_pool_trials
    )
    bo_pool_time = time.time() - t0
    print(f"  pool: {len(y)} triples ({bo_pool_rollouts} rollouts) "
          f"in {bo_pool_time:.1f}s")

    print("\n--- fitting contextual GP ---")
    ctx_bo = ContextualBayesOpt(context_dim=C.shape[1], seed=0)
    ctx_bo.fit_on_pool(X, C, y)

    # --- 3) Evaluate on held-out ---
    print("\n--- evaluating on held-out ---")
    eval_rollouts_per_cell = args.bo_pool_trials
    rows = []
    for cell in held_out:
        for seed in args.eval_seeds:
            scn = _scenario(cell, seed)
            ctx = _context_vec(scn)
            truths = scn.ground_truth_paths

            # contextual BO one-shot
            proposal_ctx = ctx_bo.suggest(ctx, num_points=1, exploit=True)[0]
            params_ctx = params_from_unit_cube(proposal_ctx)
            tr_ctx, _ = run_jpda(scn, params_ctx)
            ctx_score = compute_track_quality(tr_ctx, truths).composite

            # PPO one-shot
            proposal_ppo = ppo_propose(ppo_model, ppo_venv, ctx)
            params_ppo = params_from_unit_cube(proposal_ppo)
            tr_ppo, _ = run_jpda(scn, params_ppo)
            ppo_score = compute_track_quality(tr_ppo, truths).composite

            # vanilla BO (same per-scenario budget as BO pool trials)
            bo_res = AutoTuner(scn, BayesOpt(seed=seed)).optimize(eval_rollouts_per_cell)

            # random search (same budget)
            rs_res = AutoTuner(scn, RandomSearch(seed=seed)).optimize(eval_rollouts_per_cell)

            # default params
            tr_def, _ = run_jpda(scn, JPDAParams())
            default_score = compute_track_quality(tr_def, truths).composite

            row = {
                **asdict(cell),
                "seed": seed,
                "contextual_bo_one_shot": ctx_score,
                "ppo_one_shot": ppo_score,
                "vanilla_bo_full_budget": bo_res.best_score,
                "random_search_full_budget": rs_res.best_score,
                "default_params": default_score,
                "context_vec": ctx.tolist(),
                "ppo_proposal": proposal_ppo.tolist(),
                "ctx_proposal": proposal_ctx.tolist(),
            }
            rows.append(row)
            print(
                f"  N={cell.num_targets:2d} clut={cell.clutter_rate:4.1f} "
                f"man={cell.maneuver_intensity:4.2f} seed={seed}  "
                f"ctx={ctx_score:6.2f}  ppo={ppo_score:6.2f}  "
                f"bo={bo_res.best_score:6.2f}  rs={rs_res.best_score:6.2f}  "
                f"def={default_score:6.2f}"
            )

    # --- 4) Summary ---
    ctx_arr = np.array([r["contextual_bo_one_shot"] for r in rows])
    ppo_arr = np.array([r["ppo_one_shot"] for r in rows])
    bo_arr = np.array([r["vanilla_bo_full_budget"] for r in rows])
    rs_arr = np.array([r["random_search_full_budget"] for r in rows])
    def_arr = np.array([r["default_params"] for r in rows])

    print("\n=== summary (mean +/- std across held-out points) ===")
    print(f"  contextual BO one-shot:   {ctx_arr.mean():7.2f}  +/- {ctx_arr.std():5.2f}")
    print(f"  PPO one-shot:             {ppo_arr.mean():7.2f}  +/- {ppo_arr.std():5.2f}")
    print(f"  vanilla BO ({eval_rollouts_per_cell} trials):  {bo_arr.mean():7.2f}  +/- {bo_arr.std():5.2f}")
    print(f"  random search ({eval_rollouts_per_cell}):     {rs_arr.mean():7.2f}  +/- {rs_arr.std():5.2f}")
    print(f"  default params:           {def_arr.mean():7.2f}  +/- {def_arr.std():5.2f}")

    print("\n=== sample efficiency ===")
    print(f"  Contextual BO training rollouts: {bo_pool_rollouts}")
    print(f"  PPO training rollouts:           {args.ppo_steps}")
    print(f"  ratio (PPO/ctx):                  {args.ppo_steps / max(bo_pool_rollouts,1):.1f}x")
    print(f"  ctx vs PPO improvement: "
          f"{100*(ppo_arr.mean() - ctx_arr.mean())/max(ppo_arr.mean(),1e-9):+.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "ppo_steps": args.ppo_steps,
            "ppo_envs": args.ppo_envs,
            "bo_pool_trials": args.bo_pool_trials,
            "bo_pool_rollouts": int(bo_pool_rollouts),
            "train_seeds": list(args.train_seeds),
            "eval_seeds": list(args.eval_seeds),
            "ppo_train_time_s": float(ppo_train_time),
            "bo_pool_time_s": float(bo_pool_time),
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  saved -> {args.out}")


if __name__ == "__main__":
    main()
