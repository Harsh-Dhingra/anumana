"""Week-1 gate experiment: does warm-starting per-scenario BO from the
contextual GP reach vanilla-BO-8 quality in <= 4 trials?

Pipeline:
1. Build the contextual-BO training pool (same training cells as
   run_ppo_vs_bo.py) and fit ContextualBayesOpt.
2. For each held-out (cell, seed) in the v3 held-out set:
   - vanilla BayesOpt for `--trials` trials -> best-so-far curve
   - WarmStartBayesOpt for `--trials` trials -> best-so-far curve
   - record contextual one-shot (= warm-start trial-0 score) and default
3. Aggregate mean best-so-far per trial index across all held-out points.
4. GATE: smallest K where mean warm-start best-so-far(K) <= mean
   vanilla best-so-far(8). PASS if K <= 4.
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
    WarmStartBayesOpt,
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
        GridCell(num_targets=5, clutter_rate=6.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=3.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=10, clutter_rate=6.0, maneuver_intensity=0.1, **base),
        GridCell(num_targets=5, clutter_rate=3.0, maneuver_intensity=0.1, **base),
    ]


def _held_out_cells() -> list[GridCell]:
    base = dict(duration_steps=20, detection_probability=0.9)
    return [
        GridCell(num_targets=7, clutter_rate=2.0, maneuver_intensity=0.3, **base),
        GridCell(num_targets=9, clutter_rate=4.0, maneuver_intensity=0.3, **base),
        GridCell(num_targets=10, clutter_rate=8.0, maneuver_intensity=0.5, **base),
        GridCell(num_targets=8, clutter_rate=3.0, maneuver_intensity=1.0, **base),
    ]


def _best_so_far(scores: list[float]) -> np.ndarray:
    return np.minimum.accumulate(np.array(scores, dtype=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--pool-trials", type=int, default=8)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-bootstrap", type=int, default=2)
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/warm_start/gate.json")
    )
    args = ap.parse_args()

    train_cells = _train_cells()
    held_out = _held_out_cells()

    print("=== warm-start gate ===")
    print(f"  train cells:    {len(train_cells)}  seeds {args.train_seeds}")
    print(f"  held-out cells: {len(held_out)}  seeds {args.eval_seeds}")
    print(f"  trials/scenario: {args.trials}  n_bootstrap={args.n_bootstrap}")
    print()

    # 1) Build pool + fit contextual GP.
    print("--- building contextual-BO pool ---")
    X, C, y = [], [], []
    t0 = time.time()
    for cell in train_cells:
        for seed in args.train_seeds:
            scn = _scenario(cell, seed)
            ctx = _ctx(scn)
            res = AutoTuner(scn, BayesOpt(seed=seed)).optimize(args.pool_trials)
            for t in res.trials:
                X.append(np.asarray(t.x, float).flatten())
                C.append(ctx.copy())
                y.append(float(t.score))
    pool_rollouts = len(y)
    print(f"  pool: {pool_rollouts} triples in {time.time()-t0:.1f}s")

    ctx_bo = ContextualBayesOpt(context_dim=np.array(C).shape[1], seed=0)
    ctx_bo.fit_on_pool(np.array(X), np.array(C), np.array(y))

    # 2) Held-out: vanilla BO vs warm-start BO best-so-far curves.
    print("\n--- held-out: vanilla vs warm-start ---")
    rows = []
    for cell in held_out:
        for seed in args.eval_seeds:
            scn = _scenario(cell, seed)
            ctx = _ctx(scn)
            truths = scn.ground_truth_paths

            van = AutoTuner(scn, BayesOpt(seed=seed)).optimize(args.trials)
            ws_opt = WarmStartBayesOpt(
                ctx_bo, ctx, seed=seed, n_bootstrap=args.n_bootstrap
            )
            ws = AutoTuner(scn, ws_opt).optimize(args.trials)

            default_score = compute_track_quality(
                run_jpda(scn, JPDAParams())[0], truths
            ).composite

            van_bsf = _best_so_far([t.score for t in van.trials]).tolist()
            ws_bsf = _best_so_far([t.score for t in ws.trials]).tolist()
            rows.append(
                {
                    **asdict(cell),
                    "seed": seed,
                    "vanilla_best_so_far": van_bsf,
                    "warm_start_best_so_far": ws_bsf,
                    "contextual_one_shot": ws.trials[0].score,
                    "default": default_score,
                }
            )
            print(
                f"  N={cell.num_targets:2d} c={cell.clutter_rate:4.1f} "
                f"m={cell.maneuver_intensity:4.2f} s={seed}  "
                f"van@8={van_bsf[-1]:6.2f}  ws@1={ws_bsf[0]:6.2f}  "
                f"ws@4={ws_bsf[min(3,len(ws_bsf)-1)]:6.2f}  "
                f"ws@8={ws_bsf[-1]:6.2f}"
            )

    # 3) Aggregate per-trial-index means.
    T = args.trials
    van_curve = np.zeros(T)
    ws_curve = np.zeros(T)
    for k in range(T):
        van_curve[k] = np.mean([r["vanilla_best_so_far"][k] for r in rows])
        ws_curve[k] = np.mean([r["warm_start_best_so_far"][k] for r in rows])

    van8 = van_curve[-1]
    gate_k = None
    for k in range(T):
        if ws_curve[k] <= van8 + 1e-9:
            gate_k = k + 1
            break

    print("\n=== best-so-far (mean across held-out) ===")
    print("  trial | vanilla |  warm-start")
    for k in range(T):
        print(f"   {k+1:3d}  | {van_curve[k]:7.2f} | {ws_curve[k]:7.2f}")

    print("\n=== GATE ===")
    print(f"  vanilla BO @ {T} trials:  {van8:.2f}")
    if gate_k is None:
        print(f"  warm-start never reaches vanilla-BO-{T} within {T} trials")
        verdict = "FAIL"
    else:
        print(f"  warm-start reaches vanilla-BO-{T} quality at trial {gate_k}")
        verdict = "PASS" if gate_k <= 4 else "FAIL"
    print(f"  VERDICT: {verdict}  (PASS iff K <= 4)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "config": {
                    "trials": args.trials,
                    "pool_trials": args.pool_trials,
                    "pool_rollouts": pool_rollouts,
                    "n_bootstrap": args.n_bootstrap,
                    "train_seeds": list(args.train_seeds),
                    "eval_seeds": list(args.eval_seeds),
                },
                "vanilla_curve": van_curve.tolist(),
                "warm_start_curve": ws_curve.tolist(),
                "gate_k": gate_k,
                "verdict": verdict,
                "rows": rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n  saved -> {args.out}")


if __name__ == "__main__":
    main()
