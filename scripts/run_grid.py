"""Run a multi-scenario grid sweep comparing RandomSearch and BayesOpt.

Default = small pilot grid that finishes in roughly 10-20 minutes.
Override via the CLI, e.g.:

    python scripts/run_grid.py num_trials=20 grid.kind=full \\
        grid.duration_steps=40

Outputs results to `outputs/grid/results.csv` (path overridable).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from anumana.experiments import GridCell, run_grid
from anumana.experiments.grid import summarise


def _pilot_grid(duration_steps: int) -> list[GridCell]:
    return [
        GridCell(num_targets=n, duration_steps=duration_steps,
                 clutter_rate=c, maneuver_intensity=m,
                 detection_probability=0.9)
        for n in (5, 15)
        for c in (1.0, 5.0)
        for m in (0.1, 1.0)
    ]


def _full_grid(duration_steps: int) -> list[GridCell]:
    return [
        GridCell(num_targets=n, duration_steps=duration_steps,
                 clutter_rate=c, maneuver_intensity=m,
                 detection_probability=0.9)
        for n in (5, 10, 20, 50)
        for c in (1.0, 3.0, 8.0)
        for m in (0.1, 0.5, 2.0)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--num-trials", type=int, default=10)
    ap.add_argument("--duration-steps", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--csv", type=Path, default=Path("outputs/grid/results.csv"))
    args = ap.parse_args()

    cells = _pilot_grid(args.duration_steps) if args.kind == "pilot" else _full_grid(args.duration_steps)

    print(f"=== grid sweep ({args.kind}) ===")
    print(f"  cells:      {len(cells)}")
    print(f"  seeds:      {args.seeds}")
    print(f"  trials:     {args.num_trials}")
    print(f"  duration:   {args.duration_steps} steps")
    print(f"  csv out:    {args.csv}")
    print(f"  total runs: {len(cells) * len(args.seeds) * 2 * args.num_trials} tracker invocations")
    print()

    t0 = time.time()
    results = run_grid(
        cells=cells,
        seeds=args.seeds,
        num_trials=args.num_trials,
        csv_path=args.csv,
    )
    dt = time.time() - t0

    print()
    print(f"=== summary ===")
    print(f"  total wall-clock: {dt:.1f}s ({dt/60:.1f} min)")
    summary = summarise(results)
    for k, v in summary.items():
        print(f"  {k:32s}: {v}")


if __name__ == "__main__":
    main()
