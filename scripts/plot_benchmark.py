"""Plot the benchmark: method means with bootstrap 95% CIs (the paper's
main figure) + a per-method-by-cell-kind breakdown.

Reads the JSON from run_benchmark.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHOD_LABELS = {
    "default": "Default params",
    "random_search": "Random search (K)",
    "vanilla_bo": "Vanilla BO (K)",
    "contextual_bo_oneshot": "Contextual BO (one-shot)",
    "warm_start_bo": "Warm-start BO (K)",
    "ppo_oneshot": "PPO (one-shot)",
}

# Mirrors run_benchmark held-out cells.
CELL_KIND = {
    (7, 2.0, 0.3): "interpolation",
    (9, 4.0, 0.3): "interpolation",
    (10, 8.0, 0.5): "extrap-clutter",
    (8, 3.0, 1.0): "extrap-maneuver",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark"))
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    payload = json.loads(args.json.read_text())
    summary = payload["summary"]
    rows = payload["rows"]
    cfg = payload["config"]
    methods = list(METHOD_LABELS)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) forest plot: mean + 95% CI per method ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ys = np.arange(len(methods))[::-1]
    for y, m in zip(ys, methods):
        s = summary[m]
        ax.errorbar(
            s["mean"], y,
            xerr=[[s["mean"] - s["ci_lo"]], [s["ci_hi"] - s["mean"]]],
            fmt="o", capsize=4, ms=7,
        )
        ax.text(s["mean"], y + 0.18, f"{s['mean']:.1f}",
                ha="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_xlabel("composite score (lower = better)")
    ax.set_title(
        f"Scene-adaptive tracker tuning benchmark\n"
        f"(n={cfg['n_eval_points']} held-out points, 95% bootstrap CI)"
    )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p1 = args.out_dir / "benchmark_forest.png"
    fig.savefig(p1, dpi=140)
    print(f"  forest -> {p1}")

    # --- 2) per cell-kind grouped bars ---
    kinds = ["interpolation", "extrap-clutter", "extrap-maneuver"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.13
    xk = np.arange(len(kinds))
    for i, m in enumerate(methods):
        vals = []
        for k in kinds:
            sel = [
                r[m] for r in rows
                if CELL_KIND.get(
                    (int(r["num_targets"]), float(r["clutter_rate"]),
                     float(r["maneuver_intensity"])), "?"
                ) == k
            ]
            vals.append(np.mean(sel) if sel else np.nan)
        ax.bar(xk + (i - 2.5) * width, vals, width, label=METHOD_LABELS[m])
    ax.set_xticks(xk)
    ax.set_xticklabels(kinds)
    ax.set_ylabel("mean composite (lower = better)")
    ax.set_title("By held-out cell kind")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p2 = args.out_dir / "benchmark_by_kind.png"
    fig.savefig(p2, dpi=140)
    print(f"  by-kind -> {p2}")

    # --- text table ---
    print("\n=== benchmark table ===")
    print(f"{'method':26s} {'mean':>7s} {'95% CI':>18s} {'std':>6s}")
    for m in methods:
        s = summary[m]
        print(f"{METHOD_LABELS[m]:26s} {s['mean']:7.2f} "
              f"[{s['ci_lo']:6.2f},{s['ci_hi']:6.2f}] {s['std']:6.2f}")


if __name__ == "__main__":
    main()
