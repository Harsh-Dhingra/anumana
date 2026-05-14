"""Analyse a grid CSV produced by `scripts/run_grid.py`.

Reports per-cell BO-vs-RS win/loss/tie, mean improvement, and optionally
writes a matplotlib heatmap to `outputs/grid/heatmap.png`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def pivot_best_scores(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, seed), columns = optimizer best_score."""
    keys = [
        "num_targets",
        "duration_steps",
        "clutter_rate",
        "maneuver_intensity",
        "detection_probability",
        "seed",
    ]
    return (
        df.pivot_table(
            index=keys,
            columns="optimizer",
            values="best_score",
            aggfunc="first",
        )
        .reset_index()
    )


def summarise(pivot: pd.DataFrame) -> dict:
    rs = pivot["random_search"]
    bo = pivot["bayes_opt"]
    improvement_pct = 100.0 * (rs - bo) / np.maximum(rs, 1e-9)
    return {
        "pairs": len(pivot),
        "bo_wins": int(((bo + 1e-9) < rs).sum()),
        "rs_wins": int(((rs + 1e-9) < bo).sum()),
        "ties": int((rs == bo).sum()),
        "mean_improvement_pct": float(improvement_pct.mean()),
        "median_improvement_pct": float(improvement_pct.median()),
        "p25_improvement_pct": float(improvement_pct.quantile(0.25)),
        "p75_improvement_pct": float(improvement_pct.quantile(0.75)),
        "min_improvement_pct": float(improvement_pct.min()),
        "max_improvement_pct": float(improvement_pct.max()),
    }


def per_cell_table(pivot: pd.DataFrame) -> pd.DataFrame:
    cell_keys = ["num_targets", "clutter_rate", "maneuver_intensity"]
    pivot = pivot.copy()
    pivot["improvement_pct"] = (
        100.0
        * (pivot["random_search"] - pivot["bayes_opt"])
        / np.maximum(pivot["random_search"], 1e-9)
    )
    return (
        pivot.groupby(cell_keys)[
            ["random_search", "bayes_opt", "improvement_pct"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )


def convergence_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Plot mean best-so-far convergence curve across all (cell, seed) pairs."""
    import ast

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    by_opt: dict[str, list[np.ndarray]] = {}
    # `history` is not in the CSV (we only saved best_score), so reconstruct
    # the per-trial best-so-far from the trial CSV if present.
    if "history" not in df.columns:
        # Use best_iteration + best_score to approximate the curve.
        for opt_name in df["optimizer"].unique():
            sub = df[df["optimizer"] == opt_name]
            ax.scatter(sub["best_iteration"], sub["best_score"],
                       label=f"{opt_name} (best iter)", alpha=0.5)
        ax.set_xlabel("iteration where best was found")
        ax.set_ylabel("best score")
        ax.set_title("Best score vs iteration (per cell-seed pair)")
        ax.legend()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        print(f"  scatter -> {out_path}")
        return

    for opt_name in sorted(df["optimizer"].unique()):
        sub = df[df["optimizer"] == opt_name]
        histories = [np.array(ast.literal_eval(h)) for h in sub["history"]]
        max_len = max(len(h) for h in histories)
        cumulative = np.full((len(histories), max_len), np.nan)
        for i, h in enumerate(histories):
            best_so_far = np.minimum.accumulate(h)
            cumulative[i, :len(best_so_far)] = best_so_far
        mean = np.nanmean(cumulative, axis=0)
        std = np.nanstd(cumulative, axis=0)
        x = np.arange(1, max_len + 1)
        ax.plot(x, mean, label=opt_name, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("trial")
    ax.set_ylabel("best score so far (lower is better)")
    ax.set_title("Mean best-so-far across cells")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"  convergence -> {out_path}")


def heatmap(pivot: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = pivot.copy()
    pivot["improvement_pct"] = (
        100.0
        * (pivot["random_search"] - pivot["bayes_opt"])
        / np.maximum(pivot["random_search"], 1e-9)
    )
    grid = (
        pivot.groupby(["num_targets", "clutter_rate"])["improvement_pct"]
        .mean()
        .unstack(level="clutter_rate")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid.values, cmap="RdBu_r", aspect="auto",
                   vmin=-abs(grid.values).max(), vmax=abs(grid.values).max())
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("clutter rate")
    ax.set_ylabel("num targets")
    ax.set_title("Mean BO improvement vs RS (%)\n(positive = BO better)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:+.1f}",
                    ha="center", va="center",
                    color="black" if abs(grid.values[i, j]) < 5 else "white")
    fig.colorbar(im, ax=ax, label="improvement (%)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"  heatmap -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("outputs/grid/pilot.csv"))
    ap.add_argument("--plot", action="store_true", help="write heatmap PNG")
    ap.add_argument("--heatmap-out", type=Path,
                    default=Path("outputs/grid/heatmap.png"))
    args = ap.parse_args()

    df = load(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")
    print(f"  optimizers: {sorted(df['optimizer'].unique())}")
    print(f"  cells:      {len(df.drop_duplicates(subset=['num_targets','clutter_rate','maneuver_intensity']))}")
    print(f"  seeds:      {sorted(df['seed'].unique())}")

    pivot = pivot_best_scores(df)
    summary = summarise(pivot)
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")

    table = per_cell_table(pivot)
    print("\n=== per-cell ===")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:6.2f}".format):
        print(table.to_string(index=False))

    if args.plot:
        heatmap(pivot, args.heatmap_out)


if __name__ == "__main__":
    main()
