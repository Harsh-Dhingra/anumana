"""Analyse a results JSON produced by `scripts/run_contextual_bo.py`.

Tags each held-out cell as 'interior_interpolation' or 'extrapolation_*',
reports per-method means with bootstrap CIs, and prints a per-cell
comparison table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Defined here so the analysis is self-contained; mirrors
# scripts/run_contextual_bo.py::_held_out_cells(expanded=True).
CELL_KIND: dict[tuple, str] = {
    (7, 2.0, 0.3): "interior_interpolation",
    (9, 4.0, 0.3): "interior_interpolation",
    (10, 8.0, 0.5): "extrapolation_clutter",
    (8, 3.0, 1.0): "extrapolation_maneuver",
    # legacy small-eval cells
    (8, 3.0, 0.5): "interior_interpolation",
    (10, 6.0, 0.5): "extrapolation_clutter",
}


METHODS = [
    ("contextual_one_shot", "contextual one-shot"),
    ("vanilla_bo_best", "vanilla BO (full budget)"),
    ("random_search_best", "random search (full budget)"),
    ("default_params", "default parameters"),
]


def _bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean."""
    rng = np.random.default_rng(0)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def load(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    df = pd.DataFrame(rows)
    df["cell_kind"] = df.apply(
        lambda r: CELL_KIND.get(
            (int(r["num_targets"]), float(r["clutter_rate"]), float(r["maneuver_intensity"])),
            "unknown",
        ),
        axis=1,
    )
    return df


def summarise(df: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    out_rows = []
    iterable = [(None, df)] if group is None else df.groupby(group)
    for key, sub in iterable:
        for col, label in METHODS:
            arr = sub[col].to_numpy()
            ci_lo, ci_hi = _bootstrap_ci(arr)
            out_rows.append(
                {
                    "subset": key if key is not None else "all",
                    "method": label,
                    "n": len(arr),
                    "mean": float(np.mean(arr)) if len(arr) else float("nan"),
                    "std": float(np.std(arr)) if len(arr) else float("nan"),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                }
            )
    return pd.DataFrame(out_rows)


def pairwise_improvement(df: pd.DataFrame, baseline: str, group: str | None = None) -> pd.DataFrame:
    """Per-row contextual improvement (%) over `baseline` column."""
    df = df.copy()
    df["improvement_pct"] = (
        100.0
        * (df[baseline] - df["contextual_one_shot"])
        / np.maximum(df[baseline].abs(), 1e-9)
    )
    rows = []
    iterable = [(None, df)] if group is None else df.groupby(group)
    for key, sub in iterable:
        arr = sub["improvement_pct"].to_numpy()
        wins = int((arr > 1e-6).sum())
        losses = int((arr < -1e-6).sum())
        ties = len(arr) - wins - losses
        ci_lo, ci_hi = _bootstrap_ci(arr)
        rows.append(
            {
                "subset": key if key is not None else "all",
                "baseline": baseline,
                "n": len(arr),
                "ctx_wins": wins,
                "ctx_losses": losses,
                "ties": ties,
                "mean_improvement_pct": float(arr.mean()) if len(arr) else float("nan"),
                "ci_lo_pct": ci_lo,
                "ci_hi_pct": ci_hi,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    df = load(args.json)
    print(f"loaded {len(df)} rows from {args.json}")
    print(f"  cells: {df[['num_targets','clutter_rate','maneuver_intensity']].drop_duplicates().to_dict(orient='records')}")
    print(f"  seeds per cell: {df.groupby(['num_targets','clutter_rate','maneuver_intensity']).size().unique().tolist()}")

    print("\n=== per-method summary (all eval points) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:7.2f}".format):
        print(summarise(df).to_string(index=False))

    print("\n=== per-method by cell kind ===")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:7.2f}".format):
        print(summarise(df, group="cell_kind").to_string(index=False))

    print("\n=== contextual vs vanilla BO (full budget) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:7.2f}".format):
        print(pairwise_improvement(df, baseline="vanilla_bo_best",
                                   group="cell_kind").to_string(index=False))

    print("\n=== contextual vs default ===")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:7.2f}".format):
        print(pairwise_improvement(df, baseline="default_params",
                                   group="cell_kind").to_string(index=False))

    print("\n=== full per-cell table ===")
    keep = [
        "num_targets", "clutter_rate", "maneuver_intensity", "seed",
        "cell_kind",
        "contextual_one_shot", "vanilla_bo_best",
        "random_search_best", "default_params",
        "ctx_vs_default_pct", "ctx_vs_vanilla_pct",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 220,
                           "display.float_format", "{:7.2f}".format):
        print(df[keep].sort_values(["cell_kind", "num_targets", "seed"]).to_string(index=False))


if __name__ == "__main__":
    main()
