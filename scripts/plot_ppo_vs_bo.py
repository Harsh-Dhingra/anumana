"""Plot the PPO-vs-contextual-BO sample-efficiency comparison.

Reads the JSON from run_ppo_vs_bo.py and produces:
1. A sample-efficiency scatter: held-out mean score vs training rollouts
   (log x-axis), with vanilla-BO and default reference lines.
2. A per-cell bar chart of contextual BO vs PPO one-shot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/ppo_vs_bo"))
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    payload = json.loads(args.json.read_text())
    cfg = payload["config"]
    rows = payload["rows"]

    ctx = np.array([r["contextual_bo_one_shot"] for r in rows])
    ppo = np.array([r["ppo_one_shot"] for r in rows])
    bo = np.array([r["vanilla_bo_full_budget"] for r in rows])
    default = np.array([r["default_params"] for r in rows])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) sample-efficiency scatter ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ctx_rollouts = cfg["bo_pool_rollouts"]
    ppo_rollouts = cfg["ppo_steps"]

    ax.errorbar(ctx_rollouts, ctx.mean(), yerr=ctx.std(), fmt="o",
                ms=10, capsize=4, label="Contextual BO (one-shot)", color="C0")
    ax.errorbar(ppo_rollouts, ppo.mean(), yerr=ppo.std(), fmt="s",
                ms=10, capsize=4, label="PPO (one-shot)", color="C3")
    ax.axhline(bo.mean(), ls="--", color="C2",
               label=f"Vanilla BO full budget ({bo.mean():.1f})")
    ax.axhline(default.mean(), ls=":", color="gray",
               label=f"Default params ({default.mean():.1f})")
    ax.set_xscale("log")
    ax.set_xlabel("training tracker rollouts (log scale)")
    ax.set_ylabel("held-out composite score (lower = better)")
    ax.set_title("Sample efficiency: Contextual BO vs PPO\n"
                 "(one-shot scene-adaptive tracker tuning)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p1 = args.out_dir / "sample_efficiency.png"
    fig.savefig(p1, dpi=140)
    print(f"  sample-efficiency -> {p1}")

    # --- 2) per-cell bars ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [
        f"N{r['num_targets']} c{r['clutter_rate']:g} m{r['maneuver_intensity']:g} s{r['seed']}"
        for r in rows
    ]
    x = np.arange(len(rows))
    w = 0.2
    ax.bar(x - 1.5 * w, ctx, w, label="Contextual BO one-shot", color="C0")
    ax.bar(x - 0.5 * w, ppo, w, label="PPO one-shot", color="C3")
    ax.bar(x + 0.5 * w, bo, w, label="Vanilla BO full budget", color="C2")
    ax.bar(x + 1.5 * w, default, w, label="Default", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("composite score (lower = better)")
    ax.set_title("Per-cell one-shot comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p2 = args.out_dir / "per_cell.png"
    fig.savefig(p2, dpi=140)
    print(f"  per-cell -> {p2}")

    # --- text summary ---
    print("\n=== summary ===")
    print(f"  contextual BO one-shot: {ctx.mean():7.2f} +/- {ctx.std():5.2f} "
          f"({ctx_rollouts} training rollouts)")
    print(f"  PPO one-shot:           {ppo.mean():7.2f} +/- {ppo.std():5.2f} "
          f"({ppo_rollouts} training rollouts)")
    print(f"  vanilla BO full budget: {bo.mean():7.2f} +/- {bo.std():5.2f}")
    print(f"  default params:         {default.mean():7.2f} +/- {default.std():5.2f}")
    ratio = ppo_rollouts / max(ctx_rollouts, 1)
    print(f"  PPO used {ratio:.1f}x more training rollouts than contextual BO")


if __name__ == "__main__":
    main()
