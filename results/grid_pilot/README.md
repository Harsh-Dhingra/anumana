# Phase 1.1 grid pilot — results

Archived results from the first multi-scenario validation sweep.

## Config

- 8 cells: `{N targets ∈ (5, 15)} × {clutter ∈ (1.0, 5.0)} × {maneuver ∈ (0.10, 1.00)}`
- 2 seeds per cell: `[0, 1]`
- 2 optimizers: `random_search`, `bayes_opt`
- 12 trials per optimizer per (cell, seed) pair
- 20 timesteps per scenario
- Detection probability: 0.9
- Total: **384 tracker invocations, 32 (cell, seed, optimizer) triples**

## Result

- **BO wins 14/16 (cell, seed) pairs**, 1 tie, 1 RS win.
- Mean improvement (BO over RS): **+13.4%**.
- Median improvement: +6.4%.
- Max win: 65% (N=5, clutter=5.0, maneuver=0.10, seed=0; BO 21.05 vs RS 60.00).

## Files

- `pilot.csv` — full results table (one row per cell × seed × optimizer).
- `heatmap.png` — mean improvement % across (N targets × clutter rate).

## Reproduce

From the repo root:

```bash
python scripts/run_grid.py --kind pilot --num-trials 12 \
    --duration-steps 20 --seeds 0 1 \
    --csv outputs/grid/pilot.csv
python scripts/analyse_grid.py --csv outputs/grid/pilot.csv --plot \
    --heatmap-out outputs/grid/pilot_heatmap.png
```

Wall-clock: ~2.2 hours on a single M-series Mac (no parallelism).

Discussion in [docs/research_log.md](../../docs/research_log.md) under
the 2026-05-15 entry.
