# Design: anumana

This is the working technical design document. The eventual workshop paper
grows out of this file.

## Problem

Multi-target trackers (Kalman + JPDA/MHT/IMM) have parameters — gate sizes,
process noise scaling, hypothesis pruning thresholds, IMM model weights — whose
optimal values depend strongly on scene characteristics. A configuration tuned
for "3 fighter jets at 30 km" actively underperforms on "100 small UAS at 3 km."

In counter-UAS scenarios specifically, classical trackers struggle because:

- Target density is 10–100× higher than design assumptions.
- Individual SNR is low (small radar cross-section).
- Clutter rate is high (low-altitude operation, ground returns, birds).
- Motion is correlated across the swarm.
- Targets sit inside each other's association gates.

Manual retuning per scenario is operationally infeasible. We want the tracker
to adapt its own parameters online as the scene evolves.

## Method

**Online autotuning via contextual Bayesian optimization.**

On a sliding window of length $W$:

1. **Extract a scene-context vector** $c_t \in \mathbb{R}^k$ from recent
   measurements and tracks. Initial features:
   - estimated target density (tracks per unit volume)
   - maneuver-intensity proxy (mean absolute innovation residual)
   - clutter-rate proxy (measurements per detection vs. confirmed tracks)
   - SNR proxy (measurement amplitude statistics, when available)
2. **Query a contextual Gaussian Process** $f(\theta, c)$ over tracker
   parameters $\theta$ and context $c$, with acquisition function (UCB by
   default) selecting the next $\theta_{t+1}$.
3. **Apply $\theta_{t+1}$** to the tracker for the next window.
4. **Score the window** with a composite tracking-quality metric
   $r_t = -\alpha \cdot \text{OSPA} - \beta \cdot \text{ID switches} - \gamma \cdot \text{fragmentation}$.
5. **Update the GP posterior** with $(\theta_t, c_t, r_t)$.

Over many scenarios the GP learns a transferable mapping
$c \mapsto \theta^*(c)$.

## Parameters tuned (v1)

| Parameter | Type | Typical range |
|---|---|---|
| `gate_size` | continuous | 1.0 – 20.0 (std. deviations) |
| `process_noise_scale` | continuous, log | 0.1 – 10.0 |
| `pruning_threshold` | continuous, log | $10^{-6}$ – $10^{-1}$ |
| `imm_model_weights` | simplex over $M$ | $M = 3$ models (CV, CT, CA) |

Out of scope for v1: track init/delete thresholds, frame rate, sensor management.

## Baselines

1. **Fixed default parameters** (Stone Soup library defaults).
2. **Expert-tuned per scenario class** (manually swept grid, best of grid).
3. **Online BO without context** (vanilla GP-UCB).
4. **RL (PPO)** with the same observation/action space.

The honest comparison is against (2). (1) and (3) are sanity checks.

## Evaluation

Synthetic scenarios via Stone Soup. Scenario grid axes:

- target count: {5, 20, 50, 100, 200}
- maneuver intensity: {low, medium, high}
- clutter rate: {low, medium, high}
- SNR / detection probability: {0.95, 0.85, 0.7}

Metrics:

- OSPA, GOSPA (Stone Soup built-in)
- Identity switches (custom)
- Track fragmentation rate (custom)
- Time-to-confirmed-track for new threats (counter-UAS-specific)
- Track maintenance under saturation (custom)

## Risks (open)

1. **Reward hacking.** Composite reward shape needs ablation.
2. **Weak baseline.** Beating (1) is meaningless; beating (2) is the real bar.
3. **Distribution shift.** Contextual BO must generalise across scenario types.
4. **Compute envelope.** Each tracker run is CPU-bound; sweeping the grid
   takes hours per BO experiment.
5. **Prior art.** ~30% chance someone published this exact cut in FUSION
   2023–2025; week-1 lit review confirms or kills the framing.

## Non-goals

- Real radar data. Synthetic via Stone Soup only.
- Deployment as a defense system. The library is the deliverable; deployment
  is downstream and requires institutional partners.
- Tracker replacement. We tune existing Stone Soup trackers, not replace them.
