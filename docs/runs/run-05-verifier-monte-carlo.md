# Run 05: Stage 4 Monte Carlo Safety Verifier (`verify.py`, 1,000 Rollouts & Scaling Benchmark)

## Overview

This report documents the reproducible Stage 4 Monte Carlo verification run (`simulate_plan(fleet_plan, n=1000, seed=42)`) over the 12-crater Daedalus far-side benchmark terrain (`data/terrain/daedalus_200x200.npz`, `200 x 200` grid at `100.0 m/cell`).

The vectorised Monte Carlo safety verifier in `apps/sim/selene_sim/verify.py` evaluates candidate multi-hopper mission plans from the Argon orchestrator across $n = 1,000$ independent stochastic physics rollouts simultaneously using a single `(n_runs, n_hoppers)` NumPy batch per mission stage. Each rollout perturbs propellant specific impulse ($I_{\text{sp}} \sim \mathcal{N}(220\text{ s}, (2.2\text{ s})^2)$), ballistic landing dispersion ($\Delta x, \Delta y \sim \mathcal{N}(0, (5.0\text{ m})^2)$), and initial battery state of charge ($\pm 1.5\%$ uniform jitter) while enforcing five hard safety rules (`R1_FUEL_RESERVE` through `R5_SINGLE_CHANNEL_RELAY`) at a $\ge 99.0\%$ fleet pass threshold (`overall_pass_rate >= 0.99`).

Two candidate plans for a 4-hopper (`h0`..`h3`), 5-stage cooperative survey of six Daedalus craters (`daedalus_a`..`daedalus_f`) are evaluated back-to-back:
1. **Candidate Plan A (Unsafe Naive Plan)**: Hopper `h2` targets a steep inner rim wall on `daedalus_c` (`(-675.8 m, 1250.0 m)`, local Horn slope `29.94 deg > 15.0 deg` limit), and Hoppers `h0` and `h3` transmit simultaneously during the open Stage 2 orbiter relay window. The verifier rejects Plan A in `35.44 ms` (`0.0%` pass rate, `1,000 / 1,000` failed rollouts) and returns per-rule failure attribution by hopper and stage.
2. **Candidate Plan B (Argon-Revised Cooperative Plan)**: Argon relocates Hopper `h2` to the flat floor of `daedalus_c` (`(-1350.0 m, 1250.0 m)`, local floor slope `0.00 deg`) and staggers the Stage 2 relay downlink so `h0` transmits while `h3` waits for a subsequent pass. The verifier certifies Plan B in `12.14 ms` (`100.0%` pass rate, `1,000 / 1,000` safe rollouts) and exports `data/verification_cloud.json` (`169,957 bytes`, 50 sampled 3D parabolic hop trajectories for the Three.js lunar globe visualiser).

---

## Reproducible Command Lines

Run from the repository root (`/Users/nabilshah/.gemini/antigravity/scratch/worktrees/run-verifier-and-benchmarks`):

```bash
# 1. Execute Plan A (unsafe) vs Plan B (Argon-revised safe) 1,000-rollout Monte Carlo verification
#    and export data/verification_cloud.json
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python -c '
from pathlib import Path
from selene_sim.sim import SimConfig, TerrainGrid
from selene_sim.verify import export_verification_cloud, simulate_plan

terrain = TerrainGrid.from_npz("data/terrain/daedalus_200x200.npz")
cfg = SimConfig(seed=42)
craters = {c.id: c for c in terrain.craters}

initial_hoppers = [
    {"id": "h0", "x": craters["daedalus_a"].x_m - 1200.0, "y": craters["daedalus_a"].y_m, "fuel_kg": 15.0, "battery_pct": 100.0},
    {"id": "h1", "x": craters["daedalus_b"].x_m, "y": craters["daedalus_b"].y_m - 1200.0, "fuel_kg": 15.0, "battery_pct": 100.0},
    {"id": "h2", "x": craters["daedalus_c"].x_m - 1200.0, "y": craters["daedalus_c"].y_m, "fuel_kg": 15.0, "battery_pct": 100.0},
    {"id": "h3", "x": craters["daedalus_d"].x_m, "y": craters["daedalus_d"].y_m - 1200.0, "fuel_kg": 15.0, "battery_pct": 100.0},
]

plan_b = {
    "plan_id": "plan_b_revised_daedalus_4hopper",
    "initial_hoppers": initial_hoppers,
    "stages": [
        {
            "stage": 0, "relay_window_open": False,
            "actions": {
                "h0": {"type": "hop", "target_x": craters["daedalus_a"].x_m, "target_y": craters["daedalus_a"].y_m, "target_crater_id": "daedalus_a"},
                "h1": {"type": "hop", "target_x": craters["daedalus_b"].x_m, "target_y": craters["daedalus_b"].y_m, "target_crater_id": "daedalus_b"},
                "h2": {"type": "hop", "target_x": craters["daedalus_c"].x_m, "target_y": craters["daedalus_c"].y_m, "target_crater_id": "daedalus_c"},
                "h3": {"type": "hop", "target_x": craters["daedalus_d"].x_m, "target_y": craters["daedalus_d"].y_m, "target_crater_id": "daedalus_d"},
            },
        },
        {
            "stage": 1, "relay_window_open": False,
            "actions": {
                "h0": {"type": "scan"}, "h1": {"type": "scan"},
                "h2": {"type": "scan"}, "h3": {"type": "scan"},
            },
        },
        {
            "stage": 2, "relay_window_open": True,
            "actions": {
                "h0": {"type": "send"}, "h1": {"type": "wait"},
                "h2": {"type": "wait"}, "h3": {"type": "wait"},
            },
        },
        {
            "stage": 3, "relay_window_open": False,
            "actions": {
                "h0": {"type": "hop", "target_x": craters["daedalus_f"].x_m, "target_y": craters["daedalus_f"].y_m, "target_crater_id": "daedalus_f"},
                "h1": {"type": "wait"},
                "h2": {"type": "wait"},
                "h3": {"type": "hop", "target_x": craters["daedalus_e"].x_m, "target_y": craters["daedalus_e"].y_m, "target_crater_id": "daedalus_e"},
            },
        },
        {
            "stage": 4, "relay_window_open": False,
            "actions": {
                "h0": {"type": "scan"}, "h1": {"type": "wait"},
                "h2": {"type": "wait"}, "h3": {"type": "scan"},
            },
        },
    ],
}

res = simulate_plan(plan_b, n=1000, seed=42, terrain=terrain, cfg=cfg)
export_verification_cloud(res, Path("data/verification_cloud.json"))
print("Pass rate:", res["pass_rate"], "| Passed:", res["pass"], "| Summary:", res["summary"])
'

# 2. Run the verifier unit & performance test suite
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python \
  -m pytest apps/sim/tests/test_verify.py -v
```

---

## Hardware, Runtime Environment, and Stochastic Perturbation Model

| Property | Value | Notes |
| --- | --- | --- |
| Platform | `macOS-26.2-arm64-arm-64bit` | Apple Silicon (`arm64`) |
| Python / NumPy | `Python 3.12.14` / `NumPy 2.5.4` | Vectorised `(n_runs, n_hoppers)` float32/float64 execution |
| Terrain Asset | `data/terrain/daedalus_200x200.npz` | `200 x 200` cells, `100.0 m/cell` (`20.0 km x 20.0 km`), 12 craters |
| Rollout Count ($n$) | `1,000` | Default verification batch size (`seed = 42`) |
| $I_{\text{sp}}$ Perturbation | $\mathcal{N}(220.0\text{ s}, (2.2\text{ s})^2)$ | $\pm 1.0\%$ $1\sigma$ Gaussian jitter clipped to $[154.0\text{ s}, 286.0\text{ s}]$ |
| Landing Error ($\Delta x, \Delta y$) | $\mathcal{N}(0.0\text{ m}, (5.0\text{ m})^2)$ | Independent 2D Gaussian dispersion per hop |
| Initial Battery Jitter | $\mathcal{U}(-1.5\%, +1.5\%)$ | Uniform offset clipped to $[20.0\%, 100.0\%]$ |
| Safety Certification Gate | `overall_pass_rate >= 0.99` | Requires $\ge 990 / 1,000$ rollouts with zero rule violations |

---

## Candidate Plan Comparison: Plan A (Unsafe Naive) vs Plan B (Argon-Revised Safe)

Both candidate plans deploy 4 hoppers (`h0`..`h3`) starting `1,200 m` outside craters `daedalus_a`, `daedalus_b`, `daedalus_c`, and `daedalus_d` with `15.0 kg` hydrazine propellant (`35.0 kg` dry mass) and `100.0%` initial battery capacity over 5 mission stages:

| Stage | Orbiter Relay Window | Candidate Plan A (Unsafe Naive) | Candidate Plan B (Argon-Revised Safe) |
| --- | --- | --- | --- |
| Stage 0 | Closed (`False`) | `h0->daedalus_a`, `h1->daedalus_b`, **`h2->daedalus_c` rim wall `(-675.8, 1250.0)`**, `h3->daedalus_d` | `h0->daedalus_a`, `h1->daedalus_b`, **`h2->daedalus_c` flat floor `(-1350.0, 1250.0)`**, `h3->daedalus_d` |
| Stage 1 | Closed (`False`) | `h0: scan`, `h1: scan`, `h2: scan`, `h3: scan` | `h0: scan`, `h1: scan`, `h2: scan`, `h3: scan` |
| Stage 2 | Open (`True`) | **`h0: send`, `h1: wait`, `h2: wait`, `h3: send` (simultaneous collision)** | **`h0: send`, `h1: wait`, `h2: wait`, `h3: wait` (conflict-free single channel)** |
| Stage 3 | Closed (`False`) | `h0->daedalus_f`, `h1: wait`, `h2: wait`, `h3->daedalus_e` | `h0->daedalus_f`, `h1: wait`, `h2: wait`, `h3->daedalus_e` |
| Stage 4 | Closed (`False`) | `h0: scan`, `h1: wait`, `h2: wait`, `h3: scan` | `h0: scan`, `h1: wait`, `h2: wait`, `h3: scan` |

### Fleet-Wide Summary Metrics ($n = 1,000, \text{seed} = 42$)

| Metric | Candidate Plan A (Unsafe) | Candidate Plan B (Argon-Revised) | Safety Requirement |
| --- | --- | --- | --- |
| Overall Pass Rate (`pass_rate`) | **`0.000` (`0 / 1,000`)** | **`1.000` (`1,000 / 1,000`)** | $\ge 0.990$ (`99.0%`) |
| Certification Status (`pass`) | **`False` (REJECTED)** | **`True` (CERTIFIED)** | `True` |
| Wall-Clock Execution Time | `35.44 ms` (cold) | `12.14 ms` (`10.41 ms` median) | $< 10,000.0\text{ ms}$ (`10.0 s`) |
| Fleet Rollout Crash Rate (`crash_rate`) | `1.000` (`100.0%`) | `0.000` (`0.0%`) | `0.000` |
| Per-Hopper Crash Rate (`hopper_crash_rate`) | `0.250` (`h2` crashes in `100%`) | `0.000` (`0.0%`) | `0.000` |
| Relay Conflict Rate (`relay_conflict_rate`) | `1.000` (`Stage 2: 100.0%`) | `0.000` (`0.0%`) | `0.000` |
| Proximity Violation Rate (`proximity_violation_rate`) | `0.000` (`0.0%`) | `0.000` (`0.0%`) | `0.000` |
| Minimum Final Fuel (`min_fuel_kg`) | `9.583 kg` | `9.627 kg` | $\ge 1.500\text{ kg}$ (`10%` of `15 kg`) |
| 5th Percentile Final Fuel (`p05_final_fuel_kg`) | `9.721 kg` | `9.720 kg` | $\ge 1.500\text{ kg}$ |
| Mean Final Fuel (`mean_final_fuel_kg`) | `11.130 kg` | `11.054 kg` | - |
| Minimum Battery SoC (`min_battery_pct`) | `88.64%` | `88.65%` | $\ge 20.00\%$ |
| 5th Percentile Final Battery (`p05_final_battery_pct`) | `88.65%` | `88.66%` | $\ge 20.00\%$ |
| Mean Final Battery (`mean_final_battery_pct`) | `91.61%` | `90.76%` | - |
| Maximum Landing Slope (`max_landing_slope_deg`) | **`32.93 deg`** (`daedalus_c` wall) | **`3.76 deg`** | $\le 15.00\text{ deg}$ |
| Minimum Inter-Hopper Separation (`min_separation_m`) | `1,234.03 m` | `1,474.66 m` | $\ge 100.00\text{ m}$ |
| Mean Successful Relay Downlinks | `1.00` | `1.00` | $\ge 1.00$ |

---

## Safety Rule Breakdown (`R1` to `R5`)

The verifier evaluates five explicit safety rules across all $1,000$ rollouts and attributes every violation to specific hoppers (`failures_by_hopper`) and mission stages (`failures_by_stage`) so Argon can repair the plan in a single revision turn:

| Rule ID | Safety Rule Statement | Plan A Pass Rate | Plan A Failed Runs & Attribution | Plan B Pass Rate | Plan B Failed Runs & Attribution |
| --- | --- | --- | --- | --- | --- |
| `R1_FUEL_RESERVE` | Each hopper must retain at least `10%` fuel reserve (`1.50 kg`) after final landing. | `100.0%` (`PASS`) | `0 / 1,000` (`h0:0, h1:0, h2:0, h3:0`) | `100.0%` (`PASS`) | `0 / 1,000` (`h0:0, h1:0, h2:0, h3:0`) |
| `R2_BATTERY_RESERVE` | Battery state of charge must remain at or above `20.0%` throughout all mission stages. | `100.0%` (`PASS`) | `0 / 1,000` (Stages `0..4`: `[0, 0, 0, 0, 0]`) | `100.0%` (`PASS`) | `0 / 1,000` (Stages `0..4`: `[0, 0, 0, 0, 0]`) |
| `R3_LANDING_SLOPE` | No hopper may land on a surface slope exceeding `15.0 deg` or suffer a ballistic crash. | **`0.0%` (`FAIL`)** | **`1,000 / 1,000`** (`h2: 1,000`, `h0/h1/h3: 0`; Stages `0..4`: `[1000, 1000, 1000, 1000, 1000]`) | `100.0%` (`PASS`) | `0 / 1,000` (`h0:0, h1:0, h2:0, h3:0`; Stages `0..4`: `[0, 0, 0, 0, 0]`) |
| `R4_FLEET_SEPARATION` | Pairwise horizontal distance between any two hoppers must remain $\ge 100.0\text{ m}$. | `100.0%` (`PASS`) | `0 / 1,000` (Stages `0..4`: `[0, 0, 0, 0, 0]`) | `100.0%` (`PASS`) | `0 / 1,000` (Stages `0..4`: `[0, 0, 0, 0, 0]`) |
| `R5_SINGLE_CHANNEL_RELAY` | At most one hopper may transmit during an open orbiter relay window (no packet collisions). | **`0.0%` (`FAIL`)** | **`1,000 / 1,000`** (Stages `0..4`: `[0, 0, 1000, 0, 0]`, `h0` & `h3` collide on Stage 2) | `100.0%` (`PASS`) | `0 / 1,000` (Stages `0..4`: `[0, 0, 0, 0, 0]`) |

---

## Per-Hopper Propellant, Battery, and Slope Distributions ($n = 1,000$)

### Candidate Plan A (Unsafe Naive Plan)

In Plan A, Hopper `h2` lands at `(-675.8 m, 1250.0 m)` on the steep inner rim wall of `daedalus_c` (`radius = 864.3 m`), where the 2D Gaussian landing dispersion ($\sigma = 5.0\text{ m}$) samples Horn slopes between `26.1 deg` and `32.93 deg` (mean `29.38 deg`), triggering a `steep_slope` crash on Stage 0 in all `1,000` rollouts. Because `h2` is immobilised after Stage 0, it does not execute its Stage 1 thermal scan (`mean_final_battery_pct = 97.42%`).

| Hopper | Hops Executed | Mean Final Fuel (`kg`) | `p05` Final Fuel (`kg`) | Min Final Fuel (`kg`) | Mean Final Battery (`%`) | `p05` Final Battery (`%`) | Min Final Battery (`%`) | Mean Slope (`deg`) | Max Slope (`deg`) | Crash Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `h0` | `2` (`daedalus_a` -> `daedalus_f`) | `9.794` | `9.704` | `9.583` | `88.67%` | `88.65%` | `88.64%` | `0.305` | `3.498` | `0.0%` |
| `h1` | `1` (`daedalus_b`) | `12.293` | `12.225` | `12.140` | `91.68%` | `91.65%` | `91.63%` | `0.315` | `3.581` | `0.0%` |
| `h2` | `1` (`daedalus_c` rim wall) | `12.638` | `12.578` | `12.508` | `97.42%` | `97.29%` | `97.29%` | **`29.384`** | **`32.932`** | **`100.0%`** |
| `h3` | `2` (`daedalus_d` -> `daedalus_e`) | `9.793` | `9.699` | `9.591` | `88.68%` | `88.65%` | `88.64%` | `0.330` | `4.123` | `0.0%` |

### Candidate Plan B (Argon-Revised Cooperative Plan)

In Plan B, Hopper `h2` lands safely on the flat floor of `daedalus_c` (`(-1350.0 m, 1250.0 m)`), experiencing a mean slope of `0.235 deg` (max `3.351 deg` across all `1,000` dispersed touchdowns) and completing its Stage 1 PSR scan (`mean_final_battery_pct = 92.40%`). Hoppers `h0` and `h3` each execute two ballistic hops (totalling `~5.20 kg` hydrazine burned per 2-hop survey), retaining `9.63 kg` worst-case fuel (`6.4x` above the `1.50 kg` reserve floor).

| Hopper | Hops Executed | Mean Final Fuel (`kg`) | `p05` Final Fuel (`kg`) | Min Final Fuel (`kg`) | Mean Final Battery (`%`) | `p05` Final Battery (`%`) | Min Final Battery (`%`) | Mean Slope (`deg`) | Max Slope (`deg`) | Crash Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `h0` | `2` (`daedalus_a` -> `daedalus_f`) | `9.798` | `9.706` | `9.630` | `88.68%` | `88.65%` | `88.65%` | `0.329` | `3.586` | `0.0%` |
| `h1` | `1` (`daedalus_b`) | `12.293` | `12.225` | `12.140` | `92.48%` | `92.45%` | `92.43%` | `0.315` | `3.581` | `0.0%` |
| `h2` | `1` (`daedalus_c` flat floor) | `12.334` | `12.268` | `12.186` | `92.40%` | `92.37%` | `92.36%` | `0.235` | `3.351` | `0.0%` |
| `h3` | `2` (`daedalus_d` -> `daedalus_e`) | `9.793` | `9.694` | `9.627` | `89.49%` | `89.47%` | `89.45%` | `0.324` | `3.760` | `0.0%` |

---

## Exported Trajectory Cloud Artefact (`data/verification_cloud.json`)

The certified Plan B verification run writes `data/verification_cloud.json` (`169,957 bytes`) for the Three.js lunar globe visualiser and the Argon dashboard. Each of the `50` sampled rollout runs contains `4` hopper trajectories across `5` stages, where every ballistic hop is discretised into `9` parabolic 3D arc points `(x_m, y_m, z_m)` over the local Daedalus terrain elevation plus stationary stage markers:

| Field | Value / Structure | Description |
| --- | --- | --- |
| `plan_id` | `"plan_b_revised_daedalus_4hopper"` | Identifier of the Argon-revised 4-hopper cooperative plan |
| `n_runs` | `1000` | Total Monte Carlo rollouts evaluated |
| `pass_rate` / `pass` | `1.0` / `true` | Fleet safety certification outcome (`>= 0.99` gate) |
| `summary` | `15` aggregate keys | Fleet-wide crash, relay conflict, proximity, fuel, battery, and slope statistics |
| `rule_breakdown` | `5` rule objects (`R1`..`R5`) | Per-rule pass rates and failure counts by hopper and stage |
| `hopper_stats` | `4` hopper objects (`h0`..`h3`) | Mean, `p05`, and min fuel/battery and landing slope distributions |
| `trajectory_cloud.num_sampled_runs` | `50` | Number of 3D rollout fans exported for rendering (`runs[0..49]`) |
| `File Size` | `169,957 bytes` (`166.0 KB`) | Compact JSON payload for static web fallback and live WebSocket streaming |

---

## Verifier Wall-Clock Scaling Benchmark ($n \in \{100, 500, 1000, 2000, 5000\}$)

To verify that vectorised NumPy rollout evaluation scales well within the `< 10.0 s` interactive orchestrator budget (Issue #8 and `tests/test_verify.py::test_verify_1000_rollouts_under_10_seconds`), `simulate_plan` was benchmarked across $n \in \{100, 500, 1000, 2000, 5000\}$ rollouts (7 trials per point after warm-up) on two mission profiles:
1. **Plan B (`4` Hoppers x `5` Stages = `20` hopper-stages per rollout)**
2. **Full Stress Benchmark (`5` Hoppers x `10` Stages = `50` hopper-stages per rollout)**

| Rollouts ($n$) | `4H x 5S` Median (`ms`) | `4H x 5S` IQR (`ms`) | `4H x 5S` Rollouts/s | `4H x 5S` Hopper-Stages/s | `5H x 10S` Median (`ms`) | `5H x 10S` IQR (`ms`) | `5H x 10S` Rollouts/s | `5H x 10S` Hopper-Stages/s | Margin vs `10.0 s` Gate (`5H x 10S`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `100` | `2.492` | `1.234` | `40,131.8` | `802,635.5` | `5.487` | `0.183` | `18,226.0` | `911,300.0` | `1,822x faster` |
| `500` | `5.967` | `0.704` | `83,788.9` | `1,675,778.7` | `14.378` | `0.706` | `34,775.2` | `1,738,762.5` | `695x faster` |
| **`1,000`** | **`10.414`** | **`0.306`** | **`96,027.3`** | **`1,920,545.3`** | **`26.420`** | **`0.869`** | **`37,849.8`** | **`1,892,490.7`** | **`378x faster`** |
| `2,000` | `18.336` | `0.572` | `109,077.8` | `2,181,555.4` | `45.910` | `0.294` | `43,563.3` | `2,178,164.8` | `218x faster` |
| `5,000` | `43.638` | `1.244` | `114,579.3` | `2,291,585.1` | `113.308` | `4.966` | `44,127.7` | `2,206,383.7` | `88x faster` |

### Key Scaling Observations

- At the standard verification batch size of $n = 1,000$ rollouts, the 4-hopper 5-stage cooperative plan completes in **`10.41 ms`** (`96,027 rollouts/s`, `1.92 million hopper-stages/s`), and the 5-hopper 10-stage stress plan completes in **`26.42 ms`** (`37,850 rollouts/s`), which is **`378x` faster** than the `10.0 s` Stage 4 gate.
- Even at $n = 5,000$ rollouts (`250,000` ballistic hopper-stage evaluations), vectorised NumPy execution finishes in **`113.31 ms`**, enabling sub-second interactive plan-verify-revise loops inside the Argon orchestrator.
