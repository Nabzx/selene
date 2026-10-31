# Selene Simulation (`apps/sim`) Reproducible Run Reports (`run-01` to `run-06`)

## Overview

This directory contains the reproducible execution reports, training telemetry analyses, world-model uncertainty ablations, Monte Carlo safety verification results, and hardware throughput benchmarks for the **Selene** cooperative lunar crater hopper simulation (`apps/sim/selene_sim`).

All six runs execute deterministically (`seed = 42`) on the `200 x 200` Daedalus lunar far-side terrain grid (`data/terrain/daedalus_200x200.npz`, `100.0 m/cell` resolution, `20.0 km x 20.0 km` extent centred at `5.9 deg S, 179.4 deg E` with `12` catalogued craters `daedalus_a`..`daedalus_l`).

---

## Index of Reproducible Run Reports

| Run ID | Report Document | Stage / Issue Scope | Generated Data Artefacts | Wall-Clock Time | Headline Result |
| --- | --- | --- | --- | --- | --- |
| **Run 01** | [`run-01-mappo-quick.md`](./run-01-mappo-quick.md) | Stage 2 & 3 Quick CTDE MAPPO (`--quick`, `2` hoppers, `6` craters, `40` updates) | `data/training_log_quick.csv` | `7.04 s` (`51,200` steps) | Reward `180.76 -> 269.24` (`+88.48`), crashes `0.0155 -> 0.0000`, crater overlap `2.03 -> 0.02` (`-98.8%`) |
| **Run 02** | [`run-02-mappo-full-daedalus.md`](./run-02-mappo-full-daedalus.md) | Stage 3 Full Daedalus CTDE MAPPO (`4` hoppers, `12` craters, `80` updates) | `data/training_log.csv`, `data/episodes.json` | `44.32 s` (`307,200` steps) | Reward `228.32 -> 492.52` (`+264.20`), crashes `0.0469 -> 0.0000`, crater overlap `4.46 -> 0.03` (`-99.3%`), `10.0` craters/ep |
| **Run 03** | [`run-03-world-model-uncertainty.md`](./run-03-world-model-uncertainty.md) | Stage 4 Shared 5-MLP Ensemble World Model & Epistemic Uncertainty Map (`8,000` transitions, `25` epochs) | `data/world_model_map.json` | `4.50 s` (`25` epochs) | Held-out normalised val MSE `0.9957 -> 0.2771` (`-72.17%`), unnormalised MSE `0.0075`, OOD uncertainty `1.0591` vs ID `0.2873` (`3.69x`) |
| **Run 04** | [`run-04-mbpo-imagination-rollouts.md`](./run-04-mbpo-imagination-rollouts.md) | Issue #13 Short-Horizon MBPO Imagination Rollouts (`imagine_rollout`, $k \in \{1,2,3,5\}$, $\tau \in \{0.15,0.25,0.50,\infty\}$) | Included in `run-04` report tables | `0.41 s` (`16`-point grid) | Default $(k=2, \tau=0.50)$ retains `89.8%` of branches while reducing multi-step MSE by `20.48%` (`0.0204 -> 0.0162`) |
| **Run 05** | [`run-05-verifier-monte-carlo.md`](./run-05-verifier-monte-carlo.md) | Stage 4 Monte Carlo Safety Verifier (`simulate_plan`, $n=1,000$ rollouts, Plan A vs Plan B & $n \le 5,000$ scaling) | `data/verification_cloud.json` | `10.41 ms` (`4H x 5S`), `26.42 ms` (`5H x 10S`) | Rejects unsafe Plan A (`0.0%` pass, `R3` & `R5` failures); certifies Argon Plan B (`100.0%` pass, `1,000 / 1,000` safe runs, `378x` faster than `10.0 s` gate) |
| **Run 06** | [`run-06-benchmarks-and-test-suite.md`](./run-06-benchmarks-and-test-suite.md) | Stage 1 `VectorFleetEnv` CPU Throughput, CPU vs MPS MLP Latency, Daedalus NPZ Stats & 41-Test Suite | `data/terrain/daedalus_200x200.npz` | `41.76 s` (`41` tests) | `89,017 env steps/s` (`--quick`) & `82,479 env steps/s` (`full`), CPU `3.7x-6.2x` faster than MPS on rollout inference, `41 / 41` pytest pass |

---

## Stage 1 to Stage 4 Verification Gate Summary

Every stage verification gate defined in the Selene technical specification passes with substantial margin:

| Stage Gate | Verification Requirement | Measured Result | Target Threshold | Status | Reference Report |
| --- | --- | --- | --- | --- | --- |
| **Stage 1: Environment Stability** | `1,000` random masked steps without `NaN` or invariant breach | `1,000 / 1,000` steps clean (`0 NaNs`) | `0 NaNs`, all invariants hold | **PASS** | [`run-06`](./run-06-benchmarks-and-test-suite.md) |
| **Stage 1: Vectorised Throughput** | `VectorFleetEnv` throughput on CPU | `65,205.6 env steps/s` (`E=128, H=4`) / `82,479.3 env steps/s` (`E=256, H=4`) | `> 10,000 env steps/s` | **PASS (`8.2x`)** | [`run-06`](./run-06-benchmarks-and-test-suite.md) |
| **Stage 2: Quick MAPPO Convergence** | `--quick` (`2` hoppers, `6` craters, `40` updates) reward gain & crash drop | Reward `+88.48` (`180.76 -> 269.24`), crashes `0.0155 -> 0.0000` in `7.04 s` | Reward delta `>= +15.0`, crashes `<= 0.05`, `< 120 s` | **PASS** | [`run-01`](./run-01-mappo-quick.md) |
| **Stage 3: Full Fleet Cooperation** | `4` hoppers, `12` craters, `80` updates crater splitting & relay coordination | Reward `+264.20` (`228.32 -> 492.52`), overlap `4.46 -> 0.03` (`-99.3%`), relay conflicts `2.24 -> 0.64` (`-71.5%`) | Overlap & relay conflicts decrease, `< 20 min` (`44.32 s`) | **PASS** | [`run-02`](./run-02-mappo-full-daedalus.md) |
| **Stage 4: World Model Calibration** | 5-MLP ensemble held-out MSE & epistemic OOD vs ID separation | Normalised MSE `-72.17%` (`0.9957 -> 0.2771`), unnormalised MSE `0.0075`, OOD/ID uncertainty ratio `3.69x` (`1.0591` vs `0.2873`) | MSE drops `>= 50%` (`< 0.05` unnorm), OOD `> 1.5x` ID | **PASS** | [`run-03`](./run-03-world-model-uncertainty.md) |
| **Stage 4: MBPO Rollout Gating** | Short-horizon `imagine_rollout` uncertainty truncation | At $(k=2, \tau=0.50)$, `89.8%` branches retained, multi-step MSE reduced by `20.48%` (`-41.33%` at $k=5, \tau=0.25$) | Truncates high-uncertainty OOD branches & reduces compounding MSE | **PASS** | [`run-04`](./run-04-mbpo-imagination-rollouts.md) |
| **Stage 4: Monte Carlo Verifier** | `1,000` stochastic physics rollouts (`verify.py`) under `10.0 s` | Rejects Plan A (`0.0%` pass) in `35.44 ms`; certifies Plan B (`100.0%` pass) in `12.14 ms` (`26.42 ms` for `5H x 10S`) | `pass_rate >= 0.99` on safe plan, `< 10.0 s` (`378x` faster) | **PASS** | [`run-05`](./run-05-verifier-monte-carlo.md) |
| **Full Test Suite** | All unit, schema, orchestrator contract, and performance tests | `41 passed in 41.76s` across 10 test modules | `100%` pass (`41 / 41`) | **PASS** | [`run-06`](./run-06-benchmarks-and-test-suite.md) |

---

## Quick-Reference Commands and Generated Data Files

Run all commands from the repository root with `PYTHONPATH=apps/sim`:

| Task / Run | Command | Output File(s) |
| --- | --- | --- |
| **Rebuild Daedalus Terrain NPZ** | `PYTHONPATH=apps/sim python scripts/build_terrain_crop.py` | `data/terrain/daedalus_200x200.npz` (`485 KB`) |
| **Run 01: Quick MAPPO (`40` updates)** | `PYTHONPATH=apps/sim python -m selene_sim.mappo --quick --updates 40 --csv data/training_log_quick.csv --checkpoint-dir "" --export-episodes ""` | `data/training_log_quick.csv` |
| **Run 02: Full MAPPO (`80` updates)** | `PYTHONPATH=apps/sim python -m selene_sim.mappo --updates 80 --csv data/training_log.csv --checkpoint-dir "" --export-episodes data/episodes.json` | `data/training_log.csv`, `data/episodes.json` |
| **Format Training Summary Table** | `PYTHONPATH=apps/sim python apps/sim/scripts/training_summary.py data/training_log.csv` | Terminal table (`Stage 2/3/4: PASS`) |
| **Generate All Static Demo Artefacts** | `PYTHONPATH=apps/sim python scripts/generate_demo_bundle.py --output-dir data` | `data/episodes.json`, `data/training_log.csv`, `data/world_model_map.json`, `data/verification_cloud.json` |
| **Run Full 41-Test Suite** | `PYTHONPATH=apps/sim pytest apps/sim/tests -v` | `41 passed` |
