# Run 06: Stage 1 Environment Throughput, CPU vs MPS Device Benchmark, Terrain Crop Statistics, and 41-Test Suite

## Overview

This report documents the reproducible Stage 1 and system-wide verification benchmarks for `selene_sim`:
1. **`VectorFleetEnv` CPU Throughput Scaling**: Batched NumPy simulation step rate across `num_envs` $\in \{16, 32, 64, 128, 256\}$ for both `--quick` (`2` hoppers, `6` craters) and full Daedalus (`4` hoppers, `12` craters) configurations against the Stage 1 gate (`> 10,000 env steps/s`).
2. **CPU vs Apple Silicon `"mps"` Small-MLP Latency Benchmark**: Empirical measurement of 2-layer `Tanh` MLP (`64` and `128` hidden units, batch sizes `256` and `512`) forward+backward pass time and rollout Actor inference time with NumPy-to-Torch host transfer, justifying `select_device(prefer_cpu_for_small_mlp=True)`.
3. **Daedalus Far-Side Terrain Crop Verification (`data/terrain/daedalus_200x200.npz`)**: Byte-for-byte reproducibility check of `scripts/build_terrain_crop.py`, full distribution statistics for elevation (`height_m`), Horn $3 \times 3$ slope (`slope_deg`), and Diviner night-time temperature (`temp_c`), and the 12-crater Daedalus catalogue.
4. **Full 41-Test Suite (`pytest apps/sim/tests -v`)**: Complete execution and verbatim pass log across all 10 test modules (`41 passed in 41.76s`).

---

## Reproducible Command Lines

Run from the repository root (`/Users/nabilshah/.gemini/antigravity/scratch/worktrees/run-verifier-and-benchmarks`):

```bash
# 1. Run the Stage 1 VectorFleetEnv throughput benchmark (num_envs=64, 500 batch steps = 32,000 env steps)
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python -c "
from selene_sim.env import benchmark_vector_env
res_full = benchmark_vector_env(num_envs=64, num_steps=500, quick=False)
res_quick = benchmark_vector_env(num_envs=64, num_steps=500, quick=True)
print('Full (H=4, C=12):', res_full)
print('Quick (H=2, C=6):', res_quick)
"

# 2. Rebuild the 200x200 Daedalus terrain crop and verify exact NPZ array match
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python \
  scripts/build_terrain_crop.py --output /tmp/daedalus_200x200_check.npz

# 3. Execute the complete 41-test pytest suite
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python \
  -m pytest apps/sim/tests -v
```

---

## 1. `VectorFleetEnv` CPU Throughput Scaling (`num_envs` $\in \{16, 32, 64, 128, 256\}$)

`VectorFleetEnv` steps all `(E, H)` hoppers simultaneously in pure NumPy without Python loops over environments or agents, computing action masks `(E, H, C + 3)`, ballistic hop rocket-equation propellant burns, $3 \times 3$ Horn terrain slopes, Diviner thermal battery drain, pairwise horizontal separation distances `(E, H, H)`, and single-channel orbiter relay collisions in vectorised array operations.

Each configuration was benchmarked over `400` batched steps (`5` trials after warm-up) with uniformly sampled valid masked actions:

### Quick Mode (`--quick`: `H = 2` Hoppers, `C = 6` Craters, `obs_dim = 31`, `state_dim = 28`)

| `num_envs` ($E$) | Batch Step Latency (`ms/step`) | Median Throughput (`env steps/s`) | Throughput IQR (`env steps/s`) | Agent Decisions (`hopper steps/s`) | Margin vs `10,000 steps/s` Gate |
| --- | --- | --- | --- | --- | --- |
| `16` | `0.8127` | `19,686.7` | `111.4` | `39,373.4` | `1.97x` |
| `32` | `0.9010` | `35,515.4` | `600.2` | `71,030.8` | `3.55x` |
| `64` | `1.0706` | `59,778.6` | `200.4` | `119,557.3` | `5.98x` |
| **`128`** | **`1.4379`** | **`89,016.6`** | **`914.4`** | **`178,033.1`** | **`8.90x`** |
| `256` | `3.1661` | `80,855.8` | `2,975.3` | `161,711.5` | `8.09x` |

### Full Daedalus Mode (`H = 4` Hoppers, `C = 12` Craters, `obs_dim = 57`, `state_dim = 54`)

| `num_envs` ($E$) | Batch Step Latency (`ms/step`) | Median Throughput (`env steps/s`) | Throughput IQR (`env steps/s`) | Agent Decisions (`hopper steps/s`) | Margin vs `10,000 steps/s` Gate |
| --- | --- | --- | --- | --- | --- |
| `16` | `1.2544` | `12,754.7` | `2,352.3` | `51,018.7` | `1.28x` |
| `32` | `1.7019` | `18,802.3` | `2,610.0` | `75,209.1` | `1.88x` |
| `64` | `1.6374` | `39,086.6` | `13,116.4` | `156,346.4` | `3.91x` |
| **`128`** | **`1.9630`** | **`65,205.6`** | **`2,502.2`** | **`260,822.3`** | **`6.52x`** |
| **`256`** | **`3.1038`** | **`82,479.3`** | **`5,460.0`** | **`329,917.3`** | **`8.25x`** |

### Throughput Analysis

- Every batch size from $E = 16$ to $E = 256$ exceeds the Stage 1 requirement of `10,000 env steps/s` in both `--quick` and full 4-hopper, 12-crater modes.
- At the default Stage 3 training batch size ($E = 128$), full 4-hopper `VectorFleetEnv` achieves **`65,205.6 env steps/s`** (`260,822.3 hopper decisions/s`), `6.52x` above the Stage 1 gate. Peak throughput reaches **`89,016.6 env steps/s`** (`--quick`, $E = 128$) and **`82,479.3 env steps/s`** (full mode, $E = 256$, `329,917.3 hopper decisions/s`).

---

## 2. 2-Layer MLP (`64` & `128` Hidden Units) CPU vs Apple Silicon `"mps"` Benchmark

Both `apps/sim/selene_sim/mappo.py` and `apps/sim/selene_sim/world_model.py` use `select_device(prefer_cpu_for_small_mlp=True)` so that small 2-layer MLPs default to CPU unless overridden. To verify this design empirically on Apple Silicon (`macOS-26.2-arm64`, PyTorch `2.15.0`), we benchmarked:
1. **PPO Minibatch Update (`50` Forward + Backward + `Adam` Optimizer Steps)** on a shared 2-layer `Tanh` `Actor` (`obs_dim = 57 -> H -> H -> 15`) and centralised `Critic` (`state_dim = 54 -> H -> H -> 1`).
2. **Rollout Step `Actor` Inference (`24` Steps x `E = 128` x `H = 4` = `512` Agent Observations/Step)** including the mandatory NumPy-to-Torch observation/mask transfer and action copy back to NumPy for `VectorFleetEnv.step`.

| Hidden Units ($H$) | Minibatch Size | CPU Fwd+Bwd (`ms/step`) | MPS Fwd+Bwd (`ms/step`) | CPU Fwd+Bwd Speedup vs MPS | CPU Rollout Inference (`24` steps, `ms`) | MPS Rollout Inference (`24` steps, `ms`) | CPU Rollout Speedup vs MPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `64` (`--quick`) | `256` | **`0.9501`** (`47.50 ms` / 50) | `1.4285` (`71.43 ms` / 50) | **`1.50x faster on CPU`** | **`0.5077`** | `2.4648` | **`4.85x faster on CPU`** |
| `64` (`--quick`) | `512` | **`1.2400`** (`62.00 ms` / 50) | `1.5479` (`77.39 ms` / 50) | **`1.25x faster on CPU`** | **`0.7262`** | `2.7135` | **`3.74x faster on CPU`** |
| `128` (`full`) | `256` | **`1.1690`** (`58.45 ms` / 50) | `1.3606` (`68.03 ms` / 50) | **`1.16x faster on CPU`** | **`0.5417`** | `3.3498` | **`6.18x faster on CPU`** |
| `128` (`full`) | `512` | `1.4269` (`71.35 ms` / 50) | `1.3887` (`69.44 ms` / 50) | `0.97x` (parity) | **`0.8214`** | `4.0057` | **`4.88x faster on CPU`** |

### Device Selection Conclusion

- Because `VectorFleetEnv` executes in CPU NumPy memory, every rollout step requires transferring `(E, H, obs_dim)` observations and `(E, H, act_dim)` action masks to the policy device and returning `(E, H)` discrete actions. Metal command-buffer dispatch and host-device synchronisation make `"mps"` **`3.74x` to `6.18x` slower** than CPU during rollout collection.
- Even for pure on-device minibatch gradient updates (`batch_size = 256`), CPU Accelerate matrix multiplication is **`1.16x` to `1.50x` faster** than `"mps"` for `64`-unit and `128`-unit 2-layer MLPs. Only at `hidden_dim = 128, batch_size = 512` do pure on-device updates reach parity (`1.43 ms` vs `1.39 ms`), which is outweighed by the `4.88x` rollout inference penalty on `"mps"`.

---

## 3. Daedalus Terrain Crop Verification (`data/terrain/daedalus_200x200.npz`)

Running `scripts/build_terrain_crop.py --output /tmp/daedalus_200x200_check.npz` completes in **`21.90 ms`**, producing a `485,600-byte` compressed `.npz` bundle whose arrays match `data/terrain/daedalus_200x200.npz` byte-for-byte across all 12 keys.

### Spatial Grid and Surface Field Distributions (`200 x 200` = `40,000` Cells at `100.0 m/cell`)

| Field / Layer | Units | Min | `p05` | `p25` | Median (`p50`) | Mean | Std Dev | `p75` | `p95` | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Elevation (`height_m`) | `m` | `-169.468` | `-21.187` | `-4.333` | `+0.571` | `-4.345` | `28.384` | `+5.605` | `+12.501` | `+49.964` |
| Horn $3 \times 3$ Slope (`slope_deg`) | `deg` | `0.000` | `0.045` | `0.108` | `0.155` | `1.543` | `5.133` | `0.188` | `7.404` | `34.845` |
| Diviner Night Temp (`temp_c`) | `deg C` | `-225.000` | `-181.707` | `-171.761` | `-169.732` | `-171.641` | `8.009` | `-168.562` | `-167.244` | `-166.431` |

- **Region Centre**: Daedalus Crater (`5.9 deg S, 179.4 deg E`), `20.0 km x 20.0 km` extent (`[-10,000 m, +10,000 m]` in local East/North coordinates).
- **Hazardous Steep Cells (`slope_deg > 15.0 deg`)**: `1,484 / 40,000` cells (`3.71%` of the map), concentrated along the raised inner rim walls (`0.68 <= r/R <= 1.05`) of the 12 bowls where local slopes reach `28.72 deg` to `34.85 deg`.

### 12-Crater Daedalus Catalogue (`data/terrain/daedalus_200x200.npz`)

| Crater ID | East $x$ (`m`) | North $y$ (`m`) | Latitude (`deg`) | Longitude (`deg`) | Radius $R$ (`m`) | Depth $D$ (`m`) | Science Priority | Floor Height (`m`) | Floor Slope (`deg`) | Rim Wall Slope (`deg`) | Floor Night Temp (`deg C`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `daedalus_a` | `+1550.0` | `+550.0` | `-5.8819` | `179.4514` | `796.9` | `150.7` | `1.00` | `-150.74` | `0.00` | `32.31` | `-223.79` |
| `daedalus_b` | `+550.0` | `+1850.0` | `-5.8390` | `179.4182` | `729.7` | `142.6` | `1.10` | `-142.61` | `0.00` | `31.10` | `-223.12` |
| `daedalus_c` | `-1350.0` | `+1250.0` | `-5.8588` | `179.3552` | `864.3` | `149.0` | `1.20` | `-149.05` | `0.00` | `29.94` | `-224.65` |
| `daedalus_d` | `-1650.0` | `-650.0` | `-5.9214` | `179.3453` | `793.5` | `146.8` | `1.00` | `-146.83` | `0.00` | `31.86` | `-224.50` |
| `daedalus_e` | `-350.0` | `-1850.0` | `-5.9610` | `179.3884` | `766.1` | `129.1` | `1.10` | `-129.09` | `0.00` | `29.78` | `-221.79` |
| `daedalus_f` | `+1450.0` | `-1450.0` | `-5.9478` | `179.4481` | `852.7` | `143.0` | `1.20` | `-143.02` | `0.00` | `28.72` | `-222.80` |
| `daedalus_g` | `+3750.0` | `+1150.0` | `-5.8621` | `179.5243` | `956.6` | `169.1` | `1.40` | `-169.13` | `0.00` | `31.05` | `-225.00` |
| `daedalus_h` | `+1550.0` | `+3650.0` | `-5.7796` | `179.4514` | `923.5` | `166.3` | `1.30` | `-166.29` | `0.00` | `32.27` | `-225.00` |
| `daedalus_i` | `-2850.0` | `+2750.0` | `-5.8093` | `179.3055` | `934.7` | `169.5` | `1.50` | `-169.47` | `0.00` | `32.50` | `-225.00` |
| `daedalus_j` | `-3750.0` | `-1350.0` | `-5.9445` | `179.2757` | `882.7` | `162.9` | `1.30` | `-162.93` | `0.00` | `32.47` | `-225.00` |
| `daedalus_k` | `-1050.0` | `-3650.0` | `-6.0204` | `179.3652` | `973.4` | `162.2` | `1.40` | `-162.21` | `0.00` | `29.26` | `-225.00` |
| `daedalus_l` | `+3250.0` | `-2750.0` | `-5.9907` | `179.5077` | `898.5` | `150.0` | `1.30` | `-150.03` | `0.00` | `30.46` | `-223.09` |

---

## 4. Full 41-Test Suite Execution (`pytest apps/sim/tests -v`)

### Summary by Test Module

| Test Module | Tests | Status | Coverage Scope |
| --- | --- | --- | --- |
| `tests/test_api.py` | `1` | `1 PASSED` | FastAPI `/health` endpoint status check |
| `tests/test_demo_bundle.py` | `2` | `2 PASSED` | End-to-end static demo bundle generator (`episodes.json`, `training_log.csv`, `world_model_map.json`, `verification_cloud.json`) and honest `metadata.source` tagging |
| `tests/test_env.py` | `5` | `5 PASSED` | Stage 1 1,000-step random masked rollout (`0 NaNs`), `> 10,000 steps/s` vectorised throughput benchmark, action masking rules, crater claiming/scanning/downlink mechanics, and `render_episode_dict` schema check |
| `tests/test_episode_schema.py` | `3` | `3 PASSED` | `docs/data/EPISODE_SCHEMA.md` positive validation and negative rejection of mismatched fleet sizes or invalid action types |
| `tests/test_mappo.py` | `3` | `3 PASSED` | CTDE `Actor`/`Critic` invalid-action masking (`-1e8`), Stage 2 `--quick` reward improvement (`last_10 > first_10`), Stage 3 CSV summary, and Issue #13 MBPO `imagine_rollout` uncertainty gating |
| `tests/test_orchestrator_contract.py` | `4` | `4 PASSED` | Issue #15 end-to-end Argon orchestrator contract tests: Eureka reward hot-swap in live `VectorFleetEnv`, world model epistemic uncertainty map, 1,000-rollout verifier safe vs unsafe plan certification, and FastAPI `TestClient` endpoints |
| `tests/test_reward.py` | `5` | `5 PASSED` | Single-env and vectorised `(E, H)` reward shapes, positive scan/claim/relay rewards, crash/relay-conflict/proximity/idle penalties, and runtime `compile_and_validate_reward` hot-swapping |
| `tests/test_sim.py` | `6` | `6 PASSED` | Pure-NumPy isolation (`no torch import`), synthetic/NPZ terrain grids, analytical ballistic hop and Tsiolkovsky rocket equation checks, vectorised noise and crash conditions, Diviner thermal battery drain, and 3D parabolic keyframe interpolation |
| `tests/test_terrain_crop.py` | `4` | `4 PASSED` | Horn $3 \times 3$ slope stencil analytical accuracy, cylindrical projection and bilinear resampling, synthetic and external GeoTIFF/IMG DEM crop generation, and `data/terrain/daedalus_200x200.npz` verification |
| `tests/test_verify.py` | `6` | `6 PASSED` | Pure-NumPy isolation (`never imports world_model or torch`), `simulate_plan` determinism, shapes and physics invariants, relay conflict and proximity violation detection, 3D trajectory cloud JSON serialisation, and `< 10.0 s` 1,000-rollout performance gate |
| `tests/test_world_model.py` | `2` | `2 PASSED` | `select_device` CPU preference, `DynamicsMLP` shapes, and Stage 4 5-MLP ensemble held-out validation MSE (`val_mse < 0.05`) and OOD epistemic uncertainty separation (`OOD > 1.5x ID`) |
| **Total** | **`41`** | **`41 PASSED`** | **Complete Stage 1 to Stage 4 verification (`41.76s` wall-clock)** |

### Verbatim `pytest -v` Output

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.14, pytest-9.2.0, pluggy-1.7.0 -- /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/nabilshah/.gemini/antigravity/scratch/worktrees/run-verifier-and-benchmarks/apps/sim
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 41 items

apps/sim/tests/test_api.py::test_health PASSED                           [  2%]
apps/sim/tests/test_demo_bundle.py::test_generate_demo_bundle_creates_all_artefacts_and_matches_schema PASSED [  4%]
apps/sim/tests/test_demo_bundle.py::test_honest_source_tagging_with_checkpoint_and_cli PASSED [  7%]
apps/sim/tests/test_env.py::test_stage1_random_masked_policy_1000_steps PASSED [  9%]
apps/sim/tests/test_env.py::test_stage1_vector_env_throughput_benchmark PASSED [ 12%]
apps/sim/tests/test_env.py::test_action_masking_rules PASSED             [ 14%]
apps/sim/tests/test_env.py::test_claiming_scanning_and_sending_mechanics PASSED [ 17%]
apps/sim/tests/test_env.py::test_render_episode_dict_matches_schema PASSED [ 19%]
apps/sim/tests/test_episode_schema.py::test_example_matches_schema PASSED [ 21%]
apps/sim/tests/test_episode_schema.py::test_schema_rejects_mismatched_fleet_size PASSED [ 24%]
apps/sim/tests/test_episode_schema.py::test_schema_rejects_invalid_action_type PASSED [ 26%]
apps/sim/tests/test_mappo.py::test_mappo_config_and_actor_critic_masking PASSED [ 29%]
apps/sim/tests/test_mappo.py::test_stage2_quick_learning_and_stage3_summary PASSED [ 31%]
apps/sim/tests/test_mappo.py::test_issue13_mbpo_imagine_rollout_uncertainty_gating PASSED [ 34%]
apps/sim/tests/test_orchestrator_contract.py::test_contract_1_eureka_reward_hot_swap_in_live_vector_env PASSED [ 36%]
apps/sim/tests/test_orchestrator_contract.py::test_contract_2_world_model_error_map_and_epistemic_uncertainty PASSED [ 39%]
apps/sim/tests/test_orchestrator_contract.py::test_contract_3_verifier_1000_rollouts_safe_vs_unsafe_plan PASSED [ 41%]
apps/sim/tests/test_orchestrator_contract.py::test_contract_4_fastapi_testclient_endpoints PASSED [ 43%]
apps/sim/tests/test_reward.py::test_single_and_vectorised_shapes PASSED  [ 46%]
apps/sim/tests/test_reward.py::test_positive_reward_for_scanning_and_claiming_crater PASSED [ 48%]
apps/sim/tests/test_reward.py::test_positive_reward_for_relay_send_without_conflict PASSED [ 51%]
apps/sim/tests/test_reward.py::test_negative_penalties PASSED            [ 53%]
apps/sim/tests/test_reward.py::test_hot_swap_and_validation PASSED       [ 56%]
apps/sim/tests/test_sim.py::test_sim_does_not_import_torch PASSED        [ 58%]
apps/sim/tests/test_sim.py::test_terrain_grid_synthetic_full_and_quick PASSED [ 60%]
apps/sim/tests/test_sim.py::test_ballistic_hop_and_rocket_equation_analytical PASSED [ 63%]
apps/sim/tests/test_sim.py::test_vectorised_hop_with_noise_and_crash_conditions PASSED [ 65%]
apps/sim/tests/test_sim.py::test_thermal_battery_drain_and_relay_windows PASSED [ 68%]
apps/sim/tests/test_sim.py::test_interpolate_hop_keyframes PASSED        [ 70%]
apps/sim/tests/test_terrain_crop.py::test_horn_slope_stencil_accuracy PASSED [ 73%]
apps/sim/tests/test_terrain_crop.py::test_cylindrical_projection_and_bilinear_resampling PASSED [ 75%]
apps/sim/tests/test_terrain_crop.py::test_build_daedalus_crop_synthetic_and_external_dem PASSED [ 78%]
apps/sim/tests/test_terrain_crop.py::test_generated_daedalus_npz_bundle_out_of_the_box PASSED [ 80%]
apps/sim/tests/test_verify.py::test_verify_never_imports_world_model_or_torch PASSED [ 82%]
apps/sim/tests/test_verify.py::test_simulate_plan_determinism PASSED     [ 85%]
apps/sim/tests/test_verify.py::test_simulate_plan_shapes_and_physics PASSED [ 87%]
apps/sim/tests/test_verify.py::test_relay_conflict_and_proximity_violation_detection PASSED [ 90%]
apps/sim/tests/test_verify.py::test_sample_trajectory_cloud_json_serialisable PASSED [ 92%]
apps/sim/tests/test_verify.py::test_benchmark_1000_rollouts_under_10_seconds PASSED [ 95%]
apps/sim/tests/test_world_model.py::test_select_device_and_dynamics_mlp_shapes PASSED [ 97%]
apps/sim/tests/test_world_model.py::test_stage4_validation_mse_and_epistemic_uncertainty PASSED [100%]

======================== 41 passed, 1 warning in 41.76s ========================
```
