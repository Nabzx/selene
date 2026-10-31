# Run 03: Stage 4 Shared 5-MLP Ensemble World Model and Epistemic Uncertainty Map

## Overview

This document records the execution, architecture, and quantitative evaluation of **Run 03** (Stage 4 / Issue #11): training the shared 5-MLP ensemble world model (`WorldModelEnsemble`) on 8,000 pooled multi-hopper transitions across the 12-crater Daedalus terrain (`data/terrain/daedalus_200x200.npz`) and exporting the spatial error and epistemic uncertainty map (`data/world_model_map.json`) for the Argon orchestrator.

## Reproducible Command

Run from the repository root with `PYTHONPATH=apps/sim`:

```bash
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python -c '
import json, time
from pathlib import Path
import numpy as np
from selene_sim.env import VectorFleetEnv
from selene_sim.sim import TerrainGrid
from selene_sim.world_model import TransitionBuffer, WorldModelEnsemble

terrain = TerrainGrid.from_npz("data/terrain/daedalus_200x200.npz")
vec_env = VectorFleetEnv(
    num_envs=100,
    n_hoppers=4,
    n_craters=12,
    max_steps=20,
    quick=False,
    seed=42,
    terrain=terrain,
    record_transitions=True,
)
vec_env.reset(seed=42)
rng = np.random.default_rng(42)

for _ in range(20):
    masks = vec_env.get_action_masks()
    masks[:, :, 8:12] = False  # Restrict hop exploration to craters 0..7
    noise = rng.uniform(0.01, 1.0, size=masks.shape)
    actions = np.argmax(np.where(masks, noise, -1.0), axis=-1).astype(np.int32)
    vec_env.step(actions)

raw = vec_env.get_transitions(clear=True)
crater_ids = np.where(raw["actions"] < 12, raw["actions"], -1).astype(np.int32)
buffer = TransitionBuffer()
buffer.add(
    obs=raw["obs"],
    actions=raw["actions"],
    next_obs=raw["next_obs"],
    rewards=raw["rewards"],
    crashed=raw["crashed"],
    crater_ids=crater_ids,
)

ensemble = WorldModelEnsemble(
    obs_dim=vec_env.obs_dim,
    n_actions=vec_env.n_actions,
    ensemble_size=5,
    hidden_dim=128,
    device="cpu",
    seed=42,
)
t0 = time.perf_counter()
history = ensemble.train_epochs(buffer, epochs=25, batch_size=512, seed=42)
wm_train_time = time.perf_counter() - t0

error_map = ensemble.world_model_error_map(terrain, buffer)
error_map["source"] = "learned"
error_map["seed"] = 42
error_map["training_history"] = {
    "initial_val_mse": float(history["val_mse"][0]),
    "final_val_mse": float(history["val_mse"][-1]),
    "epochs": 25,
    "train_loss": [float(x) for x in history["train_loss"]],
    "val_mse": [float(x) for x in history["val_mse"]],
    "wall_time_s": round(wm_train_time, 4),
}

Path("data/world_model_map.json").write_text(
    json.dumps(error_map, indent=2), encoding="utf-8"
)
'
```

## Architecture and Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Terrain** | `data/terrain/daedalus_200x200.npz` | 200x200 Daedalus grid (100 m cell width, 12 catalogue craters) |
| **Random Seed** | `42` | Deterministic seed for NumPy environment rollouts, split, and PyTorch ensemble |
| **Pooled Transitions** | `8,000` (`7,200` train / `800` val) | 100 vectorised environments x 4 hoppers x 20 steps |
| **Observation Dimension (`obs_dim`)** | `57` | 6 own-state + 4 one-hot identity + 36 crater features + 9 peer features + 2 relay features |
| **Action Space (`n_actions`)** | `15` | `0..11` hop to crater `k`, `12` scan, `13` wait, `14` send |
| **Ensemble Size (`K`)** | `5` | 5 independent `DynamicsMLP` members seeded at `42 + 1009 * (k + 1)` |
| **Member Backbone** | `Linear(72, 128) -> LayerNorm -> SiLU -> Linear(128, 128) -> LayerNorm -> SiLU -> Linear(128, 59)` | Predicts `(delta_obs[57], reward[1], crash_logit[1])` |
| **Randomised Prior Functions** | Untrainable `prior_net` + `prior_action_head` (`prior_scale=0.65`) | Osband et al. (2018) frozen prior preserving epistemic disagreement on unvisited craters |
| **Optimiser & Schedule** | `AdamW(lr=3e-3, weight_decay=1e-4)` | Trained for `25` epochs with `batch_size=512` and per-member bootstrap sampling |
| **Execution Device** | `cpu` | Selected for deterministic, low-latency small-MLP batch execution |
| **Training Wall-Clock Time** | `4.5046 s` | Full 25-epoch training pass across all 5 ensemble members |

## Epoch-by-Epoch Training Loss and Validation MSE

Validation MSE drops from `0.995663` (Epoch 1 pre-update baseline) to `0.277116` (Epoch 25), representing a **72.17% reduction** in normalised validation MSE (exceeding the Stage 4 gate requirement of at least a 50% reduction in under 15 seconds). Unnormalised held-out observation prediction MSE across all 57 dimensions reaches `0.007504`.

| Epoch | Train Loss (`delta + 0.2*rew + 0.1*bce`) | Validation MSE (Normalised) | Relative MSE Change vs Epoch 1 |
| :---: | :---: | :---: | :---: |
| 1 | `1.381397` | `0.995663` | `0.00%` |
| 2 | `1.011145` | `0.626823` | `-37.04%` |
| 3 | `0.798697` | `0.503796` | `-49.40%` |
| 4 | `0.638856` | `0.403655` | `-59.46%` |
| 5 | `0.520849` | `0.349982` | `-64.85%` |
| 6 | `0.466779` | `0.323717` | `-67.49%` |
| 7 | `0.427319` | `0.312669` | `-68.60%` |
| 8 | `0.415783` | `0.303794` | `-69.49%` |
| 9 | `0.399904` | `0.298091` | `-70.06%` |
| 10 | `0.388647` | `0.294109` | `-70.46%` |
| 11 | `0.378013` | `0.291118` | `-70.76%` |
| 12 | `0.370349` | `0.289161` | `-70.96%` |
| 13 | `0.365401` | `0.287062` | `-71.17%` |
| 14 | `0.355928` | `0.285292` | `-71.35%` |
| 15 | `0.380667` | `0.285203` | `-71.36%` |
| 16 | `0.345588` | `0.283716` | `-71.50%` |
| 17 | `0.332840` | `0.281752` | `-71.70%` |
| 18 | `0.326185` | `0.281197` | `-71.76%` |
| 19 | `0.321338` | `0.280143` | `-71.86%` |
| 20 | `0.326367` | `0.280736` | `-71.80%` |
| 21 | `0.319913` | `0.279555` | `-71.92%` |
| 22 | `0.311213` | `0.277794` | `-72.10%` |
| 23 | `0.304664` | `0.278483` | `-72.03%` |
| 24 | `0.298649` | `0.277740` | `-72.11%` |
| 25 | `0.293667` | `0.277116` | `-72.17%` |

## Per-Crater Error and Epistemic Uncertainty (`craters 0..11`)

In this run, craters `0..7` (`daedalus_a` through `daedalus_h`) are visited during fleet exploration (`visit_count` between `80` and `278`), whereas outer-ring craters `8..11` (`daedalus_i` through `daedalus_l`) are unvisited (`visit_count = 0`).

- **Visited Craters (`0..7`):** Mean epistemic uncertainty = `0.2873` (range `0.2641` to `0.3254`).
- **Unvisited Craters (`8..11`):** Mean epistemic uncertainty = `1.0591` (range `0.9915` to `1.1443`), which is **3.69x higher (+268.6%)** than on visited craters, confirming strong epistemic calibration via Randomised Prior Functions.

| Crater Index | Crater ID | `x_m` (m) | `y_m` (m) | Visit Count | Mean Error (L2) | Epistemic Uncertainty | Regime |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `0` | `daedalus_a` | `1550.0` | `550.0` | `278` | `0.847510` | `0.293199` | Visited (inner ring) |
| `1` | `daedalus_b` | `550.0` | `1850.0` | `222` | `0.854527` | `0.265467` | Visited (inner ring) |
| `2` | `daedalus_c` | `-1350.0` | `1250.0` | `250` | `0.855772` | `0.303161` | Visited (inner ring) |
| `3` | `daedalus_d` | `-1650.0` | `-650.0` | `248` | `0.883117` | `0.283980` | Visited (inner ring) |
| `4` | `daedalus_e` | `-350.0` | `-1850.0` | `248` | `0.878512` | `0.264059` | Visited (inner ring) |
| `5` | `daedalus_f` | `1450.0` | `-1450.0` | `195` | `0.892670` | `0.282531` | Visited (inner ring) |
| `6` | `daedalus_g` | `3750.0` | `1150.0` | `89` | `0.723056` | `0.280779` | Visited (outer NE) |
| `7` | `daedalus_h` | `1550.0` | `3650.0` | `80` | `0.700690` | `0.325432` | Visited (outer N) |
| `8` | `daedalus_i` | `-2850.0` | `2750.0` | `0` | `0.572165` | `1.144329` | Unvisited (outer NW) |
| `9` | `daedalus_j` | `-3750.0` | `-1350.0` | `0` | `0.520714` | `1.041429` | Unvisited (outer W) |
| `10` | `daedalus_k` | `-1050.0` | `-3650.0` | `0` | `0.495763` | `0.991525` | Unvisited (outer S) |
| `11` | `daedalus_l` | `3250.0` | `-2750.0` | `0` | `0.529485` | `1.058969` | Unvisited (outer SE) |

## Top-5 Highest-Uncertainty Spatial Cells (`data/world_model_map.json`)

The 20x20 spatial grid divides the `[-10000, 10000] m` Daedalus region into `1000 m x 1000 m` cells. The top-5 highest-uncertainty cells correspond to unvisited far-eastern boundary cells (`x_m = 9500.0 m`, `visit_count = 0`), where spatial coordinates lie well outside the convex hull of training transitions.

| Rank | Grid Cell `[i, j]` | Centre `x_m` (m) | Centre `y_m` (m) | Visit Count | Mean Error | Epistemic Uncertainty | Nearest Crater ID |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `[13, 19]` | `9500.0` | `3500.0` | `0` | `0.542734` | `1.085468` | `daedalus_g` |
| 2 | `[12, 19]` | `9500.0` | `2500.0` | `0` | `0.542253` | `1.084507` | `daedalus_g` |
| 3 | `[14, 19]` | `9500.0` | `4500.0` | `0` | `0.541902` | `1.083805` | `daedalus_g` |
| 4 | `[11, 19]` | `9500.0` | `1500.0` | `0` | `0.540707` | `1.081414` | `daedalus_g` |
| 5 | `[15, 19]` | `9500.0` | `5500.0` | `0` | `0.539905` | `1.079810` | `daedalus_g` |
