# Run 01: Stage 2 Quick MAPPO Training (`--quick`, 2 Hoppers, 6 Craters, 40 Updates)

## Overview

This report documents the reproducible Stage 2 (`--quick`) Centralised Training with Decentralised Execution (CTDE) Multi-Agent PPO (MAPPO) run on `VectorFleetEnv` over the 6-crater Daedalus lunar far-side benchmark (`200 x 200` terrain grid at `100 m/cell`).

The run verifies single-file CleanRL-style MAPPO with invalid-action masking (`-1e8` logits mask before `Categorical`), running return normalisation (`ValueNormaliser`), periodic 5-MLP `WorldModelEnsemble` updates, and short-horizon Model-Based Policy Optimisation (MBPO) imagination rollouts (`imagine_rollout`) gated by ensemble epistemic uncertainty.

## Reproducible Command Lines

Run from the repository root (`/Users/nabilshah/.gemini/antigravity/scratch/worktrees/run-mappo-training`):

```bash
# 1. Execute Stage 2 --quick MAPPO training (40 updates, seed=42)
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python \
  -m selene_sim.mappo \
  --quick \
  --updates 40 \
  --csv data/training_log_quick.csv \
  --checkpoint-dir "" \
  --export-episodes ""

# 2. Print formatted terminal summary from the CSV log
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python \
  apps/sim/scripts/training_summary.py data/training_log_quick.csv
```

## Hardware, Device Selection, and Seed Configuration

| Property | Value | Notes |
|---|---|---|
| Random Seed | `42` | Seeds `torch.manual_seed(42)`, `np.random.seed(42)`, `VectorFleetEnv(seed=42)`, and `WorldModelEnsemble(seed=42)` |
| `select_device(prefer_fastest=True)` | `mps` | Micro-benchmarks 50 forward/backward passes of a 2-layer 128-unit MLP on CPU vs Apple Metal (`mps`) |
| `resolve_mappo_device(cfg.device)` | `cpu` | Compact `64`-unit Actor/Critic MLPs execute on CPU alongside contiguous NumPy `VectorFleetEnv` buffers to avoid per-step host-to-device transfer overhead |
| `WorldModelEnsemble` Device | `cpu` | 5-member `DynamicsMLP` ensemble with untrainable Randomised Prior Functions (`prior_scale=0.65`) |
| Total Environment Steps | `51,200` | `40` updates x `64` parallel environments x `20` rollout steps |
| Wall-Clock Training Time | `7.04 s` | ~`7,270` environment steps/s including PPO + MBPO + WorldModel updates |

## Hyperparameters (`MAPPOConfig(quick=True)`)

| Hyperparameter | Field | Value |
|---|---|---|
| Quick mode flag | `quick` | `True` |
| Fleet size (hoppers) | `n_hoppers` | `2` |
| Active craters in catalogue | `n_craters` | `6` |
| Episode horizon (decision steps) | `max_steps` | `20` |
| Parallel vectorised environments | `num_envs` | `64` |
| Rollout length per update | `rollout_steps` | `20` |
| Actor / Critic / WM hidden width | `hidden_dim` | `64` |
| Total PPO updates | `total_updates` | `40` |
| Wall-clock time budget | `time_budget_s` | `120.0` |
| Random seed | `seed` | `42` |
| Initial learning rate (linearly annealed) | `lr` | `3e-3` |
| Discount factor | `gamma` | `0.99` |
| GAE lambda | `gae_lambda` | `0.95` |
| PPO clipping coefficient | `clip_coef` | `0.2` |
| Entropy bonus coefficient | `ent_coef` | `0.01` |
| Value loss coefficient | `vf_coef` | `0.5` |
| Max gradient norm | `max_grad_norm` | `0.5` |
| PPO epochs per rollout | `ppo_epochs` | `4` |
| Minibatches per epoch | `num_minibatches` | `4` |
| World model update interval | `wm_update_interval` | `5` |
| World model epochs per update | `wm_epochs` | `5` |
| Short-horizon MBPO enabled | `use_mbpo` | `True` |
| MBPO rollout horizon | `mbpo_horizon` | `2` |
| MBPO epistemic uncertainty threshold | `mbpo_uncertainty_threshold` | `0.5` |
| MBPO rollout batch size | `mbpo_rollout_batch` | `64` |
| MBPO policy loss weight | `mbpo_loss_weight` | `0.1` |
| Observation dimension | `obs_dim` | `31` |
| Centralised global state dimension | `global_state_dim` | `28` |
| Discrete action space per hopper | `n_actions` | `9` (`6` hop targets + `scan`, `wait`, `send`) |

## Verbatim `scripts/training_summary.py` Output

```text
=== Selene MAPPO Training Summary (training_log_quick.csv) ===
Updates: 40 | Env Steps: 51200 | Wall Time: 7.04s
--------------------------------------------------------------------
Metric               |      First 3 |       Last 3 |        Delta
--------------------------------------------------------------------
episode_reward       |     180.7622 |     269.2446 |     +88.4825
crashes              |       0.0155 |       0.0000 |      -0.0155
craters_visited      |       4.6993 |       6.0000 |      +1.3007
crater_overlap       |       2.0278 |       0.0238 |      -2.0040
relay_conflicts      |       0.3556 |       0.0074 |      -0.3482
entropy              |       0.8336 |       0.0171 |      -0.8165
value_loss           |       0.4471 |       0.0228 |      -0.4243
wm_error             |       1.1397 |       0.0145 |      -1.1252
--------------------------------------------------------------------
Stage 2 Check (Reward Up & Crashes Low): PASS
Stage 3 Check (Crater Overlap Down):     PASS
Stage 4 Check (World Model Error Down):  PASS
```

## Stage 2 & Stage 3 Verification Analysis

1. **Stage 2 Verification (Reward Gain and Crash Elimination):**
   - Mean fleet `episode_reward` across the first 3 updates (`180.7622`) rises to `269.2446` across the final 3 updates, achieving a net gain of `+88.4825` (exceeding the `+15.0` threshold by nearly 6x).
   - Mean fleet `crashes` per episode drops from `0.0155` in the first 3 updates to `0.0000` from update 10 onwards (`-0.0155` delta), confirming that invalid-action masking and slope/propellant safety constraints keep the fleet crash-free.
   - Distinct `craters_visited` increases from `4.6993` to all `6.0000` available craters in the quick catalogue.
2. **Stage 3 Verification (Cooperative Crater Splitting and Relay Turn-Taking):**
   - `crater_overlap` (duplicate crater visits across the 2 hoppers) drops by `98.8%` from `2.0278` (first 3 updates) to `0.0238` (last 3 updates), proving that the hoppers partition the 6 craters into disjoint sectors rather than visiting the same targets.
   - `relay_conflicts` (simultaneous transmissions during open Earth-orbiter relay windows) drops by `97.9%` from `0.3556` to `0.0074`, demonstrating staggered TDMA downlink behaviour.
3. **Stage 4 Verification (World Model Calibration):**
   - Held-out `WorldModelEnsemble` validation error (`wm_error`) decreases monotonically across periodic updates (updates 1, 5, 10, 15, 20, 25, 30, 35, 40) from `1.139707` to `0.013129` (`0.0145` mean over last 3 updates).

## Full Milestone Progression Table (`data/training_log_quick.csv`)

| `update` | `env_steps` | `wall_time_s` | `episode_reward` | `crashes` | `craters_visited` | `crater_overlap` | `relay_conflicts` | `entropy` | `value_loss` | `wm_error` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1280 | 0.1541 | 169.6399 | 0.0156 | 4.3594 | 2.4219 | 0.3438 | 0.8533 | 0.4908 | 1.139707 |
| 2 | 2560 | 0.2323 | 189.8847 | 0.0 | 4.9846 | 1.8154 | 0.3692 | 0.8747 | 0.4187 | 1.139707 |
| 3 | 3840 | 0.3092 | 182.7619 | 0.0308 | 4.7538 | 1.8462 | 0.3538 | 0.7727 | 0.4318 | 1.139707 |
| 4 | 5120 | 0.3873 | 183.5648 | 0.0147 | 4.9853 | 1.7647 | 0.6029 | 0.7752 | 0.4002 | 1.139707 |
| 5 | 6400 | 0.7084 | 207.6865 | 0.0139 | 5.25 | 1.3889 | 0.5417 | 0.6773 | 0.3146 | 0.319482 |
| 6 | 7680 | 0.7919 | 202.2442 | 0.0132 | 5.25 | 1.2763 | 0.4079 | 0.6051 | 0.2592 | 0.319482 |
| 7 | 8960 | 0.876 | 213.2908 | 0.0116 | 5.4186 | 1.0 | 0.6744 | 0.5599 | 0.2963 | 0.319482 |
| 8 | 10240 | 0.9613 | 206.6955 | 0.0 | 5.46 | 1.02 | 0.5 | 0.4993 | 0.2113 | 0.319482 |
| 9 | 11520 | 1.0445 | 188.0249 | 0.0111 | 5.4222 | 0.9778 | 0.6 | 0.4153 | 0.2906 | 0.319482 |
| 10 | 12800 | 1.5654 | 194.9039 | 0.0 | 5.5053 | 0.4842 | 0.4211 | 0.4034 | 0.2356 | 0.179978 |
| 11 | 14080 | 1.6817 | 201.5503 | 0.0 | 5.5149 | 0.7525 | 0.396 | 0.3793 | 0.194 | 0.179978 |
| 12 | 15360 | 1.77 | 212.2864 | 0.0 | 5.7 | 0.5 | 0.37 | 0.3505 | 0.1748 | 0.179978 |
| 13 | 16640 | 1.8559 | 204.3713 | 0.0 | 5.5789 | 0.386 | 0.4211 | 0.3065 | 0.1587 | 0.179978 |
| 14 | 17920 | 1.9499 | 228.6136 | 0.0 | 5.8015 | 0.2279 | 0.3603 | 0.2915 | 0.1287 | 0.179978 |
| 15 | 19200 | 2.5379 | 235.796 | 0.0 | 5.8819 | 0.2083 | 0.3542 | 0.2253 | 0.1348 | 0.106525 |
| 16 | 20480 | 2.6193 | 242.6148 | 0.0 | 5.9051 | 0.1392 | 0.1392 | 0.171 | 0.0766 | 0.106525 |
| 17 | 21760 | 2.7023 | 256.4628 | 0.0 | 5.9651 | 0.093 | 0.0349 | 0.1342 | 0.0683 | 0.106525 |
| 18 | 23040 | 2.7793 | 262.7177 | 0.0 | 5.9831 | 0.0734 | 0.0282 | 0.1067 | 0.0447 | 0.106525 |
| 19 | 24320 | 2.8562 | 264.4019 | 0.0 | 6.0 | 0.0751 | 0.0289 | 0.0923 | 0.0434 | 0.106525 |
| 20 | 25600 | 3.4496 | 264.6625 | 0.0 | 5.9888 | 0.118 | 0.0955 | 0.0849 | 0.068 | 0.053431 |
| 21 | 26880 | 3.5297 | 265.1316 | 0.0 | 5.9828 | 0.0862 | 0.023 | 0.074 | 0.0345 | 0.053431 |
| 22 | 28160 | 3.6088 | 269.087 | 0.0 | 6.0 | 0.073 | 0.0225 | 0.0608 | 0.0258 | 0.053431 |
| 23 | 29440 | 3.6877 | 268.092 | 0.0 | 6.0 | 0.05 | 0.0222 | 0.0568 | 0.031 | 0.053431 |
| 24 | 30720 | 3.7664 | 267.5243 | 0.0 | 5.9943 | 0.04 | 0.0571 | 0.0477 | 0.0278 | 0.053431 |
| 25 | 32000 | 4.3557 | 267.1374 | 0.0 | 6.0 | 0.0769 | 0.0275 | 0.0417 | 0.0272 | 0.023915 |
| 26 | 33280 | 4.4353 | 267.784 | 0.0 | 6.0 | 0.0615 | 0.0279 | 0.0378 | 0.0307 | 0.023915 |
| 27 | 34560 | 4.5122 | 268.2621 | 0.0 | 6.0 | 0.0489 | 0.0163 | 0.0337 | 0.0246 | 0.023915 |
| 28 | 35840 | 4.5903 | 268.4321 | 0.0 | 5.9945 | 0.044 | 0.0055 | 0.0322 | 0.0257 | 0.023915 |
| 29 | 37120 | 4.6678 | 268.8575 | 0.0 | 6.0 | 0.0393 | 0.0056 | 0.031 | 0.0232 | 0.023915 |
| 30 | 38400 | 5.2364 | 270.1405 | 0.0 | 6.0 | 0.0444 | 0.0 | 0.0285 | 0.0232 | 0.017714 |
| 31 | 39680 | 5.3171 | 269.5279 | 0.0 | 6.0 | 0.0608 | 0.0055 | 0.0261 | 0.0235 | 0.017714 |
| 32 | 40960 | 5.3954 | 268.9374 | 0.0 | 6.0 | 0.0447 | 0.0168 | 0.0253 | 0.0208 | 0.017714 |
| 33 | 42240 | 5.475 | 267.1037 | 0.0 | 5.9946 | 0.0272 | 0.0326 | 0.0224 | 0.0402 | 0.017714 |
| 34 | 43520 | 5.5539 | 269.2865 | 0.0 | 6.0 | 0.0269 | 0.0161 | 0.0228 | 0.0233 | 0.017714 |
| 35 | 44800 | 6.1636 | 269.5642 | 0.0 | 6.0 | 0.0227 | 0.0057 | 0.0207 | 0.0217 | 0.015214 |
| 36 | 46080 | 6.244 | 270.106 | 0.0 | 6.0 | 0.0222 | 0.0 | 0.0203 | 0.0205 | 0.015214 |
| 37 | 47360 | 6.3231 | 268.6244 | 0.0 | 6.0 | 0.0435 | 0.0054 | 0.0213 | 0.0232 | 0.015214 |
| 38 | 48640 | 6.4017 | 270.1528 | 0.0 | 6.0 | 0.0273 | 0.0 | 0.0177 | 0.0202 | 0.015214 |
| 39 | 49920 | 6.4781 | 268.3355 | 0.0 | 6.0 | 0.0166 | 0.0166 | 0.0168 | 0.0254 | 0.015214 |
| 40 | 51200 | 7.0419 | 269.2456 | 0.0 | 6.0 | 0.0275 | 0.0055 | 0.0168 | 0.0228 | 0.013129 |
