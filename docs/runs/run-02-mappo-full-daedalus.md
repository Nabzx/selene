# Run 02: Stage 3 Full Daedalus MAPPO Training (`4` Hoppers, `12` Craters, `80` Updates)

## Overview

This report documents the reproducible Stage 3 full-fleet Centralised Training with Decentralised Execution (CTDE) Multi-Agent PPO (MAPPO) run for a `4`-hopper lunar fleet over the `12`-crater Daedalus far-side crater catalogue (`200 x 200` terrain grid at `100 m/cell`, matching `data/terrain/daedalus_200x200.npz`).

The run trains a shared 2-layer `128`-unit `Tanh` `Actor` and centralised `Critic` across `128` vectorised parallel environments (`307,200` environment steps over `80` updates), periodically fits the 5-MLP `WorldModelEnsemble` with short-horizon Model-Based Policy Optimisation (`imagine_rollout`), logs full telemetry to `data/training_log.csv`, and exports 3 replay episodes (`data/episodes.json`) validated against `docs/data/EPISODE_SCHEMA.md`.

## Reproducible Command Lines

Run from the repository root (`/Users/nabilshah/.gemini/antigravity/scratch/worktrees/run-mappo-training`):

```bash
# 1. Execute full 4-hopper, 12-crater Daedalus MAPPO training (80 updates, seed=42)
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python \
  -m selene_sim.mappo \
  --updates 80 \
  --csv data/training_log.csv \
  --checkpoint-dir "" \
  --export-episodes data/episodes.json

# 2. Print formatted terminal summary from the CSV log
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python \
  apps/sim/scripts/training_summary.py data/training_log.csv

# 3. Validate exported data/episodes.json against EPISODE_SCHEMA.md
PYTHONPATH=apps/sim /Users/nabilshah/Random/selene/apps/sim/.venv/bin/python \
  -c "import json; from pathlib import Path; from tests.test_episode_schema import validate_episode_doc; validate_episode_doc(json.loads(Path('data/episodes.json').read_text()))"
```

## Hardware, Device Selection, and Seed Configuration

| Property | Value | Notes |
|---|---|---|
| Random Seed | `42` | Seeds `torch.manual_seed(42)`, `np.random.seed(42)`, `VectorFleetEnv(seed=42)`, `FleetEnv(seed=42)`, and `WorldModelEnsemble(seed=42)` |
| Terrain Crop | `Daedalus` (`200 x 200`, `100 m/cell`) | 12-crater Daedalus catalogue (`daedalus_a` .. `daedalus_l`) matching `TerrainGrid.from_npz("data/terrain/daedalus_200x200.npz")` |
| `select_device(prefer_fastest=True)` | `mps` | Micro-benchmarks 50 forward/backward steps of a 2-layer 128-unit MLP on CPU vs Apple Metal (`mps`) |
| `resolve_mappo_device(cfg.device)` | `cpu` | Executes `128`-unit Actor/Critic MLPs on CPU alongside contiguous NumPy `VectorFleetEnv` buffers |
| `WorldModelEnsemble` Device | `cpu` | 5-member `DynamicsMLP` ensemble (`128` hidden units) with Randomised Prior Functions (`prior_scale=0.65`) |
| Total Environment Steps | `307,200` | `80` updates x `128` parallel environments x `30` rollout steps (`1,228,800` individual hopper transitions) |
| Wall-Clock Training Time | `44.32 s` | ~`6,931` fleet environment steps/s (`27,725` hopper transitions/s) including PPO + MBPO + WorldModel updates |

## Hyperparameters (`MAPPOConfig(quick=False, total_updates=80)`)

| Hyperparameter | Field | Value |
|---|---|---|
| Quick mode flag | `quick` | `False` |
| Fleet size (hoppers) | `n_hoppers` | `4` |
| Active craters in catalogue | `n_craters` | `12` |
| Episode horizon (decision steps) | `max_steps` | `30` |
| Parallel vectorised environments | `num_envs` | `128` |
| Rollout length per update | `rollout_steps` | `30` |
| Actor / Critic / WM hidden width | `hidden_dim` | `128` |
| Total PPO updates | `total_updates` | `80` |
| Wall-clock time budget | `time_budget_s` | `1200.0` |
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
| Observation dimension | `obs_dim` | `57` (`6` own + `4` ID + `36` crater + `9` peer + `2` relay) |
| Centralised global state dimension | `global_state_dim` | `54` (`28` hopper + `24` crater + `2` relay) |
| Discrete action space per hopper | `n_actions` | `15` (`12` hop targets + `scan`, `wait`, `send`) |

## Verbatim `scripts/training_summary.py` Output

```text
=== Selene MAPPO Training Summary (training_log.csv) ===
Updates: 80 | Env Steps: 307200 | Wall Time: 44.32s
--------------------------------------------------------------------
Metric               |      First 3 |       Last 3 |        Delta
--------------------------------------------------------------------
episode_reward       |     228.3248 |     492.5240 |    +264.1992
crashes              |       0.0469 |       0.0000 |      -0.0469
craters_visited      |       7.7526 |       9.9974 |      +2.2448
crater_overlap       |       4.4635 |       0.0312 |      -4.4323
relay_conflicts      |       2.2422 |       0.6380 |      -1.6042
entropy              |       0.4986 |       0.0662 |      -0.4324
value_loss           |       0.5134 |       0.0710 |      -0.4424
wm_error             |   34401.8477 |       0.0344 |  -34401.8132
--------------------------------------------------------------------
Stage 2 Check (Reward Up & Crashes Low): PASS
Stage 3 Check (Crater Overlap Down):     PASS
Stage 4 Check (World Model Error Down):  PASS
```

## Stage 2, Stage 3, and Episode Export Verification Analysis

1. **Stage 2 Verification (Fleet Reward Scaling and Zero-Crash Convergence):**
   - Mean fleet `episode_reward` climbs from `228.3248` (first 3 updates) to `492.5240` (last 3 updates), an improvement of `+264.1992` (`+115.7%`).
   - Mean fleet `crashes` drops from `0.0469` (first 3 updates) to `0.0000` for every update from update 18 through update 80 (`-0.0469` delta).
   - Mean distinct `craters_visited` per episode increases from `7.7526` to `9.9974` (`10.0` at update 80) within the single-tank propellant budget (`36.0 kg` initial fuel per hopper).
2. **Stage 3 Verification (Multi-Agent Crater Splitting and Relay Coordination):**
   - **Crater Splitting (`crater_overlap`):** Duplicate crater visits across the 4 hoppers drop by `99.3%` from `4.4635` per episode in the first 3 updates to `0.0312` in the last 3 updates (`0.0078` at update 80). Each of the 4 hoppers claims and scans a distinct subset of craters across the Daedalus basin without redundant hops.
   - **Relay Turn-Taking (`relay_conflicts`):** Simultaneous downlink collisions during open relay windows drop by `71.5%` from `2.2422` per episode in the first 3 updates to `0.6380` in the last 3 updates (`0.5000` at update 79), confirming that hoppers batch scan data and stagger their `send` actions.
3. **Stage 4 Verification (5-MLP Ensemble World Model Convergence):**
   - Held-out ensemble validation MSE (`wm_error`) drops from `34401.8477` after the initial 1-epoch warm-up at update 1 to `0.470239` at update 5, `0.126434` at update 20, `0.041784` at update 40, and `0.034444` at update 80.
4. **Exported `data/episodes.json` Verification:**
   - Validated against `docs/data/EPISODE_SCHEMA.md` using `validate_episode_doc` in `apps/sim/tests/test_episode_schema.py`.
   - Contains `schema_version: 1`, `meta.source: "trained"`, `meta.fleet_size: 4`, all `12` Daedalus craters, `80` logged training episode rewards, and `3` full 4-hopper trajectories (`index: 0, 1, 2`) with `0.5 s` parabolic flight keyframes (`ground`, `ascent`, `coast`, `descent`), world model position predictions (`wm_pos`, `wm_error`), and rewards `507.62`, `487.63`, and `507.60` visiting 10 distinct craters per episode with zero crashes.

## Full Milestone Progression Table (`data/training_log.csv`)

| `update` | `env_steps` | `wall_time_s` | `episode_reward` | `crashes` | `craters_visited` | `crater_overlap` | `relay_conflicts` | `entropy` | `value_loss` | `wm_error` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3840 | 0.3562 | 195.0323 | 0.0547 | 7.2656 | 5.1172 | 2.8125 | 0.5251 | 0.6258 | 34401.847656 |
| 2 | 7680 | 0.6417 | 203.8507 | 0.0625 | 7.75 | 4.4609 | 2.2891 | 0.4993 | 0.5081 | 34401.847656 |
| 3 | 11520 | 0.9226 | 286.0914 | 0.0234 | 8.2422 | 3.8125 | 1.625 | 0.4714 | 0.4064 | 34401.847656 |
| 4 | 15360 | 1.2123 | 332.1895 | 0.0234 | 8.6172 | 3.1562 | 1.6016 | 0.4512 | 0.3747 | 34401.847656 |
| 5 | 19200 | 1.9828 | 308.792 | 0.0156 | 8.7656 | 2.7812 | 1.1172 | 0.4082 | 0.4276 | 0.470239 |
| 6 | 23040 | 2.3311 | 313.1155 | 0.0 | 8.7109 | 2.4609 | 1.5 | 0.3737 | 0.3882 | 0.470239 |
| 7 | 26880 | 2.6078 | 328.9723 | 0.0078 | 8.8516 | 2.2266 | 1.3984 | 0.3533 | 0.3788 | 0.470239 |
| 8 | 30720 | 2.882 | 346.1386 | 0.0078 | 9.0859 | 1.9531 | 1.6953 | 0.3335 | 0.3441 | 0.470239 |
| 9 | 34560 | 3.1622 | 337.6577 | 0.0078 | 9.0859 | 1.875 | 2.0078 | 0.3112 | 0.3212 | 0.470239 |
| 10 | 38400 | 4.1277 | 358.6545 | 0.0 | 9.1719 | 1.9062 | 1.6953 | 0.2911 | 0.3173 | 0.299875 |
| 11 | 42240 | 4.399 | 345.5553 | 0.0078 | 9.2656 | 1.8516 | 2.5312 | 0.2828 | 0.3128 | 0.299875 |
| 12 | 46080 | 4.672 | 313.4426 | 0.0 | 9.1641 | 1.7031 | 2.4297 | 0.265 | 0.3643 | 0.299875 |
| 13 | 49920 | 4.9574 | 374.6721 | 0.0078 | 9.2578 | 1.75 | 1.4609 | 0.2482 | 0.2908 | 0.299875 |
| 14 | 53760 | 5.4042 | 389.2775 | 0.0078 | 9.2422 | 1.4766 | 0.8906 | 0.2323 | 0.2423 | 0.299875 |
| 15 | 57600 | 6.5235 | 377.3403 | 0.0 | 9.2422 | 1.25 | 1.6172 | 0.2254 | 0.2464 | 0.196265 |
| 16 | 61440 | 6.7996 | 406.651 | 0.0 | 9.4922 | 0.9766 | 0.8281 | 0.2078 | 0.202 | 0.196265 |
| 17 | 65280 | 7.0772 | 436.8244 | 0.0078 | 9.6797 | 0.7812 | 0.9297 | 0.1896 | 0.1445 | 0.196265 |
| 18 | 69120 | 7.376 | 424.4261 | 0.0 | 9.6094 | 0.75 | 1.0938 | 0.1769 | 0.1554 | 0.196265 |
| 19 | 72960 | 7.6836 | 415.3711 | 0.0 | 9.6016 | 0.8125 | 1.4453 | 0.1625 | 0.1645 | 0.196265 |
| 20 | 76800 | 8.7242 | 425.2076 | 0.0 | 9.75 | 0.6094 | 2.0078 | 0.1697 | 0.1433 | 0.126434 |
| 21 | 80640 | 9.0344 | 422.2622 | 0.0 | 9.7969 | 0.5469 | 2.2891 | 0.1551 | 0.1268 | 0.126434 |
| 22 | 84480 | 9.34 | 426.744 | 0.0 | 9.8828 | 0.2578 | 2.7656 | 0.1554 | 0.1089 | 0.126434 |
| 23 | 88320 | 9.6338 | 449.0928 | 0.0 | 9.9297 | 0.4297 | 1.7266 | 0.1405 | 0.0968 | 0.126434 |
| 24 | 92160 | 9.9124 | 453.0985 | 0.0 | 9.9297 | 0.7031 | 2.0 | 0.1355 | 0.0782 | 0.126434 |
| 25 | 96000 | 10.9711 | 463.4501 | 0.0 | 9.9766 | 0.6562 | 1.6875 | 0.1249 | 0.0789 | 0.069636 |
| 26 | 99840 | 11.2587 | 463.2748 | 0.0 | 9.9297 | 0.8828 | 1.6016 | 0.1249 | 0.0768 | 0.069636 |
| 27 | 103680 | 11.5582 | 437.7744 | 0.0 | 9.9609 | 1.0547 | 2.4922 | 0.1358 | 0.0939 | 0.069636 |
| 28 | 107520 | 11.9772 | 465.5596 | 0.0 | 9.9297 | 1.1172 | 1.4062 | 0.1175 | 0.0834 | 0.069636 |
| 29 | 111360 | 12.2814 | 442.209 | 0.0 | 9.875 | 1.2656 | 2.1016 | 0.1205 | 0.1003 | 0.069636 |
| 30 | 115200 | 13.3188 | 421.0744 | 0.0 | 9.9688 | 1.3359 | 3.2188 | 0.1233 | 0.118 | 0.069636 |
| 31 | 119040 | 13.5977 | 473.637 | 0.0 | 9.9609 | 1.0547 | 1.3281 | 0.104 | 0.0801 | 0.069636 |
| 32 | 122880 | 13.8691 | 470.7979 | 0.0 | 9.9766 | 1.0547 | 1.5391 | 0.0961 | 0.0864 | 0.069636 |
| 33 | 126720 | 14.1615 | 483.5035 | 0.0 | 9.9922 | 0.9922 | 1.2266 | 0.0961 | 0.0613 | 0.069636 |
| 34 | 130560 | 14.458 | 476.1923 | 0.0 | 9.9688 | 1.0391 | 1.4141 | 0.0898 | 0.0729 | 0.069636 |
| 35 | 134400 | 15.6385 | 476.7787 | 0.0 | 9.9844 | 1.0547 | 1.4766 | 0.0812 | 0.0718 | 0.069636 |
| 36 | 138240 | 15.9104 | 487.2209 | 0.0 | 9.9844 | 1.0 | 1.2891 | 0.0821 | 0.0618 | 0.069636 |
| 37 | 142080 | 16.1854 | 489.7571 | 0.0 | 9.9844 | 1.0469 | 0.875 | 0.0773 | 0.0667 | 0.069636 |
| 38 | 145920 | 16.486 | 479.2304 | 0.0 | 9.9922 | 1.0078 | 1.4766 | 0.0773 | 0.0651 | 0.069636 |
| 39 | 149760 | 16.8499 | 491.799 | 0.0 | 9.9766 | 1.0156 | 0.8906 | 0.0733 | 0.0562 | 0.069636 |
| 40 | 153600 | 17.8652 | 493.2446 | 0.0 | 9.9766 | 1.0234 | 0.8594 | 0.0728 | 0.0621 | 0.041784 |
| 41 | 157440 | 18.1357 | 454.6154 | 0.0 | 10.0 | 0.9922 | 2.2422 | 0.0811 | 0.0851 | 0.041784 |
| 42 | 161280 | 18.4042 | 450.5182 | 0.0 | 9.9531 | 0.8672 | 2.1953 | 0.0888 | 0.097 | 0.041784 |
| 43 | 165120 | 18.6728 | 461.9626 | 0.0 | 10.0 | 0.7656 | 1.9844 | 0.0896 | 0.0897 | 0.041784 |
| 44 | 168960 | 18.9542 | 473.4823 | 0.0 | 9.9766 | 0.9141 | 1.4297 | 0.0792 | 0.0854 | 0.041784 |
| 45 | 172800 | 20.3986 | 473.2002 | 0.0 | 9.9766 | 0.5938 | 1.4609 | 0.0873 | 0.0795 | 0.041784 |
| 46 | 176640 | 20.8154 | 481.2225 | 0.0 | 9.9688 | 0.8672 | 1.1875 | 0.0737 | 0.0668 | 0.041784 |
| 47 | 180480 | 21.2256 | 487.8889 | 0.0 | 10.0 | 0.7578 | 0.8906 | 0.0719 | 0.0624 | 0.041784 |
| 48 | 184320 | 21.6175 | 474.27 | 0.0 | 9.9688 | 0.7734 | 1.4297 | 0.0757 | 0.0722 | 0.041784 |
| 49 | 188160 | 22.0545 | 489.8892 | 0.0 | 9.9922 | 0.5625 | 0.8047 | 0.0763 | 0.0621 | 0.041784 |
| 50 | 192000 | 23.4957 | 484.0074 | 0.0 | 10.0 | 0.4766 | 1.0859 | 0.0784 | 0.0664 | 0.041784 |
| 51 | 195840 | 23.8996 | 487.2839 | 0.0 | 9.9922 | 0.3203 | 0.8594 | 0.0783 | 0.0649 | 0.041784 |
| 52 | 199680 | 24.3371 | 475.2598 | 0.0 | 10.0 | 0.4844 | 1.3359 | 0.0776 | 0.0755 | 0.041784 |
| 53 | 203520 | 24.7688 | 480.3115 | 0.0 | 9.9609 | 0.3984 | 1.0312 | 0.0761 | 0.0766 | 0.041784 |
| 54 | 207360 | 25.2807 | 494.2847 | 0.0 | 9.9922 | 0.2422 | 0.4844 | 0.0701 | 0.0696 | 0.041784 |
| 55 | 211200 | 26.9235 | 489.7894 | 0.0 | 9.9844 | 0.1875 | 0.3516 | 0.0786 | 0.0737 | 0.034701 |
| 56 | 215040 | 27.3129 | 494.8396 | 0.0 | 9.9922 | 0.1719 | 0.4297 | 0.0728 | 0.0701 | 0.034701 |
| 57 | 218880 | 27.7901 | 489.852 | 0.0 | 9.9922 | 0.1641 | 0.5938 | 0.0738 | 0.0661 | 0.034701 |
| 58 | 222720 | 28.3432 | 486.8524 | 0.0 | 9.9766 | 0.1719 | 0.8438 | 0.0726 | 0.0682 | 0.034701 |
| 59 | 226560 | 28.8827 | 472.3593 | 0.0 | 10.0 | 0.1016 | 1.2031 | 0.078 | 0.0887 | 0.034701 |
| 60 | 230400 | 30.8174 | 492.9779 | 0.0 | 10.0 | 0.1016 | 0.7031 | 0.0704 | 0.0656 | 0.034701 |
| 61 | 234240 | 31.2415 | 491.8949 | 0.0 | 9.9922 | 0.1328 | 0.4062 | 0.0718 | 0.0674 | 0.034701 |
| 62 | 238080 | 31.7841 | 492.7148 | 0.0 | 10.0 | 0.0391 | 0.6406 | 0.0706 | 0.0671 | 0.034701 |
| 63 | 241920 | 32.2019 | 486.7983 | 0.0 | 9.9922 | 0.1016 | 0.8672 | 0.0714 | 0.0745 | 0.034701 |
| 64 | 245760 | 32.6049 | 494.8183 | 0.0 | 9.9922 | 0.1641 | 0.625 | 0.0736 | 0.0645 | 0.034701 |
| 65 | 249600 | 34.0729 | 496.2129 | 0.0 | 10.0 | 0.3516 | 0.6094 | 0.0729 | 0.0615 | 0.034701 |
| 66 | 253440 | 34.4824 | 486.457 | 0.0 | 9.9766 | 0.3281 | 0.875 | 0.0702 | 0.0722 | 0.034701 |
| 67 | 257280 | 34.8666 | 492.5593 | 0.0 | 10.0 | 0.2812 | 0.6562 | 0.0751 | 0.0654 | 0.034701 |
| 68 | 261120 | 35.285 | 494.3718 | 0.0 | 10.0 | 0.2188 | 0.6406 | 0.0735 | 0.0656 | 0.034701 |
| 69 | 264960 | 35.7143 | 498.7406 | 0.0 | 10.0 | 0.2188 | 0.4766 | 0.0661 | 0.0586 | 0.034701 |
| 70 | 268800 | 37.1762 | 480.9264 | 0.0 | 10.0 | 0.1562 | 1.0156 | 0.0676 | 0.0762 | 0.034701 |
| 71 | 272640 | 37.5923 | 453.8217 | 0.0 | 10.0 | 0.1484 | 1.9766 | 0.072 | 0.1011 | 0.034701 |
| 72 | 276480 | 38.0109 | 481.4179 | 0.0 | 10.0 | 0.1016 | 1.1797 | 0.0703 | 0.0739 | 0.034701 |
| 73 | 280320 | 38.4021 | 492.9061 | 0.0 | 10.0 | 0.1172 | 0.7188 | 0.0689 | 0.0644 | 0.034701 |
| 74 | 284160 | 38.7766 | 479.8428 | 0.0 | 10.0 | 0.0781 | 1.2109 | 0.0691 | 0.0701 | 0.034701 |
| 75 | 288000 | 40.3165 | 490.3859 | 0.0 | 9.9922 | 0.0469 | 0.7891 | 0.0705 | 0.0658 | 0.034444 |
| 76 | 291840 | 40.6867 | 481.9563 | 0.0 | 9.9922 | 0.0703 | 0.9688 | 0.0704 | 0.0795 | 0.034444 |
| 77 | 295680 | 41.1144 | 477.7367 | 0.0 | 9.9922 | 0.0625 | 1.1562 | 0.067 | 0.0819 | 0.034444 |
| 78 | 299520 | 41.6483 | 489.6915 | 0.0 | 10.0 | 0.0312 | 0.7578 | 0.0637 | 0.0709 | 0.034444 |
| 79 | 303360 | 42.3297 | 495.7608 | 0.0 | 9.9922 | 0.0547 | 0.5 | 0.0656 | 0.0718 | 0.034444 |
| 80 | 307200 | 44.3227 | 492.1198 | 0.0 | 10.0 | 0.0078 | 0.6562 | 0.0693 | 0.0703 | 0.034444 |
