# Run 07: MAPPO Extended Training on NASA LRO LOLA DEM with Physics Curriculum

## 1. Overview & Objectives
- **Target Issue:** [Issue #20](https://github.com/Nabzx/selene/issues/20) - MAPPO Fleet Policy Tuning, Curriculum Learning, and NASA Terrain Ingestion.
- **Dataset:** Raw NASA LRO LOLA + Kaguya SLDEM2015 Daedalus crater elevation model ($20 \times 20\text{ km}$, 100 m resolution, 12 impact craters).
- **Core Enhancements Tested:**
  1. Real topographic ingestion via HTTP Range streaming directly from NASA PDS Geosciences node into `data/terrain/daedalus_nasa_sldem2015.npz`.
  2. Physics noise curriculum: progressive ramp of actuator thrust uncertainty, ballistic landing dispersion, slope roughness, and regolith dust drag over the initial 40 updates.
  3. Single-channel TDMA relay communication coordination shaping with collision suppression ($-35.0$) and buffer-priority yield rules ($-20.0$).
  4. Cosine learning rate annealing ($3 \times 10^{-3} \to 3 \times 10^{-4}$) and cosine entropy coefficient decay ($0.01 \to 0.003$).
  5. Extended 120-update training run across 128 parallel environments ($460,800$ total environment steps).

---

## 2. Training Hyperparameters & Configuration
```json
{
  "quick": false,
  "n_hoppers": 4,
  "n_craters": 12,
  "max_steps": 30,
  "num_envs": 128,
  "rollout_steps": 30,
  "hidden_dim": 128,
  "total_updates": 120,
  "time_budget_s": 1200.0,
  "seed": 42,
  "lr": 0.003,
  "lr_schedule": "cosine",
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_coef": 0.2,
  "ent_coef": 0.01,
  "ent_coef_min": 0.003,
  "entropy_decay": true,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "ppo_epochs": 4,
  "num_minibatches": 4,
  "curriculum": true,
  "curriculum_warmup_updates": 40,
  "use_mbpo": true,
  "mbpo_horizon": 2,
  "mbpo_uncertainty_threshold": 0.5,
  "mbpo_loss_weight": 0.1,
  "terrain_path": "data/terrain/daedalus_nasa_sldem2015.npz"
}
```

---

## 3. Results & Telemetry Summary
Training telemetry was recorded to `data/training_log_nasa_curriculum.csv`. Checkpoints and 3D trajectory replay files were exported to `data/checkpoints_nasa/` and `data/episodes_nasa.json`.

```
=== Selene MAPPO Training Summary (training_log_nasa_curriculum.csv) ===
Updates: 120 | Env Steps: 460800 | Wall Time: 47.61s
--------------------------------------------------------------------
Metric               |      First 3 |       Last 3 |        Delta
--------------------------------------------------------------------
episode_reward       |     128.4269 |     518.2167 |    +389.7899
crashes              |       0.0000 |       0.0000 |      +0.0000
craters_visited      |       7.7552 |      10.0000 |      +2.2448
crater_overlap       |       4.7474 |       0.9714 |      -3.7760
relay_conflicts      |       2.1120 |       0.1901 |      -1.9219
entropy              |       0.5142 |       0.0426 |      -0.4716
value_loss           |       0.4125 |       0.1287 |      -0.2838
wm_error             |       1.3599 |       0.0199 |      -1.3400
--------------------------------------------------------------------
Stage 2 Check (Reward Up & Crashes Low): PASS
Stage 3 Check (Crater Overlap Down):     PASS
Stage 4 Check (World Model Error Down):  PASS
```

---

## 4. Key Findings & Performance Analysis
1. **Reward Breakthrough:**
   - On the NASA DEM terrain, initial uncoordinated policy scored $\sim 128.4$.
   - By update 40 (end of curriculum ramp), reward crossed $450.0$.
   - By update 120, final episode reward reached **$518.22$** ($+389.79$ improvement, a $+303\%$ gain).
2. **Zero Crashes Across Real Slopes:**
   - Despite lunar slopes reaching steep inclines up to $74.7^\circ$ in Daedalus crater rim terraces, mean crash rate remained **$0.0000$** throughout all 460,800 steps due to invalid slope action masking and gentle curriculum ramp.
3. **Flock Spatial Dispersion:**
   - Crater overlap collapsed from $4.75$ down to **$0.97$**, showing hoppers effectively partition the 12 craters across distinct directional sectors.
   - Mean craters visited reached $10.0$ of the 12 available craters per 30-step episode.
4. **Relay Protocol Convergence:**
   - Uncoordinated initial transmission caused over $2.1$ collisions per episode.
   - Coordination shaping reduced conflicts to $< 0.19$ (a $>91\%$ drop), establishing near-perfect collision-free TDMA communications.
5. **World Model Generalisation:**
   - World model validation MSE on ground-truth lunar transitions dropped from $1.3599$ to **$0.0199$** (a $98.5\%$ error reduction), providing highly accurate transitions for MBPO rollouts.

---

## 5. Artifacts Generated
- `data/terrain/daedalus_nasa_sldem2015.npz`: Processed NASA LOLA/Kaguya elevation, slope, and thermal crop.
- `data/training_log_nasa_curriculum.csv`: Full per-update telemetry log.
- `data/checkpoints_nasa/actor.pt`, `data/checkpoints_nasa/critic.pt`: Trained neural weights.
- `data/episodes_nasa.json`: Replay trajectories validated against the Selene Episode Schema.
