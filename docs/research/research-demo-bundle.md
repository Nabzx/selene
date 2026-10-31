# Demo bundle & reliability design (`scripts/export_demo_bundle.py`)

## Overview

In a 3-minute hackathon pitch, waiting on live reinforcement learning training or multi-minute Monte Carlo simulations on stage is a critical failure risk. At the same time, our engineering rules require strict honesty about what is genuinely learned versus scripted (`source: "learned" | "heuristic_fallback" | "scripted"`).

To guarantee an instant, zero-latency demo while showcasing genuine MAPPO policy learning, 5-MLP world-model uncertainty reduction, Eureka reward evolution, and 1,000-rollout Monte Carlo verification, we generate a self-contained **demo bundle** in `data/` via `apps/sim/scripts/export_demo_bundle.py`.

## Artefacts in the pre-baked demo bundle

| File path | Consumer | Contents |
|---|---|---|
| `data/episodes.json` | Friend A (`apps/web` 3D viewer) & Friend B (`apps/orchestrator`) | Multi-episode progression array (`ep_001` early high-crash uncoordinated rollout, `ep_008` mid-training after Argon reward rewrite, `ep_020` late cooperative fleet rollout with zero crashes and full crater split) interpolated at $\Delta t = 0.5\text{ s}$ keyframes. |
| `data/training_log.csv` | `scripts/training_summary.py`, Argon reflection, & HUD charts | Complete update-by-update CSV log (`update`, `env_steps`, `wall_time_s`, `episode_reward`, `crashes`, `craters_visited`, `crater_overlap`, `relay_conflicts`, `entropy`, `value_loss`, `wm_error`). |
| `data/world_model_map.json` | Argon planner & 3D uncertainty overlay | Output of `world_model_error_map()` showing low error on visited floor craters and high ensemble disagreement ($\sigma_{\text{pos}}$) on unvisited steep rim craters. |
| `data/verification_cloud.json` | Friend A 3D trajectory fan & Friend B verifier | Output summary and 50-rollout 3D `[x, y, z]` trajectory cloud from `simulate_plan(policy_rollout_plan, n=1000, seed=42)`, contrasting a rejected risky plan (`pass: false`, 18.4% rim crash rate) against the verified safe plan (`pass: true`, 0.4% failure rate). |

## Narrative arc across exported episodes in `data/episodes.json`

The 3D timeline scrubbing bar in `apps/web` lets the presenter switch directly between three canonical episodes exported from checkpoints along the training run:

1. **Episode 1 (`ep_001_early`, checkpoint update 1):**
   - **Behaviour:** Uncoordinated exploration. Two hoppers hop toward the same nearby crater (`crater_overlap > 0`), one hopper attempts a long hop onto a steep crater rim wall ($\text{slope} > 15^\circ$) and crashes (`crashed: true`), and two hoppers transmit simultaneously during the relay window (`relay_conflict: true`).
   - **World model:** High prediction error (`wm_error` $25\text{ m}$ to $45\text{ m}$) as the 5-MLP ensemble has seen almost no transitions.
2. **Episode 8 (`ep_008_reward_swap`, after Argon Eureka reward hot-swap):**
   - **Behaviour:** Argon observes the crash rate and crater overlap in `training_summary()` and loads an updated reward function (`load_reward_code`) adding an explicit steep-slope margin penalty and unclaimed-crater claim bonus. Crash rate drops sharply, though minor relay contention remains.
   - **World model:** Prediction error falls to $10\text{ m}$ to $18\text{ m}$ on floor craters while remaining elevated on unvisited outer rims.
3. **Episode 20 (`ep_020_converged`, final checkpoint):**
   - **Behaviour:** All hoppers partition the Daedalus crater targets cleanly with zero overlap (`crater_overlap == 0`), maintain fuel reserves above $15\%$, stagger their orbiter transmissions across relay windows (`relay_conflicts == 0`), and complete the mission with zero crashes.
   - **World model:** Low prediction error ($< 6\text{ m}$) along the fleet's trajectories.

## Honest `source` tagging contract

Every episode in `data/episodes.json` and every summary dictionary carries an explicit provenance tag:
- `"source": "learned"`: Produced by rolling out trained `Actor` weights (`mappo.py`) and `SharedWorldModel` weights (`world_model.py`) in `LunarSimulator`.
- `"source": "heuristic_fallback"`: Produced if a user runs the exporter without checkpoints and selects the greedy heuristic planner fallback.
- `"source": "scripted"`: Reserved strictly for static schema fixtures (such as `data/episodes.example.json`).

`scripts/export_demo_bundle.py` trains (or loads a trained checkpoint from `data/checkpoints/`), runs deterministic evaluation rollouts in `sim.py`, interpolates $60\text{ s}$ decision hops into smooth $0.5\text{ s}$ parabolic keyframes (`z(t)` ballistic arc above terrain height), validates the output against `apps/sim/tests/test_episode_schema.py`, and writes all four bundle files in under $60\text{ s}$ (or $< 5\text{ s}$ when exporting from saved checkpoints).

## Recommendation

Build `apps/sim/scripts/export_demo_bundle.py` with both `--from-checkpoints` (fast $< 5\text{ s}$ export from trained weights) and `--train-quick` (end-to-end 2-minute training + export) modes, and validate all generated files in CI so the live demo is always backed by verified, schema-compliant data.
