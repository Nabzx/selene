# Orchestrator integration contract (`apps/orchestrator` <-> `selene_sim`)

## Overview

In Selene's three-track architecture, `selene_sim` provides the fast numerical simulation, 5-MLP ensemble world model, MAPPO policy trainer, and vectorised Monte Carlo rollout engine, while Friend B's orchestrator (`apps/orchestrator`) runs Gemini 4 Argon (`thinking_level="high"`) for three high-level reasoning loops:
1. **Eureka-style reward evolution:** Inspecting fleet training summaries and failure modes, then hot-swapping `compute_reward` in `selene_sim/reward.py`.
2. **Epistemic uncertainty & mission planning:** Reading `world_model_error_map()` from `selene_sim/world_model.py` to identify under-explored or high-error craters and propose a structured `fleet_plan`.
3. **Safety verification & rule synthesis:** Running `simulate_plan()` in `selene_sim/verify.py` (which calls `sim.py` directly, never the learned world model, per ADR 0003) and checking the returned `(n, H, N)` trace arrays against safety rules.

All cross-module calls are exposed via `apps/sim/selene_sim/api.py` and tested by `apps/sim/tests/test_orchestrator_contract.py`.

## 1. Eureka reward hot-swap loop (`reward.py`)

When Argon rewrites the fleet reward to reduce crashes, eliminate crater overlap, or resolve relay contention, it passes a Python source string defining `compute_reward(state, actions, next_state)` to `load_reward_code(code_str, summary, episode)`.

### Contract invariants
- **Vectorised shape support:** `compute_reward` must accept both single-environment `RewardState` arrays of shape `(N, ...)` and vectorised `VectorFleetEnv` batches of shape `(E, N, ...)` without Python loops over environments.
- **Return signature:** Must return `(rewards, components)` where:
  - `rewards` is a `float32` NumPy array of shape `(..., N)` with finite values (`np.all(np.isfinite(rewards))`).
  - `components` is a `dict[str, float]` mapping human-readable reward term names (e.g. `"scan_bonus"`, `"claim_bonus"`, `"crash_penalty"`, `"overlap_penalty"`, `"relay_bonus"`, `"relay_conflict_penalty"`, `"fuel_cost"`, `"spread_bonus"`) to scalar Python `float` means across the batch.
- **Sandboxed compilation & atomic rollback:** `load_reward_code()` compiles `code_str` in a restricted global namespace containing only `np`, `numpy`, `math`, and `RewardState` (blocking `__import__`, `open`, `eval`, `exec`, `os`, `sys`, `subprocess`). It immediately executes a dry-run smoke test on synthetic `(N=4)` and `(E=8, N=4)` dummy `RewardState` instances. If compilation, shape validation, or finiteness checks fail, the previous active reward function remains untouched and a descriptive `ValueError` is raised so Argon can self-correct.

## 2. World model epistemic uncertainty payload (`world_model_error_map()`)

Argon queries `world_model_error_map()` to inspect where the 5-MLP ensemble (`SharedWorldModel`) has high epistemic disagreement ($\sigma_{\text{pos}}$) or high realised prediction error against ground-truth physics.

### JSON-serialisable payload schema
```python
{
    "source": "learned",  # or "untrained" before transitions are collected
    "total_transitions": 24000,
    "train_epochs": 25,
    "val_mse_history": [0.184, 0.092, 0.041, 0.019],
    "mean_error_m": 14.2,
    "mean_uncertainty_m": 11.8,
    "per_crater": [
        {
            "crater_id": "c0",
            "index": 0,
            "pos_m": [3400.0, 5100.0, 112.0],
            "radius_m": 650.0,
            "visit_count": 142,
            "mean_error_m": 6.4,
            "uncertainty_m": 5.1,
            "predicted_crash_prob": 0.02
        },
        # ... one entry per crater c0..c{C-1}
    ],
    "hotspots": [
        # Top-5 craters sorted by descending uncertainty_m / mean_error_m
        {"crater_id": "c7", "index": 7, "visit_count": 4, "mean_error_m": 38.5, "uncertainty_m": 31.2}
    ],
    "grid_error_summary": {
        "shape": [200, 200],
        "max_error_m": 52.1,
        "p90_uncertainty_m": 24.6
    }
}
```
- **Monotonicity invariant for Argon:** Less-visited craters (`visit_count < 10`) and steep rim craters exhibit systematically higher `uncertainty_m` than well-visited flat floor craters (`visit_count > 50`), giving Argon a clear signal to schedule cautious scouting hops or extra fuel margins.

## 3. Verifier Monte Carlo trace arrays (`simulate_plan()`)

When Argon proposes a multi-hopper route plan (or evaluates a policy rollout plan from `training_summary()["policy_rollout_plan"]`), the verifier calls `simulate_plan(fleet_plan, n=1000, seed=42)`.

### Input `fleet_plan` schema
```python
{
    "hoppers": [
        {
            "id": "h0",
            "start_crater": 0,
            "steps": [
                {"action": "scan"},
                {"action": "hop", "target_crater": 3},
                {"action": "scan"},
                {"action": "send"}
            ]
        },
        # ... h1..h{N-1}
    ]
}
```

### Output trace dictionary for vectorised safety rules
`simulate_plan()` steps all `n = 1000` disturbed rollouts simultaneously as `(n, H, N)` NumPy arrays in under $2\text{ s}$ on CPU (well inside the $10\text{ s}$ ceiling):
- `fuel`: `(n, H, N)` `float32` -> enables rule check `np.mean( np.any(trace["fuel"] < 0.10, axis=(1, 2)) )` (fuel reserve violation rate).
- `temp_c`: `(n, H, N)` `float32` -> enables thermal survival check (`temp_c < -55.0`).
- `landing_slope_deg`: `(n, H, N)` `float32` and `crashed`: `(n, H, N)` `bool` -> enables rim landing stability check (`landing_slope_deg > 15.0`).
- `min_separation_m`: `(n, H)` `float32` -> enables fleet collision proximity check (`min_separation_m < 100.0`).
- `relay_conflict`: `(n, H)` `bool` -> enables single-channel orbiter uplink contention check.
- `cloud_samples`: `list[dict]` of 50 representative 3D trajectory arcs per hopper for rendering the uncertainty cloud in the 3D view.

## 4. Training summary payload (`training_summary()`)

`training_summary()` reads the latest `data/training_log.csv` and trainer state to return a compact JSON-serialisable dictionary for Argon's reflection prompt:
- `source`: `"learned" | "heuristic_fallback" | "scripted"`
- `updates_completed`: `int`, `env_steps`: `int`, `elapsed_s`: `float`
- `episode_reward_start`: `float`, `episode_reward_latest`: `float`
- `crash_rate_start`: `float`, `crash_rate_latest`: `float`
- `craters_visited_mean`: `float`, `crater_overlap_rate`: `float`, `relay_conflict_rate`: `float`
- `reward_components`: `dict[str, float]`
- `reward_versions`: `list[dict]` (`[{"episode": 0, "summary": "baseline"}, ...]`)
- `policy_rollout_plan`: `dict` (greedy/sampled fleet plan extracted from the current actor policy for direct input into `simulate_plan()`).

## Recommendation

Enforce this four-part contract via `apps/sim/tests/test_orchestrator_contract.py` so Friend B's orchestrator and the simulation/RL track can integrate with zero schema mismatches.
