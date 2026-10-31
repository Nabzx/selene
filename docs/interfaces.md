# Workstream Interface Contract

Contract between the three Selene workstreams:

- **Simulator, RL and Rollouts (Nabil):** hand-built physics simulator, vectorised multi-agent environment, MAPPO fleet trainer, shared learned world model, and Monte Carlo rollout engine (`apps/sim/selene_sim/`).
- **Orchestrator (Orchestrator teammate):** Gemini 4 Argon orchestrator for writing the safety spec, rewriting the reward, planning and revising fleet routes, running verifier rule checks to produce a verdict, and reviewing hopper logs and world-model error (`apps/orchestrator/` or `apps/sim/selene_sim/orchestrator.py`).
- **3D View (3D teammate):** no-build Three.js view in `apps/web/` that replays `data/episodes.json` and renders the verifier trajectory cloud.

All modules use the vocabulary in [`CONTEXT.md`](../CONTEXT.md).

---

## 1. Shared conventions

### Coordinate frame
- Local east-north-up (ENU) Cartesian frame in metres, centred at `(0, 0, 0)` on the `20 x 20 km` far-side terrain crop (`200 x 200` cells at `100 m` per cell).
- `x` is east (`[-10000.0, 10000.0] m`), `y` is north (`[-10000.0, 10000.0] m`), `z` is up (surface elevation in metres from SLDEM2015 / `data/terrain.json`).
- Cell indices `(row, col)` map to cell centres:
  - `x_m = (col + 0.5) * 100.0 - 10000.0`
  - `y_m = (row + 0.5) * 100.0 - 10000.0`

### Units
- **Distance and position:** metres (`float32`).
- **Time:** seconds of simulated mission time (`float32`), starting at `t = 0.0`.
- **Angles:** radians (`float32`) in `data/episodes.json` and state vectors (`yaw` in `[0, 2*pi)`). Slope limits in physics constants are documented in both radians and degrees for readability.
- **Fuel and battery:** fractions in `[0.0, 1.0]` (`float32`) across all external interfaces (`RewardState`, rollout traces, `data/episodes.json`). Internal simulator helpers convert from `initial_fuel_kg = 15.0` and `100%` battery capacity.
- **Temperature:** degrees Celsius (`float32`), typically `-225.0 °C` to `-155.0 °C` during lunar night.

### Seed convention
- Every simulator, rollout, and training entry point takes an explicit integer `seed: int` (default `42`) and constructs an isolated `np.random.default_rng(seed)`.
- In `simulate_plan(fleet_plan, n, seed)`, rollout `r` (`0 <= r < n`) draws from a deterministic stream seeded by `np.random.SeedSequence([seed, r])` (or a single `default_rng(seed)` stepped across batch shape `(n, ...)`). Given `(fleet_plan, n, seed)`, any single rollout `r` can be replayed identically.

### Identifiers
- **Hopper ids:** `"h0"`, `"h1"`, `"h2"`, `"h3"`, `"h4"` (4 hoppers in default full mode `h0..h3`, up to 5 hoppers `h0..h4`; 2 hoppers `h0..h1` in `--quick` mode). Array axis order `0..N-1` always matches `["h0", ..., f"h{N-1}"]`.
- **Crater ids:** `"c0"` to `"c11"` (12 craters in default Daedalus crop, up to `"c14"` for 15 craters; `"c0"` to `"c5"` in `--quick` mode). Array axis order `0..C-1` always matches `["c0", ..., f"c{C-1}"]`.

---

## 2. What Simulator and RL supply to the Orchestrator

All functions live in `apps/sim/selene_sim/` and return NumPy arrays in Python or JSON-serialisable dicts over FastAPI (`apps/sim/selene_sim/api.py`). `E` is the number of parallel environments, `N` is `fleet_size` (`4` or `5`), `C` is the number of craters (`12`), `S` is the number of route stages in a fleet plan, and `n` is the number of Monte Carlo rollouts (`1000`).

### 2.1 Reward contract (`selene_sim/reward.py`)

#### State fields readable by the reward (`RewardState`)
The reward function may read only the fields on `RewardState`. Leading dimensions `(...)` are either `(N,)` for a single environment or `(E, N)` for vectorised training (`(C,)` or `(E, C)` for crater arrays; `()` or `(E,)` for fleet-wide flags).

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `pos` | `(..., N, 3)` | `float32` | Hopper ENU position `[x, y, z]` in metres |
| `fuel` | `(..., N)` | `float32` | Remaining fuel fraction in `[0.0, 1.0]` |
| `battery` | `(..., N)` | `float32` | Remaining battery fraction in `[0.0, 1.0]` |
| `temp_c` | `(..., N)` | `float32` | Hopper internal temperature in `°C` |
| `crashed` | `(..., N)` | `bool` | `True` if the hopper has crashed or tipped over |
| `alive` | `(..., N)` | `bool` | `True` if the hopper is active (`~crashed` and `battery > 0`) |
| `at_crater` | `(..., N)` | `int32` | Crater index `0..C-1` if within `0.85 * radius`, else `-1` |
| `unsent_data` | `(..., N)` | `int32` | Count of scanned crater payloads buffered onboard |
| `in_relay_window` | `(...,)` | `bool` | `True` if the relay orbiter is above the horizon |
| `relay_conflict` | `(...,)` | `bool` | `True` if `>= 2` hoppers sent at once or any sent outside a relay window |
| `craters_claimed` | `(..., C)` | `int32` | Hopper index `0..N-1` that claimed crater `c`, or `-1` |
| `craters_done` | `(..., C)` | `bool` | `True` once crater `c` has been scanned |
| `crater_priorities` | `(..., C)` | `float32` | Science priority weight for crater `c` (default `1.0` to `1.5`) |
| `min_peer_distance` | `(..., N)` | `float32` | Horizontal distance in metres to nearest other alive hopper |

#### Signatures
```python
def compute_reward(
    state: RewardState,
    actions: np.ndarray,      # shape (..., N), dtype int32: 0=wait, 1=scan, 2=send, 3..C+2=hop_to_crater[c]
    next_state: RewardState,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Returns:
      rewards: np.ndarray of shape (..., N) and dtype float32 (per-hopper step reward).
      components: dict[str, float] of mean scalar values per reward term across the batch.
    """

def load_reward_code(
    code_str: str,
    summary: str = "",
    episode: int = 0,
) -> None:
    """
    Compiles code_str in a restricted namespace (np, numpy, math, RewardState),
    smoke-tests compute_reward on dummy 1D (N,) and 2D (E, N) inputs for shape and
    finiteness, swaps the active reward function atomically, and records
    {"episode": episode, "summary": summary} in reward_versions.
    Raises ValueError without altering the active reward if validation fails.
    """
```

#### Hot-reloading between training chunks
MAPPO training runs in chunks of `K` updates (for example 25 updates = 50 episodes). At the end of a chunk, the trainer calls `training_summary()`. If the orchestrator decides to rewrite the reward, it passes the new Python source string to `load_reward_code(code_str, summary, episode)`. The next training chunk immediately calls the newly swapped `compute_reward` without restarting the process or losing optimiser state.

#### JSON example (hot-reload payload and response)
```json
{
  "request": {
    "episode": 200,
    "summary": "Increase separation penalty and add relay turn-taking bonus",
    "code_str": "def compute_reward(state, actions, next_state):\n    ..."
  },
  "response": {
    "status": "loaded",
    "episode": 200,
    "sample_components": {
      "scan_reward": 4.2,
      "claim_reward": 1.1,
      "relay_reward": 2.5,
      "fuel_penalty": -0.08,
      "crash_penalty": 0.0,
      "battery_penalty": 0.0,
      "separation_penalty": -0.15,
      "relay_conflict_penalty": 0.0,
      "total_reward": 7.57
    }
  }
}
```

### 2.2 Reward terms list

| Term key | Sign | Meaning |
|---|---|---|
| `claim_reward` | `+` | Hopper reaches and claims a previously unclaimed crater |
| `scan_reward` | `+` | Hopper completes a payload scan at an unscanned crater (weighted by `crater_priorities`) |
| `relay_reward` | `+` | Hopper cleanly transmits buffered scan data during an open relay window with no conflict |
| `fuel_penalty` | `-` | Proportional to fuel fraction burned (`state.fuel - next_state.fuel`) |
| `crash_penalty` | `large -` | Applied on transition `~state.crashed -> next_state.crashed` (steep slope, out of fuel, or out of bounds) |
| `battery_penalty` | `-` | Applied when `next_state.battery < 0.20` (below the 20% reserve limit) |
| `separation_penalty` | `-` | Applied when two alive hoppers come closer than `100.0 m` (`min_peer_distance < 100.0`) |
| `relay_conflict_penalty` | `-` | Applied when `>= 2` hoppers transmit in the same step or any hopper transmits outside a relay window |

### 2.3 Monte Carlo rollout engine (`selene_sim/verify.py`)

#### Signature
```python
def simulate_plan(
    fleet_plan: dict[str, Any],
    n: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Replays fleet_plan n times in parallel in the ground-truth simulator (never the
    world model) with sampled thrust error, landing scatter, slope noise, and dust.
    """
```

#### Input `fleet_plan` format
Each hopper has a starting crater (or `[x, y]` position) and an ordered list of synchronised stage steps (`hop`, `scan`, `send`, `wait`).

```json
{
  "plan_id": "plan_v1",
  "revision": 0,
  "hoppers": [
    {
      "id": "h0",
      "start_pos_xy": [-200.0, -200.0],
      "craters_assigned": ["c0", "c5"],
      "steps": [
        { "stage": 0, "action": "hop", "target_crater": "c0", "target_xy": [1650.0, 450.0], "wait_s": 0.0 },
        { "stage": 1, "action": "scan", "target_crater": "c0", "target_xy": [1650.0, 450.0], "wait_s": 15.0 },
        { "stage": 2, "action": "send", "target_crater": "c0", "target_xy": [1650.0, 450.0], "wait_s": 10.0 }
      ]
    },
    {
      "id": "h1",
      "start_pos_xy": [200.0, 100.0],
      "craters_assigned": ["c1"],
      "steps": [
        { "stage": 0, "action": "hop", "target_crater": "c1", "target_xy": [650.0, 1750.0], "wait_s": 0.0 },
        { "stage": 1, "action": "scan", "target_crater": "c1", "target_xy": [650.0, 1750.0], "wait_s": 15.0 },
        { "stage": 2, "action": "wait", "target_crater": "c1", "target_xy": [650.0, 1750.0], "wait_s": 20.0 }
      ]
    }
  ]
}
```

#### Output `traces` shapes and fields
| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `n_rollouts` | scalar | `int` | Number of Monte Carlo rollouts `n` |
| `seed` | scalar | `int` | Base seed used for the rollout batch |
| `hopper_ids` | `list[str]` | `str` | `["h0", ..., f"h{N-1}"]` |
| `pos` | `(n, S+1, N, 3)` | `float32` | `[x, y, z]` position in metres at start (`0`) and after each stage `1..S` |
| `fuel` | `(n, S+1, N)` | `float32` | Remaining fuel fraction in `[0.0, 1.0]` |
| `battery` | `(n, S+1, N)` | `float32` | Remaining battery fraction in `[0.0, 1.0]` |
| `temp_c` | `(n, S, N)` | `float32` | Hopper temperature in `°C` at each stage |
| `landing_slope_deg` | `(n, S, N)` | `float32` | Surface slope in degrees at landing cell (`landing_slope_rad` also available as `np.radians(landing_slope_deg)`) |
| `hop_time_s` | `(n, S, N)` | `float32` | Ballistic flight duration in seconds for each stage |
| `stage_time_s` | `(n, S+1)` | `float32` | Cumulative mission time in seconds at each stage boundary |
| `min_separation_m` | `(n, S)` | `float32` | Minimum horizontal distance in metres between any two hoppers in stage `s` |
| `pair_distances_m` | `(n, S, N, N)` | `float32` | Pairwise horizontal distance matrix between hoppers (`inf` on diagonal) |
| `collision` | `(n, S)` | `bool` | `True` if `min_separation_m < 100.0` at stage `s` |
| `crashed` | `(n, S, N)` | `bool` | `True` if hopper `h` crashed at or before stage `s` |
| `crash_reason` | `(n, S, N)` | `U16` | `""`, `"out_of_fuel"`, `"steep_slope"`, or `"out_of_bounds"` |
| `in_relay_window` | `(n, S)` | `bool` | `True` if relay orbiter is overhead during stage `s` |
| `relay_conflict` | `(n, S)` | `bool` | `True` if `>= 2` hoppers sent at once or any sent when `~in_relay_window` |
| `cloud_samples` | `list[dict]` | `dict` | Up to 50 subsampled rollouts formatted for the 3D trajectory cloud |

#### JSON summary example (`traces` summary over HTTP)
```json
{
  "n_rollouts": 1000,
  "seed": 42,
  "hopper_ids": ["h0", "h1", "h2", "h3"],
  "num_stages": 3,
  "summary": {
    "crash_rate": 0.004,
    "relay_conflict_rate": 0.0,
    "collision_rate": 0.002,
    "min_fuel": 0.24,
    "p05_fuel": 0.31,
    "min_battery": 0.42,
    "max_landing_slope_deg": 13.8,
    "min_separation_m": 94.2
  }
}
```

### 2.4 Physics constants and limits (`SimConfig`)

| Constant | Field in `SimConfig` | Value | Units / Meaning |
|---|---|---|---|
| Lunar gravity | `g_moon` | `1.62` | `m/s^2` |
| Specific impulse | `isp_s` | `220.0` | `s` (`v_e = 2157.46 m/s`) |
| Dry mass / initial fuel | `dry_mass_kg` / `initial_fuel_kg` | `35.0` / `15.0` | `kg` (`fuel = 1.0` equals `15.0 kg`) |
| Max hop range | `max_hop_range_m` | `3000.0` | `m` per single ballistic hop |
| Launch angle | `launch_angle_deg` | `45.0` (`0.7854 rad`) | Ballistic launch elevation angle |
| Fuel reserve limit | `min_fuel_reserve` | `0.10` | Fraction of initial fuel (`1.5 kg`) required on final landing |
| Battery reserve limit | `min_battery_reserve` | `0.20` | Fraction of battery (`20%`) required throughout mission |
| Safe landing slope | `max_landing_slope_deg` | `15.0` (`0.2618 rad`) | Maximum safe surface slope before tip-over crash |
| Separation distance | `min_separation_m` | `100.0` | `m` minimum horizontal distance between any two hoppers |
| Relay window period | `relay_period_s` | `120.0` | `s` full orbit cycle (`[k*120, k*120 + 40]` is open) |
| Relay open duration | `relay_open_duration_s` | `40.0` | `s` open window duration per period |
| Noise parameters | `thrust_sigma_rel`, `scatter_sigma_rel`, `scatter_sigma_base_m`, `slope_noise_sigma_deg`, `dust_max_extra_burn` | `0.02`, `0.015`, `8.0 m`, `1.0 deg`, `0.03` | Sampled per hop in noisy rollouts |

### 2.5 Crater catalogue (`get_crater_list()`)

#### Signature
```python
def get_crater_list(seed: int = 42) -> list[dict[str, Any]]:
    """Return the crater catalogue with positions, radii, slopes, and hop reachability graph."""
```

#### JSON example
```json
[
  {
    "id": "c0",
    "name": "Hollow 0",
    "pos": [1650.0, 450.0, -142.0],
    "radius_m": 800.0,
    "depth_m": 145.0,
    "floor_slope_deg": 3.2,
    "priority": 1.0,
    "reachable_from_origin": true,
    "reachable_craters": ["c1", "c5", "c6"]
  },
  {
    "id": "c6",
    "name": "Wraith Pit",
    "pos": [3650.0, 1050.0, -161.0],
    "radius_m": 950.0,
    "depth_m": 165.0,
    "floor_slope_deg": 4.1,
    "priority": 1.4,
    "reachable_from_origin": false,
    "reachable_craters": ["c0", "c1", "c7"]
  }
]
```

### 2.6 Training summary (`training_summary()`)

#### Signature
```python
def training_summary() -> dict[str, Any]:
    """Return reward curves, failure rates, reward term breakdown, and sample episodes for Argon."""
```

#### JSON example
```json
{
  "source": "trained",
  "episodes_completed": 400,
  "env_steps": 12000,
  "episode_rewards": [-38.2, -11.4, 14.8, 52.6, 71.3],
  "crash_rate": 0.06,
  "crater_overlap_rate": 0.08,
  "relay_conflict_rate": 0.03,
  "craters_visited_mean": 7.4,
  "reward_components": {
    "scan_reward": 48.0,
    "claim_reward": 12.0,
    "relay_reward": 18.5,
    "fuel_penalty": -3.2,
    "crash_penalty": -3.0,
    "battery_penalty": -0.5,
    "separation_penalty": -0.3,
    "relay_conflict_penalty": -0.2
  },
  "reward_versions": [
    { "episode": 0, "summary": "Baseline reward" },
    { "episode": 200, "summary": "Added relay conflict and separation penalties" }
  ],
  "sample_episodes": [
    {
      "index": 400,
      "outcome": "completed",
      "total_reward": 71.3,
      "craters_visited": ["c0", "c1", "c2", "c3", "c6", "c7"],
      "crashes": 0,
      "relay_conflicts": 0
    }
  ]
}
```

### 2.7 World-model error map (`world_model_error_map()`)

#### Signature
```python
def world_model_error_map() -> dict[str, Any]:
    """
    Return per-crater and per-cell (20x20 coarse grid in the local ENU frame)
    world-model prediction error and ensemble disagreement (epistemic uncertainty).
    """
```

#### Frame and grid layout
- Same local ENU frame (`[-10000.0, 10000.0] m` in `x` and `y`).
- Per-cell arrays have shape `(20, 20)` (`1000 m x 1000 m` bins) indexed `[row_y, col_x]`, plus a `top_error_cells` list with exact `(x_m, y_m)` bin centres.

#### JSON example
```json
{
  "frame": "local_enu_m",
  "bounds_xy_m": [-10000.0, 10000.0],
  "overall_val_mse": 0.021,
  "per_crater": [
    {
      "crater_id": "c0",
      "pos": [1650.0, 450.0, -142.0],
      "visit_count": 148,
      "mean_error_m": 5.8,
      "uncertainty_m": 4.3
    },
    {
      "crater_id": "c8",
      "pos": [-2950.0, 2650.0, -168.0],
      "visit_count": 2,
      "mean_error_m": 39.4,
      "uncertainty_m": 33.1
    }
  ],
  "top_error_cells": [
    {
      "grid_ij": [12, 3],
      "centre_xy_m": [-6500.0, 2500.0],
      "visit_count": 0,
      "mean_error_m": 44.2,
      "uncertainty_m": 36.8,
      "nearest_crater_id": "c8"
    }
  ],
  "grid_20x20": {
    "cell_size_m": 1000.0,
    "mean_error_m": [[12.1, 11.4], [10.8, 9.5]],
    "uncertainty_m": [[9.8, 9.2], [8.7, 7.9]],
    "visit_count": [[14, 18], [22, 31]]
  }
}
```

### 2.8 Policy route suggestion (`suggest_plan()`)

#### Signature
```python
def suggest_plan(state: dict[str, Any] | None = None, seed: int = 42) -> dict[str, Any]:
    """
    Roll out the trained MAPPO policy greedily in the simulator from `state`
    (or default initial fleet state) and return a candidate `fleet_plan` dict
    in the exact format accepted by `simulate_plan()`.
    """
```

---

## 3. What the Orchestrator supplies to Simulator and RL

### 3.1 Safety spec and rule check signature
Argon reads the mission documents and emits a safety spec: a list of rules where each rule has an `id`, a plain-English `statement`, a `severity` (`"hard"`), and a vectorised Python check function `check(trace)`.

#### Signature
```python
def check(trace: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """
    Evaluates one safety rule across all n rollouts in `trace` (output of simulate_plan).

    Returns:
      passed: bool (True if pass_rate >= 0.99 across the n rollouts)
      reason: str (human-readable explanation of failures, or "ok")
      stats: dict with:
        - "rule_id": str
        - "pass_rate": float in [0.0, 1.0]
        - "failed_rollouts": list[int] (indices of failing rollouts, for red cloud lines)
        - "failures_by_hopper": dict[str, int] (e.g. {"h0": 0, "h1": 14})
        - "failures_by_stage": list[int] (length S)
    """
```

#### Core safety rules the traces support
1. **`R1_FUEL_RESERVE`:** `trace["fuel"][:, -1, :] >= 0.10` (every hopper finishes with at least 10% fuel).
2. **`R2_BATTERY_RESERVE`:** `np.all(trace["battery"] >= 0.20, axis=(1, 2))` (battery never drops below 20%).
3. **`R3_LANDING_SLOPE`:** `np.all(trace["landing_slope_deg"] <= 15.0, axis=(1, 2))` and `~np.any(trace["crashed"], axis=(1, 2))` (every landing is on a slope `<= 15 deg` with no crash).
4. **`R4_FLEET_SEPARATION`:** `np.all(trace["min_separation_m"] >= 100.0, axis=1)` (no two hoppers come within `100 m` of each other).
5. **`R5_SINGLE_CHANNEL_RELAY`:** `~np.any(trace["relay_conflict"], axis=1)` (at most one hopper transmits at a time, and only inside an open relay window).

#### Verdict format produced by the verifier
```json
{
  "plan_id": "plan_v1",
  "revision": 0,
  "passed": false,
  "pass_rate": 0.942,
  "required_pass_rate": 0.99,
  "failed_rollout_indices": [12, 47, 103],
  "rule_results": [
    {
      "rule_id": "R1_FUEL_RESERVE",
      "statement": "Each hopper must retain at least 10% fuel reserve after final landing.",
      "passed": true,
      "pass_rate": 0.998,
      "reason": "ok",
      "failures_by_hopper": { "h0": 2, "h1": 0, "h2": 0, "h3": 0 },
      "failures_by_stage": [0, 0, 2]
    },
    {
      "rule_id": "R3_LANDING_SLOPE",
      "statement": "No hopper may land on a surface slope exceeding 15 degrees.",
      "passed": false,
      "pass_rate": 0.944,
      "reason": "Hopper h2 exceeded 15 deg landing slope on stage 1 in 56/1000 rollouts.",
      "failures_by_hopper": { "h0": 0, "h1": 0, "h2": 56, "h3": 0 },
      "failures_by_stage": [0, 56, 0]
    }
  ]
}
```

### 3.2 Format of Argon's fleet plans and revisions
Argon emits fleet plans in the `fleet_plan` JSON format shown in Section 2.3, plus metadata explaining its reasoning and (for revisions) what it changed after a failed verdict:
```json
{
  "plan_id": "plan_v1",
  "revision": 1,
  "previous_verdict_pass_rate": 0.942,
  "rationale": "Reassigned h2 from steep rim crater c8 to flat floor crater c2 after R3_LANDING_SLOPE failed on stage 1; staggered h0 and h1 relay sends across stages 2 and 3.",
  "hoppers": [ ... ]
}
```

### 3.3 When and how a reward rewrite is triggered and handed over
- **Trigger conditions (evaluated between training chunks of 50 episodes):**
  1. **Reward plateau:** mean episode reward improves by less than `5%` over two consecutive chunks.
  2. **High crash or violation rate:** `crash_rate > 0.15`, `crater_overlap_rate > 0.25`, or `relay_conflict_rate > 0.15`.
  3. **Behavioural loophole:** hoppers camp in place (`craters_visited_mean < 2.0`) to avoid fuel or crash penalties.
- **Handover:**
  1. Orchestrator calls `training_summary()` and inspects `episode_rewards`, `reward_components`, and `sample_episodes`.
  2. Argon writes a self-contained Python `compute_reward(state, actions, next_state)` function and a one-line `summary`.
  3. Orchestrator calls `load_reward_code(code_str, summary, episode=current_episode)` (or `POST /reward/reload`).
  4. If `load_reward_code` raises `ValueError`, the error string is returned to Argon in the same interaction (`previous_interaction_id`) for immediate self-correction (up to 2 retries).

---

## 4. What both supply to the 3D view

The 3D view (`apps/web/`) reads `data/episodes.json` (and `data/terrain.json` for surface heights). It has no build step and no direct coupling to Python internals.

### 4.1 Current schema (`docs/data/EPISODE_SCHEMA.md`)
Training writes `meta`, `craters`, `relay_windows`, `training` (including `reward_versions` from Argon's rewrites), and `episodes[].drones[]` with `0.5 s` interpolated flight keyframes (`pos`, `yaw`, `phase`, `fuel`, `battery`, `temp_c`, `action`, `reward`, `wm_pos`, `wm_error`, `event`). Note: `drones[]` is the literal JSON key in `data/episodes.json` for backwards compatibility with `test_episode_schema.py`; UI copy and docs always call them **hoppers**.

### 4.2 Proposed schema addition: `verification` (trajectory cloud and verdict)
To let the 3D view draw the trajectory cloud (green for rollouts that pass every rule in the safety spec, red for rollouts that break at least one rule) and show the uplink approval gate without adding a second file format, we propose adding an optional top-level `"verification"` key to `data/episodes.json` (or serving the same object from `GET /api/verification/latest`):

```json
{
  "verification": {
    "plan_id": "plan_v1",
    "revision": 1,
    "source": "trained",
    "seed": 42,
    "n_rollouts": 1000,
    "verdict": {
      "passed": true,
      "pass_rate": 0.994,
      "required_pass_rate": 0.99,
      "approved_by_human": false,
      "rule_results": [
        {
          "rule_id": "R1_FUEL_RESERVE",
          "statement": "Each hopper must retain at least 10% fuel reserve after final landing.",
          "passed": true,
          "pass_rate": 0.998
        }
      ]
    },
    "trajectory_cloud": [
      {
        "rollout_index": 0,
        "passed": true,
        "failed_rules": [],
        "drones": [
          {
            "id": "h0",
            "passed": true,
            "arcs": [
              [[ -200.0, -200.0, -120.0 ], [ 725.0, 125.0, 290.0 ], [ 1648.2, 451.4, -142.0 ]]
            ]
          }
        ]
      },
      {
        "rollout_index": 12,
        "passed": false,
        "failed_rules": ["R3_LANDING_SLOPE"],
        "drones": [
          {
            "id": "h0",
            "passed": false,
            "crash_pos": [ 1679.5, 488.1, -135.0 ],
            "arcs": [
              [[ -200.0, -200.0, -120.0 ], [ 740.0, 140.0, 295.0 ], [ 1679.5, 488.1, -135.0 ]]
            ]
          }
        ]
      }
    ]
  }
}
```
- Each entry in `trajectory_cloud` represents one sampled rollout (typically 50 rollouts: all failing rollouts up to 25, plus evenly spaced passing rollouts).
- Each hop arc is a polyline of `9` to `15` `[x, y, z]` points in metres so the 3D view can draw smooth ballistic ribbons in green (`passed: true`) or red (`passed: false`) with a single `THREE.Line` per hop.

---

## 5. Stubs for day one

So no workstream blocks another on day one, each boundary has a deterministic stub that returns well-formed data in `< 50 ms`:

| Stub | Where it lives | What it does |
|---|---|---|
| `stub_simulate_plan(fleet_plan, n=1000, seed=42)` | `apps/sim/selene_sim/verify.py` (callable even before full terrain/RL is wired, using `TerrainGrid.synthetic(seed)`) | Runs vectorised closed-form hops with seeded Gaussian landing scatter, thrust noise, and slope checks, returning a complete `traces` dict and 50 `cloud_samples`. |
| `stub_suggest_plan(seed=42)` | `apps/sim/selene_sim/verify.py` | Generates a valid 4-hopper, 3-stage `fleet_plan` assigning hoppers `h0..h3` to distinct inner-ring craters `c0..c3` with staggered `send` steps. |
| `stub_training_summary()` and `stub_world_model_error_map()` | `apps/sim/selene_sim/api.py` | Return well-formed JSON payloads (labelled `"source": "scripted"`) matching Sections 2.6 and 2.7 so the orchestrator can develop and test Argon prompts immediately. |
| `stub_argon_plan_and_spec()` | `apps/sim/selene_sim/stubs.py` (or `apps/orchestrator/stubs.py`) | Returns the 5 baseline safety rules (`R1` to `R5` with executable `check(trace)` functions) and a cached Argon `fleet_plan` (labelled `"source": "scripted"`) so the verifier loop and UI can run offline without waiting on live API calls. |
| `data/episodes.example.json` | `data/episodes.example.json` (already committed and tested by `apps/sim/tests/test_episode_schema.py`) | Lets the 3D view build and test multi-hopper replay immediately. |

---

## 6. Open questions

1. **In-process Python call vs HTTP between Orchestrator and Simulator:** Will the orchestrator live inside `apps/sim/selene_sim/` and pass NumPy `trace` dicts directly in memory to `check(trace)`, or run in a separate process that calls FastAPI JSON endpoints? (In-process avoids serialising `(1000, S, N, 3)` arrays over JSON.)
2. **Crater id naming in `TerrainGrid` vs `EPISODE_SCHEMA.md`:** `EPISODE_SCHEMA.md` uses `"c1".."c4"`, whereas `sim.py` currently uses `"daedalus_a".."daedalus_l"`. Should we standardise on `"c0".."c11"` as `id` everywhere and move `"daedalus_a"` / Halloween names to the display `name` field?
3. **Fuel and battery units in internal dataclasses:** `EPISODE_SCHEMA.md` uses fractions `0.0` to `1.0` for `fuel` and `battery`, while `sim.py` and `reward.py` internally use `fuel_kg` (`0..15 kg`) and `battery_pct` (`0..100%`). Confirming that the boundary functions (`simulate_plan`, `RewardState`, `episodes.json`) all normalise to `[0.0, 1.0]`.
4. **Relay window schedule scale:** `episodes.example.json` has relay windows of `900 s` every `5400 s` (real lunar orbit scale), whereas `SimConfig` uses `40 s` open every `120 s` so multi-hop episodes cross several relay windows within a few minutes of simulated mission time. Which period should the verifier and 3D timeline display?
5. **Trajectory cloud delivery to the 3D view:** Does the 3D teammate prefer reading `verification.trajectory_cloud` from `data/episodes.json`, or fetching it from a dedicated `GET /api/verification/latest` endpoint when a route is checked live?
6. **Payload decision (Issue #2) and Z3 layer (Issue #4):** Does the orchestrator safety spec need a payload-specific rule (for example minimum scan duration or thermal range during scan), and are we keeping Option A (pure Monte Carlo) or adding Z3 checks for fuel reserve in Phase 3?
