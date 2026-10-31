# Argon Orchestrator Handbook

Everything needed to build and integrate the Gemini 4 Argon orchestrator workstream in Selene.

---

## 1. Project context and non-negotiable invariants

Selene trains a fleet of four to five rocket-propelled lunar hoppers to explore the far side of the Moon during lunar night. Because the far side never faces Earth, all communication goes through a single-channel relay orbiter during periodic relay windows.

### Core invariants (`AGENTS.md` and `docs/adr/`)
1. **Argon is the single orchestrator (`ADR-0002`, `ADR-0008`):** Only `gemini-4-argon` plans, reads mission documents, writes and rewrites the reward, revises routes after failed checks, and reviews world-model errors. Do not delegate any of these five jobs to a smaller model or a non-Gemini model.
2. **Simulator and world model stay separate (`ADR-0006`):**
   - The **simulator** (`selene_sim/sim.py`) is hand-built ground truth (SLDEM2015 terrain, ballistic hop kinematics at `1.62 m/s^2`, rocket equation fuel burn, night thermal battery drain, relay windows).
   - The **world model** (`selene_sim/world_model.py`) is a learned 5-MLP neural ensemble trained on simulator transitions. It is allowed to be wrong on unvisited terrain.
   - The **verifier** (`selene_sim/verify.py`) always runs rollouts in the ground-truth simulator, never in the learned world model.
3. **Nothing uplinks without a pass and a person (`ADR-0003`):** A fleet plan reaches flight execution only after the verifier passes it (`>= 99%` pass rate across `1000` noisy rollouts) **and** a human operator presses approve during an open relay window.
4. **Scripted is labelled scripted:** Any cached Argon response used as a fallback for demo reliability must carry `"source": "scripted"` (or `"cached": true`) in the payload and on screen.
5. **Vocabulary (`CONTEXT.md`):**
   - Use **far side** (never *dark side*, except as a deliberate Halloween joke in UI copy).
   - Use **lunar night**, **hopper** (never *quadcopter* or *rover*), **fleet**, **orchestrator** (never *commander agent* or *planner agent*), **MAPPO**, **fleet plan**, **hop**, **route**, **simulator**, **world model**, **world-model error**, **relay window**, **safety spec**, **rollout**, **episode**, **trajectory cloud**, **verdict**, **uplink**, **payload**.
6. **House style:** British English (`vectorised`, `centred`, `metres`, `catalogue`, `normalised`, `optimiser`, `behaviour`, `labelled`), no em dashes, short and direct. Every commit is authored solely by Nabil Shah with no AI co-author trailers.

---

## 2. Gemini 4 Argon and Interactions API rules

All model calls must follow the repository's `.agents/skills/gemini-api-dev/SKILL.md` rules:

- **SDK version:** `google-genai >= 2.30.1` (`from google import genai`). Never use the deprecated `google-generativeai` package.
- **Model string:** `"gemini-4-argon"`.
- **API method:** Always use `client.interactions.create(...)`.
- **Forbidden legacy parameters:** Never write `client.models.generate_content`, `thinking_budget`, `temperature`, `top_p`, `top_k`, or `types.GenerateContentConfig`.
- **Thinking levels:** Pass `generation_config={"thinking_level": "high"}` (or `"medium"` / `"low"`) as a plain dict:
  - `"high"` for reading mission documents and synthesising executable safety rules.
  - `"medium"` for reward design/rewrites, fleet trajectory planning, verification loop revisions, and world-model error review.
- **Multi-turn state:** Chain turns using `previous_interaction_id=prev.id` (for example when feeding a verifier failure verdict or a `load_reward_code` validation error back to Argon). Remember that `system_instruction`, `tools`, and `generation_config` are interaction-scoped and must be passed on each turn.
- **Background execution for long calls:** Start long calls with `background=True` and poll `client.interactions.get(interaction.id)` until `interaction.status == "completed"`.

### Canonical Python patterns (`google-genai >= 2.30.1`)

```python
import time
from google import genai

client = genai.Client()

# 1. Standard synchronous interaction with Gemini 4 Argon
interaction = client.interactions.create(
    model="gemini-4-argon",
    system_instruction="You are the Selene mission orchestrator.",
    input="Write the fleet safety specification from the mission constraints.",
    generation_config={
        "thinking_level": "high",
        "max_output_tokens": 32768,
    },
)
print(interaction.output_text)

# 2. Multi-turn revision using previous_interaction_id
revision = client.interactions.create(
    model="gemini-4-argon",
    system_instruction="You are the Selene mission orchestrator.",
    input="Rule R3_LANDING_SLOPE failed on stage 1 for h2 in 56/1000 rollouts. Revise the fleet plan.",
    previous_interaction_id=interaction.id,
    generation_config={
        "thinking_level": "medium",
        "max_output_tokens": 32768,
    },
)
print(revision.output_text)

# 3. Background call with polling (for live demo latency hiding)
bg = client.interactions.create(
    model="gemini-4-argon",
    input="Plan a 4-hopper route covering craters c0..c7 with zero relay conflicts.",
    background=True,
    generation_config={
        "thinking_level": "medium",
        "max_output_tokens": 32768,
    },
)
while bg.status in ("in_progress", "queued"):
    time.sleep(2.0)
    bg = client.interactions.get(bg.id)
if bg.status == "completed":
    print(bg.output_text)
```

---

## 3. The five orchestrator jobs

| # | Job | Thinking level | Input from Simulator / RL | Output from Orchestrator |
|---|---|---|---|---|
| 1 | **Safety spec synthesis** | `high` | Mission constraints doc + `SimConfig` limits | List of safety rules (`id`, `statement`, executable `check(trace)` function) |
| 2 | **Reward design and rewrite** | `medium` | `training_summary()` (`episode_rewards`, `reward_components`, failure rates, `sample_episodes`) | Python source string defining `compute_reward(state, actions, next_state)` + `summary`, passed to `load_reward_code()` |
| 3 | **Fleet trajectory planning** | `medium` | `get_crater_list()`, `SimConfig`, `world_model_error_map()`, optional `suggest_plan()` | Structured `fleet_plan` JSON assigning hoppers `h0..h3` to craters and stage actions |
| 4 | **Verification loop and revision** | `medium` | `simulate_plan(fleet_plan, n=1000, seed=42)` traces + `verdict` from running `check(trace)` on each safety rule | Revised `fleet_plan` (if `pass_rate < 0.99`) + `rationale`, repeated until passing |
| 5 | **Log and world-model review** | `medium` | `world_model_error_map()` + episode step logs (`pos` vs `wm_pos`, `wm_error`) | Plain-English diagnosis of where the world model is wrong (unvisited steep rims) and how the route accounts for it |

---

## 4. Shared conventions

### Coordinate frame
- Local east-north-up (ENU) Cartesian frame in metres, centred at `(0, 0, 0)` on the `20 x 20 km` far-side terrain crop (`200 x 200` cells at `100 m` per cell).
- `x` is east (`[-10000.0, 10000.0] m`), `y` is north (`[-10000.0, 10000.0] m`), `z` is up (surface elevation in metres from SLDEM2015 / `data/terrain.json`).
- Grid cell `(row, col)` maps to cell centre:
  - `x_m = (col + 0.5) * 100.0 - 10000.0`
  - `y_m = (row + 0.5) * 100.0 - 10000.0`

### Units
- **Distance and position:** metres (`float32`).
- **Time:** seconds of simulated mission time (`float32`), starting at `t = 0.0`.
- **Angles:** radians (`float32`) in `data/episodes.json` and state vectors (`yaw` in `[0, 2*pi)`). Slope limits in physics constants are given in both radians and degrees.
- **Fuel and battery:** fractions in `[0.0, 1.0]` (`float32`) across all external interfaces (`RewardState`, rollout traces, `data/episodes.json`). Internal simulator helpers convert from `initial_fuel_kg = 15.0` and `100%` battery capacity.
- **Temperature:** degrees Celsius (`float32`), typically `-225.0 °C` to `-155.0 °C` during lunar night.

### Seed convention
- Every simulator, rollout, and training entry point takes an explicit integer `seed: int` (default `42`) and constructs an isolated `np.random.default_rng(seed)`.
- Given `(fleet_plan, n, seed)`, `simulate_plan(fleet_plan, n, seed)` is 100% deterministic, and any individual rollout `r` (`0 <= r < n`) can be replayed identically.

### Identifiers
- **Hopper ids:** `"h0"`, `"h1"`, `"h2"`, `"h3"`, `"h4"` (4 hoppers `h0..h3` in default full mode, up to 5 hoppers `h0..h4`; 2 hoppers `h0..h1` in `--quick` mode). Array axis `0..N-1` always matches `["h0", ..., f"h{N-1}"]`.
- **Crater ids:** `"c0"` to `"c11"` (12 craters in default Daedalus crop, up to `"c14"` for 15 craters; `"c0"` to `"c5"` in `--quick` mode). Array axis `0..C-1` always matches `["c0", ..., f"c{C-1}"]`.

---

## 5. What Simulator and RL supply to the Orchestrator

All functions live in `apps/sim/selene_sim/`. `E` is the number of parallel environments, `N` is `fleet_size` (`4` or `5`), `C` is the number of craters (`12`), `S` is the number of route stages in a fleet plan, and `n` is the number of Monte Carlo rollouts (`1000`).

### 5.1 Reward contract (`apps/sim/selene_sim/reward.py`)

#### State fields readable by the reward (`RewardState`)
The reward function may read only the attributes of `RewardState`. Leading dimensions `(...)` are either `(N,)` for a single environment or `(E, N)` for vectorised training (`(C,)` or `(E, C)` for crater arrays; `()` or `(E,)` for fleet-wide flags). Never use Python loops over environments `E`; use vectorised NumPy operations.

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
    finiteness, swaps the active reward function atomically, and appends
    {"episode": episode, "summary": summary} to reward_versions.
    Raises ValueError without altering the active reward if validation fails.
    """
```

#### How hot-reloading works between training chunks
1. MAPPO training runs in chunks of `K` updates (for example 50 episodes per chunk).
2. Between chunks, the orchestrator calls `training_summary()` and checks whether learning has stalled or a constraint is being violated:
   - **Plateau:** mean episode reward improves by `< 5%` over two consecutive chunks.
   - **High violation rate:** `crash_rate > 0.15`, `crater_overlap_rate > 0.25`, or `relay_conflict_rate > 0.15`.
   - **Loophole:** hoppers sit still (`craters_visited_mean < 2.0`) to avoid fuel burn.
3. Argon writes a new `def compute_reward(state, actions, next_state): ...` string and calls `load_reward_code(code_str, summary, episode)`.
4. If `load_reward_code` raises a `ValueError` (syntax error, wrong output shape, or `NaN`), pass the error message back to Argon via `previous_interaction_id` for immediate self-correction (up to 2 retries).

#### Reward terms list
| Term key | Sign | Meaning |
|---|---|---|
| `claim_reward` | `+` | Hopper reaches and claims a previously unclaimed crater |
| `scan_reward` | `+` | Hopper completes a payload scan at an unscanned crater (weighted by `crater_priorities`) |
| `relay_reward` | `+` | Hopper cleanly transmits buffered scan data during an open relay window with no conflict |
| `fuel_penalty` | `-` | Proportional to fuel fraction burned (`state.fuel - next_state.fuel`) |
| `crash_penalty` | `large -` | Applied on transition `~state.crashed -> next_state.crashed` |
| `battery_penalty` | `-` | Applied when `next_state.battery < 0.20` (below the 20% reserve limit) |
| `separation_penalty` | `-` | Applied when two alive hoppers come closer than `100.0 m` (`min_peer_distance < 100.0`) |
| `relay_conflict_penalty` | `-` | Applied when `>= 2` hoppers transmit in the same step or any hopper transmits outside a relay window |

---

### 5.2 Monte Carlo rollout engine (`apps/sim/selene_sim/verify.py`)

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
    Completes 1000 rollouts in < 1 s on CPU.
    """
```

#### Input `fleet_plan` JSON format
```json
{
  "plan_id": "plan_v1",
  "revision": 0,
  "rationale": "Assign inner-ring craters c0..c3 to h0..h3 and stagger relay transmissions across stages 2..5.",
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

#### Output `traces` fields and shapes
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

---

### 5.3 Physics constants and limits (`SimConfig`)

Use these exact numbers in prompts so Argon's plans and safety rules match the simulator:

| Constant | Field in `SimConfig` | Value | Units / Meaning |
|---|---|---|---|
| Lunar gravity | `g_moon` | `1.62` | `m/s^2` |
| Specific impulse | `isp_s` | `220.0` | `s` (`v_e = 2157.46 m/s`) |
| Dry mass / initial fuel | `dry_mass_kg` / `initial_fuel_kg` | `35.0` / `15.0` | `kg` (`fuel = 1.0` equals `15.0 kg`) |
| Max hop range | `max_hop_range_m` | `3000.0` | `m` per single ballistic hop |
| Launch angle | `launch_angle_deg` | `45.0` (`0.7854 rad`) | Ballistic launch elevation angle |
| Gravity-loss margin | `gravity_loss_margin` | `0.08` | `Delta v = (v_0 + v_land) * 1.08` |
| Fuel reserve limit | `min_fuel_reserve` | `0.10` | Fraction of initial fuel (`1.5 kg`) required on final landing |
| Battery reserve limit | `min_battery_reserve` | `0.20` | Fraction of battery (`20%`) required throughout mission |
| Safe landing slope | `max_landing_slope_deg` | `15.0` (`0.2618 rad`) | Maximum safe surface slope before tip-over crash |
| Separation distance | `min_separation_m` | `100.0` | `m` minimum horizontal distance between any two hoppers |
| Relay window period | `relay_period_s` | `120.0` | `s` full orbit cycle (`[k*120, k*120 + 40]` is open) |
| Relay open duration | `relay_open_duration_s` | `40.0` | `s` open window duration per period |
| Noise parameters | `thrust_sigma_rel`, `scatter_sigma_rel`, `scatter_sigma_base_m`, `slope_noise_sigma_deg`, `dust_max_extra_burn` | `0.02`, `0.015`, `8.0 m`, `1.0 deg`, `0.03` | Sampled per hop in noisy rollouts |

---

### 5.4 Crater catalogue (`get_crater_list()`)

```python
def get_crater_list(seed: int = 42) -> list[dict[str, Any]]:
    """Return the crater catalogue with positions, radii, slopes, and hop reachability graph."""
```

Example entry:
```json
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
}
```

---

### 5.5 Training summary (`training_summary()`) and World-model error map (`world_model_error_map()`)

```python
def training_summary() -> dict[str, Any]:
    """Return reward curves, failure rates, reward term breakdown, and sample episodes for Argon."""

def world_model_error_map() -> dict[str, Any]:
    """Return per-crater and 20x20 per-cell prediction error and ensemble disagreement in local ENU metres."""

def suggest_plan(state: dict[str, Any] | None = None, seed: int = 42) -> dict[str, Any]:
    """Roll out the trained MAPPO policy greedily and return a candidate fleet_plan dict."""
```

- **Monotonicity invariant on `world_model_error_map()`:** Less-visited craters (`visit_count < 10`) and steep crater rims systematically have higher `uncertainty_m` and `mean_error_m` than frequently visited flat crater floors (`visit_count > 50`). Argon should cite these numbers when explaining world-model errors and when choosing conservative hop targets or extra fuel margins.

---

## 6. Safety rule contract and example implementation

Each rule generated by Argon (or loaded from the baseline spec) must implement `check(trace)`:

```python
import numpy as np
from typing import Any

def check_fuel_reserve(trace: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """R1_FUEL_RESERVE: Each hopper must retain at least 10% fuel reserve after final landing."""
    final_fuel = trace["fuel"][:, -1, :]          # shape (n, N)
    run_ok = np.all(final_fuel >= 0.10, axis=1)   # shape (n,)
    pass_rate = float(np.mean(run_ok))
    passed = pass_rate >= 0.99

    failed_rollouts = np.flatnonzero(~run_ok).astype(int).tolist()
    hopper_fails = np.sum(final_fuel < 0.10, axis=0)
    failures_by_hopper = {
        hid: int(hopper_fails[idx])
        for idx, hid in enumerate(trace["hopper_ids"])
    }
    stage_fails = np.sum(np.any(trace["fuel"][:, 1:, :] < 0.10, axis=2), axis=0)
    failures_by_stage = stage_fails.astype(int).tolist()

    reason = (
        "ok"
        if passed
        else f"Fuel dropped below 10% reserve in {len(failed_rollouts)}/{trace['n_rollouts']} rollouts."
    )
    return passed, reason, {
        "rule_id": "R1_FUEL_RESERVE",
        "pass_rate": pass_rate,
        "failed_rollouts": failed_rollouts,
        "failures_by_hopper": failures_by_hopper,
        "failures_by_stage": failures_by_stage,
    }
```

### All five baseline rules to include in the safety spec
1. `R1_FUEL_RESERVE`: `trace["fuel"][:, -1, :] >= 0.10`
2. `R2_BATTERY_RESERVE`: `np.all(trace["battery"] >= 0.20, axis=(1, 2))`
3. `R3_LANDING_SLOPE`: `np.all(trace["landing_slope_deg"] <= 15.0, axis=(1, 2))` and `~np.any(trace["crashed"], axis=(1, 2))`
4. `R4_FLEET_SEPARATION`: `np.all(trace["min_separation_m"] >= 100.0, axis=1)`
5. `R5_SINGLE_CHANNEL_RELAY`: `~np.any(trace["relay_conflict"], axis=1)`

### Overall verdict rule
A fleet plan passes verification if and only if the joint pass rate across all rules is `>= 0.99` (at least 990 of 1000 rollouts satisfy every rule simultaneously).

---

## 7. Handoff to the 3D view (`data/episodes.json` + `verification`)

When a fleet plan is verified (both on the failing first attempt and on the passing revised attempt), export the `verification` block (see [`docs/interfaces.md`](interfaces.md#42-proposed-schema-addition-verification-trajectory-cloud-and-verdict)) so the 3D view can render:
- Green trajectory ribbons for rollouts with `passed: true`.
- Red trajectory ribbons and crash markers for rollouts with `passed: false`.
- The per-rule pass rates and the human **Approve Uplink** button (enabled only when `verdict.passed == true` and a relay window is open).

---

## 8. Day-one stubs and how to run tests

You can build and test the entire orchestrator pipeline immediately without waiting for long RL runs:

- **Simulator & rollout engine:** `apps/sim/selene_sim/sim.py` and `apps/sim/selene_sim/verify.py` (already implemented, runs 1000 rollouts in `< 0.1 s` using `TerrainGrid.synthetic(seed=42)`).
- **Reward hot-swap:** `apps/sim/selene_sim/reward.py` (`compute_reward`, `load_reward_code`, `reset_default_reward`).
- **Example 3D replay file:** `data/episodes.example.json`.

### Running the test suite
```bash
cd /Users/nabilshah/Random/selene/apps/sim
uv run pytest -v
```
