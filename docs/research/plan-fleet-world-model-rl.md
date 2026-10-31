# Implementation Plan: Fleet Simulator, World Model, MAPPO, and Verifier Rollouts

## 1. Design constraints for fast laptop training

Everything is designed to train in minutes (5 to 15 minutes for clear reward gains, 20 to 40 minutes for full fleet cooperation, 1 to 3 minutes for the world model) on an Apple M1 laptop:

- **Action space (crater graph, not grid cells):** Each hopper picks from $A = C + 3$ discrete actions (`wait`, `scan`, `send`, or `hop_to_crater[c]` for $c \in \{0, \dots, C-1\}$, with $C = 6$ in `--quick` mode and $C = 12\text{ to }15$ in full mode). Out-of-range hops, insufficient-fuel hops, already-scanned craters, and sends outside relay windows are masked out. Terrain height, landing slope, crash risk, and night temperature still come from the $200 \times 200$ `TerrainGrid` in `sim.py`.
- **Observation space (~50 floats per hopper, no CNNs):** Flat `float32` vector containing own position, fuel, battery, temperature, unsent data, one-hot hopper id, other hoppers' positions, crater claimed/done flags and relative positions, and time to the next relay window. No image-like terrain patches.
- **Episode horizon (20 to 40 decisions):** Each hop is computed once in closed form (ballistic range, rocket-equation fuel, landing slope check), not integrated step by step during training. $0.5\text{ s}$ flight-phase keyframes are generated only when exporting `data/episodes.json` for the 3D view.
- **Environment throughput (> 10,000 env steps/s on CPU):** `VectorFleetEnv` steps $E = 64\text{ to }256$ environments in parallel as contiguous `(E, N, ...)` numpy arrays in one process, verified by a benchmark test.
- **Networks and device selection:** 2-layer MLPs with 64 to 128 hidden units. At startup, benchmark `"cpu"` against `"mps"` and run on whichever is faster.
- **Small runs first (`--quick` flag, CSV logs, checkpoints):**
  - `--quick`: 2 hoppers, 6 craters, synthetic terrain, 2-minute time budget. Used for all debugging.
  - Full mode: 4 to 5 hoppers, 12 to 15 craters, Daedalus terrain grid.
  - Logs a CSV every $K$ updates (`update`, `env_steps`, `elapsed_s`, `episode_reward`, `crashes`, `craters_visited`, `crater_overlap`, `relay_conflicts`, `entropy`, `value_loss`, `wm_error`) and saves checkpoints at fixed intervals.
  - `scripts/training_summary.py` prints a plain-text summary table from the CSV. Long runs happen in a separate terminal, never inside an agent tool call.

---

## 2. Module breakdown and data flow

All simulation, learning, and verification modules live in `apps/sim/selene_sim/` with tests in `apps/sim/tests/`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Friend B: Argon Orchestrator                         │
└──────┬───────────────────────┬────────────────────────┬───────────▲─────┘
       │ load_reward_code()    │ simulate_plan()        │           │
       ▼                       ▼                        │           │
┌──────────────┐        ┌──────────────┐                │     training_summary()
│  reward.py   │        │  verify.py   │                │     world_model_error_map()
└──────┬───────┘        └──────┬───────┘                │           │
       │ compute_reward()      │ vectorised numpy       │           │
       ▼                       ▼                        ▼           │
┌──────────────┐        ┌──────────────┐         ┌──────────────────┴─┐
│    env.py    │───────►│    sim.py    │◄────────│      mappo.py      │
│(VectorFleet) │ >10k/s │ (GroundTruth)│ eval    │ (2-layer MLP 64-128│
└──────┬───────┘        └──────┬───────┘         │  --quick / CSV log)│
       │ transitions           │ terrain         └──────┬──────────┬──┘
       ▼                       ▼                        │          │ writes
┌──────────────────┐    ┌──────────────┐                │          ▼
│  world_model.py  │    │   data/      │◄───────────────┘   data/episodes.json
│ (5-MLP Ensemble) │    │  grid.npz    │                    (Friend A: 3D View)
└──────────────────┘    └──────────────┘
```

| File | Responsibility |
|---|---|
| `selene_sim/sim.py` | `TerrainGrid` (seeded synthetic Daedalus generator + `data/grid.npz` loader, 12 to 15 craters), `SimConfig`, `LunarSimulator`, closed-form vectorised hop physics, and $0.5\text{ s}$ arc interpolation helper for episode export. |
| `selene_sim/env.py` | `VectorFleetEnv` (steps $E = 64\text{ to }256$ envs in parallel as numpy arrays at $> 10\,000$ steps/s) and `FleetEnv` (PettingZoo `ParallelEnv` wrapper) on the crater-graph action space with action masking. |
| `selene_sim/reward.py` | `RewardState`, vectorised `compute_reward(state, actions, next_state)`, and `load_reward_code(code_str, summary, episode)` for runtime hot-swapping by Argon. |
| `selene_sim/world_model.py` | `SharedWorldModel`: ensemble of 5 small 2-layer MLPs trained on `(obs, action) -> next_obs, reward, crash_prob` in 1 to 3 minutes, exposing ensemble disagreement uncertainty and per-crater/per-cell error maps. |
| `selene_sim/mappo.py` | Single-file CleanRL-style MAPPO with `--quick` flag, CPU vs MPS benchmark selector, CSV logging, periodic checkpointing, `training_summary()`, and `export_episodes_json()`. |
| `selene_sim/verify.py` | `simulate_plan(fleet_plan, n=1000, seed=42)`: vectorised numpy Monte Carlo rollout engine (< 10 s for 1000 runs) using only `sim.py`. |
| `scripts/training_summary.py` | Reads the training CSV and prints progress metrics (`episode_reward`, `crashes`, `craters_visited`, `crater_overlap`, `relay_conflicts`, `wm_error`). |

---

## 3. Exact interfaces and data shapes for Friend A and Friend B

### 3.1 Interface to Friend B: `reward.py`

Supports both single-environment `(N, ...)` and vectorised `(E, N, ...)` shapes:

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RewardState:
    pos: np.ndarray              # (..., N, 3) float32, metres in local ENU frame
    fuel: np.ndarray             # (..., N) float32 in [0.0, 1.0]
    battery: np.ndarray          # (..., N) float32 in [0.0, 1.0]
    temp_c: np.ndarray           # (..., N) float32, hopper internal temp in C
    crashed: np.ndarray          # (..., N) bool
    at_crater: np.ndarray        # (..., N) int32, index of current crater (-1 if none)
    unsent_data: np.ndarray      # (..., N) int32, un-transmitted crater scans held
    in_relay_window: np.ndarray  # (...,) bool, True if orbiter is overhead
    crater_pos: np.ndarray       # (C, 3) float32, crater centres in metres
    crater_radii: np.ndarray     # (C,) float32, crater radii in metres
    crater_claimed: np.ndarray   # (..., C) bool, claimed by a hopper in the fleet
    crater_scanned: np.ndarray   # (..., C) bool, cumulative scanned flags
    newly_claimed_by: np.ndarray # (..., N) bool, True if hopper reached an unclaimed crater this step
    newly_scanned_by: np.ndarray # (..., N) bool, True if hopper completed a new scan this step
    data_sent_by: np.ndarray     # (..., N) int32, scans transmitted cleanly this step
    relay_conflict: np.ndarray   # (...,) bool, True if >=2 hoppers transmitted at once or outside window
    min_peer_distance: np.ndarray# (..., N) float32, distance in metres to nearest other hopper

def compute_reward(
    state: RewardState,
    actions: np.ndarray,         # (..., N) int32 discrete action indices
    next_state: RewardState,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Returns:
      rewards: (..., N) float32 array of per-hopper scalar rewards
      components: dict mapping term names to their mean float value for Argon reflection.
    """

def load_reward_code(code_str: str, summary: str = "", episode: int = 0) -> None:
    """
    Compiles code_str in a restricted namespace, smoke-tests compute_reward on
    dummy RewardState inputs, swaps the active reward function, and appends
    {'episode': episode, 'summary': summary} to reward_versions.
    """
```

### 3.2 Interface to Friend B: `verify.py`

```python
def simulate_plan(
    fleet_plan: dict,
    n: int = 1000,
    seed: int = 42,
    grid: TerrainGrid | None = None,
) -> dict[str, Any]:
```
Returns a trace dictionary of numpy arrays across `n` rollouts, `H` route steps, and `N` hoppers:
- `"n_rollouts"`: `int` (`1000`)
- `"hopper_ids"`: `list[str]` (`["h0", "h1", ...]`)
- `"step_times"`: `np.ndarray` `(n, H)` `float32`
- `"action_types"`: `np.ndarray` `(H, N)` `int32` (`0=wait, 1=scan, 2=send, 3=hop`)
- `"pos"`: `np.ndarray` `(n, H, N, 3)` `float32`
- `"fuel"`: `np.ndarray` `(n, H, N)` `float32`
- `"battery"`: `np.ndarray` `(n, H, N)` `float32`
- `"temp_c"`: `np.ndarray` `(n, H, N)` `float32`
- `"landing_slope_deg"`: `np.ndarray` `(n, H, N)` `float32`
- `"crashed"`: `np.ndarray` `(n, H, N)` `bool`
- `"min_separation_m"`: `np.ndarray` `(n, H)` `float32`
- `"pair_distances_m"`: `np.ndarray` `(n, H, N, N)` `float32`
- `"relay_active_count"`: `np.ndarray` `(n, H)` `int32`
- `"in_relay_window"`: `np.ndarray` `(n, H)` `bool`
- `"relay_conflict"`: `np.ndarray` `(n, H)` `bool`
- `"cloud_samples"`: `list[dict]`, 50 rollouts formatted as `[x, y, z]` arcs per hopper for the 3D trajectory cloud.

### 3.3 Interface to Friend B: `training_summary()` and `world_model_error_map()`

- `training_summary() -> dict[str, Any]`: reads the latest training CSV / trainer state and returns `episode_rewards`, `reward_components`, `craters_visited_mean`, `crater_overlap_rate`, `crash_rate`, `relay_conflict_rate`, `reward_versions`, `sample_episodes`, and `policy_rollout_plan`.
- `world_model_error_map() -> dict[str, Any]`: returns per-crater and per-cell (`200 x 200`) prediction error (`mean_error_m`), ensemble disagreement (`uncertainty_m`), `visit_count`, and the top-5 high-uncertainty / high-error `hotspots`.

---

## 4. Observation and action spaces with sizes

Full mode defaults: $N = 4$ hoppers (configurable up to 5), $C = 12$ craters (configurable up to 15), $H = 30$ decisions per episode.
`--quick` mode defaults: $N = 2$ hoppers, $C = 6$ craters, $H = 25$ decisions per episode.

### 4.1 Action space per hopper: `Discrete(C + 3)`

| Index | Action | Meaning | Masked when |
|---|---|---|---|
| `0` | `wait` | Hold position for $\Delta t = 60\text{ s}$ | Never (`always True`) |
| `1` | `scan` | Scan current crater ($\Delta t = 60\text{ s}$) | Hopper crashed, not at a crater, or crater already scanned |
| `2` | `send` | Send buffered scans in relay window ($\Delta t = 60\text{ s}$) | Hopper crashed, `unsent_data == 0`, or outside relay window |
| `3 .. C+2` | `hop_to_crater[c]` | Ballistic hop to crater $c \in \{0..C-1\}$ | Hopper crashed, already at crater $c$, crater $c$ already scanned, distance $R > R_{\max}$ ($4500\text{ m}$), or required fuel $> \text{fuel}$ |

- Full mode ($C = 12$): `Discrete(15)` (or `Discrete(18)` for $C = 15$).
- `--quick` mode ($C = 6$): `Discrete(9)`.

### 4.2 Observation space per hopper: flat vector of ~50 floats

| Slice | Dim ($N=4, C=12$) | Dim (`--quick`: $N=2, C=6$) | Content |
|---|---|---|---|
| Own state | `7` | `7` | Normalised `x`, `y`, `z`, `fuel`, `battery`, `temp_c`, `unsent_data` |
| One-hot agent id | `N = 4` | `N = 2` | One-hot hopper identity |
| Other hoppers | `3(N - 1) = 9` | `3(N - 1) = 3` | Relative `dx`, `dy`, and `alive` flag for each other hopper |
| Crater features | `3C = 36` | `3C = 18` | Per crater $c$: normalised distance to $c$, `claimed` flag, `scanned` flag |
| Relay features | `2` | `2` | `in_relay_window` flag and normalised time to next relay window |
| **Total `obs_dim`** | **58 floats** | **32 floats** | Flat `float32` vector, no image patches |

- **Centralised critic `state_dim`:** All $N$ hoppers' core states (`7 * N`), crater `claimed` and `scanned` flags (`2 * C`), and relay state (`2`), giving `54` floats in full mode ($N=4, C=12$) and `28` floats in `--quick` mode ($N=2, C=6$).

---

## 5. Stage success checks and cut order

### Success checks (enforced as issue acceptance criteria)

1. **Stage 1 (Random policy & vectorised env):** `VectorFleetEnv` runs with a random masked policy with zero errors and passes the throughput benchmark at **over 10,000 env steps per second on CPU**.
2. **Stage 2 (MAPPO `--quick`):** With `--quick` (2 hoppers, 6 craters, synthetic terrain, 2-minute budget), mean episode reward rises by a clear margin within 5 minutes and crash rate drops.
3. **Stage 3 (MAPPO full):** With 4 hoppers and 12 craters, hoppers split craters (low crater overlap across the fleet) and relay conflicts fall across training.
4. **Stage 4 (World model):** Ensemble of 5 small MLPs trains in 1 to 3 minutes; validation error falls across training epochs, and ensemble disagreement (uncertainty) is higher on less-visited craters than on frequently visited craters.

### Cut order if time runs short

1. **Issue #13 (MBPO short-horizon imagination rollouts):** Stretch goal, last in build order. Cut first if late; MAPPO already trains in minutes directly on the $>10\,000\text{ steps/s}$ vectorised simulator while `world_model.py` trains on the logged transitions.
2. **Raw PDS GeoTIFF raster cropping:** Use the deterministic Daedalus-profile `TerrainGrid` (`data/grid.npz`) if raw SLDEM2015 downloads block.
3. **Reward rewrite loop iterations:** Keep one Argon-written reward swap if time is short.
