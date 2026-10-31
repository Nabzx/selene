# Implementation Plan: Fleet Simulator, World Model, MAPPO, and Verifier Rollouts

## 1. Module breakdown and data flow

All simulation, learning, and verification modules live in `apps/sim/selene_sim/` with unit and performance tests in `apps/sim/tests/`. None of these modules call Gemini at runtime; Friend B's orchestrator imports and calls them.

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
│  (FleetEnv)  │ steps  │ (GroundTruth)│ eval    │ (Shared Policy +   │
└──────┬───────┘        └──────┬───────┘         │ Centralised Critic)│
       │ transitions           │ terrain         └──────┬──────────┬──┘
       ▼                       ▼                        │ imagines │
┌──────────────────┐    ┌──────────────┐                │          │ writes
│  world_model.py  │    │   data/      │◄───────────────┘          ▼
│ (5-MLP Ensemble) │◄───│  grid.npz    │                    data/episodes.json
└──────────────────┘    └──────────────┘                   (Friend A: 3D View)
```

| File | Responsibility |
|---|---|
| `selene_sim/sim.py` | `TerrainGrid` (synthetic seeded generator + `data/grid.npz` loader), `Crater`, `RelaySchedule`, `SimConfig`, `LunarSimulator`, and vectorised ballistic/fuel/battery/thermal functions. |
| `selene_sim/env.py` | `FleetEnv`: PettingZoo `ParallelEnv`-compatible multi-agent environment for $N \in \{4, 5\}$ hoppers with action masking, crater claims/scans, separation tracking, and single-channel relay windows. |
| `selene_sim/reward.py` | `RewardState`, default `compute_reward(state, actions, next_state)`, and `load_reward_code(code_str, summary)` for runtime reward hot-swapping by the orchestrator. |
| `selene_sim/world_model.py` | `SharedWorldModel`: ensemble of $K=5$ PyTorch MLPs trained on pooled hopper transitions. Provides `predict()` with ensemble disagreement uncertainty and `error_map()` per grid cell. |
| `selene_sim/mappo.py` | CleanRL-style single-file MAPPO (`SharedActor`, `CentralisedCritic`, `MAPPOTrainer`) on `"mps"` or `"cpu"`. Supports direct simulator training and short $k$-step MBPO imagination rollouts, plus `training_summary()` and `export_episodes_json()`. |
| `selene_sim/verify.py` | `simulate_plan(fleet_plan, n=1000, seed=42)`: vectorised numpy Monte Carlo rollout engine over the real simulator physics (never the world model). |

---

## 2. Exact interfaces and data shapes for Friend A and Friend B

### 2.1 Interface to Friend B: `reward.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RewardState:
    pos: np.ndarray              # (N, 3) float32, metres in local ENU frame
    fuel: np.ndarray             # (N,) float32 in [0.0, 1.0]
    battery: np.ndarray          # (N,) float32 in [0.0, 1.0]
    temp_c: np.ndarray           # (N,) float32, hopper internal temp in C
    crashed: np.ndarray          # (N,) bool
    unsent_data: np.ndarray      # (N,) int32, un-transmitted crater scans held
    in_relay_window: bool        # True if orbiter is overhead at current step
    crater_pos: np.ndarray       # (C, 3) float32, crater centres in metres
    crater_radii: np.ndarray     # (C,) float32, crater radii in metres
    crater_scanned: np.ndarray   # (C,) bool, cumulative scanned flags
    newly_scanned_by: np.ndarray # (N,) bool, True for hopper that completed a new scan this step
    data_sent_by: np.ndarray     # (N,) int32, scans transmitted cleanly this step
    relay_conflict: bool         # True if >=2 hoppers transmitted at once or outside a window
    min_peer_distance: np.ndarray# (N,) float32, distance in metres to nearest other hopper

def compute_reward(
    state: RewardState,
    actions: np.ndarray,         # (N,) int32 discrete action indices
    next_state: RewardState,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Returns:
      rewards: (N,) float32 array of per-hopper scalar rewards
      components: dict mapping term names (e.g. 'crater_scan', 'fuel_cost')
                  to their fleet-mean float value for Argon reflection.
    """

def load_reward_code(code_str: str, summary: str = "", episode: int = 0) -> None:
    """
    Compiles code_str in a restricted namespace, smoke-tests compute_reward on
    dummy RewardState inputs, replaces the active reward function, and appends
    {'episode': episode, 'summary': summary} to reward_versions.
    """
```

### 2.2 Interface to Friend B: `verify.py`

**Input `fleet_plan` dict:**
```python
{
    "plan_id": "plan_001",
    "seed": 42,
    "hoppers": [
        {
            "id": "h0",
            "start_pos": [0.0, 0.0, -120.0],
            "initial_fuel": 1.0,
            "initial_battery": 0.95,
            "route": [
                {"step": 0, "t": 0.0, "type": "hop", "target": [1800.0, 600.0], "crater_id": "c2"},
                {"step": 1, "t": 60.0, "type": "scan", "crater_id": "c2"},
                {"step": 2, "t": 120.0, "type": "send"},
                {"step": 3, "t": 180.0, "type": "wait"},
            ],
        },
        # ... h1 .. h_{N-1}
    ],
}
```

**Signature and output `traces` dict:**
```python
def simulate_plan(
    fleet_plan: dict,
    n: int = 1000,
    seed: int = 42,
    grid: TerrainGrid | None = None,
) -> dict[str, Any]:
```
Returns a dictionary of numpy arrays across `n` rollouts, `H` route steps, and `N` hoppers:
- `"n_rollouts"`: `int` (`1000`)
- `"hopper_ids"`: `list[str]` (`["h0", "h1", ...]`)
- `"step_times"`: `np.ndarray` of shape `(n, H)`, `float32`, simulated mission time in seconds
- `"action_types"`: `np.ndarray` of shape `(H, N)`, `int32` (`0=wait, 1=scan, 2=send, 3=hop`)
- `"pos"`: `np.ndarray` of shape `(n, H, N, 3)`, `float32`, `[x, y, z]` in metres
- `"fuel"`: `np.ndarray` of shape `(n, H, N)`, `float32` in `[0.0, 1.0]`
- `"battery"`: `np.ndarray` of shape `(n, H, N)`, `float32` in `[0.0, 1.0]`
- `"temp_c"`: `np.ndarray` of shape `(n, H, N)`, `float32`, internal temperature in °C
- `"landing_slope_deg"`: `np.ndarray` of shape `(n, H, N)`, `float32`, terrain slope in degrees at landing cell (`0.0` on non-hop steps)
- `"crashed"`: `np.ndarray` of shape `(n, H, N)`, `bool` (slope exceeded limit or fuel exhausted mid-hop)
- `"min_separation_m"`: `np.ndarray` of shape `(n, H)`, `float32`, minimum pairwise distance between any two hoppers at step `h`
- `"pair_distances_m"`: `np.ndarray` of shape `(n, H, N, N)`, `float32`, full pairwise horizontal/3D distance matrix
- `"relay_active_count"`: `np.ndarray` of shape `(n, H)`, `int32`, number of hoppers executing `send` at step `h`
- `"in_relay_window"`: `np.ndarray` of shape `(n, H)`, `bool`
- `"relay_conflict"`: `np.ndarray` of shape `(n, H)`, `bool` (`(relay_active_count > 1) | ((relay_active_count == 1) & ~in_relay_window)`)
- `"cloud_samples"`: `list[dict]`, first 50 rollouts serialised as `[x, y, z]` arcs per hopper so the backend can pass them straight to the 3D trajectory cloud after Friend B marks each rollout pass/fail.

### 2.3 Interface to Friend B: `training_summary()` and `world_model_error_map()`

```python
def training_summary() -> dict[str, Any]:
    """
    Returns curves and sample episodes for Argon's reward reflection and planning:
    {
        "episodes_trained": int,
        "episode_rewards": list[float],
        "reward_components": dict[str, list[float]],
        "craters_visited_mean": list[float],
        "crash_rate": list[float],
        "relay_conflict_rate": list[float],
        "reward_versions": list[dict], # [{"episode": int, "summary": str}]
        "sample_episodes": list[dict], # recent logged episodes
        "policy_rollout_plan": dict,   # greedy fleet_plan generated by current MAPPO policy
    }
    """

def world_model_error_map() -> dict[str, Any]:
    """
    Returns per-cell prediction error and ensemble uncertainty for Argon's log review:
    {
        "grid_shape": [200, 200],
        "cell_size_m": 100.0,
        "mean_error_m": list[list[float | None]],  # 200x200 logged L2 error vs simulator
        "uncertainty_m": list[list[float]],        # 200x200 ensemble std dev in metres
        "visit_count": list[list[int]],            # 200x200 transition counts per cell
        "hotspots": [                              # Top 5 cells/craters with highest error
            {
                "cell": [ix, iy],
                "pos": [x, y, z],
                "nearest_crater_id": "c3",
                "slope_deg": 18.4,
                "visit_count": 2,
                "mean_error_m": 44.2,
                "uncertainty_m": 39.8,
            }
        ],
    }
    """
```

---

## 3. Observation and action spaces with sizes

Default configuration: $N = 4$ hoppers (supports $N = 5$), $C = 8$ craters in the $20 \times 20\text{ km}$ crop (`[-10000, 10000]` m in $x$ and $y$, $200 \times 200$ cells at $100\text{ m}$).

### 3.1 Action space per hopper: `Discrete(19)` (or `Discrete(3 + 8 + 8)`)

| Index | Action | Target / meaning | Masked when |
|---|---|---|---|
| `0` | `wait` | Hold position for $\Delta t = 30\text{ s}$ | Never (`always True`) |
| `1` | `scan` | Scan current crater ($\Delta t = 30\text{ s}$) | Hopper crashed, or not inside an unscanned crater |
| `2` | `send` | Transmit buffered scans to relay orbiter ($\Delta t = 30\text{ s}$) | Hopper crashed, `unsent_data == 0`, or no open relay window |
| `3 .. 10` | `hop_dir_short` | Hop $1200\text{ m}$ in 8 compass directions ($0, \frac{\pi}{4}, \dots, \frac{7\pi}{4}$) | Out of grid bounds, insufficient fuel, or hopper crashed |
| `11 .. 18` | `hop_crater[c]` | Hop directly to crater $c \in \{0..7\}$ (clamped to max single-hop range $3500\text{ m}$ along vector to $c$) | Crater $c$ already scanned, insufficient fuel for that hop, or hopper crashed |

Total discrete actions per hopper: $A = 3 + 8 + C = 19$ for $C = 8$.

### 3.2 Observation space per hopper: `Box(low=-5.0, high=5.0, shape=(104,), dtype=float32)`

| Slice | Dim | Content |
|---|---|---|
| `0:7` | 7 | Own state: normalised `x/10000`, `y/10000`, `z/1000`, `fuel`, `battery`, `(temp_c + 180)/100`, `unsent_data/C` |
| `7:7+N` | 4 | One-hot hopper index ($N=4$) |
| `11:61` | 50 | Local $5 \times 5$ terrain patch centred on hopper: relative elevation $(z_{\text{cell}} - z)/200$ and slope in radians for each of the 25 cells |
| `61:61+3(N-1)` | 9 | Other $N-1$ hoppers: relative $(\Delta x/10000, \Delta y/10000)$ and `active` flag (`1.0` if not crashed) |
| `70:70+4C` | 32 | All $C=8$ craters: relative $(\Delta x/10000, \Delta y/10000)$, `radius/1000`, and `scanned` flag (`0.0` or `1.0`) |
| `102:104` | 2 | Relay status: `in_relay_window` (`0.0` or `1.0`), normalised `time_to_next_window / 3600` |

- **Actor input dimension:** `104` floats ($N=4, C=8$; formula `59 + 4*N + 4*C`).
- **Centralised critic input dimension (`global_state`):** Concatenation of all $N$ hoppers' core states (`N * 7 = 28`), pairwise distances (`N*(N-1)/2 = 6`), all crater statuses (`C * 4 = 32`), and global time/relay features (`3`), giving `69` floats for $N=4, C=8$.

---

## 4. Agreed `drones[]` extension to `docs/data/EPISODE_SCHEMA.md`

To support a fleet of 4 to 5 hoppers in `data/episodes.json`, each episode entry in `episodes[]` uses `drones[]` as its primary per-hopper list:

```json
{
  "schema_version": 1,
  "meta": {
    "region": "Daedalus",
    "generated_at": "2026-10-31T14:00:00Z",
    "source": "trained",
    "seed": 7,
    "fleet_size": 4
  },
  "craters": [ ... ],
  "relay_windows": [ ... ],
  "training": { ... },
  "episodes": [
    {
      "index": 500,
      "outcome": "completed",
      "total_reward": 84.6,
      "craters_visited": ["c1", "c2", "c4", "c6"],
      "drones": [
        {
          "id": "h0",
          "name": "Hopper 0",
          "outcome": "completed",
          "craters_visited": ["c1", "c2"],
          "steps": [
            {
              "t": 0, "pos": [0, 0, -120], "yaw": 0.3, "phase": "ground",
              "fuel": 1.0, "battery": 0.9, "temp_c": -20,
              "action": { "type": "hop", "target": [1800, 600] },
              "reward": 0, "wm_pos": null, "wm_error": null, "event": null
            }
          ]
        }
      ]
    }
  ]
}
```

Additional schema extension:
- Allow `"send"` in `step.action.type` (`{ "type": "send" }`) alongside `"hop"`, `"scan"`, and `"wait"`, since sending data in a relay window is one of the four hopper actions.

---

## 5. Time estimates per issue and cut order

Total build window for our scope: roughly **5.5 hours**, leaving buffer for integration with Friend A and Friend B.

| Order | Issue | Est. time | Phase |
|---|---|---|---|
| 1 | Update episode schema and example file for fleet `drones[]` and `send` action | 20 min | `phase-1` |
| 2 | Build physics simulator (`sim.py`) with synthetic terrain and `grid.npz` loader | 60 min | `phase-1` |
| 3 | Build vectorised Monte Carlo rollout engine (`verify.py`) and benchmark (< 10 s for 1000 runs) | 45 min | `phase-1` |
| 4 | Build hot-swappable reward module (`reward.py`) with Eureka reflection stats | 30 min | `phase-2` |
| 5 | Build multi-hopper parallel environment (`env.py`) with action masking | 50 min | `phase-2` |
| 6 | Build shared 5-MLP ensemble world model (`world_model.py`) and per-cell error map | 45 min | `phase-3` |
| 7 | Build CleanRL-style MAPPO trainer (`mappo.py`) and `data/episodes.json` exporter | 60 min | `phase-3` |
| 8 | Add short-horizon MBPO imagination rollouts inside `mappo.py` | 30 min | `phase-3` |

### Cut order if time runs short (matching `ROADMAP.md` and Part 2)

1. **First cut: Issue 8 (MBPO imagination training).** Train MAPPO directly on the fast vectorised numpy simulator (which runs thousands of steps per second on CPU), while still training `world_model.py` online on the collected transitions so the world-model ghost (`wm_pos`, `wm_error`) and Argon's error map work identically.
2. **Second cut: Real SLDEM2015 / Diviner raster download processing.** Keep the seeded deterministic synthetic far-side crater terrain (or pre-baked `data/grid.npz`) if raw PDS GeoTIFF downloads are slow on venue Wi-Fi.
3. **Third cut: Scale fleet size from 5 to 4 hoppers and 6 craters during live training.** Cuts episode horizon and speeds up MAPPO convergence.

---

## 6. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **PyTorch not yet in `apps/sim/pyproject.toml`** | `world_model.py` and `mappo.py` need PyTorch | Add `torch` via `uv add torch` in `apps/sim` at the start of the world-model/RL issues; keep `sim.py`, `env.py`, `reward.py`, and `verify.py` pure numpy so they have zero torch dependency. |
| **Apple M1 MPS overhead on tiny MLP batches** | Small single-step forward passes can be slower on MPS than CPU due to kernel launch latency | Benchmark `"mps"` vs `"cpu"` in `mappo.py`; use `"cpu"` for inference during environment rollouts and `"mps"` (or `"cpu"` if faster) for minibatch SGD updates. |
| **MAPPO learns to sit still (`wait`) to avoid fuel and crash penalties** | Flat reward curves, hoppers never leave spawn | Include potential-based distance shaping towards unclaimed craters in the baseline `reward.py`, and mask `scan`/`send` when invalid so random exploration must hop. |
| **Policy exploits world-model hallucinations on unvisited crater rims** | High imagined reward, immediate crash in simulator | Restrict imagination rollouts to $k \le 2$ steps branched from real states, penalise high ensemble disagreement $\sigma_{\text{pos}}$, and always evaluate and log episodes in the real simulator. |
| **Schema change breaks Friend A's existing 3D viewer** | Merge conflict or broken replay | Keep top-level `episodes[].steps` populated with `drones[0].steps` alongside the new `episodes[].drones[]` array. |

---

## 7. Resolved design decisions with Nabil

1. **PyTorch dependency in `apps/sim/pyproject.toml`:** Confirmed. Add `torch` via `uv add torch` for `world_model.py` and `mappo.py`, keeping `sim.py`, `env.py`, `reward.py`, and `verify.py` in pure numpy.
2. **Default fleet size and crater count:** Confirmed $N = 4$ hoppers (configurable up to 5) and $C = 8$ craters.
3. **Action space parameterisation:** Confirmed `Discrete(19)` (`wait`, `scan`, `send`, 8 compass hops of $1.2\text{ km}$, and 8 crater-targeted hops up to $3.5\text{ km}$) in `env.py`, while `verify.py` takes exact `[x, y]` coordinates from `fleet_plan`.
4. **Multi-hopper `episodes.json` structure:** Confirmed everyone uses multi-agent `episodes[].drones[]` as the primary schema structure.
5. **Phase-0 defaults:** Confirmed Daedalus crater (`region="Daedalus"`, $5.9^\circ\text{S}, 179.4^\circ\text{E}$), Option A (Monte Carlo rollouts without Z3) for `verify.py`, and integer scan counts (`unsent_data`) for payload tracking.
