# Episode schema

The contract between training (Python) and the 3D view (JavaScript). Training writes `data/episodes.json`. The view reads it and nothing else for replay. Change this file before changing either side.

An example lives at [`data/episodes.example.json`](../../data/episodes.example.json). `apps/sim/tests/test_episode_schema.py` checks it.

## Units and frames

- Positions in metres, in a local east-north-up frame centred on the terrain crop. `x` east, `y` north, `z` up.
- Angles in radians. Time in seconds of simulated mission time.
- Terrain heights come from `data/terrain.json` (same frame), so the view can drop the hopper onto the surface.

## Top level

```json
{
  "schema_version": 1,
  "meta": { ... },
  "craters": [ ... ],
  "relay_windows": [ ... ],
  "training": { ... },
  "episodes": [ ... ]
}
```

### `meta`

| Field | Type | Meaning |
|---|---|---|
| `region` | string | Far-side region name, for example `"Daedalus"` |
| `generated_at` | string | ISO 8601 timestamp |
| `source` | `"trained"` or `"scripted"` | `scripted` must be shown as illustrative in the view |
| `seed` | int | Simulator seed |

### `craters[]`

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable id, for example `"c3"` |
| `name` | string | Display name (Halloween names welcome, for example `"Hollow 3"`) |
| `x`, `y`, `z` | number | Centre on the surface |
| `radius` | number | Rim radius in metres |

### `relay_windows[]`

| Field | Type | Meaning |
|---|---|---|
| `start`, `end` | number | Mission time in seconds |

### `training`

| Field | Type | Meaning |
|---|---|---|
| `episode_rewards` | number[] | Mean reward per logged episode, for the reward curve |
| `reward_versions` | object[] | `{ "episode": int, "summary": string }` each time Argon rewrote the reward |

### `episodes[]`

Only a sample of episodes is logged (for example 1, 50, 200, 500), not every one.

| Field | Type | Meaning |
|---|---|---|
| `index` | int | Training episode number |
| `outcome` | `"completed"`, `"crashed"`, `"battery"`, `"timeout"` | How it ended |
| `total_reward` | number | |
| `craters_visited` | string[] | Crater ids in visit order |
| `steps` | object[] | See below |

### `episodes[].steps[]`

| Field | Type | Meaning |
|---|---|---|
| `t` | number | Mission time |
| `pos` | `[x, y, z]` | Hopper position from the simulator (truth) |
| `yaw` | number | Heading |
| `phase` | `"ground"`, `"ascent"`, `"coast"`, `"descent"` | For the engine plume and animation |
| `fuel` | number | 0 to 1 |
| `battery` | number | 0 to 1 |
| `temp_c` | number | Hopper internal temperature |
| `action` | object | `{ "type": "hop", "target": [x, y] }`, `{ "type": "scan" }` or `{ "type": "wait" }` |
| `reward` | number | Reward for this step |
| `wm_pos` | `[x, y, z]` or null | Where the world model predicted the hopper would be |
| `wm_error` | number or null | Distance between `pos` and `wm_pos` in metres |
| `event` | string or null | `"landed"`, `"crashed"`, `"scan"`, `"relay_contact"`, or null |

The view interpolates between steps. Steps are logged at hop boundaries and every 0.5 s during flight.
