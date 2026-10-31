# Multi-agent environment API and action masking

## PettingZoo ParallelEnv conventions

In our setting, all hoppers in the fleet act simultaneously at each decision step (choosing whether to hop, scan, wait, or send data). PettingZoo's `ParallelEnv` API is the standard interface for simultaneous-move multi-agent environments:

- `possible_agents`: `["h0", "h1", "h2", "h3"]` (configurable $N \in \{4, 5\}$).
- `agents`: active hoppers at the current step (a crashed or depleted hopper stays in the state for collision checks and logging, or its actions are masked to `wait`).
- `reset(seed=None, options=None) -> tuple[dict[str, np.ndarray], dict[str, dict]]`: returns `(observations, infos)` keyed by hopper id.
- `step(actions: dict[str, int]) -> tuple[obs, rewards, terminations, truncations, infos]`: steps all hoppers at once and returns dictionaries keyed by hopper id.
- Keep allhopper state in contiguous numpy arrays of shape `(N, ...)` under the hood so the environment can also expose a fast array-based step for MAPPO training without dictionary-creation overhead on every inner step.

## Discrete action space and action masking

Hopping to an arbitrary $(x, y)$ coordinate on a $200 \times 200$ grid ($40\,000$ cells) is too large for discrete PPO and awkward for continuous PPO when craters are discrete targets and out-of-range hops are illegal. Two practical discrete parameterisations work well:

1. **Local relative hop grid + crater targets:** Discrete actions covering:
   - `0`: `wait` (1 step, minimal battery drain)
   - `1`: `scan` (valid when inside an unscanned crater radius)
   - `2`: `send` (valid during an open relay window when the hopper holds unsent data)
   - `3 .. 3 + M - 1`: hop towards crater $c_m$ (or up to $d_{\max}$ along the vector to $c_m$, or a polar ring of 16 directional short/medium hops).
   Giving direct hop actions for the $K_{\text{dirs}} = 16$ compass/range offsets (for example 8 directions $\times$ 2 distances: 1.5 km and 3.0 km) plus `wait`, `scan`, `send` yields a compact fixed action space of size $A = 19$ (or $3 + C$ where $C$ is the number of craters).

**Action masking (Huang & Ontañón, 2022):**
At every step, `env` computes a boolean mask `action_mask` of shape `(N, A)` returned inside `infos[agent]["action_mask"]` (and available as an array):
- A hop action is masked (`False`) if the target cell is outside the $200 \times 200$ crop, requires more fuel than the hopper currently has, or if the hopper has crashed.
- `scan` is masked unless the hopper is inside an unvisited crater.
- `send` is masked unless a relay window is currently active and the hopper holds unsent payload data.
- `wait` is always `True` so the categorical distribution always has at least one valid action.

In `mappo.py`, masked logits are replaced with `-1e8` before creating `torch.distributions.Categorical(logits=masked_logits)`.

## Observation design per hopper

Each hopper observes a flat `float32` vector `obs[i]` normalised to roughly $[-1, 1]$:
- **Own state (7 floats):** normalised $(x, y, z)$, fuel $[0, 1]$, battery $[0, 1]$, internal temperature, unsent data count.
- **One-hot agent id ($N$ floats):** lets the shared policy differentiate hoppers.
- **Local terrain patch ($5 \times 5 \times 2 = 50$ floats):** relative elevation and local slope around the hopper's current cell.
- **Other hoppers ($(N-1) \times 3$ floats):** relative $(\Delta x, \Delta y)$ and active flag for each teammate.
- **Craters ($C \times 4$ floats):** relative $(\Delta x, \Delta y)$, radius, and status flag (unclaimed / claimed by teammate / scanned) for each crater in the crop.
- **Relay state (2 floats):** binary flag `relay_open` and normalised time to the next relay window.

For $N = 4$ hoppers and $C = 8$ craters, `obs_dim` $= 7 + 4 + 50 + 9 + 32 + 2 = 104$ floats, which is small enough for a 2-layer MLP with 128 hidden units to run in under 0.2 ms per step.

## Sources

- Terry et al. (2021), *PettingZoo: Gym for Multi-Agent Reinforcement Learning*, NeurIPS. [https://arxiv.org/abs/2009.14471](https://arxiv.org/abs/2009.14471)
- Farama Foundation, *PettingZoo Parallel API Documentation*. [https://pettingzoo.farama.org/api/parallel/](https://pettingzoo.farama.org/api/parallel/)
- Huang & Ontañón (2022), *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*, FLAIRS. [https://arxiv.org/abs/2006.14171](https://arxiv.org/abs/2006.14171)

## Recommendation

Follow the PettingZoo `ParallelEnv` signature (`reset`, `step`, `agents`, `possible_agents`, `infos[h]["action_mask"]`) in `selene_sim/env.py`, backed by vectorised `(N, ...)` numpy state arrays and a compact 104-float observation vector with invalid action masking.
