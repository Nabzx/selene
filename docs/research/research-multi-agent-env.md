# Multi-agent environment API and vectorised crater-graph design

## Crater-graph action space

Rather than picking arbitrary $(x, y)$ grid cells out of $200 \times 200 = 40\,000$ cells, each hopper acts on the **crater graph** of $C$ craters in the crop ($C = 6$ in `--quick` mode, $C = 12\text{ to }15$ in full mode):

- `0`: `wait` (hold position for one decision step)
- `1`: `scan` (scan the crater the hopper currently occupies)
- `2`: `send` (transmit buffered scan data during an open relay window)
- `3 .. 3 + C - 1`: `hop_to_crater[c]` (ballistic hop from current position to crater $c \in \{0, \dots, C-1\}$, landing on the real terrain grid cell with optional landing scatter)

For $C = 14$ craters, each hopper has $A = C + 3 = 17$ discrete actions (`Discrete(17)`). For `--quick` ($C = 6$), $A = 9$.

Terrain still governs every hop: hop distance $R$, elevation change $\Delta z$, rocket-equation fuel burn, landing cell slope, crash risk, and night surface temperature are all looked up on the $200 \times 200$ `TerrainGrid` in `sim.py`.

## Invalid action masking (Huang & Ontañón, 2022)

At every step, the vectorised environment computes a boolean mask `action_mask` of shape `(E, N, A)`:
- `wait` (`0`) is always `True`.
- `scan` (`1`) is `True` only if the hopper is alive, sits at crater $c$, and crater $c$ is not yet scanned.
- `send` (`2`) is `True` only if the hopper is alive, holds `unsent_data > 0`, and a relay window is currently open.
- `hop_to_crater[c]` (`3 + c`) is `True` only if the hopper is alive, is not already at crater $c$, crater $c$ is not yet scanned, horizontal distance $R \le R_{\max}$, and required fuel $\Delta \text{fuel} \le \text{fuel}$.

In `mappo.py`, masked action logits are set to $-10^8$ before `Categorical(logits=...)`.

## Compact flat observation (~50 numbers, no CNNs)

Each hopper receives a flat `float32` vector with no image-like terrain patches:
- **Own state (`7 + N` floats):** normalised `x`, `y`, `z`, `fuel`, `battery`, `temp_c`, `unsent_data`, plus one-hot hopper id ($N$ floats).
- **Other hoppers (`3 * (N - 1)` floats):** relative `dx`, `dy`, and `alive` flag for each other hopper in the fleet.
- **Crater states (`3 * C` floats):** for each crater $c \in \{0, \dots, C-1\}$, normalised relative `(dx, dy)` (or distance) and binary flags `(claimed_by_fleet, scanned_done)`.
- **Relay state (`2` floats):** `in_relay_window` flag and normalised time to the next relay window.

For $N = 4, C = 12$, `obs_dim` $= 11 + 9 + 36 + 2 = 58$ floats (for `--quick` with $N = 2, C = 6$, `obs_dim` $= 9 + 3 + 18 + 2 = 32$ floats).

## Vectorised numpy execution (> 10,000 env steps/s on CPU)

Episodes last $H = 20\text{ to }40$ decision steps. Each hop is evaluated in closed form in one step (range, $\Delta v$, fuel burn, landing slope check, thermal battery drain), rather than integrated at $0.5\text{ s}$ ticks during training. ($0.5\text{ s}$ flight-phase interpolation is only computed when exporting sampled episodes to `data/episodes.json` for the 3D view.)

Stepping $E = 64\text{ to }256$ environments in parallel as contiguous `(E, N, ...)` numpy arrays in one process eliminates per-environment Python loops and reaches well over $10\,000$ environment steps per second on an Apple M1 CPU. A single-environment PettingZoo `ParallelEnv` wrapper sits smoothly on top of `VectorFleetEnv(num_envs=1)`.

## Sources

- Terry et al. (2021), *PettingZoo: Gym for Multi-Agent Reinforcement Learning*, NeurIPS. [https://arxiv.org/abs/2009.14471](https://arxiv.org/abs/2009.14471)
- Huang & Ontañón (2022), *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*, FLAIRS. [https://arxiv.org/abs/2006.14171](https://arxiv.org/abs/2006.14171)

## Recommendation

Build `VectorFleetEnv` in `selene_sim/env.py` stepping $E = 64\text{ to }256$ environments as `(E, N, ...)` numpy arrays with the `C + 3` crater-graph action space, invalid action masking, a ~50-float observation vector, 20 to 40 decisions per episode, and a benchmark test enforcing $> 10\,000$ env steps/s on CPU.
