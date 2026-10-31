# Fast vectorised Monte Carlo rollouts in numpy

## Performance target

The verifier replays a fleet plan $M = 1000$ times with sampled noise (thrust error, landing scatter, dust, sensor noise) in the ground-truth physics simulator. ADR-0003 sets a hard wall-clock budget of under 10 seconds for 1000 rollouts. In practice, pure vectorised numpy on an Apple M1 can run 1000 rollouts of a 10-step, 5-hopper fleet plan in **under 50 milliseconds** (more than $100\times$ faster than the 10 s budget) if we avoid Python loops over runs.

## Array layout and broadcasting

A fleet plan has $N$ hoppers executing a sequence of up to $H$ timed steps (hops, scans, waits, relay transmissions). Instead of running a single-plan simulator inside `for m in range(1000):`, we allocate state tensors with leading dimension $(M, N)$:

- `pos`: `float32` of shape `(M, N, 3)` in metres.
- `fuel`: `float32` of shape `(M, N)` in $[0, 1]$.
- `battery`: `float32` of shape `(M, N)` in $[0, 1]$.
- `temp_c`: `float32` of shape `(M, N)`.

For each step $t \in \{0, \dots, H-1\}$ (a loop of length $H \le 15$):

1. **Sample all noise in one call:** Using `rng = np.random.default_rng(seed)`, draw `thrust_err = rng.normal(0.0, sigma_thrust, size=(M, N))`, `scatter_xy = rng.normal(0.0, sigma_scatter, size=(M, N, 2))`, and `dust_factor = rng.uniform(0.0, max_dust, size=(M, N))`.
2. **Vectorised terrain lookup:** Convert landing coordinates `(x, y)` of shape `(M, N)` to integer grid indices:
   ```python
   ix = np.clip(((x - x_min) / cell_size).astype(np.int32), 0, W - 1)
   iy = np.clip(((y - y_min) / cell_size).astype(np.int32), 0, H_grid - 1)
   landing_z = terrain_height[iy, ix]       # shape (M, N)
   landing_slope = terrain_slope[iy, ix]    # shape (M, N)
   landing_temp = terrain_temp[iy, ix]      # shape (M, N)
   ```
   Numpy fancy indexing over a $200 \times 200$ array for $1000 \times 5 = 5000$ points takes roughly $20\ \mu\text{s}$.
3. **Vectorised physics update:** Compute horizontal range $R$, launch speed $v = \sqrt{R g / \sin(2\theta)}$, $\Delta v$ with `thrust_err`, fuel burn from the rocket equation `dm = m * (1.0 - np.exp(-dv / v_e))`, and battery drain from base load, heater load (as a function of `landing_temp` and `dust_factor`), and hop cost.
4. **Vectorised fleet-level pairwise checks:**
   - **Pairwise hopper distance:** Compute `diff = pos[:, :, None, :2] - pos[:, None, :, :2]` of shape `(M, N, N, 2)`, Euclidean distance `dist = np.linalg.norm(diff, axis=-1)`, and mask the diagonal `np.eye(N, dtype=bool)` with `np.inf`. Minimum separation per rollout step is `dist.min(axis=(-2, -1))` of shape `(M,)`.
   - **Relay conflict count:** Count how many hoppers have `action == SEND` during the same step: `send_count = is_sending.sum(axis=-1)` of shape `(M,)`. Conflict occurs where `send_count > 1` or when outside a relay window.

## Output trace structure for Friend B

`simulate_plan(fleet_plan, n=1000, seed=42)` returns a `RolloutBatch` (dataclass or dictionary of numpy arrays of shape `(M, H, N, ...)` plus per-hop summary arrays) so Friend B's safety spec check functions can evaluate rules either via vectorised numpy expressions (`traces["fuel"][:, :, :] >= 0.10`) or per rollout.

## Sources

- Harris et al. (2020), *Array programming with NumPy*, Nature 585, 357-362. [https://doi.org/10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- NumPy Generator (`np.random.default_rng`) documentation. [https://numpy.org/doc/stable/reference/random/generator.html](https://numpy.org/doc/stable/reference/random/generator.html)

## Recommendation

Share the core physics formulas between `sim.py` and `verify.py` as vectorised numpy functions that operate on arbitrary leading dimensions `(...)`. Run `simulate_plan` across shape `(n, N)` simultaneously with `np.random.default_rng(seed)` so 1000 noisy rollouts finish in under 0.1 seconds using only the ground-truth simulator.
