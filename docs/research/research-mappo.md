# MAPPO for the hopper fleet

## Core algorithm

Multi-Agent PPO (MAPPO) applies Clipped Proximal Policy Optimization under Centralised Training with Decentralised Execution (CTDE):

- **Shared policy (actor) $\pi_\theta(a_i \mid o_i)$:** One 2-layer MLP (64 to 128 hidden units, `Tanh` activations) shared across all $N$ hoppers (parameter sharing), with a one-hot hopper index included in each hopper's ~50-float observation $o_i$.
- **Centralised value function (critic) $V_\phi(s_i)$:** A 2-layer MLP (64 to 128 hidden units) receiving the compact global fleet state $s$ plus hopper identity, outputting a scalar value estimate per hopper during training.
- **Per-agent GAE and clipped objective:** Generalized Advantage Estimation (GAE) runs over `(E, T, N)` rollout buffers from the vectorised numpy environment ($E = 64\text{ to }256$ parallel environments, $T = 20\text{ to }40$ steps).

## Hyperparameters that matter (Yu et al., 2022)

1. **Value normalisation:** Running mean/std normalisation on critic targets keeps value updates stable when reward scales change or when Argon rewrites `reward.py`.
2. **Compact global state:** Concatenating fleet positions, fuel, battery, crater claimed/done flags, and relay window timer keeps the critic input under 80 floats.
3. **Epoch count and clipping:** 4 update epochs per rollout, clipping $\epsilon = 0.2$, learning rate $5 \times 10^{-4}$, GAE $\gamma = 0.99, \lambda = 0.95$, entropy bonus $0.01$.
4. **Action masking before softmax:** Set invalid action logits to $-10^8$ prior to the categorical distribution so out-of-range hops, unfunded hops, and invalid `scan`/`send` actions have zero probability.
5. **CPU vs MPS auto-benchmark:** For 2-layer MLPs with 64 to 128 units, CPU often beats Apple Silicon `"mps"` due to kernel launch overhead. `mappo.py` benchmarks a forward/backward pass on `"cpu"` vs `"mps"` at startup and picks the faster device automatically.

## Fast iteration workflow (`--quick`, CSV logging, checkpoints)

- **`--quick` mode:** Runs 2 hoppers, 6 craters, synthetic terrain, with a 2-minute wall-clock budget. Used for all debugging and rapid verification that mean reward rises and crash rate drops within 2 to 5 minutes.
- **Full mode:** Scales to 4 to 5 hoppers, 12 to 15 craters, and the Daedalus terrain grid, reaching cooperative crater splitting and relay turn-taking within 20 to 40 minutes.
- **CSV telemetry and summary CLI:** Every $K$ updates, `mappo.py` appends a row to a CSV log (`update`, `env_steps`, `elapsed_s`, `episode_reward`, `crashes`, `craters_visited`, `crater_overlap`, `relay_conflicts`, `entropy`, `value_loss`, `wm_error`) and saves model checkpoints at fixed intervals. A lightweight CLI helper (`scripts/training_summary.py`) prints a table summary of the CSV so long runs executed in a separate terminal can be inspected at any time.

## Sources

- Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS Datasets and Benchmarks. [https://arxiv.org/abs/2103.01955](https://arxiv.org/abs/2103.01955)
- Official MAPPO repository (`on-policy`). [https://github.com/marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)
- Huang et al. (2022), *CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms*, JMLR. [https://jmlr.org/papers/v23/21-1342.html](https://jmlr.org/papers/v23/21-1342.html)

## Recommendation

Implement `selene_sim/mappo.py` with 2-layer 64-128 unit MLPs, automatic CPU vs MPS benchmarking, `--quick` mode (2 hoppers, 6 craters, 2-minute budget), periodic CSV logging and checkpointing, and a CLI summary reader.
