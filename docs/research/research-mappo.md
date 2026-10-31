# MAPPO for the hopper fleet

## Core algorithm

Multi-Agent PPO (MAPPO) applies standard Clipped Proximal Policy Optimization under Centralised Training with Decentralised Execution (CTDE):

- **Shared policy (actor) $\pi_\theta(a_i \mid o_i)$:** One network shared across all $N$ hoppers (parameter sharing), with a one-hot hopper index appended to each observation $o_i$ so agents can specialise roles.
- **Centralised value function (critic) $V_\phi(s)$:** Receives the global state $s$ (all hoppers' positions, fuel, battery, crater claim/done flags, and time to the next relay window) and outputs a scalar value estimate per hopper. At runtime (and during verification), only $\pi_\theta$ is needed.
- **Per-agent GAE and clipped objective:** Generalized Advantage Estimation (GAE) runs over each hopper's trajectory using its own critic value $V_\phi(s_t^{(i)})$. Rollout batches flatten across environments and agents to $(B \cdot T \cdot N, \dots)$ for minibatch SGD.

## Hyperparameters that matter (Yu et al., 2022)

Yu et al. showed that tuned PPO matches or beats complex off-policy MARL methods when five practical details are handled properly:

1. **Value normalisation (PopArt or running mean/std):** Reward scales shift during training (especially when the orchestrator rewrites `reward.py`). Normalising value targets by a running standard deviation keeps critic gradients stable.
2. **Global state design:** Concatenating raw observations from all agents creates redundant dimensions. A compact global state $s$ containing fleet coordinates, resources, crater states, and agent-specific relative features trains faster.
3. **Epoch count and clipping:** Fewer epochs than single-agent PPO (4 to 5 epochs instead of 10 to 15) prevent joint policy divergence when all agents update at once. Clipping coefficient $\epsilon = 0.2$, learning rate $5 \times 10^{-4}$ with linear annealing, GAE $\gamma = 0.99, \lambda = 0.95$.
4. **Update batching:** Do not split minibatches into tiny slices; use 1 to 2 minibatches per epoch so advantage normalisation across $(B \cdot T \cdot N)$ stays low-variance.
5. **Action masking before softmax:** Set invalid action logits to $-10^8$ prior to the categorical distribution so out-of-range hops have zero probability and zero gradient contribution.

## Minimal single-file implementation

Heavy MARL libraries (RLlib, BenchMARL, EPyMARL) pull in Ray or Hydra and make hot-swapping `reward.py` awkward. A CleanRL-style single file (`mappo.py`, roughly 250 lines of PyTorch) needs only:

- Two 2-layer MLPs with `Tanh` activations and orthogonal weight initialisation (`gain=sqrt(2)` on hidden layers, `0.01` on policy head, `1.0` on value head).
- A vectorised rollout loop over $E = 8$ parallel simulator or world-model environments for $T = 128$ steps.
- Device selection preferring `"mps"` on Apple Silicon (M1) for network passes, falling back to `"cpu"` if tensor transfer overhead dominates on small batch sizes.

## Sources

- Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS Datasets and Benchmarks. [https://arxiv.org/abs/2103.01955](https://arxiv.org/abs/2103.01955)
- Official MAPPO repository (`on-policy`). [https://github.com/marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)
- Huang et al. (2022), *CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms*, JMLR. [https://jmlr.org/papers/v23/21-1342.html](https://jmlr.org/papers/v23/21-1342.html)

## Recommendation

Build `selene_sim/mappo.py` as a self-contained CleanRL-style PyTorch module with parameter sharing across hoppers, a compact global critic input, running value target normalisation, 4 update epochs per rollout, $\epsilon = 0.2$, and logit masking for invalid hops.
