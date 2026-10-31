# World models for a hopper fleet

## Candidates compared

| Approach | Architecture | Strengths | One-day hackathon fit |
|---|---|---|---|
| **PETS** (Chua et al., 2018) | Ensemble of $K=5$ probabilistic MLPs predicting state deltas | Fast training on low-dimensional states; ensemble disagreement directly measures epistemic uncertainty | **High.** Trains in seconds on an M1 laptop |
| **MBPO** (Janner et al., 2019) | PETS dynamics ensemble + short $k$-step rollouts branched from real replay states | Avoids compounding model error; feeds standard policy gradients without changing the actor | **High.** 1- to 3-step rollouts slot directly into MAPPO |
| **DreamerV3** (Hafner et al., 2023) | Recurrent State-Space Model (RSSM) with discrete latent states, symlog predictions, imagination actor-critic | Handles raw image observations and long credit assignment | **Low.** Overkill for low-dimensional tabular state; slow to debug and tune in one day |

## Why an MLP ensemble fits Selene

Our hopper state is low-dimensional and physical: position $(x, y, z)$, fuel, battery, internal temperature, plus a local $5 \times 5$ terrain patch (height and slope) and action vector. There are no camera images requiring a latent autoencoder.

Following PETS, we train an ensemble of $K = 5$ small MLPs $\{f_{\psi_1}, \dots, f_{\psi_5}\}$ on transitions pooled from all hoppers in the fleet:

1. **Delta prediction:** Each network takes $(s_{\text{local}}, a)$ and predicts state changes $\Delta \hat{s} = (\Delta x, \Delta y, \Delta z, \Delta \text{fuel}, \Delta \text{battery}, \Delta T)$ plus a landing crash probability $\hat{p}_{\text{crash}} \in [0, 1]$. Next state is $\hat{s}' = s + \Delta \hat{s}$.
2. **Independent initialisation and bootstrapped batches:** Each ensemble member is initialised with a different seed and trained on random minibatches from the shared replay buffer using MSE on normalised state deltas and binary cross-entropy on crash labels.
3. **Epistemic uncertainty from disagreement:** The ensemble prediction is the mean $\bar{s}' = \frac{1}{K}\sum_k \hat{s}'_k$, and the uncertainty at a target cell is the spatial standard deviation across the $K$ predicted landing positions:
   $$\sigma_{\text{pos}} = \sqrt{\frac{1}{K}\sum_{k=1}^K \|\hat{\mathbf{p}}_k - \bar{\mathbf{p}}\|_2^2}$$
   On unvisited crater rims, the 5 networks extrapolate differently, producing high $\sigma_{\text{pos}}$ and large prediction error $\|\mathbf{p}_{\text{sim}} - \bar{\mathbf{p}}\|_2$ when a hopper actually hops there.

## Short imagination rollouts (MBPO style)

Long rollouts inside a learned model compound errors and let the policy exploit false high-reward states (Janner et al., 2019). For imagination training:

- Sample real states from the simulator buffer.
- Roll out the shared MAPPO policy for $k \in \{1, 2, 3\}$ hops inside the ensemble (randomly picking one ensemble member per step, or stepping the ensemble mean).
- Penalise steps where ensemble disagreement $\sigma_{\text{pos}}$ exceeds a threshold (MOPO/MBPO uncertainty penalty) or terminate the imagined branch.
- Evaluate the policy every $M$ updates in the real simulator. The verifier never touches `world_model.py`.

## Sources

- Chua et al. (2018), *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)*, NeurIPS. [https://arxiv.org/abs/1805.12114](https://arxiv.org/abs/1805.12114)
- Janner et al. (2019), *When to Trust Your Model: Model-Based Policy Optimization (MBPO)*, NeurIPS. [https://arxiv.org/abs/1906.08253](https://arxiv.org/abs/1906.08253)
- Hafner et al. (2023), *Mastering Diverse Domains through World Models (DreamerV3)*, arXiv. [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

## Recommendation

Implement `selene_sim/world_model.py` as a 5-MLP ensemble predicting state deltas and crash probability from pooled hopper transitions. Compute epistemic uncertainty via ensemble standard deviation, log per-step prediction error (`wm_pos`, `wm_error`) for `episodes.json` and the per-cell error map for the orchestrator, and use 1- to 3-step MBPO branched rollouts for imagination training (with a clean flag to train directly on the simulator if cut for time).
