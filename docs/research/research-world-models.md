# World models for a hopper fleet

## Candidates compared

| Approach | Architecture | Strengths | One-day laptop fit |
|---|---|---|---|
| **PETS** (Chua et al., 2018) | Ensemble of $K=5$ small MLPs predicting `(obs, action) -> next_obs, reward` | Trains in 1 to 3 minutes on low-dimensional observations; ensemble disagreement directly measures epistemic uncertainty | **High.** Fast, reliable, interpretable |
| **MBPO** (Janner et al., 2019) | PETS dynamics ensemble + short $k$-step rollouts branched from real states | Prevents compounding error over long horizons | **Stretch goal.** Planned as the final issue; cut if late |
| **DreamerV3** (Hafner et al., 2023) | Recurrent State-Space Model (RSSM) with latent discrete states | Designed for visual observations | **Low.** Too slow and complex for ~50-float observations |

## 5-MLP ensemble design (`world_model.py`)

All hoppers in the fleet pool their simulator transitions `(obs, action, next_obs, reward, crashed, target_crater)` into a shared replay buffer:

1. **Architecture:** $K = 5$ independent 2-layer MLPs (64 to 128 hidden units) taking `(obs, one_hot_action)` and predicting $(\Delta \text{obs}, \hat{r}, \hat{p}_{\text{crash}})$, including predicted landing position change $(\Delta x, \Delta y, \Delta z)$.
2. **Training speed (1 to 3 minutes):** Trained with Adam on bootstrapped minibatches from the shared buffer. Validation MSE on held-out transitions is logged across epochs to verify that prediction error falls steadily.
3. **Uncertainty and error map:**
   - **Ensemble disagreement (uncertainty):** Standard deviation across the 5 ensemble members' predictions:
     $$\sigma_{\text{pos}} = \sqrt{\frac{1}{K}\sum_{k=1}^K \|\hat{\mathbf{p}}_k - \bar{\mathbf{p}}\|_2^2}$$
   - **Realised world-model error:** L2 distance $\|\mathbf{p}_{\text{sim}} - \bar{\mathbf{p}}\|_2$ between simulator truth and ensemble mean prediction.
   - Less-visited craters and steep rims naturally exhibit higher ensemble disagreement $\sigma_{\text{pos}}$ and higher prediction error, which is mapped per crater and per grid cell for Argon to review and logged as `wm_pos` and `wm_error` in `data/episodes.json`.

## Imagination training as a stretch goal

Because our vectorised numpy simulator runs at $> 10\,000$ steps/s on CPU, MAPPO can train directly on the simulator while `world_model.py` learns from the logged transitions. Training the policy inside `world_model.py` via short MBPO branched rollouts is isolated as the final issue (#13) so it can be cut cleanly if time is tight.

## Sources

- Chua et al. (2018), *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)*, NeurIPS. [https://arxiv.org/abs/1805.12114](https://arxiv.org/abs/1805.12114)
- Janner et al. (2019), *When to Trust Your Model: Model-Based Policy Optimization (MBPO)*, NeurIPS. [https://arxiv.org/abs/1906.08253](https://arxiv.org/abs/1906.08253)
- Hafner et al. (2023), *Mastering Diverse Domains through World Models (DreamerV3)*, arXiv. [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

## Recommendation

Implement `selene_sim/world_model.py` as a 5-MLP ensemble predicting `(obs, action) -> next_obs, reward, crash_prob` that trains in 1 to 3 minutes, verifies falling validation error and higher uncertainty on less-visited craters, and keeps MBPO imagination rollouts as a separate final issue.
