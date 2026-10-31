# Reward design pitfalls and Eureka-style hot-swapping

## Common reward pitfalls in multi-hopper lunar exploration

Multi-agent exploration with fuel, battery, and survival constraints has four classic local optima that stall RL training:

1. **Sitting still to avoid fuel and crash penalties (the survival trap):** If fuel burn is penalised every hop and crashing carries a large negative penalty (for example $-50$), an untrained policy that crashes frequently learns that choosing `wait` every step yields $0$ reward, beating an exploratory policy that averages $-10$.
   - *Fix:* Potential-based distance shaping $\Phi(s') - \Phi(s)$ towards the nearest unclaimed crater, plus a large positive milestone reward for first-time crater scans that outweighs the fuel cost of getting there by at least $10\times$.
2. **Crater camping (repeatedly scanning the same crater):** If `scan` gives positive reward without checking whether the crater has already been scanned, a hopper reaches one easy crater and scans it forever.
   - *Fix:* Track `crater_scanned[c]` in the environment state; only award the scan bonus on the transition from unscanned to scanned, and mask `scan` thereafter.
3. **Herd behaviour (all hoppers chasing the nearest crater):** With parameter sharing, identical observations cause hoppers to pick the same crater and collide.
   - *Fix:* Include one-hot hopper IDs in the observation, apply a proximity penalty when any two hoppers come within the safety separation distance (for example $250\text{ m}$), and give a team-shared bonus for distinct craters visited across the fleet.
4. **Relay hogging and link collisions:** If sending data is rewarded regardless of whether another hopper is transmitting, two hoppers transmit in the same window and cause a relay conflict.
   - *Fix:* Only credit data transmission when exactly one hopper transmits during an open relay window; apply a relay-conflict penalty to both hoppers if two transmit at once.

## Eureka-style reward hot-swapping (Ma et al., 2023)

In Eureka, the LLM writes executable reward code against a fixed **reward signature**, receives **reward reflection** statistics (component-wise reward breakdown, episode survival rates, craters visited, fuel remaining), and rewrites the function when learning stalls.

To make `selene_sim/reward.py` safe and clean for Argon to rewrite at runtime:

1. **Frozen signature and pure typed inputs:**
   ```python
   def compute_reward(
       state: RewardState,
       actions: np.ndarray,
       next_state: RewardState,
   ) -> tuple[np.ndarray, dict[str, float]]:
   ```
   where `state` and `next_state` expose read-only numpy arrays (`pos`, `fuel`, `battery`, `temp_c`, `crashed`, `in_relay_window`, `unsent_data`, `crater_pos`, `crater_radii`, `crater_scanned`, `newly_scanned_by`, `relay_conflict`, `min_separation`) and the return value is `(rewards_per_hopper, term_breakdown_dict)`.
2. **Component breakdown for reflection:** Returning `term_breakdown_dict` (for example `{"crater_scan": 12.0, "fuel_cost": -1.4, "crash_penalty": -20.0, "proximity_penalty": 0.0}`) lets `training_summary()` report the exact mean contribution of each reward term to Argon, just as Eureka does.
3. **Safe hot-reload with validation:** Provide `load_reward_fn(code_str: str)` which compiles the candidate Python string in a restricted namespace (`np`, `math`, `RewardState`), runs a smoke test on dummy `RewardState` inputs, checks that output shape is `(N,)` of finite `float32` values and a `dict[str, float]`, and swaps the active reward function in the environment while logging an entry to `training.reward_versions`.

## Sources

- Ma et al. (2023), *Eureka: Human-Level Reward Design via Coding Large Language Models*, ICLR 2024. [https://arxiv.org/abs/2310.12931](https://arxiv.org/abs/2310.12931)
- Ng, Harada & Russell (1999), *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*, ICML. [https://ai.stanford.edu/~ang/papers/icml99-shaping.pdf](https://ai.stanford.edu/~ang/papers/icml99-shaping.pdf)

## Recommendation

Put `compute_reward(state, actions, next_state) -> tuple[np.ndarray, dict[str, float]]` in `selene_sim/reward.py` with a frozen `RewardState` dataclass and a validated `load_reward_code(source: str)` loader so Friend B's Argon orchestrator can inspect per-term reward statistics and hot-swap the reward function mid-training.
