"""Vectorised fleet reward computation and Eureka-style runtime hot-swapping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class RewardState:
    """Fleet state snapshot used for pure NumPy reward computation.

    Supports both single-environment `(H,)` and vectorised `(E, H)` shapes,
    where `E` is the number of parallel environments, `H` is the number of
    hoppers, and `C` is the number of craters.
    """

    pos_xy: np.ndarray            # (..., H, 2) float32, metres in local frame
    fuel_kg: np.ndarray           # (..., H) float32
    battery_pct: np.ndarray       # (..., H) float32 in [0.0, 100.0]
    temp_c: np.ndarray            # (..., H) float32
    unsent_data: np.ndarray       # (..., H) float32 or int32
    crashed: np.ndarray           # (..., H) bool
    alive: np.ndarray             # (..., H) bool
    craters_done: np.ndarray      # (..., C) bool
    craters_claimed: np.ndarray   # (..., C) int32 (-1 if unclaimed, else hopper index)
    crater_priorities: np.ndarray # (..., C) or (C,) float32
    relay_open: np.ndarray | bool # (...,) bool
    relay_conflict: np.ndarray | bool  # (...,) bool
    at_crater_id: np.ndarray      # (..., H) int32 (-1 if not at a crater, else crater index)


RewardFn = Callable[[RewardState, np.ndarray, RewardState], tuple[np.ndarray, dict[str, float]]]

reward_versions: list[dict[str, object]] = []


def _to_batch(state: RewardState, actions: np.ndarray) -> tuple[RewardState, np.ndarray, bool]:
    """Promote 1D `(H,)` inputs to 2D `(1, H)` for unified vectorised computation."""
    fuel = np.asarray(state.fuel_kg, dtype=np.float32)
    single_env = fuel.ndim == 1
    if not single_env:
        return state, np.asarray(actions, dtype=np.int32), False

    num_craters = np.asarray(state.craters_done).shape[-1]
    priorities = np.asarray(state.crater_priorities, dtype=np.float32)
    if priorities.ndim == 1:
        priorities = priorities[None, :]

    batched_state = RewardState(
        pos_xy=np.asarray(state.pos_xy, dtype=np.float32)[None, ...],
        fuel_kg=fuel[None, ...],
        battery_pct=np.asarray(state.battery_pct, dtype=np.float32)[None, ...],
        temp_c=np.asarray(state.temp_c, dtype=np.float32)[None, ...],
        unsent_data=np.asarray(state.unsent_data, dtype=np.float32)[None, ...],
        crashed=np.asarray(state.crashed, dtype=bool)[None, ...],
        alive=np.asarray(state.alive, dtype=bool)[None, ...],
        craters_done=np.asarray(state.craters_done, dtype=bool).reshape(1, num_craters),
        craters_claimed=np.asarray(state.craters_claimed, dtype=np.int32).reshape(1, num_craters),
        crater_priorities=priorities.reshape(1, num_craters),
        relay_open=np.asarray(state.relay_open, dtype=bool).reshape(1),
        relay_conflict=np.asarray(state.relay_conflict, dtype=bool).reshape(1),
        at_crater_id=np.asarray(state.at_crater_id, dtype=np.int32)[None, ...],
    )
    batched_actions = np.asarray(actions, dtype=np.int32)[None, ...]
    return batched_state, batched_actions, True


def default_compute_reward(
    state: RewardState,
    actions: np.ndarray,
    next_state: RewardState,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute vectorised per-hopper rewards and component breakdown statistics."""
    s0, act, single_env = _to_batch(state, actions)
    s1, _, _ = _to_batch(next_state, actions)

    num_envs, num_hoppers = s0.fuel_kg.shape
    num_craters = s0.craters_done.shape[-1]

    priorities = np.asarray(s1.crater_priorities, dtype=np.float32)
    if priorities.ndim == 1:
        priorities = np.broadcast_to(priorities[None, :], (num_envs, num_craters))

    # 1. Crater scan milestone reward: craters_done transitions False -> True
    newly_done = (~s0.craters_done) & s1.craters_done  # (E, C)
    newly_done_weighted = newly_done.astype(np.float32) * priorities  # (E, C)

    # Credit the hopper currently at the newly completed crater (or claimed owner as fallback)
    hopper_ids = np.arange(num_hoppers, dtype=np.int32)[None, :, None]  # (1, H, 1)
    at_crater_1 = s1.at_crater_id[:, :, None]  # (E, H, 1)
    at_crater_0 = s0.at_crater_id[:, :, None]  # (E, H, 1)
    crater_ids = np.arange(num_craters, dtype=np.int32)[None, None, :]  # (1, 1, C)

    at_match = (at_crater_1 == crater_ids) | (at_crater_0 == crater_ids)  # (E, H, C)
    claimed_match = s1.craters_claimed[:, None, :] == hopper_ids  # (E, H, C)
    credit_mask = at_match | ((~np.any(at_match, axis=1, keepdims=True)) & claimed_mask_fallback(claimed_match, num_hoppers))

    scan_reward = 25.0 * np.sum(
        credit_mask.astype(np.float32) * newly_done_weighted[:, None, :],
        axis=-1,
    )  # (E, H)

    # 2. Crater claim shaping reward: craters_claimed transitions -1 -> hopper_idx
    unclaimed_before = s0.craters_claimed == -1  # (E, C)
    newly_claimed_by_h = unclaimed_before[:, None, :] & (s1.craters_claimed[:, None, :] == hopper_ids)  # (E, H, C)
    claim_reward = 5.0 * np.sum(
        newly_claimed_by_h.astype(np.float32) * priorities[:, None, :],
        axis=-1,
    )  # (E, H)

    # 3. Relay transmission reward: unsent_data drops during open relay window with no conflict
    relay_open = np.asarray(s0.relay_open, dtype=bool).reshape(num_envs, 1) | np.asarray(
        s1.relay_open, dtype=bool
    ).reshape(num_envs, 1)
    relay_conflict = np.asarray(s1.relay_conflict, dtype=bool).reshape(num_envs, 1)
    data_sent = np.maximum(0.0, s0.unsent_data - s1.unsent_data)  # (E, H)
    clean_send = relay_open & (~relay_conflict) & (s1.unsent_data <= 0.0) & (s0.unsent_data > 0.0)
    relay_reward = np.where(clean_send, 15.0 * data_sent, 0.0).astype(np.float32)

    # 4. Fuel burn penalty
    fuel_burned = np.maximum(0.0, s0.fuel_kg - s1.fuel_kg)
    fuel_penalty = (-0.2 * fuel_burned).astype(np.float32)

    # 5. Crash transition penalty: crashed transitions False -> True
    newly_crashed = (~s0.crashed) & s1.crashed
    crash_penalty = np.where(newly_crashed, -50.0, 0.0).astype(np.float32)

    # 6. Low battery penalty: battery_pct < 20.0
    low_battery_deficit = np.maximum(0.0, 20.0 - s1.battery_pct) / 20.0
    battery_penalty = np.where(s1.battery_pct < 20.0, -5.0 * (1.0 + low_battery_deficit), 0.0).astype(np.float32)

    # 7. Close separation penalty: < 100.0 m between alive hoppers
    if num_hoppers > 1:
        diff = s1.pos_xy[:, :, None, :] - s1.pos_xy[:, None, :, :]  # (E, H, H, 2)
        dist = np.linalg.norm(diff, axis=-1)  # (E, H, H)
        eye_mask = np.eye(num_hoppers, dtype=bool)[None, :, :]
        both_alive = s1.alive[:, :, None] & s1.alive[:, None, :] & (~eye_mask)
        close_pair = both_alive & (dist < 100.0)
        any_close = np.any(close_pair, axis=-1)  # (E, H)
        min_dist = np.where(both_alive, dist, np.inf).min(axis=-1)  # (E, H)
        closeness = np.where(any_close, (100.0 - np.minimum(min_dist, 100.0)) / 100.0, 0.0)
        separation_penalty = np.where(any_close, -5.0 * (1.0 + closeness), 0.0).astype(np.float32)
    else:
        separation_penalty = np.zeros((num_envs, num_hoppers), dtype=np.float32)

    # 8. Relay conflict penalty: relay_conflict is True (especially for hoppers choosing send action C + 2)
    send_action_idx = num_craters + 2
    chose_send = (act == send_action_idx) | (act == 2)
    conflict_mask = relay_conflict & (chose_send | (~np.any(chose_send, axis=-1, keepdims=True)))
    relay_conflict_penalty = np.where(conflict_mask, -10.0, 0.0).astype(np.float32)

    total_reward = (
        scan_reward
        + claim_reward
        + relay_reward
        + fuel_penalty
        + crash_penalty
        + battery_penalty
        + separation_penalty
        + relay_conflict_penalty
    ).astype(np.float32)

    components: dict[str, float] = {
        "scan_reward": float(np.mean(scan_reward)),
        "claim_reward": float(np.mean(claim_reward)),
        "relay_reward": float(np.mean(relay_reward)),
        "fuel_penalty": float(np.mean(fuel_penalty)),
        "crash_penalty": float(np.mean(crash_penalty)),
        "battery_penalty": float(np.mean(battery_penalty)),
        "separation_penalty": float(np.mean(separation_penalty)),
        "relay_conflict_penalty": float(np.mean(relay_conflict_penalty)),
        "total_reward": float(np.mean(total_reward)),
    }

    if single_env:
        return total_reward[0], components
    return total_reward, components


def claimed_mask_fallback(claimed_match: np.ndarray, num_hoppers: int) -> np.ndarray:
    """Return claimed hopper mask or all-hoppers mask if nobody claimed the newly done crater."""
    any_claimed = np.any(claimed_match, axis=1, keepdims=True)
    all_hoppers = np.ones_like(claimed_match, dtype=bool)
    return np.where(any_claimed, claimed_match, all_hoppers)


_active_reward_fn: RewardFn = default_compute_reward


def compute_reward(
    state: RewardState,
    actions: np.ndarray,
    next_state: RewardState,
) -> tuple[np.ndarray, dict[str, float]]:
    """Delegate to the currently active reward function."""
    return _active_reward_fn(state, actions, next_state)


def reset_default_reward() -> None:
    """Restore the default fleet reward function."""
    global _active_reward_fn
    _active_reward_fn = default_compute_reward


def _make_dummy_state(batch_shape: tuple[int, ...], num_hoppers: int = 3, num_craters: int = 4) -> RewardState:
    """Construct a synthetic RewardState for smoke-testing hot-swapped reward code."""
    h_shape = (*batch_shape, num_hoppers)
    c_shape = (*batch_shape, num_craters)
    return RewardState(
        pos_xy=np.zeros((*h_shape, 2), dtype=np.float32),
        fuel_kg=np.full(h_shape, 50.0, dtype=np.float32),
        battery_pct=np.full(h_shape, 80.0, dtype=np.float32),
        temp_c=np.zeros(h_shape, dtype=np.float32),
        unsent_data=np.zeros(h_shape, dtype=np.float32),
        crashed=np.zeros(h_shape, dtype=bool),
        alive=np.ones(h_shape, dtype=bool),
        craters_done=np.zeros(c_shape, dtype=bool),
        craters_claimed=np.full(c_shape, -1, dtype=np.int32),
        crater_priorities=np.ones(c_shape, dtype=np.float32),
        relay_open=np.zeros(batch_shape, dtype=bool) if batch_shape else np.asarray(False, dtype=bool),
        relay_conflict=np.zeros(batch_shape, dtype=bool) if batch_shape else np.asarray(False, dtype=bool),
        at_crater_id=np.full(h_shape, -1, dtype=np.int32),
    )


def load_reward_code(code_str: str, summary: str = "", episode: int = 0) -> RewardFn:
    """Compile and validate a candidate reward function before swapping it in at runtime.

    Raises:
        ValueError: If compilation fails, `compute_reward` is missing, or smoke tests fail.
    """
    global _active_reward_fn

    if not isinstance(code_str, str) or not code_str.strip():
        raise ValueError("Reward code string must be a non-empty string.")

    namespace: dict[str, object] = {
        "__builtins__": __builtins__,
        "np": np,
        "numpy": np,
        "math": math,
        "RewardState": RewardState,
        "default_compute_reward": default_compute_reward,
    }

    try:
        compiled = compile(code_str, "<hot_swap_reward>", "exec")
        exec(compiled, namespace)
    except Exception as exc:
        raise ValueError(f"Failed to compile or execute reward code: {exc}") from exc

    candidate = namespace.get("compute_reward")
    if not callable(candidate):
        raise ValueError("Reward code must define a callable `compute_reward(state, actions, next_state)`.")

    # Smoke-test on both 1D (H,) and 2D (E, H) dummy states
    for batch_shape in ((), (2,)):
        dummy_s0 = _make_dummy_state(batch_shape, num_hoppers=3, num_craters=4)
        dummy_s1 = _make_dummy_state(batch_shape, num_hoppers=3, num_craters=4)
        dummy_actions = np.zeros((*batch_shape, 3), dtype=np.int32)
        expected_shape = (*batch_shape, 3)

        try:
            result = candidate(dummy_s0, dummy_actions, dummy_s1)
        except Exception as exc:
            raise ValueError(f"Candidate compute_reward raised an exception on shape {expected_shape}: {exc}") from exc

        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("compute_reward must return a tuple of (rewards, info_dict).")

        rewards, info = result
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        if rewards_arr.shape != expected_shape:
            raise ValueError(
                f"compute_reward returned shape {rewards_arr.shape}, expected {expected_shape}."
            )
        if not np.all(np.isfinite(rewards_arr)):
            raise ValueError("compute_reward returned non-finite values on dummy inputs.")
        if not isinstance(info, dict):
            raise ValueError("compute_reward second return value must be a dict[str, float].")

    def _wrapped_candidate(
        state: RewardState,
        actions: np.ndarray,
        next_state: RewardState,
    ) -> tuple[np.ndarray, dict[str, float]]:
        r, info = candidate(state, actions, next_state)
        return np.asarray(r, dtype=np.float32), {str(k): float(v) for k, v in info.items()}

    _active_reward_fn = _wrapped_candidate
    reward_versions.append({"episode": int(episode), "summary": str(summary), "code": code_str})
    return _active_reward_fn
