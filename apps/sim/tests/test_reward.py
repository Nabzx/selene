"""Unit tests for fleet reward computation and hot-swapping."""

from __future__ import annotations

import numpy as np
import pytest

from selene_sim import reward
from selene_sim.reward import (
    RewardState,
    compute_reward,
    default_compute_reward,
    load_reward_code,
    reset_default_reward,
)


def make_single_state(
    num_hoppers: int = 3,
    num_craters: int = 4,
    *,
    pos_xy: np.ndarray | None = None,
    fuel_kg: np.ndarray | None = None,
    battery_pct: np.ndarray | None = None,
    temp_c: np.ndarray | None = None,
    unsent_data: np.ndarray | None = None,
    crashed: np.ndarray | None = None,
    alive: np.ndarray | None = None,
    craters_done: np.ndarray | None = None,
    craters_claimed: np.ndarray | None = None,
    crater_priorities: np.ndarray | None = None,
    relay_open: bool | np.ndarray = False,
    relay_conflict: bool | np.ndarray = False,
    at_crater_id: np.ndarray | None = None,
) -> RewardState:
    """Create a 1D (H,) RewardState with safe default values."""
    if pos_xy is None:
        # Place hoppers 500 m apart so there is no separation penalty by default.
        pos_xy = np.array([[i * 500.0, 0.0] for i in range(num_hoppers)], dtype=np.float32)
    if fuel_kg is None:
        fuel_kg = np.full(num_hoppers, 50.0, dtype=np.float32)
    if battery_pct is None:
        battery_pct = np.full(num_hoppers, 80.0, dtype=np.float32)
    if temp_c is None:
        temp_c = np.full(num_hoppers, 10.0, dtype=np.float32)
    if unsent_data is None:
        unsent_data = np.zeros(num_hoppers, dtype=np.float32)
    if crashed is None:
        crashed = np.zeros(num_hoppers, dtype=bool)
    if alive is None:
        alive = ~crashed
    if craters_done is None:
        craters_done = np.zeros(num_craters, dtype=bool)
    if craters_claimed is None:
        craters_claimed = np.full(num_craters, -1, dtype=np.int32)
    if crater_priorities is None:
        crater_priorities = np.ones(num_craters, dtype=np.float32)
    if at_crater_id is None:
        at_crater_id = np.full(num_hoppers, -1, dtype=np.int32)

    return RewardState(
        pos_xy=np.asarray(pos_xy, dtype=np.float32),
        fuel_kg=np.asarray(fuel_kg, dtype=np.float32),
        battery_pct=np.asarray(battery_pct, dtype=np.float32),
        temp_c=np.asarray(temp_c, dtype=np.float32),
        unsent_data=np.asarray(unsent_data, dtype=np.float32),
        crashed=np.asarray(crashed, dtype=bool),
        alive=np.asarray(alive, dtype=bool),
        craters_done=np.asarray(craters_done, dtype=bool),
        craters_claimed=np.asarray(craters_claimed, dtype=np.int32),
        crater_priorities=np.asarray(crater_priorities, dtype=np.float32),
        relay_open=np.asarray(relay_open, dtype=bool),
        relay_conflict=np.asarray(relay_conflict, dtype=bool),
        at_crater_id=np.asarray(at_crater_id, dtype=np.int32),
    )


def make_batch_state(num_envs: int = 4, num_hoppers: int = 3, num_craters: int = 4) -> RewardState:
    """Create a 2D (E, H) vectorised RewardState."""
    single = make_single_state(num_hoppers=num_hoppers, num_craters=num_craters)
    return RewardState(
        pos_xy=np.broadcast_to(single.pos_xy, (num_envs, num_hoppers, 2)).copy(),
        fuel_kg=np.broadcast_to(single.fuel_kg, (num_envs, num_hoppers)).copy(),
        battery_pct=np.broadcast_to(single.battery_pct, (num_envs, num_hoppers)).copy(),
        temp_c=np.broadcast_to(single.temp_c, (num_envs, num_hoppers)).copy(),
        unsent_data=np.broadcast_to(single.unsent_data, (num_envs, num_hoppers)).copy(),
        crashed=np.broadcast_to(single.crashed, (num_envs, num_hoppers)).copy(),
        alive=np.broadcast_to(single.alive, (num_envs, num_hoppers)).copy(),
        craters_done=np.broadcast_to(single.craters_done, (num_envs, num_craters)).copy(),
        craters_claimed=np.broadcast_to(single.craters_claimed, (num_envs, num_craters)).copy(),
        crater_priorities=np.broadcast_to(single.crater_priorities, (num_envs, num_craters)).copy(),
        relay_open=np.zeros(num_envs, dtype=bool),
        relay_conflict=np.zeros(num_envs, dtype=bool),
        at_crater_id=np.broadcast_to(single.at_crater_id, (num_envs, num_hoppers)).copy(),
    )


@pytest.fixture(autouse=True)
def _ensure_default_reward() -> None:
    """Reset the active reward function before and after every test."""
    reset_default_reward()
    yield
    reset_default_reward()


def test_single_and_vectorised_shapes() -> None:
    """compute_reward supports both 1D (H,) and 2D (E, H) inputs and returns matching shapes."""
    s1 = make_single_state(num_hoppers=3, num_craters=4)
    a1 = np.zeros(3, dtype=np.int32)
    r1, info1 = compute_reward(s1, a1, s1)
    assert r1.shape == (3,)
    assert r1.dtype == np.float32
    assert isinstance(info1, dict)
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in info1.items())

    sb = make_batch_state(num_envs=5, num_hoppers=3, num_craters=4)
    ab = np.zeros((5, 3), dtype=np.int32)
    rb, infob = compute_reward(sb, ab, sb)
    assert rb.shape == (5, 3)
    assert rb.dtype == np.float32
    assert isinstance(infob, dict)
    np.testing.assert_allclose(rb[0], r1, rtol=1e-5, atol=1e-6)


def test_positive_reward_for_scanning_and_claiming_crater() -> None:
    """Scanning a newly completed crater and claiming an unclaimed crater yield positive rewards."""
    s0 = make_single_state(
        num_hoppers=3,
        num_craters=4,
        crater_priorities=np.array([1.0, 2.0, 1.0, 1.0], dtype=np.float32),
        at_crater_id=np.array([1, -1, -1], dtype=np.int32),
    )
    # Hopper 0 scans crater 1 (False -> True), hopper 2 claims crater 3 (-1 -> 2).
    craters_done_next = np.array([False, True, False, False], dtype=bool)
    craters_claimed_next = np.array([-1, 0, -1, 2], dtype=np.int32)
    s1 = make_single_state(
        num_hoppers=3,
        num_craters=4,
        crater_priorities=np.array([1.0, 2.0, 1.0, 1.0], dtype=np.float32),
        craters_done=craters_done_next,
        craters_claimed=craters_claimed_next,
        at_crater_id=np.array([1, -1, 3], dtype=np.int32),
    )
    # Actions: C = 4, scan action is C + 1 = 5 (or 1), wait is C = 4 (or 0).
    actions = np.array([5, 4, 3], dtype=np.int32)

    base_r, _ = compute_reward(s0, actions, s0)
    r, info = compute_reward(s0, actions, s1)

    # Hopper 0 scanned crater 1 and claimed crater 1 -> clear positive reward over baseline.
    assert r[0] > base_r[0] + 5.0
    # Hopper 2 claimed crater 3 -> positive claim reward over baseline.
    assert r[2] > base_r[2]
    assert info["scan_reward"] > 0.0
    assert info["claim_reward"] > 0.0


def test_positive_reward_for_relay_send_without_conflict() -> None:
    """Sending unsent data during an open relay window with no conflict gives positive reward."""
    s0 = make_single_state(
        num_hoppers=2,
        num_craters=4,
        unsent_data=np.array([2.0, 0.0], dtype=np.float32),
        relay_open=True,
        relay_conflict=False,
    )
    s1 = make_single_state(
        num_hoppers=2,
        num_craters=4,
        unsent_data=np.array([0.0, 0.0], dtype=np.float32),
        relay_open=True,
        relay_conflict=False,
    )
    # C = 4, send action is C + 2 = 6
    actions = np.array([6, 4], dtype=np.int32)
    base_r, _ = compute_reward(s0, actions, s0)
    r, info = compute_reward(s0, actions, s1)

    assert r[0] > base_r[0] + 5.0
    assert info["relay_reward"] > 0.0


def test_negative_penalties() -> None:
    """Fuel burn, crash transition, low battery, close separation, and relay conflict incur negative penalties."""
    num_hoppers = 3
    num_craters = 4
    s_clean = make_single_state(
        num_hoppers=num_hoppers,
        num_craters=num_craters,
        unsent_data=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        relay_open=True,
    )
    wait_actions = np.full(num_hoppers, num_craters, dtype=np.int32)
    r_clean, _ = compute_reward(s_clean, wait_actions, s_clean)

    # 1. Fuel burn penalty
    s_fuel = make_single_state(
        num_hoppers=num_hoppers,
        num_craters=num_craters,
        fuel_kg=np.array([40.0, 50.0, 50.0], dtype=np.float32),
        unsent_data=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        relay_open=True,
    )
    r_fuel, info_fuel = compute_reward(s_clean, wait_actions, s_fuel)
    assert r_fuel[0] < r_clean[0]
    assert info_fuel["fuel_penalty"] < 0.0

    # 2. Crash transition penalty (False -> True)
    s_crash = make_single_state(
        num_hoppers=num_hoppers,
        num_craters=num_craters,
        crashed=np.array([True, False, False], dtype=bool),
        alive=np.array([False, True, True], dtype=bool),
        unsent_data=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        relay_open=True,
    )
    r_crash, info_crash = compute_reward(s_clean, wait_actions, s_crash)
    assert r_crash[0] < r_clean[0] - 10.0
    assert info_crash["crash_penalty"] < 0.0

    # 3. Low battery penalty (< 20.0%)
    s_low_bat = make_single_state(
        num_hoppers=num_hoppers,
        num_craters=num_craters,
        battery_pct=np.array([10.0, 80.0, 80.0], dtype=np.float32),
        unsent_data=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        relay_open=True,
    )
    r_bat, info_bat = compute_reward(s_clean, wait_actions, s_low_bat)
    assert r_bat[0] < r_clean[0]
    assert info_bat["battery_penalty"] < 0.0

    # 4. Close separation penalty (< 100.0 m between alive hoppers)
    s_close = make_single_state(
        num_hoppers=num_hoppers,
        num_craters=num_craters,
        pos_xy=np.array([[0.0, 0.0], [30.0, 40.0], [500.0, 0.0]], dtype=np.float32),
        unsent_data=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        relay_open=True,
    )
    r_close, info_close = compute_reward(s_clean, wait_actions, s_close)
    assert r_close[0] < r_clean[0]
    assert r_close[1] < r_clean[1]
    assert r_close[2] == pytest.approx(r_clean[2])
    assert info_close["separation_penalty"] < 0.0

    # 5. Relay conflict penalty (multiple hoppers choose send action C + 2 when relay_conflict is True)
    send_action = num_craters + 2
    conflict_actions = np.array([send_action, send_action, num_craters], dtype=np.int32)
    s_conflict = make_single_state(
        num_hoppers=num_hoppers,
        num_craters=num_craters,
        unsent_data=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        relay_open=True,
        relay_conflict=True,
    )
    r_conflict, info_conflict = compute_reward(s_clean, conflict_actions, s_conflict)
    assert r_conflict[0] < r_clean[0]
    assert r_conflict[1] < r_clean[1]
    assert info_conflict["relay_conflict_penalty"] < 0.0


def test_hot_swap_and_validation() -> None:
    """load_reward_code hot-swaps compute_reward and rejects invalid/crashing code."""
    s = make_single_state(num_hoppers=2, num_craters=3)
    a = np.zeros(2, dtype=np.int32)
    default_r, _ = default_compute_reward(s, a, s)

    custom_code = """
import numpy as np

def compute_reward(state, actions, next_state):
    rewards = np.full(state.fuel_kg.shape, 42.0, dtype=np.float32)
    return rewards, {"custom_constant": 42.0}
"""
    load_reward_code(custom_code)

    swapped_r, swapped_info = compute_reward(s, a, s)
    np.testing.assert_allclose(swapped_r, np.array([42.0, 42.0], dtype=np.float32))
    assert swapped_info["custom_constant"] == 42.0

    # Calling via module attribute also hits the swapped version
    mod_r, _ = reward.compute_reward(s, a, s)
    np.testing.assert_allclose(mod_r, np.array([42.0, 42.0], dtype=np.float32))

    # Reject syntax errors or runtime crashes and preserve the currently active reward function
    bad_code_crash = """
def compute_reward(state, actions, next_state):
    raise RuntimeError("boom")
"""
    with pytest.raises(ValueError):
        load_reward_code(bad_code_crash)

    bad_code_shape = """
import numpy as np
def compute_reward(state, actions, next_state):
    return np.zeros((1,), dtype=np.float32), {}
"""
    with pytest.raises(ValueError):
        load_reward_code(bad_code_shape)

    # Active function must still be the valid custom one
    still_swapped_r, _ = compute_reward(s, a, s)
    np.testing.assert_allclose(still_swapped_r, np.array([42.0, 42.0], dtype=np.float32))

    # Reset back to default
    reset_default_reward()
    restored_r, _ = compute_reward(s, a, s)
    np.testing.assert_allclose(restored_r, default_r)
