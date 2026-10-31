"""Unit and benchmark tests for the vectorised lunar fleet environment (env.py)."""

from __future__ import annotations

import time

import numpy as np

from selene_sim.env import FleetEnv, VectorFleetEnv
from tests.test_episode_schema import validate_episode_doc


def _sample_masked_actions(masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from valid actions across an arbitrary batch shape (..., A)."""
    # Add small random uniform values to valid actions and pick argmax
    noise = rng.uniform(0.01, 1.0, size=masks.shape)
    scores = np.where(masks, noise, -1.0)
    return np.argmax(scores, axis=-1).astype(np.int32)


def test_stage1_random_masked_policy_1000_steps() -> None:
    """Stage 1 check: random masked policy runs 1,000 steps in FleetEnv and VectorFleetEnv with zero NaNs."""
    rng = np.random.default_rng(123)

    # 1. VectorFleetEnv in both quick=True and quick=False modes
    for quick in (True, False):
        vec_env = VectorFleetEnv(num_envs=8, quick=quick, seed=42, record_transitions=True)
        obs, info = vec_env.reset(seed=42)
        expected_h = 2 if quick else 4
        expected_c = 6 if quick else 12
        assert vec_env.n_hoppers == expected_h
        assert vec_env.n_craters == expected_c
        assert vec_env.n_actions == expected_c + 3
        assert obs.shape == (8, expected_h, vec_env.obs_dim)
        assert info["action_masks"].shape == (8, expected_h, vec_env.n_actions)
        assert vec_env.get_global_state().shape == (8, vec_env.global_state_dim)
        assert np.all(np.isfinite(obs))

        completed_episodes = 0
        for _ in range(1000):
            masks = vec_env.get_action_masks()
            assert np.all(masks[..., expected_c + 1])  # wait is always valid
            actions = _sample_masked_actions(masks, rng)
            obs, rewards, terminated, truncated, step_info = vec_env.step(actions)

            assert obs.shape == (8, expected_h, vec_env.obs_dim)
            assert rewards.shape == (8, expected_h)
            assert terminated.shape == (8,)
            assert truncated.shape == (8,)
            assert np.all(np.isfinite(obs))
            assert np.all(np.isfinite(rewards))
            completed_episodes += len(step_info["episode_metrics"])

        assert completed_episodes > 0
        transitions = vec_env.get_transitions()
        assert transitions["obs"].shape[1] == vec_env.obs_dim
        assert np.all(np.isfinite(transitions["obs"]))
        assert np.all(np.isfinite(transitions["next_obs"]))

    # 2. Single-environment PettingZoo ParallelEnv-style FleetEnv in quick=True and quick=False modes
    for quick in (True, False):
        env = FleetEnv(quick=quick, seed=42)
        obs_dict, info_dict = env.reset(seed=42)
        assert len(env.agents) == env.n_hoppers
        assert env.get_global_state().shape == (env.global_state_dim,)

        for _ in range(1000):
            action_dict: dict[str, int] = {}
            for agent_id in env.possible_agents:
                mask = env.get_action_mask(agent_id)
                assert mask.shape == (env.n_actions,)
                assert bool(mask[env.n_craters + 1]) is True
                action_dict[agent_id] = int(_sample_masked_actions(mask, rng))

            obs_dict, rewards_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
            for agent_id in env.possible_agents:
                assert obs_dict[agent_id].shape == (env.obs_dim,)
                assert np.all(np.isfinite(obs_dict[agent_id]))
                assert np.isfinite(rewards_dict[agent_id])

            if all(term_dict.values()) or all(trunc_dict.values()):
                obs_dict, info_dict = env.reset()


def test_stage1_vector_env_throughput_benchmark() -> None:
    """Stage 1 benchmark: VectorFleetEnv(num_envs=128) exceeds 10,000 env steps/second on CPU."""
    num_envs = 128
    n_iters = 250
    rng = np.random.default_rng(7)
    vec_env = VectorFleetEnv(num_envs=num_envs, quick=False, seed=42)
    vec_env.reset(seed=42)

    # Warm-up steps
    for _ in range(10):
        masks = vec_env.get_action_masks()
        actions = _sample_masked_actions(masks, rng)
        vec_env.step(actions)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        masks = vec_env.get_action_masks()
        actions = _sample_masked_actions(masks, rng)
        vec_env.step(actions)
    elapsed_s = time.perf_counter() - t0

    total_env_steps = num_envs * n_iters
    steps_per_sec = total_env_steps / max(elapsed_s, 1e-9)
    print(
        f"\n[Benchmark] VectorFleetEnv(num_envs={num_envs}): "
        f"{steps_per_sec:,.1f} env steps/s ({total_env_steps} steps in {elapsed_s:.3f} s)"
    )
    assert steps_per_sec > 10_000.0, f"Expected > 10,000 env steps/s, got {steps_per_sec:,.1f}"


def test_action_masking_rules() -> None:
    """Verify invalid action masking for hop, scan, wait, and send actions."""
    vec_env = VectorFleetEnv(num_envs=1, n_hoppers=2, n_craters=6, quick=True, seed=42)
    vec_env.reset(seed=42)
    c = vec_env.n_craters
    scan_idx = c
    wait_idx = c + 1
    send_idx = c + 2

    # At reset: hoppers are near origin (not at any crater) and hold 0 unsent_data
    masks = vec_env.get_action_masks()
    assert masks.shape == (1, 2, c + 3)
    assert np.all(masks[0, :, wait_idx])  # wait is always allowed
    assert not np.any(masks[0, :, scan_idx])  # cannot scan when not at a crater
    assert not np.any(masks[0, :, send_idx])  # cannot send when unsent_data == 0
    assert np.any(masks[0, 0, :c])  # at least one inner-ring crater is reachable from origin

    # Place hopper 0 at crater 0 with unsent_data = 1 and relay open
    vec_env.pos_xy[0, 0] = vec_env._crater_xy[0]
    vec_env.pos_z[0, 0] = vec_env._crater_z[0]
    vec_env.at_crater_id[0, 0] = 0
    vec_env.unsent_data[0, 0] = 1.0
    vec_env.relay_open[0] = True

    masks = vec_env.get_action_masks()
    assert bool(masks[0, 0, scan_idx]) is True  # can scan unscanned crater 0
    assert bool(masks[0, 0, 0]) is False  # cannot hop to crater 0 while already at crater 0
    assert bool(masks[0, 0, send_idx]) is True  # can send when unsent_data > 0 and relay_open

    # Close relay window -> cannot send
    vec_env.relay_open[0] = False
    masks = vec_env.get_action_masks()
    assert bool(masks[0, 0, send_idx]) is False

    # Mark crater 0 as done -> cannot scan crater 0 and no hopper can hop to crater 0
    vec_env.craters_done[0, 0] = True
    masks = vec_env.get_action_masks()
    assert bool(masks[0, 0, scan_idx]) is False
    assert not np.any(masks[0, :, 0])

    # Drain fuel of hopper 1 -> cannot hop anywhere
    vec_env.fuel_kg[0, 1] = 0.01
    masks = vec_env.get_action_masks()
    assert not np.any(masks[0, 1, :c])
    assert bool(masks[0, 1, wait_idx]) is True

    # Crash hopper 0 -> only wait is valid
    vec_env.crashed[0, 0] = True
    vec_env.alive[0, 0] = False
    vec_env.relay_open[0] = True
    masks = vec_env.get_action_masks()
    expected_dead_mask = np.zeros(c + 3, dtype=bool)
    expected_dead_mask[wait_idx] = True
    np.testing.assert_array_equal(masks[0, 0], expected_dead_mask)


def test_claiming_scanning_and_sending_mechanics() -> None:
    """Hopping claims a crater, scanning marks it done and adds unsent_data, sending transmits it."""
    from selene_sim.sim import SimConfig

    cfg = SimConfig(noise_enabled=False)
    vec_env = VectorFleetEnv(num_envs=1, n_hoppers=2, n_craters=6, quick=True, seed=42, config=cfg)
    vec_env.reset(seed=42)
    c = vec_env.n_craters
    scan_idx = c
    wait_idx = c + 1
    send_idx = c + 2

    # Step 1: Hopper 0 hops to crater 0, Hopper 1 hops to crater 1
    actions = np.array([[0, 1]], dtype=np.int32)
    _, rewards, _, _, _ = vec_env.step(actions)
    assert int(vec_env.at_crater_id[0, 0]) == 0
    assert int(vec_env.at_crater_id[0, 1]) == 1
    assert int(vec_env.craters_claimed[0, 0]) == 0
    assert int(vec_env.craters_claimed[0, 1]) == 1

    # Step 2: Both hoppers scan their respective craters (mission_time becomes 120 s -> relay open)
    actions = np.array([[scan_idx, scan_idx]], dtype=np.int32)
    _, rewards, _, _, _ = vec_env.step(actions)
    assert bool(vec_env.craters_done[0, 0]) is True
    assert bool(vec_env.craters_done[0, 1]) is True
    assert float(vec_env.unsent_data[0, 0]) == 1.0
    assert float(vec_env.unsent_data[0, 1]) == 1.0
    assert bool(vec_env.relay_open[0]) is True

    # Step 3: Both hoppers try to send simultaneously -> relay conflict, data remains unsent
    actions = np.array([[send_idx, send_idx]], dtype=np.int32)
    vec_env.step(actions)
    assert bool(vec_env.relay_conflict[0]) is True
    assert float(vec_env.unsent_data[0, 0]) == 1.0
    assert float(vec_env.unsent_data[0, 1]) == 1.0

    # Force relay open and have only Hopper 0 send while Hopper 1 waits -> clean transmission
    vec_env.mission_time_s[0] = 0.0
    vec_env.relay_open[0] = True
    actions = np.array([[send_idx, wait_idx]], dtype=np.int32)
    vec_env.step(actions)
    assert bool(vec_env.relay_conflict[0]) is False
    assert float(vec_env.unsent_data[0, 0]) == 0.0
    assert float(vec_env.unsent_data[0, 1]) == 1.0


def test_render_episode_dict_matches_schema() -> None:
    """render_episode_dict expands hops into 0.5 s keyframes and matches the episode schema."""
    env = FleetEnv(quick=True, seed=42)
    env.reset(seed=42)
    c = env.n_craters
    scan_idx = c
    wait_idx = c + 1
    send_idx = c + 2

    # Execute a short sequence: hop -> scan -> send
    env.step({"hopper_0": 0, "hopper_1": 1})
    env.step({"hopper_0": scan_idx, "hopper_1": scan_idx})
    env.step({"hopper_0": send_idx, "hopper_1": wait_idx})

    ep = env.render_episode_dict(episode_id=5, phase="final_policy", source="learned")
    assert ep["episode_id"] == 5
    assert ep["phase"] == "final_policy"
    assert ep["source"] == "learned"
    assert ep["fleet_size"] == 2
    assert len(ep["drones"]) == 2
    assert "metrics" in ep
    assert ep["metrics"]["craters_visited"] >= 2

    # Hopper 0 hopped, scanned, and sent -> should have multi-keyframe flight trajectory (> 10 steps)
    assert len(ep["drones"][0]["steps"]) > 10
    phases_seen = {s["phase"] for s in ep["drones"][0]["steps"]}
    assert "ascent" in phases_seen
    assert "coast" in phases_seen
    assert "descent" in phases_seen
    assert "ground" in phases_seen

    # Validate full top-level document against test_episode_schema.validate_episode_doc
    doc = env.render_episodes_document(episodes=[ep], source="trained")
    validate_episode_doc(doc)
