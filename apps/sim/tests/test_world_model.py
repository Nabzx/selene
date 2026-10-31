"""Tests for the 5-MLP ensemble world model and spatial uncertainty map (Issue #11)."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from selene_sim.sim import (
    SimConfig,
    TerrainGrid,
    compute_battery_drain,
    compute_hop,
    is_relay_open,
    time_to_next_relay,
)
from selene_sim.world_model import (
    DynamicsMLP,
    TransitionBuffer,
    WorldModelEnsemble,
    select_device,
)


def _build_hopper_obs(
    pos_xy: np.ndarray,
    fuel_kg: np.ndarray,
    battery_pct: np.ndarray,
    t_s: np.ndarray,
    unsent_data: np.ndarray,
    terrain: TerrainGrid,
    config: SimConfig,
) -> np.ndarray:
    """Construct realistic 32-float quick-mode hopper observations from simulator state."""
    batch_n = pos_xy.shape[0]
    half_ext = terrain.half_extent_m
    z_m = terrain.height_at(pos_xy[:, 0], pos_xy[:, 1])
    temp_c = terrain.temp_at(pos_xy[:, 0], pos_xy[:, 1])

    own = np.stack(
        [
            pos_xy[:, 0] / half_ext,
            pos_xy[:, 1] / half_ext,
            z_m / 200.0,
            fuel_kg / config.initial_fuel_kg,
            battery_pct / 100.0,
            (temp_c + 200.0) / 50.0,
            unsent_data.astype(np.float32) / 6.0,
        ],
        axis=-1,
    ).astype(np.float32)

    agent_one_hot = np.zeros((batch_n, 2), dtype=np.float32)
    agent_one_hot[:, 0] = 1.0

    peer_rel = np.zeros((batch_n, 3), dtype=np.float32)
    peer_rel[:, 2] = 1.0

    crater_feats = []
    for crater in terrain.craters:
        dx = (crater.x_m - pos_xy[:, 0]) / half_ext
        dy = (crater.y_m - pos_xy[:, 1]) / half_ext
        dist_norm = np.hypot(dx, dy).astype(np.float32)
        claimed = np.zeros(batch_n, dtype=np.float32)
        scanned = np.zeros(batch_n, dtype=np.float32)
        crater_feats.extend([dist_norm, claimed, scanned])
    crater_arr = np.stack(crater_feats, axis=-1).astype(np.float32)

    relay_open = np.asarray(is_relay_open(t_s, config), dtype=np.float32)
    relay_wait = (
        np.asarray(time_to_next_relay(t_s, config), dtype=np.float32)
        / config.relay_period_s
    )
    relay_arr = np.stack([relay_open, relay_wait], axis=-1).astype(np.float32)

    return np.concatenate([own, agent_one_hot, peer_rel, crater_arr, relay_arr], axis=-1)


def _generate_transitions(
    terrain: TerrainGrid,
    target_craters: tuple[int, ...],
    n_samples: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate realistic ballistic hop transitions on the lunar terrain using sim.py."""
    rng = np.random.default_rng(seed)
    cfg = SimConfig(noise_enabled=True)

    crater_xy = np.array(
        [[c.x_m, c.y_m] for c in terrain.craters], dtype=np.float64
    )
    target_ids = rng.choice(np.array(target_craters, dtype=np.int32), size=n_samples)
    actions = target_ids + 3  # Discrete action 3 + c = hop_to_crater[c]

    # Sample start positions within feasible hop range of the selected target crater
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    radii = rng.uniform(400.0, 2400.0, size=n_samples)
    target_pos_xy = crater_xy[target_ids]
    start_pos_xy = np.clip(
        target_pos_xy + np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1),
        -terrain.half_extent_m + 200.0,
        terrain.half_extent_m - 200.0,
    )

    fuel_kg = rng.uniform(4.0, cfg.initial_fuel_kg, size=n_samples)
    battery_pct = rng.uniform(35.0, 100.0, size=n_samples)
    t_s = rng.uniform(0.0, 600.0, size=n_samples)
    unsent_data = rng.integers(0, 4, size=n_samples)

    obs = _build_hopper_obs(
        start_pos_xy, fuel_kg, battery_pct, t_s, unsent_data, terrain, cfg
    )

    hop_out = compute_hop(
        start_pos_xy=start_pos_xy,
        target_pos_xy=target_pos_xy,
        fuel_kg=fuel_kg,
        terrain=terrain,
        config=cfg,
        rng=rng,
    )
    landed_xy = hop_out["landed_pos_xy"]
    flight_dt = np.maximum(hop_out["flight_time_s"], 60.0)
    next_fuel_kg = np.maximum(0.0, fuel_kg - hop_out["fuel_used_kg"])
    end_temp_c = terrain.temp_at(landed_xy[:, 0], landed_xy[:, 1])
    batt_drain = compute_battery_drain(
        dt_s=flight_dt,
        temp_c=end_temp_c,
        did_hop=np.ones(n_samples, dtype=bool),
        did_scan=np.zeros(n_samples, dtype=bool),
        did_send=np.zeros(n_samples, dtype=bool),
        config=cfg,
    )
    next_battery_pct = np.clip(battery_pct - batt_drain, 0.0, 100.0)
    next_t_s = t_s + flight_dt
    crashed = hop_out["crashed"].astype(bool)

    next_obs = _build_hopper_obs(
        landed_xy, next_fuel_kg, next_battery_pct, next_t_s, unsent_data, terrain, cfg
    )

    priorities = np.array([terrain.craters[idx].priority for idx in target_ids], dtype=np.float32)
    rewards = (
        10.0 * priorities * (~crashed).astype(np.float32)
        - 0.4 * hop_out["fuel_used_kg"].astype(np.float32)
        - 50.0 * crashed.astype(np.float32)
    )

    return {
        "obs": obs.astype(np.float32),
        "actions": actions.astype(np.int32),
        "next_obs": next_obs.astype(np.float32),
        "rewards": rewards.astype(np.float32),
        "crashed": crashed,
        "crater_ids": target_ids.astype(np.int32),
        "pos_xy": landed_xy.astype(np.float32),
    }


def test_select_device_and_dynamics_mlp_shapes() -> None:
    """Verify device micro-benchmarking and single DynamicsMLP output shapes."""
    device = select_device(prefer_fastest=True)
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "mps")

    mlp = DynamicsMLP(obs_dim=32, n_actions=9, hidden_dim=128)
    obs = torch.randn(16, 32)
    one_hot_act = torch.nn.functional.one_hot(torch.arange(16) % 9, num_classes=9).float()
    x = torch.cat([obs, one_hot_act], dim=-1)
    delta_obs, reward, crash_logit = mlp(x)

    assert delta_obs.shape == (16, 32)
    assert reward.shape == (16,)
    assert crash_logit.shape == (16,)


def test_stage4_validation_mse_and_epistemic_uncertainty() -> None:
    """Verify Stage 4 validation MSE drop, training latency, unvisited crater uncertainty, and error map."""
    terrain = TerrainGrid.synthetic(seed=42, quick=True)
    assert len(terrain.craters) == 6

    # Generate 6,000 transitions visiting craters 0, 1, 2, 3 while leaving craters 4, 5 unvisited
    visited_data = _generate_transitions(
        terrain=terrain,
        target_craters=(0, 1, 2, 3),
        n_samples=6000,
        seed=42,
    )

    buffer = TransitionBuffer()
    buffer.add(
        obs=visited_data["obs"],
        actions=visited_data["actions"],
        next_obs=visited_data["next_obs"],
        rewards=visited_data["rewards"],
        crashed=visited_data["crashed"],
        crater_ids=visited_data["crater_ids"],
        pos_xy=visited_data["pos_xy"],
    )
    assert len(buffer) == 6000

    train_split, val_split = buffer.sample_split(val_fraction=0.1, seed=42)
    assert train_split["obs"].shape[0] == 5400
    assert val_split["obs"].shape[0] == 600

    ensemble = WorldModelEnsemble(
        obs_dim=32,
        n_actions=9,
        ensemble_size=5,
        hidden_dim=128,
        lr=3e-3,
        weight_decay=1e-4,
        seed=42,
    )

    t0 = time.perf_counter()
    history = ensemble.train_epochs(buffer, epochs=25, batch_size=512, seed=42)
    elapsed_s = time.perf_counter() - t0

    assert elapsed_s < 15.0, f"Training took {elapsed_s:.2f}s, expected < 15.0s"
    assert "train_loss" in history and "val_mse" in history
    assert len(history["val_mse"]) >= 25

    initial_val_mse = float(history["val_mse"][0])
    final_val_mse = float(history["val_mse"][-1])
    assert final_val_mse <= 0.5 * initial_val_mse, (
        f"Expected final val_mse ({final_val_mse:.6f}) to drop by >= 50% from initial ({initial_val_mse:.6f})"
    )

    # Evaluate epistemic uncertainty on visited craters (0, 1, 2, 3) vs unvisited craters (4, 5)
    eval_visited = _generate_transitions(
        terrain=terrain,
        target_craters=(0, 1, 2, 3),
        n_samples=600,
        seed=101,
    )
    eval_unvisited = _generate_transitions(
        terrain=terrain,
        target_craters=(4, 5),
        n_samples=600,
        seed=202,
    )

    pred_next_v, pred_rew_v, pred_crash_v, unc_visited = ensemble.predict(
        eval_visited["obs"], eval_visited["actions"]
    )
    pred_next_u, pred_rew_u, pred_crash_u, unc_unvisited = ensemble.predict(
        eval_unvisited["obs"], eval_unvisited["actions"]
    )

    assert pred_next_v.shape == (600, 32)
    assert pred_rew_v.shape == (600,)
    assert pred_crash_v.shape == (600,)
    assert unc_visited.shape == (600,)
    assert unc_unvisited.shape == (600,)

    mean_unc_visited = float(np.mean(unc_visited))
    mean_unc_unvisited = float(np.mean(unc_unvisited))
    assert mean_unc_unvisited >= 1.25 * mean_unc_visited, (
        f"Expected unvisited crater uncertainty ({mean_unc_unvisited:.4f}) to exceed "
        f"visited crater uncertainty ({mean_unc_visited:.4f}) by at least 25%"
    )

    # Verify world_model_error_map structure and JSON serialisability
    error_map = ensemble.world_model_error_map(terrain, buffer)
    serialised = json.dumps(error_map)
    assert isinstance(serialised, str) and len(serialised) > 100

    assert isinstance(error_map["overall_val_mse"], float)
    assert len(error_map["crater_metrics"]) == 6
    assert len(error_map["uncertainty_grid"]) == 20
    assert len(error_map["uncertainty_grid"][0]) == 20
    assert len(error_map["mean_error_grid"]) == 20
    assert len(error_map["mean_error_grid"][0]) == 20
    assert len(error_map["visit_count_grid"]) == 20
    assert len(error_map["visit_count_grid"][0]) == 20
    assert len(error_map["top_error_cells"]) == 5

    # Unvisited craters 4 and 5 should report visit_count == 0 and higher epistemic_uncertainty
    c_metrics = {m["id"]: m for m in error_map["crater_metrics"]}
    assert c_metrics[terrain.craters[4].id]["visit_count"] == 0
    assert c_metrics[terrain.craters[5].id]["visit_count"] == 0
    assert c_metrics[terrain.craters[0].id]["visit_count"] > 0

    print(
        f"\n[Stage 4 Metrics] train_time={elapsed_s:.2f}s | "
        f"val_mse: {initial_val_mse:.6f} -> {final_val_mse:.6f} | "
        f"uncertainty visited={mean_unc_visited:.4f}, unvisited={mean_unc_unvisited:.4f}"
    )
