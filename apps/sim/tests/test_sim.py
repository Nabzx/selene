"""Unit tests for the ground-truth lunar physics simulator (sim.py)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from selene_sim.sim import (
    CraterInfo,
    SimConfig,
    TerrainGrid,
    check_relay_conflicts,
    compute_battery_drain,
    compute_hop,
    interpolate_hop_keyframes,
    is_relay_open,
    time_to_next_relay,
)


def test_sim_does_not_import_torch() -> None:
    """sim.py must stay pure NumPy and never import torch or world_model."""
    import ast
    import selene_sim.sim as sim_mod

    assert "torch" not in sim_mod.__dict__
    assert "world_model" not in sim_mod.__dict__
    sim_path = Path(sim_mod.__file__).resolve()
    tree = ast.parse(sim_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("torch", "selene_sim.world_model"))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith(("torch", "selene_sim.world_model"))


def test_terrain_grid_synthetic_full_and_quick(tmp_path: Path) -> None:
    """TerrainGrid generates 12-15 craters in full mode, 6 in quick mode, and round-trips to npz."""
    terrain_full = TerrainGrid.synthetic(seed=42, quick=False)
    assert terrain_full.height_m.shape == (200, 200)
    assert terrain_full.slope_deg.shape == (200, 200)
    assert terrain_full.temp_c.shape == (200, 200)
    assert terrain_full.cell_size_m == 100.0
    assert 12 <= len(terrain_full.craters) <= 15

    # Determinism check
    terrain_full_again = TerrainGrid.synthetic(seed=42, quick=False)
    np.testing.assert_allclose(terrain_full.height_m, terrain_full_again.height_m)
    assert len(terrain_full.craters) == len(terrain_full_again.craters)

    # Quick mode check (6 craters)
    terrain_quick = TerrainGrid.synthetic(seed=42, quick=True)
    assert len(terrain_quick.craters) == 6

    # Check lookup helpers on arrays
    xs = np.array([0.0, 2500.0, -3000.0])
    ys = np.array([0.0, -1500.0, 4000.0])
    h = terrain_full.height_at(xs, ys)
    s = terrain_full.slope_at(xs, ys)
    t = terrain_full.temp_at(xs, ys)
    assert h.shape == (3,)
    assert s.shape == (3,)
    assert np.all(s >= 0.0)
    assert np.all(t < -150.0)

    # Crater floor should be colder and lower than mean plain
    c0 = terrain_full.craters[0]
    assert terrain_full.temp_at(np.array([c0.x_m]), np.array([c0.y_m]))[0] < -175.0
    cid = terrain_full.crater_at(np.array([c0.x_m]), np.array([c0.y_m]))
    assert int(cid[0]) == 0

    # Round-trip save and load via npz
    npz_path = tmp_path / "daedalus_200x200.npz"
    terrain_full.to_npz(npz_path)
    loaded = TerrainGrid.from_npz(npz_path)
    np.testing.assert_allclose(loaded.height_m, terrain_full.height_m)
    assert len(loaded.craters) == len(terrain_full.craters)
    assert loaded.craters[0].id == terrain_full.craters[0].id


def test_ballistic_hop_and_rocket_equation_analytical() -> None:
    """Verify closed-form ballistic hop kinematics and Tsiolkovsky propellant burn against theory."""
    cfg = SimConfig(noise_enabled=False)
    # Create a completely flat terrain so analytical flat-ground equations hold exactly
    zeros = np.zeros((200, 200), dtype=np.float32)
    temps = np.full((200, 200), -180.0, dtype=np.float32)
    craters = [CraterInfo(id="crater_0", x_m=1000.0, y_m=0.0, radius_m=500.0, depth_m=50.0, priority=1.0)]
    flat_terrain = TerrainGrid(height_m=zeros, slope_deg=zeros, temp_c=temps, craters=craters)

    start_pos = np.array([[0.0, 0.0]], dtype=np.float64)
    target_pos = np.array([[1620.0, 0.0]], dtype=np.float64)  # R = 1620 m
    fuel_kg = np.array([15.0], dtype=np.float64)
    rng = np.random.default_rng(42)

    res = compute_hop(
        start_pos_xy=start_pos,
        target_pos_xy=target_pos,
        fuel_kg=fuel_kg,
        terrain=flat_terrain,
        config=cfg,
        rng=rng,
    )

    # Analytical values for R = 1620 m, g = 1.62 m/s^2, theta = 45 deg:
    # v_0 = sqrt(R * g / sin(90 deg)) = sqrt(1620 * 1.62) = 51.22499389946279 m/s
    # t_flight = 2 * v_0 * sin(45 deg) / g = 2 * 51.2249939 * 0.70710678 / 1.62 = 44.72135955 s
    # delta_v = 2 * v_0 * (1 + k_margin) = 2 * 51.2249939 * 1.08 = 110.6459868 m/s
    # m_0 = m_dry + fuel = 35 + 15 = 50 kg
    # v_e = I_sp * g_0 = 220 * 9.80665 = 2157.463 m/s
    # dm = m_0 * (1 - exp(-delta_v / v_e))
    r_dist = 1620.0
    theta_rad = math.radians(cfg.launch_angle_deg)
    v_0 = math.sqrt(r_dist * cfg.g_moon / math.sin(2.0 * theta_rad))
    expected_t = 2.0 * v_0 * math.sin(theta_rad) / cfg.g_moon
    expected_dv = 2.0 * v_0 * (1.0 + cfg.gravity_loss_margin)
    v_e = cfg.isp_s * cfg.g0_earth
    m_0 = cfg.dry_mass_kg + 15.0
    expected_dm = m_0 * (1.0 - math.exp(-expected_dv / v_e))

    np.testing.assert_allclose(res["landed_pos_xy"][0], [1620.0, 0.0], atol=1e-5)
    assert abs(float(res["flight_time_s"][0]) - expected_t) < 1e-5
    assert abs(float(res["delta_v_ms"][0]) - expected_dv) < 1e-5
    assert abs(float(res["fuel_used_kg"][0]) - expected_dm) < 1e-5
    assert not bool(res["crashed"][0])


def test_vectorised_hop_with_noise_and_crash_conditions() -> None:
    """Verify compute_hop works across (E, H) shapes, applies noise deterministically, and flags crashes."""
    cfg = SimConfig(noise_enabled=True)
    terrain = TerrainGrid.synthetic(seed=7, quick=False)

    start_xy = np.zeros((16, 4, 2), dtype=np.float64)
    target_xy = np.full((16, 4, 2), 1200.0, dtype=np.float64)
    fuel_kg = np.full((16, 4), 15.0, dtype=np.float64)

    res1 = compute_hop(start_xy, target_xy, fuel_kg, terrain, cfg, np.random.default_rng(99))
    res2 = compute_hop(start_xy, target_xy, fuel_kg, terrain, cfg, np.random.default_rng(99))

    assert res1["landed_pos_xy"].shape == (16, 4, 2)
    assert res1["fuel_used_kg"].shape == (16, 4)
    np.testing.assert_allclose(res1["landed_pos_xy"], res2["landed_pos_xy"])

    # With noise enabled, landed positions scatter around target_xy
    scatter = np.linalg.norm(res1["landed_pos_xy"] - target_xy, axis=-1)
    assert float(np.mean(scatter)) > 0.5

    # Insufficient fuel should trigger a crash
    low_fuel = np.full((2, 2), 0.01, dtype=np.float64)
    res_low = compute_hop(
        np.zeros((2, 2, 2)),
        np.full((2, 2, 2), 2500.0),
        low_fuel,
        terrain,
        cfg,
        np.random.default_rng(1),
    )
    assert np.all(res_low["crashed"])


def test_thermal_battery_drain_and_relay_windows() -> None:
    """Battery drain grows as lunar night temperature drops below -170 C; relay schedule detects conflicts."""
    cfg = SimConfig(noise_enabled=False)
    dt = np.array([30.0, 30.0])
    temp_warm = np.array([-170.0, -220.0])  # second hopper is in a cold -220 C crater floor
    did_hop = np.array([False, False])
    did_scan = np.array([False, False])
    did_send = np.array([False, False])

    drain = compute_battery_drain(
        dt_s=dt,
        temp_c=temp_warm,
        did_hop=did_hop,
        did_scan=did_scan,
        did_send=did_send,
        config=cfg,
    )
    assert drain.shape == (2,)
    assert drain[1] > drain[0]  # colder hopper burns more heater energy

    # Hopping adds per-hop avionics cost
    drain_hop = compute_battery_drain(
        dt_s=dt,
        temp_c=temp_warm,
        did_hop=np.array([True, False]),
        did_scan=did_scan,
        did_send=did_send,
        config=cfg,
    )
    assert abs((drain_hop[0] - drain[0]) - cfg.battery_hop_pct) < 1e-5

    # Relay window schedule: period = 120 s, open duration = 40 s -> open in [0, 40), closed in [40, 120)
    assert bool(is_relay_open(10.0, cfg)) is True
    assert bool(is_relay_open(60.0, cfg)) is False
    assert abs(float(time_to_next_relay(10.0, cfg)) - 0.0) < 1e-6
    assert abs(float(time_to_next_relay(60.0, cfg)) - (cfg.relay_period_s - 60.0)) < 1e-6

    # Single-channel relay conflict detection
    # Case 1: 1 hopper sends during open window -> success, no conflict
    s1, c1 = check_relay_conflicts(np.array([[True, False, False, False]]), np.array([True]))
    assert bool(s1[0, 0]) is True
    assert bool(c1[0]) is False

    # Case 2: 2 hoppers send simultaneously -> conflict, neither succeeds
    s2, c2 = check_relay_conflicts(np.array([[True, True, False, False]]), np.array([True]))
    assert not np.any(s2)
    assert bool(c2[0]) is True


def test_interpolate_hop_keyframes() -> None:
    """Verify 0.5 s trajectory keyframe interpolation produces smooth ballistic arcs."""
    terrain = TerrainGrid.synthetic(seed=42, quick=True)
    cfg = SimConfig(noise_enabled=False)
    start_xy = np.array([0.0, 0.0])
    end_xy = np.array([800.0, 600.0])

    keyframes = interpolate_hop_keyframes(
        start_pos_xy=start_xy,
        landed_pos_xy=end_xy,
        flight_time_s=20.0,
        start_time_s=10.0,
        start_fuel_pct=90.0,
        end_fuel_pct=84.0,
        start_battery_pct=95.0,
        end_battery_pct=92.5,
        terrain=terrain,
        config=cfg,
        dt_s=0.5,
    )
    assert len(keyframes) == 41  # 0.0 to 20.0 inclusive at 0.5 s steps
    assert keyframes[0]["t"] == pytest.approx(10.0)
    assert keyframes[-1]["t"] == pytest.approx(30.0)
    assert keyframes[0]["x"] == pytest.approx(0.0)
    assert keyframes[-1]["x"] == pytest.approx(800.0)
    # Mid-flight altitude must be strictly above the chord connecting start and end terrain height
    mid_kf = keyframes[len(keyframes) // 2]
    chord_z = 0.5 * (keyframes[0]["z"] + keyframes[-1]["z"])
    assert mid_kf["z"] > chord_z + 10.0
