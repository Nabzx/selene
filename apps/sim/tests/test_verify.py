"""Unit and benchmark tests for the vectorised Monte Carlo rollout engine (verify.py)."""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from selene_sim.sim import SimConfig, TerrainGrid
from selene_sim.verify import (
    FleetPlan,
    HopCommand,
    RolloutTraces,
    sample_trajectory_cloud,
    simulate_plan,
)


def _make_sample_fleet_plan(terrain: TerrainGrid) -> FleetPlan:
    """Create a 4-hopper, 5-stage fleet plan visiting Daedalus craters."""
    c0, c1, c2, c3 = terrain.craters[:4]
    start_positions = np.array(
        [
            [-150.0, -150.0],
            [150.0, -150.0],
            [-150.0, 150.0],
            [150.0, 150.0],
        ],
        dtype=np.float64,
    )
    stages = [
        # Stage 0: hop from landing site to assigned inner-ring craters
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="hop", target_crater_id=c0.id),
            HopCommand(hopper_id=1, target_xy=(c1.x_m, c1.y_m), action="hop", target_crater_id=c1.id),
            HopCommand(hopper_id=2, target_xy=(c2.x_m, c2.y_m), action="hop", target_crater_id=c2.id),
            HopCommand(hopper_id=3, target_xy=(c3.x_m, c3.y_m), action="hop", target_crater_id=c3.id),
        ],
        # Stage 1: scan assigned craters in parallel
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="scan", target_crater_id=c0.id),
            HopCommand(hopper_id=1, target_xy=(c1.x_m, c1.y_m), action="scan", target_crater_id=c1.id),
            HopCommand(hopper_id=2, target_xy=(c2.x_m, c2.y_m), action="scan", target_crater_id=c2.id),
            HopCommand(hopper_id=3, target_xy=(c3.x_m, c3.y_m), action="scan", target_crater_id=c3.id),
        ],
        # Stage 2: hopper 0 sends during open relay window while others wait
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="send"),
            HopCommand(hopper_id=1, target_xy=(c1.x_m, c1.y_m), action="wait"),
            HopCommand(hopper_id=2, target_xy=(c2.x_m, c2.y_m), action="wait"),
            HopCommand(hopper_id=3, target_xy=(c3.x_m, c3.y_m), action="wait"),
        ],
        # Stage 3: wait for next relay window
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="wait", wait_s=90.0),
            HopCommand(hopper_id=1, target_xy=(c1.x_m, c1.y_m), action="wait", wait_s=90.0),
            HopCommand(hopper_id=2, target_xy=(c2.x_m, c2.y_m), action="wait", wait_s=90.0),
            HopCommand(hopper_id=3, target_xy=(c3.x_m, c3.y_m), action="wait", wait_s=90.0),
        ],
        # Stage 4: hopper 1 sends while others wait
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="wait"),
            HopCommand(hopper_id=1, target_xy=(c1.x_m, c1.y_m), action="send"),
            HopCommand(hopper_id=2, target_xy=(c2.x_m, c2.y_m), action="wait"),
            HopCommand(hopper_id=3, target_xy=(c3.x_m, c3.y_m), action="wait"),
        ],
    ]
    return FleetPlan(num_hoppers=4, start_positions_xy=start_positions, stages=stages)


def test_verify_never_imports_world_model_or_torch() -> None:
    """verify.py must remain pure NumPy and never import world_model or torch."""
    import selene_sim.verify as verify_mod

    assert "world_model" not in verify_mod.__dict__
    assert "torch" not in verify_mod.__dict__

    verify_path = Path(verify_mod.__file__).resolve()
    tree = ast.parse(verify_path.read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.add(node.module)
            for alias in node.names:
                imported_names.add(f"{node.module}.{alias.name}")

    for name in imported_names:
        assert "world_model" not in name
        assert "torch" not in name


def test_simulate_plan_determinism() -> None:
    """Calling simulate_plan twice with the same seed returns identical RolloutTraces."""
    terrain = TerrainGrid.synthetic(seed=42, quick=False)
    plan = _make_sample_fleet_plan(terrain)

    traces_a = simulate_plan(plan, n=100, seed=42, terrain=terrain)
    traces_b = simulate_plan(plan, n=100, seed=42, terrain=terrain)

    np.testing.assert_array_equal(traces_a.positions_xyz, traces_b.positions_xyz)
    np.testing.assert_array_equal(traces_a.fuel_kg, traces_b.fuel_kg)
    np.testing.assert_array_equal(traces_a.battery_pct, traces_b.battery_pct)
    np.testing.assert_array_equal(traces_a.landing_slope_deg, traces_b.landing_slope_deg)
    np.testing.assert_array_equal(traces_a.hop_durations_s, traces_b.hop_durations_s)
    np.testing.assert_array_equal(traces_a.min_separation_m, traces_b.min_separation_m)
    np.testing.assert_array_equal(traces_a.crashed, traces_b.crashed)
    np.testing.assert_array_equal(traces_a.crash_reason, traces_b.crash_reason)
    np.testing.assert_array_equal(traces_a.relay_conflict, traces_b.relay_conflict)
    np.testing.assert_array_equal(traces_a.data_downlinked, traces_b.data_downlinked)


def test_simulate_plan_shapes_and_physics() -> None:
    """Verify output array shapes, physical invariants, dict input support, and summary dict."""
    terrain = TerrainGrid.synthetic(seed=42, quick=False)
    plan = _make_sample_fleet_plan(terrain)
    n_runs = 64
    num_stages = 5
    num_hoppers = 4

    traces = simulate_plan(plan, n=n_runs, seed=42, terrain=terrain)
    assert isinstance(traces, RolloutTraces)

    assert traces.positions_xyz.shape == (n_runs, num_stages + 1, num_hoppers, 3)
    assert traces.fuel_kg.shape == (n_runs, num_stages + 1, num_hoppers)
    assert traces.battery_pct.shape == (n_runs, num_stages + 1, num_hoppers)
    assert traces.landing_slope_deg.shape == (n_runs, num_stages, num_hoppers)
    assert traces.hop_durations_s.shape == (n_runs, num_stages, num_hoppers)
    assert traces.min_separation_m.shape == (n_runs, num_stages)
    assert traces.crashed.shape == (n_runs, num_stages, num_hoppers)
    assert traces.crash_reason.shape == (n_runs, num_stages, num_hoppers)
    assert traces.relay_conflict.shape == (n_runs, num_stages)
    assert traces.data_downlinked.shape == (n_runs, num_stages, num_hoppers)

    # Fuel and battery are monotonically non-increasing across stages
    assert np.all(np.diff(traces.fuel_kg, axis=1) <= 1e-5)
    assert np.all(np.diff(traces.battery_pct, axis=1) <= 1e-5)
    # Stage 0 is a hop, so fuel drops strictly between stage 0 and stage 1
    assert np.all(traces.fuel_kg[:, 1, :] < traces.fuel_kg[:, 0, :])
    assert np.all(traces.hop_durations_s[:, 0, :] > 10.0)

    # Summary dictionary for Friend B's safety spec runner must be JSON-serialisable
    summary = traces.to_summary_dict()
    assert summary["num_runs"] == n_runs
    assert summary["num_stages"] == num_stages
    assert summary["num_hoppers"] == num_hoppers
    assert 0.0 <= summary["crash_rate"] <= 1.0
    json.dumps(summary)

    # Passing a dictionary matching FleetPlan schema produces identical traces
    plan_dict = plan.to_dict()
    traces_from_dict = simulate_plan(plan_dict, n=n_runs, seed=42, terrain=terrain)
    np.testing.assert_allclose(traces.positions_xyz, traces_from_dict.positions_xyz)
    np.testing.assert_allclose(traces.fuel_kg, traces_from_dict.fuel_kg)


def test_relay_conflict_and_proximity_violation_detection() -> None:
    """Detect relay contention when 2 hoppers send at once and proximity violation (< 100 m)."""
    terrain = TerrainGrid.synthetic(seed=42, quick=True)
    c0 = terrain.craters[0]
    cfg = SimConfig(noise_enabled=False)

    # Two hoppers hop to the exact same coordinates at stage 0, scan at stage 1,
    # and both transmit simultaneously at stage 2.
    plan = FleetPlan(
        num_hoppers=2,
        start_positions_xy=np.array([[-200.0, 0.0], [200.0, 0.0]], dtype=np.float64),
        stages=[
            [
                HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="hop", target_crater_id=c0.id),
                HopCommand(hopper_id=1, target_xy=(c0.x_m, c0.y_m), action="hop", target_crater_id=c0.id),
            ],
            [
                HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="scan", target_crater_id=c0.id),
                HopCommand(hopper_id=1, target_xy=(c0.x_m, c0.y_m), action="scan", target_crater_id=c0.id),
            ],
            [
                HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="send"),
                HopCommand(hopper_id=1, target_xy=(c0.x_m, c0.y_m), action="send"),
            ],
        ],
    )

    traces = simulate_plan(plan, n=32, seed=42, terrain=terrain, config=cfg)

    # Both hoppers land at (c0.x_m, c0.y_m) -> separation is 0 m (< 100 m violation)
    assert np.all(traces.min_separation_m[:, 0] < 100.0)
    np.testing.assert_allclose(traces.min_separation_m[:, 0], 0.0, atol=1e-5)

    # Both hoppers send in stage 2 -> relay conflict in all runs and no data downlinked
    assert np.all(traces.relay_conflict[:, 2])
    assert not np.any(traces.data_downlinked[:, 2, :])


def test_sample_trajectory_cloud_json_serialisable() -> None:
    """sample_trajectory_cloud returns JSON-serialisable trajectory fans ready for data/episodes.json."""
    terrain = TerrainGrid.synthetic(seed=42, quick=False)
    plan = _make_sample_fleet_plan(terrain)
    traces = simulate_plan(plan, n=120, seed=42, terrain=terrain)

    cloud = sample_trajectory_cloud(traces, max_runs=50)
    assert isinstance(cloud, list)
    assert len(cloud) == 50

    payload = json.dumps(cloud)
    decoded = json.loads(payload)
    assert len(decoded) == 50
    first_run = decoded[0]
    assert "run_index" in first_run
    assert "crashed" in first_run
    assert "hoppers" in first_run
    assert len(first_run["hoppers"]) == 4
    first_hopper = first_run["hoppers"][0]
    assert len(first_hopper["path_xyz"]) == 6  # S + 1 = 6 waypoints


def test_benchmark_1000_rollouts_under_10_seconds() -> None:
    """1000 noisy rollouts of a 5-hopper, 10-stage FleetPlan must complete in < 10.0 seconds."""
    terrain = TerrainGrid.synthetic(seed=42, quick=False)
    craters = terrain.craters
    num_hoppers = 5
    num_stages = 10

    angles = np.linspace(0.0, 2.0 * np.pi, num_hoppers, endpoint=False)
    start_xy = np.stack([250.0 * np.cos(angles), 250.0 * np.sin(angles)], axis=-1)

    stages: list[list[HopCommand]] = []
    for s in range(num_stages):
        stage_cmds: list[HopCommand] = []
        for h in range(num_hoppers):
            crater = craters[(h + (s // 2) * num_hoppers) % len(craters)]
            if s % 3 == 0:
                stage_cmds.append(
                    HopCommand(
                        hopper_id=h,
                        target_xy=(crater.x_m, crater.y_m),
                        action="hop",
                        target_crater_id=crater.id,
                    )
                )
            elif s % 3 == 1:
                stage_cmds.append(
                    HopCommand(
                        hopper_id=h,
                        target_xy=(crater.x_m, crater.y_m),
                        action="scan",
                        target_crater_id=crater.id,
                    )
                )
            else:
                act = "send" if h == (s % num_hoppers) else "wait"
                stage_cmds.append(
                    HopCommand(
                        hopper_id=h,
                        target_xy=(crater.x_m, crater.y_m),
                        action=act,
                    )
                )
        stages.append(stage_cmds)

    plan = FleetPlan(num_hoppers=num_hoppers, start_positions_xy=start_xy, stages=stages)

    # Warm-up call
    simulate_plan(plan, n=10, seed=42, terrain=terrain)

    t0 = time.perf_counter()
    traces = simulate_plan(plan, n=1000, seed=42, terrain=terrain)
    elapsed_s = time.perf_counter() - t0
    elapsed_ms = elapsed_s * 1000.0
    print(f"\n[benchmark] 1000 rollouts x 5 hoppers x 10 stages: {elapsed_ms:.2f} ms ({elapsed_s:.4f} s)")

    assert traces.positions_xyz.shape == (1000, 11, 5, 3)
    assert elapsed_s < 10.0
