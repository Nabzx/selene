"""End-to-end integration tests for the Argon orchestrator contract (Issue #16).

Verifies the cross-module loops between Friend B's orchestrator and selene_sim:
1. Eureka reward hot-swap in live VectorFleetEnv rollouts (`reward.py` + `env.py`).
2. 5-MLP ensemble world model error map and epistemic uncertainty (`world_model.py`).
3. Vectorised 1000-run Monte Carlo safety verification in < 10 s (`verify.py`).
4. FastAPI HTTP endpoints (`/health`, `/verify`, `/reward/swap`, `/world-model/error-map`, `/episodes/sample`).
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest

from selene_sim.api import (
    app,
    compute_world_model_report,
    get_health,
    run_verify_plan,
    simulate_episode_bundle,
    swap_reward_code,
)
from selene_sim.env import VectorFleetEnv
from selene_sim.reward import (
    load_reward_code,
    reset_default_reward,
    reward_versions,
)
from selene_sim.sim import TerrainGrid
from selene_sim.verify import (
    FleetPlan,
    HopCommand,
    sample_trajectory_cloud,
    simulate_plan,
)


@pytest.fixture(autouse=True)
def _restore_default_reward_fixture() -> None:
    """Ensure the default fleet reward function is active before and after each test."""
    reset_default_reward()
    yield
    reset_default_reward()


def _make_safe_fleet_plan(terrain: TerrainGrid) -> FleetPlan:
    """Build a safe 4-hopper, 4-stage FleetPlan visiting separated flat crater centres."""
    c0, c1, c2, c3 = terrain.craters[:4]
    start_xy = np.array(
        [
            [-200.0, -200.0],
            [200.0, -200.0],
            [-200.0, 200.0],
            [200.0, 200.0],
        ],
        dtype=np.float64,
    )
    stages = [
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="hop", target_crater_id=c0.id),
            HopCommand(hopper_id=1, target_xy=(c1.x_m, c1.y_m), action="hop", target_crater_id=c1.id),
            HopCommand(hopper_id=2, target_xy=(c2.x_m, c2.y_m), action="hop", target_crater_id=c2.id),
            HopCommand(hopper_id=3, target_xy=(c3.x_m, c3.y_m), action="hop", target_crater_id=c3.id),
        ],
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="scan", target_crater_id=c0.id),
            HopCommand(hopper_id=1, target_xy=(c1.x_m, c1.y_m), action="scan", target_crater_id=c1.id),
            HopCommand(hopper_id=2, target_xy=(c2.x_m, c2.y_m), action="scan", target_crater_id=c2.id),
            HopCommand(hopper_id=3, target_xy=(c3.x_m, c3.y_m), action="scan", target_crater_id=c3.id),
        ],
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="wait", wait_s=65.0),
            HopCommand(hopper_id=1, target_xy=(c1.x_m, c1.y_m), action="wait", wait_s=65.0),
            HopCommand(hopper_id=2, target_xy=(c2.x_m, c2.y_m), action="wait", wait_s=65.0),
            HopCommand(hopper_id=3, target_xy=(c3.x_m, c3.y_m), action="wait", wait_s=65.0),
        ],
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="send"),
            HopCommand(hopper_id=1, target_xy=(c1.x_m, c1.y_m), action="wait"),
            HopCommand(hopper_id=2, target_xy=(c2.x_m, c2.y_m), action="wait"),
            HopCommand(hopper_id=3, target_xy=(c3.x_m, c3.y_m), action="wait"),
        ],
    ]
    return FleetPlan(num_hoppers=4, start_positions_xy=start_xy, stages=stages)


def _make_unsafe_fleet_plan(terrain: TerrainGrid) -> FleetPlan:
    """Build an unsafe FleetPlan with steep rim landing, proximity collision, and relay contention."""
    c0 = terrain.craters[0]
    # Find the steepest cell on the terrain grid to force a deterministic slope crash
    steep_flat_idx = int(np.argmax(terrain.slope_deg))
    steep_iy, steep_ix = np.unravel_index(steep_flat_idx, terrain.slope_deg.shape)
    half_ext = terrain.half_extent_m
    steep_x = float((steep_ix + 0.5) * terrain.cell_size_m - half_ext)
    steep_y = float((steep_iy + 0.5) * terrain.cell_size_m - half_ext)

    # Start hopper 2 within 800 m of the steep point so it reaches the steep slope in 1 hop
    start_h2_x = float(np.clip(steep_x - 600.0, -half_ext + 300.0, half_ext - 300.0))
    start_h2_y = float(np.clip(steep_y, -half_ext + 300.0, half_ext - 300.0))

    start_xy = np.array(
        [
            [-200.0, -200.0],
            [200.0, -200.0],
            [start_h2_x, start_h2_y],
            [200.0, 200.0],
        ],
        dtype=np.float64,
    )
    stages = [
        # Stage 0: hoppers 0 and 1 land at the exact same crater (< 100 m separation),
        # while hopper 2 hops onto a steep wall/out-of-bounds target.
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="hop", target_crater_id=c0.id),
            HopCommand(hopper_id=1, target_xy=(c0.x_m, c0.y_m), action="hop", target_crater_id=c0.id),
            HopCommand(hopper_id=2, target_xy=(steep_x, steep_y), action="hop"),
            HopCommand(hopper_id=3, target_xy=(9950.0, 9950.0), action="hop"),
        ],
        # Stage 1: hoppers 0 and 1 both scan the same crater
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="scan", target_crater_id=c0.id),
            HopCommand(hopper_id=1, target_xy=(c0.x_m, c0.y_m), action="scan", target_crater_id=c0.id),
            HopCommand(hopper_id=2, target_xy=(steep_x, steep_y), action="wait"),
            HopCommand(hopper_id=3, target_xy=(9950.0, 9950.0), action="wait"),
        ],
        # Stage 2: hoppers 0 and 1 both transmit simultaneously over the single-channel relay
        [
            HopCommand(hopper_id=0, target_xy=(c0.x_m, c0.y_m), action="send"),
            HopCommand(hopper_id=1, target_xy=(c0.x_m, c0.y_m), action="send"),
            HopCommand(hopper_id=2, target_xy=(steep_x, steep_y), action="wait"),
            HopCommand(hopper_id=3, target_xy=(9950.0, 9950.0), action="wait"),
        ],
    ]
    return FleetPlan(num_hoppers=4, start_positions_xy=start_xy, stages=stages)


def test_contract_1_eureka_reward_hot_swap_in_live_vector_env() -> None:
    """Contract 1: Eureka reward hot-swap updates live VectorFleetEnv step rewards without restart."""
    env = VectorFleetEnv(num_envs=8, quick=True, seed=42)
    env.reset(seed=42)

    wait_actions = np.full(
        (env.num_envs, env.n_hoppers), env.wait_action, dtype=np.int32
    )
    _, default_rewards, _, _, default_info = env.step(wait_actions)
    assert default_rewards.shape == (env.num_envs, env.n_hoppers)
    assert np.all(np.isfinite(default_rewards))
    assert "total_reward" in default_info["reward_components"]

    # Hot-swap a custom Eureka candidate reward while `env` remains live in memory
    eureka_code = """
import numpy as np

def compute_reward(state, actions, next_state):
    alive_bonus = np.where(next_state.alive, 77.5, -25.0).astype(np.float32)
    fuel_term = (-0.05 * np.maximum(0.0, state.fuel_kg - next_state.fuel_kg)).astype(np.float32)
    total = (alive_bonus + fuel_term).astype(np.float32)
    return total, {
        "eureka_alive_bonus": float(np.mean(alive_bonus)),
        "eureka_fuel_term": float(np.mean(fuel_term)),
        "total_reward": float(np.mean(total)),
    }
"""
    prev_versions = len(reward_versions)
    swap_res = swap_reward_code(
        eureka_code, summary="eureka_iteration_1_alive_bonus", episode=12
    )
    assert swap_res["status"] == "swapped"
    assert swap_res["version_count"] == prev_versions + 1

    _, swapped_rewards, _, _, swapped_info = env.step(wait_actions)
    np.testing.assert_allclose(swapped_rewards, 77.5, rtol=1e-5, atol=1e-5)
    assert swapped_info["reward_components"]["eureka_alive_bonus"] == pytest.approx(77.5)
    assert not np.allclose(swapped_rewards, default_rewards)

    # Invalid candidate code must raise ValueError and leave the swapped reward intact
    with pytest.raises(ValueError):
        load_reward_code("def compute_reward(state, actions, next_state):\n    return 123\n")

    _, still_swapped, _, _, _ = env.step(wait_actions)
    np.testing.assert_allclose(still_swapped, 77.5, rtol=1e-5, atol=1e-5)

    # Resetting restores the baseline reward function immediately on the next step
    reset_default_reward()
    _, restored_rewards, _, _, restored_info = env.step(wait_actions)
    np.testing.assert_allclose(restored_rewards, default_rewards, rtol=1e-5, atol=1e-5)
    assert "scan_reward" in restored_info["reward_components"]


def test_contract_2_world_model_error_map_and_epistemic_uncertainty() -> None:
    """Contract 2: world_model_error_map payload matches schema and flags unvisited craters."""
    report = compute_world_model_report(
        seed=42,
        quick=True,
        visited_craters=(0, 1, 2, 3),
        n_samples=2400,
        epochs=20,
    )

    # Payload must be strictly JSON-serialisable
    encoded = json.dumps(report)
    decoded: dict[str, Any] = json.loads(encoded)

    required_keys = {
        "overall_val_mse",
        "crater_metrics",
        "uncertainty_grid",
        "mean_error_grid",
        "visit_count_grid",
        "top_error_cells",
    }
    assert required_keys.issubset(decoded.keys())
    assert isinstance(decoded["overall_val_mse"], float)
    assert decoded["overall_val_mse"] >= 0.0

    # Validation MSE must decrease across training epochs
    val_history = decoded["val_mse_history"]
    assert len(val_history) >= 20
    assert val_history[-1] < val_history[0]

    # Check 20x20 spatial grids and top-5 error hotspots
    assert len(decoded["uncertainty_grid"]) == 20
    assert len(decoded["uncertainty_grid"][0]) == 20
    assert len(decoded["mean_error_grid"]) == 20
    assert len(decoded["visit_count_grid"]) == 20
    assert len(decoded["top_error_cells"]) == 5

    # Check per-crater metrics and epistemic uncertainty monotonicity (unvisited > visited)
    crater_metrics = decoded["crater_metrics"]
    assert len(crater_metrics) == 6
    visited_unc = [
        m["epistemic_uncertainty"] for m in crater_metrics if m["crater_index"] in (0, 1, 2, 3)
    ]
    unvisited_unc = [
        m["epistemic_uncertainty"] for m in crater_metrics if m["crater_index"] in (4, 5)
    ]
    unvisited_counts = [
        m["visit_count"] for m in crater_metrics if m["crater_index"] in (4, 5)
    ]
    assert all(cnt == 0 for cnt in unvisited_counts)
    assert float(np.mean(unvisited_unc)) > float(np.mean(visited_unc)) * 1.15


def test_contract_3_verifier_1000_rollouts_safe_vs_unsafe_plan() -> None:
    """Contract 3: simulate_plan runs n=1000 rollouts in < 10 s and separates safe vs unsafe plans."""
    terrain = TerrainGrid.synthetic(seed=42, quick=False)
    safe_plan = _make_safe_fleet_plan(terrain)
    unsafe_plan = _make_unsafe_fleet_plan(terrain)

    t0 = time.perf_counter()
    safe_traces = simulate_plan(safe_plan, n=1000, seed=42, terrain=terrain)
    safe_elapsed_s = time.perf_counter() - t0
    assert safe_elapsed_s < 10.0

    assert safe_traces.positions_xyz.shape == (1000, 5, 4, 3)
    assert safe_traces.fuel_kg.shape == (1000, 5, 4)
    assert safe_traces.battery_pct.shape == (1000, 5, 4)
    assert safe_traces.landing_slope_deg.shape == (1000, 4, 4)
    assert safe_traces.crashed.shape == (1000, 4, 4)
    assert safe_traces.min_separation_m.shape == (1000, 4)
    assert safe_traces.relay_conflict.shape == (1000, 4)

    safe_summary = run_verify_plan(safe_plan, n=1000, seed=42)
    assert safe_summary["num_runs"] == 1000
    assert safe_summary["crash_rate"] <= 0.01
    assert safe_summary["relay_conflict_rate"] == 0.0
    assert safe_summary["proximity_violation_rate"] == 0.0
    assert safe_summary["pass_rate"] >= 0.99
    assert len(safe_summary["trajectory_cloud"]) == 25

    unsafe_summary = run_verify_plan(unsafe_plan, n=1000, seed=42)
    assert unsafe_summary["num_runs"] == 1000
    assert unsafe_summary["relay_conflict_rate"] == 1.0
    assert unsafe_summary["proximity_violation_rate"] > 0.80
    assert unsafe_summary["crash_rate"] > 0.50
    assert unsafe_summary["pass_rate"] == 0.0


def test_contract_4_fastapi_testclient_endpoints() -> None:
    """Contract 4: FastAPI TestClient checks on /health, /verify, /reward/swap, and /episodes/sample."""
    client = TestClient(app)

    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}
    assert get_health() == {"status": "ok"}

    # 2. POST /verify
    terrain = TerrainGrid.synthetic(seed=42, quick=False)
    safe_plan = _make_safe_fleet_plan(terrain)
    verify_resp = client.post(
        "/verify",
        json={"fleet_plan": safe_plan.to_dict(), "n": 200, "seed": 42},
    )
    assert verify_resp.status_code == 200
    verify_json = verify_resp.json()
    assert verify_json["num_runs"] == 200
    assert verify_json["pass_rate"] >= 0.99
    assert len(verify_json["trajectory_cloud"]) == 25

    # 3. POST /reward/swap (valid and invalid payloads)
    valid_code = """
import numpy as np
def compute_reward(state, actions, next_state):
    r = np.full(state.fuel_kg.shape, 12.5, dtype=np.float32)
    return r, {"const_bonus": 12.5}
"""
    swap_resp = client.post(
        "/reward/swap",
        json={"code": valid_code, "summary": "api_test_swap", "episode": 3},
    )
    assert swap_resp.status_code == 200
    assert swap_resp.json()["status"] == "swapped"

    bad_swap_resp = client.post(
        "/reward/swap",
        json={"code": "def compute_reward(state, actions, next_state): raise RuntimeError('bad')"},
    )
    assert bad_swap_resp.status_code == 400

    # 4. GET /world-model/error-map
    wm_resp = client.get("/world-model/error-map?seed=42&quick=true")
    assert wm_resp.status_code == 200
    wm_json = wm_resp.json()
    assert "overall_val_mse" in wm_json
    assert len(wm_json["crater_metrics"]) == 6
    assert len(wm_json["top_error_cells"]) == 5

    # 5. GET /episodes/sample
    ep_resp = client.get("/episodes/sample?seed=42&quick=true")
    assert ep_resp.status_code == 200
    ep_doc = ep_resp.json()
    assert ep_doc["schema_version"] == 1
    assert ep_doc["meta"]["region"] == "Daedalus"
    assert len(ep_doc["craters"]) == 6
    assert len(ep_doc["episodes"]) == 1
    first_ep = ep_doc["episodes"][0]
    assert len(first_ep["drones"]) == 2
    assert len(first_ep["drones"][0]["steps"]) > 0
    assert isinstance(simulate_episode_bundle(seed=42, quick=True), dict)
