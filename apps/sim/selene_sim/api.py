"""FastAPI routes and Python helper functions for the Argon orchestrator contract."""

from __future__ import annotations

from typing import Any, Mapping

from fastapi import Body, FastAPI, HTTPException
import numpy as np

from selene_sim.env import FleetEnv, VectorFleetEnv
from selene_sim.reward import (
    load_reward_code,
    reset_default_reward,
    reward_versions,
)
from selene_sim.sim import (
    SimConfig,
    TerrainGrid,
    compute_battery_drain,
    compute_hop,
)
from selene_sim.verify import (
    FleetPlan,
    sample_trajectory_cloud,
    simulate_plan,
)
from selene_sim.world_model import TransitionBuffer, WorldModelEnsemble

app = FastAPI(title="Selene")


def get_health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


def run_verify_plan(
    fleet_plan: Mapping[str, Any] | FleetPlan,
    n: int = 1000,
    seed: int = 42,
    max_cloud_runs: int = 25,
) -> dict[str, Any]:
    """Run vectorised Monte Carlo verification of a candidate FleetPlan in pure NumPy."""
    if isinstance(fleet_plan, Mapping) and "fleet_plan" in fleet_plan:
        raw_plan = fleet_plan["fleet_plan"]
        n = int(fleet_plan.get("n", n))
        seed = int(fleet_plan.get("seed", seed))
    else:
        raw_plan = fleet_plan

    traces = simulate_plan(raw_plan, n=int(n), seed=int(seed))
    summary = traces.to_summary_dict()

    any_crashed = (
        np.any(traces.crashed, axis=(1, 2))
        if traces.crashed.shape[1] > 0
        else np.zeros(traces.positions_xyz.shape[0], dtype=bool)
    )
    any_conflict = (
        np.any(traces.relay_conflict, axis=1)
        if traces.relay_conflict.shape[1] > 0
        else np.zeros(traces.positions_xyz.shape[0], dtype=bool)
    )
    any_prox = (
        np.any(traces.min_separation_m < 100.0, axis=1)
        if traces.min_separation_m.shape[1] > 0
        else np.zeros(traces.positions_xyz.shape[0], dtype=bool)
    )
    safe_runs = ~(any_crashed | any_conflict | any_prox)
    summary["pass_rate"] = float(np.mean(safe_runs))
    summary["trajectory_cloud"] = sample_trajectory_cloud(
        traces, max_runs=int(max_cloud_runs)
    )
    return summary


def swap_reward_code(
    code_str: str,
    summary: str = "orchestrator_hot_swap",
    episode: int = 0,
) -> dict[str, Any]:
    """Compile, smoke-test, and hot-swap the active fleet reward function."""
    load_reward_code(code_str, summary=summary, episode=episode)
    return {
        "status": "swapped",
        "episode": int(episode),
        "summary": str(summary),
        "version_count": len(reward_versions),
    }


def compute_world_model_report(
    seed: int = 42,
    quick: bool = True,
    visited_craters: tuple[int, ...] = (0, 1, 2, 3),
    n_samples: int = 2400,
    epochs: int = 20,
) -> dict[str, Any]:
    """Train a 5-MLP WorldModelEnsemble on simulator transitions and return its error map."""
    vec_env = VectorFleetEnv(
        num_envs=32,
        quick=quick,
        seed=seed,
        record_transitions=False,
    )
    terrain = vec_env.terrain
    cfg = SimConfig(noise_enabled=True)
    rng = np.random.default_rng(seed)

    obs, _ = vec_env.reset(seed=seed)
    base_obs = obs.reshape(-1, vec_env.obs_dim)
    half_ext = terrain.half_extent_m

    crater_xy = np.array(
        [[c.x_m, c.y_m] for c in terrain.craters[: vec_env.n_craters]],
        dtype=np.float64,
    )
    valid_craters = tuple(c for c in visited_craters if 0 <= c < vec_env.n_craters)
    if not valid_craters:
        valid_craters = (0,)

    target_ids = rng.choice(
        np.array(valid_craters, dtype=np.int32), size=int(n_samples)
    )
    wm_actions = (target_ids + 3).astype(np.int32)

    angles = rng.uniform(0.0, 2.0 * np.pi, size=int(n_samples))
    radii = rng.uniform(400.0, 2200.0, size=int(n_samples))
    target_pos_xy = crater_xy[target_ids]
    start_pos_xy = np.clip(
        target_pos_xy
        + np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1),
        -half_ext + 200.0,
        half_ext - 200.0,
    )

    sample_idx = rng.integers(0, base_obs.shape[0], size=int(n_samples))
    obs_batch = base_obs[sample_idx].copy()
    fuel_kg = rng.uniform(5.0, cfg.initial_fuel_kg, size=int(n_samples)).astype(
        np.float32
    )
    battery_pct = rng.uniform(40.0, 100.0, size=int(n_samples)).astype(np.float32)

    obs_batch[:, 0] = (start_pos_xy[:, 0] / half_ext).astype(np.float32)
    obs_batch[:, 1] = (start_pos_xy[:, 1] / half_ext).astype(np.float32)
    obs_batch[:, 2] = fuel_kg / cfg.initial_fuel_kg
    obs_batch[:, 3] = battery_pct * 0.01

    hop_out = compute_hop(
        start_pos_xy=start_pos_xy,
        target_pos_xy=target_pos_xy,
        fuel_kg=fuel_kg,
        terrain=terrain,
        config=cfg,
        rng=rng,
    )
    landed_xy = hop_out["landed_pos_xy"].astype(np.float32)
    next_fuel_kg = np.maximum(0.0, fuel_kg - hop_out["fuel_used_kg"].astype(np.float32))
    flight_dt = np.maximum(hop_out["flight_time_s"], 60.0)
    end_temp_c = terrain.temp_at(landed_xy[:, 0], landed_xy[:, 1])
    drain = compute_battery_drain(
        dt_s=flight_dt,
        temp_c=end_temp_c,
        did_hop=np.ones(int(n_samples), dtype=bool),
        did_scan=np.zeros(int(n_samples), dtype=bool),
        did_send=np.zeros(int(n_samples), dtype=bool),
        config=cfg,
    ).astype(np.float32)
    next_battery_pct = np.clip(battery_pct - drain, 0.0, 100.0)
    crashed = hop_out["crashed"].astype(bool)

    next_obs_batch = obs_batch.copy()
    next_obs_batch[:, 0] = landed_xy[:, 0] / half_ext
    next_obs_batch[:, 1] = landed_xy[:, 1] / half_ext
    next_obs_batch[:, 2] = next_fuel_kg / cfg.initial_fuel_kg
    next_obs_batch[:, 3] = next_battery_pct * 0.01
    next_obs_batch[:, 4] = (end_temp_c + 200.0) * 0.02

    priorities = np.array(
        [terrain.craters[idx].priority for idx in target_ids], dtype=np.float32
    )
    rewards = (
        10.0 * priorities * (~crashed).astype(np.float32)
        - 0.2 * hop_out["fuel_used_kg"].astype(np.float32)
        - 50.0 * crashed.astype(np.float32)
    ).astype(np.float32)

    buffer = TransitionBuffer()
    buffer.add(
        obs=obs_batch,
        actions=wm_actions,
        next_obs=next_obs_batch,
        rewards=rewards,
        crashed=crashed,
        crater_ids=target_ids,
        pos_xy=landed_xy,
    )

    ensemble = WorldModelEnsemble(
        obs_dim=vec_env.obs_dim,
        n_actions=vec_env.n_actions,
        ensemble_size=5,
        hidden_dim=128,
        seed=seed,
    )
    history = ensemble.train_epochs(buffer, epochs=int(epochs), batch_size=512, seed=seed)
    report = ensemble.world_model_error_map(terrain, buffer)
    report["val_mse_history"] = [float(x) for x in history["val_mse"]]
    return report


def simulate_episode_bundle(
    seed: int = 42,
    quick: bool = True,
    max_steps: int = 12,
) -> dict[str, Any]:
    """Run a masked episode in FleetEnv and return an EPISODE_SCHEMA-compliant document."""
    env = FleetEnv(quick=quick, seed=seed, max_steps=max_steps)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    for _ in range(env.max_steps):
        if not env.agents:
            break
        masks = env.get_action_masks()
        noise = rng.uniform(0.01, 1.0, size=masks.shape)
        scores = np.where(masks, noise, -1.0)
        actions = np.argmax(scores, axis=-1).astype(np.int32)
        env.step(actions)

    ep_dict = env.render_episode_dict(
        episode_id=0, phase="final_policy", source="learned"
    )
    versions_summary = [
        {"episode": int(v["episode"]), "summary": str(v["summary"])}
        for v in reward_versions
    ]
    return env.render_episodes_document(
        episodes=[ep_dict],
        source="trained",
        reward_versions=versions_summary,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """FastAPI health check endpoint."""
    return get_health()


@app.post("/verify")
def verify_endpoint(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Run vectorised Monte Carlo verification on a submitted FleetPlan."""
    try:
        return run_verify_plan(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reward/swap")
def reward_swap_endpoint(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Hot-swap the active fleet reward function from Python source code."""
    code_str = payload.get("code") or payload.get("code_str") or ""
    summary = str(payload.get("summary", "orchestrator_hot_swap"))
    episode = int(payload.get("episode", 0))
    try:
        return swap_reward_code(code_str= str(code_str), summary=summary, episode=episode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/world-model/error-map")
def world_model_error_map_endpoint(
    seed: int = 42,
    quick: bool = True,
) -> dict[str, Any]:
    """Return the 5-MLP ensemble world model error and epistemic uncertainty map."""
    return compute_world_model_report(seed=seed, quick=quick)


@app.get("/episodes/sample")
def episodes_sample_endpoint(
    seed: int = 42,
    quick: bool = True,
) -> dict[str, Any]:
    """Simulate a masked fleet episode and return an episodes.json bundle."""
    return simulate_episode_bundle(seed=seed, quick=quick)


__all__ = [
    "app",
    "compute_world_model_report",
    "get_health",
    "reset_default_reward",
    "run_verify_plan",
    "simulate_episode_bundle",
    "swap_reward_code",
]
