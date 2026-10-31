"""Vectorised Monte Carlo rollout engine for the safety verifier in pure NumPy.

Replays a candidate FleetPlan across n=1000 stochastic rollouts in parallel
using only the ground-truth physics simulator (selene_sim.sim). Never imports
or invokes the learned world model or PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from selene_sim.sim import (
    SimConfig,
    TerrainGrid,
    check_relay_conflicts,
    compute_battery_drain,
    compute_hop,
    is_relay_open,
)

CRASH_REASON_NAMES: dict[int, str] = {
    0: "",
    1: "out_of_fuel",
    2: "steep_slope",
    3: "out_of_bounds",
}

DEFAULT_SCAN_DURATION_S: float = 15.0
DEFAULT_SEND_DURATION_S: float = 10.0
DEFAULT_WAIT_DURATION_S: float = 20.0


@dataclass(frozen=True)
class HopCommand:
    """Single-stage command for one lunar hopper within a synchronised FleetPlan."""

    hopper_id: int
    target_xy: tuple[float, float]
    action: str  # "hop" | "scan" | "send" | "wait"
    target_crater_id: str | None = None
    wait_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise command to a JSON-compatible dictionary."""
        return {
            "hopper_id": int(self.hopper_id),
            "target_xy": [float(self.target_xy[0]), float(self.target_xy[1])],
            "action": str(self.action),
            "target_crater_id": self.target_crater_id,
            "wait_s": float(self.wait_s),
        }

    @classmethod
    def from_any(cls, obj: HopCommand | Mapping[str, Any], default_hopper_id: int = 0) -> HopCommand:
        """Coerce a HopCommand instance or dictionary into a HopCommand."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, Mapping):
            raw_xy = obj.get("target_xy", (0.0, 0.0))
            return cls(
                hopper_id=int(obj.get("hopper_id", default_hopper_id)),
                target_xy=(float(raw_xy[0]), float(raw_xy[1])),
                action=str(obj.get("action", "wait")),
                target_crater_id=obj.get("target_crater_id"),
                wait_s=float(obj.get("wait_s", 0.0)),
            )
        raise TypeError(f"Expected HopCommand or mapping, got {type(obj)!r}")


@dataclass
class FleetPlan:
    """Synchronised multi-stage route and action plan for a fleet of hoppers."""

    num_hoppers: int
    start_positions_xy: np.ndarray  # shape (H, 2)
    stages: list[list[HopCommand]] = field(default_factory=list)  # length S, each length H

    def __post_init__(self) -> None:
        self.num_hoppers = int(self.num_hoppers)
        self.start_positions_xy = np.asarray(self.start_positions_xy, dtype=np.float64).reshape(
            self.num_hoppers, 2
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the fleet plan to a JSON-compatible dictionary."""
        return {
            "num_hoppers": int(self.num_hoppers),
            "start_positions_xy": self.start_positions_xy.tolist(),
            "stages": [[cmd.to_dict() for cmd in stage] for stage in self.stages],
        }

    @classmethod
    def from_any(cls, plan: FleetPlan | Mapping[str, Any]) -> FleetPlan:
        """Coerce a FleetPlan dataclass or dictionary into a validated FleetPlan."""
        if isinstance(plan, cls):
            return plan
        if not isinstance(plan, Mapping):
            raise TypeError(f"Expected FleetPlan or mapping, got {type(plan)!r}")

        raw_starts = np.asarray(plan["start_positions_xy"], dtype=np.float64)
        num_hoppers = int(plan.get("num_hoppers", raw_starts.shape[0]))
        raw_stages = plan.get("stages", [])
        stages: list[list[HopCommand]] = []
        for raw_stage in raw_stages:
            stage_cmds = [
                HopCommand.from_any(cmd, default_hopper_id=idx)
                for idx, cmd in enumerate(raw_stage)
            ]
            stages.append(stage_cmds)
        return cls(
            num_hoppers=num_hoppers,
            start_positions_xy=raw_starts,
            stages=stages,
        )


@dataclass
class RolloutTraces:
    """Batch of Monte Carlo rollout traces for safety specification evaluation."""

    positions_xyz: np.ndarray      # (n, S+1, H, 3)
    fuel_kg: np.ndarray            # (n, S+1, H)
    battery_pct: np.ndarray        # (n, S+1, H)
    landing_slope_deg: np.ndarray  # (n, S, H)
    hop_durations_s: np.ndarray    # (n, S, H)
    min_separation_m: np.ndarray   # (n, S)
    crashed: np.ndarray            # (n, S, H) bool
    crash_reason: np.ndarray       # (n, S, H) str
    relay_conflict: np.ndarray     # (n, S) bool
    data_downlinked: np.ndarray    # (n, S, H) bool

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style attribute lookup for convenience in verifier checks."""
        return getattr(self, key)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return scalar and per-stage aggregate statistics for Friend B's verifier."""
        n_runs, s_plus_1, n_hoppers, _ = self.positions_xyz.shape
        num_stages = s_plus_1 - 1

        any_crashed_per_run = (
            np.any(self.crashed, axis=(1, 2))
            if num_stages > 0
            else np.zeros(n_runs, dtype=bool)
        )
        any_conflict_per_run = (
            np.any(self.relay_conflict, axis=1)
            if num_stages > 0
            else np.zeros(n_runs, dtype=bool)
        )
        min_sep_per_run = (
            np.min(self.min_separation_m, axis=1)
            if num_stages > 0
            else np.full(n_runs, np.inf, dtype=np.float64)
        )

        final_fuel = self.fuel_kg[:, -1, :]
        final_battery = self.battery_pct[:, -1, :]
        max_slope = (
            float(np.max(self.landing_slope_deg)) if num_stages > 0 else 0.0
        )

        return {
            "num_runs": int(n_runs),
            "num_stages": int(num_stages),
            "num_hoppers": int(n_hoppers),
            "crash_rate": float(np.mean(any_crashed_per_run)),
            "hopper_crash_rate": (
                float(np.mean(np.any(self.crashed, axis=1))) if num_stages > 0 else 0.0
            ),
            "relay_conflict_rate": float(np.mean(any_conflict_per_run)),
            "proximity_violation_rate": (
                float(np.mean(min_sep_per_run < 100.0)) if num_stages > 0 else 0.0
            ),
            "min_fuel_kg": float(np.min(final_fuel)),
            "mean_final_fuel_kg": float(np.mean(final_fuel)),
            "p05_final_fuel_kg": float(np.percentile(final_fuel, 5.0)),
            "min_battery_pct": float(np.min(final_battery)),
            "mean_final_battery_pct": float(np.mean(final_battery)),
            "p05_final_battery_pct": float(np.percentile(final_battery, 5.0)),
            "max_landing_slope_deg": max_slope,
            "min_separation_m": (
                float(np.min(self.min_separation_m)) if num_stages > 0 else float("inf")
            ),
            "total_downlinks_mean": (
                float(np.mean(np.sum(self.data_downlinked, axis=(1, 2))))
                if num_stages > 0
                else 0.0
            ),
            "stage_crash_rates": (
                np.mean(np.any(self.crashed, axis=2), axis=0).astype(float).tolist()
                if num_stages > 0
                else []
            ),
            "stage_relay_conflict_rates": (
                np.mean(self.relay_conflict, axis=0).astype(float).tolist()
                if num_stages > 0
                else []
            ),
        }


def _normalise_stage_commands(
    stage: list[HopCommand],
    num_hoppers: int,
    fallback_xy: np.ndarray,
) -> list[HopCommand]:
    """Order stage commands by hopper_id 0..H-1, filling any omitted hopper with a wait."""
    by_id: dict[int, HopCommand] = {}
    for idx, cmd in enumerate(stage):
        hid = int(cmd.hopper_id) if 0 <= int(cmd.hopper_id) < num_hoppers else idx
        by_id[hid] = cmd

    ordered: list[HopCommand] = []
    for h in range(num_hoppers):
        if h in by_id:
            ordered.append(by_id[h])
        else:
            ordered.append(
                HopCommand(
                    hopper_id=h,
                    target_xy=(float(fallback_xy[h, 0]), float(fallback_xy[h, 1])),
                    action="wait",
                )
            )
    return ordered


def simulate_plan(
    fleet_plan: FleetPlan | Mapping[str, Any],
    n: int = 1000,
    seed: int = 42,
    terrain: TerrainGrid | None = None,
    config: SimConfig | None = None,
) -> RolloutTraces:
    """Vectorised Monte Carlo execution of a FleetPlan across n noisy rollouts in parallel.

    Runs only the ground-truth physics functions in `selene_sim.sim` with leading
    batch dimension `(n, H)` so 1000 rollouts complete in milliseconds.
    """
    plan = FleetPlan.from_any(fleet_plan)
    cfg = config if config is not None else SimConfig()
    grid = terrain if terrain is not None else TerrainGrid.synthetic(seed=seed, quick=False)
    rng = np.random.default_rng(seed)

    num_runs = int(n)
    num_hoppers = int(plan.num_hoppers)
    num_stages = len(plan.stages)

    positions_xyz = np.zeros((num_runs, num_stages + 1, num_hoppers, 3), dtype=np.float32)
    fuel_kg = np.zeros((num_runs, num_stages + 1, num_hoppers), dtype=np.float32)
    battery_pct = np.zeros((num_runs, num_stages + 1, num_hoppers), dtype=np.float32)
    landing_slope_deg = np.zeros((num_runs, num_stages, num_hoppers), dtype=np.float32)
    hop_durations_s = np.zeros((num_runs, num_stages, num_hoppers), dtype=np.float32)
    min_separation_m = np.zeros((num_runs, num_stages), dtype=np.float32)
    crashed = np.zeros((num_runs, num_stages, num_hoppers), dtype=bool)
    crash_reason_codes = np.zeros((num_runs, num_stages, num_hoppers), dtype=np.int32)
    relay_conflict = np.zeros((num_runs, num_stages), dtype=bool)
    data_downlinked = np.zeros((num_runs, num_stages, num_hoppers), dtype=bool)

    cur_xy = np.broadcast_to(
        plan.start_positions_xy[None, :, :], (num_runs, num_hoppers, 2)
    ).copy()
    cur_z = grid.height_at(cur_xy[:, :, 0], cur_xy[:, :, 1]).astype(np.float64)

    positions_xyz[:, 0, :, 0] = cur_xy[:, :, 0].astype(np.float32)
    positions_xyz[:, 0, :, 1] = cur_xy[:, :, 1].astype(np.float32)
    positions_xyz[:, 0, :, 2] = cur_z.astype(np.float32)

    cur_fuel = np.full((num_runs, num_hoppers), cfg.initial_fuel_kg, dtype=np.float64)
    cur_battery = np.full((num_runs, num_hoppers), 100.0, dtype=np.float64)
    cur_crashed = np.zeros((num_runs, num_hoppers), dtype=bool)
    cur_crash_code = np.zeros((num_runs, num_hoppers), dtype=np.int32)
    has_unsent_data = np.zeros((num_runs, num_hoppers), dtype=bool)
    mission_time_s = np.zeros(num_runs, dtype=np.float64)

    fuel_kg[:, 0, :] = cur_fuel.astype(np.float32)
    battery_pct[:, 0, :] = cur_battery.astype(np.float32)

    eye_mask = np.eye(num_hoppers, dtype=bool)[None, :, :]
    nominal_xy = plan.start_positions_xy.copy()

    for s_idx, raw_stage in enumerate(plan.stages):
        cmds = _normalise_stage_commands(raw_stage, num_hoppers, nominal_xy)

        actions = [cmd.action.lower().strip() for cmd in cmds]
        is_hop_cmd = np.array([a == "hop" for a in actions], dtype=bool)
        is_scan_cmd = np.array([a == "scan" for a in actions], dtype=bool)
        is_send_cmd = np.array([a == "send" for a in actions], dtype=bool)
        wait_override_s = np.array([max(0.0, float(cmd.wait_s)) for cmd in cmds], dtype=np.float64)

        cmd_target_xy = np.array(
            [[cmd.target_xy[0], cmd.target_xy[1]] for cmd in cmds],
            dtype=np.float64,
        )
        nominal_xy = np.where(is_hop_cmd[:, None], cmd_target_xy, nominal_xy)

        active = ~cur_crashed
        do_hop = active & is_hop_cmd[None, :]
        do_scan = active & is_scan_cmd[None, :]
        do_send = active & is_send_cmd[None, :]

        # Target coordinates for compute_hop: non-hopping or crashed hoppers keep cur_xy (zero hop)
        target_xy = np.where(do_hop[:, :, None], cmd_target_xy[None, :, :], cur_xy)

        hop_res = compute_hop(
            start_pos_xy=cur_xy,
            target_pos_xy=target_xy,
            fuel_kg=cur_fuel,
            terrain=grid,
            config=cfg,
            rng=rng,
        )

        new_xy = np.where(do_hop[:, :, None], hop_res["landed_pos_xy"], cur_xy)
        new_z = grid.height_at(new_xy[:, :, 0], new_xy[:, :, 1]).astype(np.float64)
        flight_t = np.where(do_hop, hop_res["flight_time_s"], 0.0)
        fuel_burned = np.where(do_hop, hop_res["fuel_used_kg"], 0.0)
        slope_deg = np.where(
            do_hop,
            hop_res["landing_slope_deg"],
            grid.slope_at(new_xy[:, :, 0], new_xy[:, :, 1]).astype(np.float64),
        )

        newly_crashed = do_hop & hop_res["crashed"]
        cur_crashed = cur_crashed | newly_crashed
        cur_crash_code = np.where(
            newly_crashed & (cur_crash_code == 0),
            hop_res["crash_reason_code"],
            cur_crash_code,
        )

        cur_fuel = np.maximum(0.0, cur_fuel - fuel_burned)
        cur_xy = new_xy
        cur_z = new_z

        # Stage duration per hopper: flight time for hops, fixed or custom wait for ground actions
        default_ground_dt = np.where(
            is_scan_cmd,
            DEFAULT_SCAN_DURATION_S,
            np.where(is_send_cmd, DEFAULT_SEND_DURATION_S, DEFAULT_WAIT_DURATION_S),
        )
        ground_dt = np.where(wait_override_s > 0.0, wait_override_s, default_ground_dt)
        hopper_dt = np.where(
            do_hop,
            flight_t + wait_override_s[None, :],
            np.where(active, ground_dt[None, :], 0.0),
        )
        stage_dt = np.max(hopper_dt, axis=1)  # synchronised stage clock advance per rollout

        temp_c = grid.temp_at(cur_xy[:, :, 0], cur_xy[:, :, 1]).astype(np.float64)
        drain_pct = compute_battery_drain(
            dt_s=stage_dt[:, None],
            temp_c=temp_c,
            did_hop=do_hop,
            did_scan=do_scan,
            did_send=do_send,
            config=cfg,
        )
        cur_battery = np.where(
            active,
            np.clip(cur_battery - drain_pct, 0.0, 100.0),
            cur_battery,
        )

        # Scans buffer science data; sends transmit over the single-channel orbiter relay
        has_unsent_data = has_unsent_data | (do_scan & (~cur_crashed))
        relay_open = np.asarray(is_relay_open(mission_time_s, config=cfg), dtype=bool)
        send_attempts = do_send & (~cur_crashed)
        send_success, conflict = check_relay_conflicts(send_attempts, relay_open)
        downlinked_now = send_success & has_unsent_data
        has_unsent_data = np.where(downlinked_now, False, has_unsent_data)

        mission_time_s = mission_time_s + stage_dt

        # Pairwise separation across hoppers in the horizontal plane
        if num_hoppers > 1:
            diff_xy = cur_xy[:, :, None, :] - cur_xy[:, None, :, :]  # (n, H, H, 2)
            pair_dist = np.linalg.norm(diff_xy, axis=-1)  # (n, H, H)
            pair_dist_masked = np.where(eye_mask, np.inf, pair_dist)
            stage_min_sep = np.min(pair_dist_masked, axis=(-2, -1))
        else:
            stage_min_sep = np.full(num_runs, np.inf, dtype=np.float64)

        positions_xyz[:, s_idx + 1, :, 0] = cur_xy[:, :, 0].astype(np.float32)
        positions_xyz[:, s_idx + 1, :, 1] = cur_xy[:, :, 1].astype(np.float32)
        positions_xyz[:, s_idx + 1, :, 2] = cur_z.astype(np.float32)
        fuel_kg[:, s_idx + 1, :] = cur_fuel.astype(np.float32)
        battery_pct[:, s_idx + 1, :] = cur_battery.astype(np.float32)

        landing_slope_deg[:, s_idx, :] = slope_deg.astype(np.float32)
        hop_durations_s[:, s_idx, :] = flight_t.astype(np.float32)
        min_separation_m[:, s_idx] = stage_min_sep.astype(np.float32)
        crashed[:, s_idx, :] = cur_crashed
        crash_reason_codes[:, s_idx, :] = cur_crash_code
        relay_conflict[:, s_idx] = conflict
        data_downlinked[:, s_idx, :] = downlinked_now

    reason_lookup = np.array(
        [CRASH_REASON_NAMES[i] for i in range(4)],
        dtype="<U16",
    )
    crash_reason = reason_lookup[np.clip(crash_reason_codes, 0, 3)]

    return RolloutTraces(
        positions_xyz=positions_xyz,
        fuel_kg=fuel_kg,
        battery_pct=battery_pct,
        landing_slope_deg=landing_slope_deg,
        hop_durations_s=hop_durations_s,
        min_separation_m=min_separation_m,
        crashed=crashed,
        crash_reason=crash_reason,
        relay_conflict=relay_conflict,
        data_downlinked=data_downlinked,
    )


def sample_trajectory_cloud(
    traces: RolloutTraces,
    max_runs: int = 50,
) -> list[dict[str, Any]]:
    """Subsample up to max_runs rollout trajectories into a JSON-serialisable fan for 3D replay."""
    num_runs, num_waypoints, num_hoppers, _ = traces.positions_xyz.shape
    count = max(0, min(int(max_runs), int(num_runs)))
    if count == 0:
        return []

    indices = np.linspace(0, num_runs - 1, count, dtype=np.int64)
    has_stages = traces.crashed.shape[1] > 0

    cloud: list[dict[str, Any]] = []
    for run_idx in indices:
        r = int(run_idx)
        run_crashed = bool(np.any(traces.crashed[r])) if has_stages else False
        hoppers_out: list[dict[str, Any]] = []
        for h in range(num_hoppers):
            pts = np.round(traces.positions_xyz[r, :, h, :].astype(np.float64), 2).tolist()
            h_crashed = bool(np.any(traces.crashed[r, :, h])) if has_stages else False
            h_reason = (
                str(traces.crash_reason[r, -1, h])
                if has_stages and h_crashed
                else ""
            )
            hoppers_out.append(
                {
                    "hopper_id": f"h{h}",
                    "crashed": h_crashed,
                    "crash_reason": h_reason,
                    "final_fuel_kg": round(float(traces.fuel_kg[r, -1, h]), 3),
                    "final_battery_pct": round(float(traces.battery_pct[r, -1, h]), 2),
                    "path_xyz": pts,
                }
            )
        cloud.append(
            {
                "run_index": r,
                "crashed": run_crashed,
                "num_waypoints": int(num_waypoints),
                "hoppers": hoppers_out,
            }
        )
    return cloud
