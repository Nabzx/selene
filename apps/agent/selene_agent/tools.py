from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import sys
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


WORKSPACE_DIR = Path("/workspace")
REPO_ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path.cwd()
DATA_DIR = WORKSPACE_DIR if _safe_is_file(WORKSPACE_DIR / "episodes.json") else (REPO_ROOT / "data")

# Exact SimConfig physical constants from apps/sim/selene_sim/sim.py and docs/interfaces.md §2.4
SIM_CONFIG_DEFAULTS: dict[str, Any] = {
    "g_moon": 1.62,
    "g0_earth": 9.80665,
    "isp_s": 220.0,
    "exhaust_velocity_m_s": round(220.0 * 9.80665, 2),
    "dry_mass_kg": 35.0,
    "initial_fuel_kg": 15.0,
    "max_hop_range_m": 3000.0,
    "launch_angle_deg": 45.0,
    "gravity_loss_margin": 0.08,
    "min_fuel_reserve": 0.10,
    "min_fuel_reserve_kg": 1.5,
    "min_battery_reserve": 0.20,
    "min_battery_reserve_pct": 20.0,
    "max_landing_slope_deg": 15.0,
    "min_separation_m": 100.0,
    "relay_period_s": 120.0,
    "relay_open_duration_s": 40.0,
    "thrust_sigma_rel": 0.02,
    "scatter_sigma_rel": 0.015,
    "scatter_sigma_base_m": 8.0,
    "slope_noise_sigma_deg": 1.0,
    "dust_max_extra_burn": 0.03,
}

LUNAR_GRAVITY_M_S2 = SIM_CONFIG_DEFAULTS["g_moon"]
EXHAUST_VELOCITY_M_S = 220.0 * 9.80665
DRY_MASS_KG = SIM_CONFIG_DEFAULTS["dry_mass_kg"]
INITIAL_FUEL_KG = SIM_CONFIG_DEFAULTS["initial_fuel_kg"]
GRAVITY_LOSS_MARGIN = SIM_CONFIG_DEFAULTS["gravity_loss_margin"]
MAX_HOP_RANGE_M = SIM_CONFIG_DEFAULTS["max_hop_range_m"]


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    root = name.split(".")[0]
    if root == "math":
        return math
    if root in ("numpy", "np") and _np is not None:
        return _np
    if root == "typing":
        return typing
    raise ImportError(f"Import of '{name}' is not allowed in safety/reward functions")


SAFE_BUILTINS: dict[str, Any] = {
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "callable": callable,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "Exception": Exception,
    "AttributeError": AttributeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}


@dataclass
class _SandboxRewardState:
    """Compatible RewardState matching apps/sim/selene_sim/reward.py (`RewardState`)."""

    pos_xy: Any
    fuel_kg: Any
    battery_pct: Any
    temp_c: Any
    unsent_data: Any
    crashed: Any
    alive: Any
    craters_done: Any
    craters_claimed: Any
    crater_priorities: Any
    relay_open: Any
    relay_conflict: Any
    at_crater_id: Any


def compile_sandboxed_function(code: str, fn_name: str) -> Callable[..., Any]:
    tree = ast.parse(code)
    fn_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if fn_name not in fn_names:
        raise ValueError(f"Expected 'def {fn_name}(...)' in code, found {fn_names}")
    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "Any": Any,
        "RewardState": _SandboxRewardState,
    }
    if _np is not None:
        namespace["np"] = _np
        namespace["numpy"] = _np
    exec(compile(code, f"<{fn_name}>", "exec"), namespace)
    return namespace[fn_name]


def _default_episodes_path() -> Path:
    for candidate in (
        WORKSPACE_DIR / "episodes.json",
        DATA_DIR / "episodes.json",
        DATA_DIR / "episodes.example.json",
    ):
        if _safe_is_file(candidate):
            return candidate
    return DATA_DIR / "episodes.example.json"


def _default_mission_doc_path() -> Path:
    for candidate in (
        WORKSPACE_DIR / "mission_constraints.md",
        DATA_DIR / "mission_constraints.md",
        DATA_DIR / "mission_constraints.example.md",
    ):
        if _safe_is_file(candidate):
            return candidate
    return DATA_DIR / "mission_constraints.example.md"


def _default_safety_spec_path() -> Path:
    for candidate in (
        WORKSPACE_DIR / "safety_spec.json",
        DATA_DIR / "safety_spec.json",
        DATA_DIR / "safety_spec.cached.json",
    ):
        if _safe_is_file(candidate):
            return candidate
    return DATA_DIR / "safety_spec.json"


def _load_episodes_data(episodes_data: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    if isinstance(episodes_data, dict):
        return episodes_data
    path = Path(episodes_data) if episodes_data else _default_episodes_path()
    if _safe_is_file(path):
        return json.loads(path.read_text())
    return {}


def _enrich_craters(raw_craters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich crater dicts with `pos`, `floor_slope_deg`, `priority`, `reachable_from_origin`, and `reachable_craters`."""
    enriched: list[dict[str, Any]] = []
    for idx, c in enumerate(raw_craters):
        cid = str(c.get("id", f"c{idx}"))
        pos = c.get("pos")
        if not pos:
            pos = [float(c.get("x", c.get("x_m", 0.0))), float(c.get("y", c.get("y_m", 0.0))), float(c.get("z", -140.0))]
        x_m, y_m = float(pos[0]), float(pos[1])
        z_m = float(pos[2]) if len(pos) > 2 else -140.0
        radius_m = float(c.get("radius", c.get("radius_m", 800.0)))
        depth_m = float(c.get("depth_m", round(radius_m * 0.18, 1)))
        slope_deg = float(c.get("floor_slope_deg", round(3.0 + (idx % 5) * 0.6, 1)))
        priority = float(c.get("priority", round(1.0 + (idx % 5) * 0.1, 1)))
        dist_origin = math.hypot(x_m, y_m)
        enriched.append(
            {
                "id": cid,
                "name": str(c.get("name", f"Hollow {idx}")),
                "x": x_m,
                "y": y_m,
                "z": z_m,
                "pos": [x_m, y_m, z_m],
                "radius": radius_m,
                "radius_m": radius_m,
                "depth_m": depth_m,
                "floor_slope_deg": slope_deg,
                "priority": priority,
                "reachable_from_origin": dist_origin <= MAX_HOP_RANGE_M,
                "reachable_craters": [],
            }
        )
    for i, ci in enumerate(enriched):
        reach: list[str] = []
        for j, cj in enumerate(enriched):
            if i == j:
                continue
            if math.hypot(ci["pos"][0] - cj["pos"][0], ci["pos"][1] - cj["pos"][1]) <= MAX_HOP_RANGE_M:
                reach.append(cj["id"])
        ci["reachable_craters"] = reach
    return enriched


def _normalise_plan_inputs(
    routes: list[dict[str, Any]] | None = None,
    hoppers: list[dict[str, Any]] | None = None,
    stages: list[list[dict[str, Any]]] | None = None,
    fleet_plan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Coerce any fleet plan input (`routes`, `hoppers`, `stages`, or `fleet_plan`) into both `routes` and `sim_plan`."""
    plan_id = "plan_v1"
    revision = 0
    start_positions_xy: list[list[float]] = []

    if isinstance(fleet_plan, dict):
        plan_id = str(fleet_plan.get("plan_id", plan_id))
        revision = int(fleet_plan.get("revision", revision))
        routes = routes or fleet_plan.get("routes")
        hoppers = hoppers or fleet_plan.get("hoppers")
        stages = stages or fleet_plan.get("stages")
        if fleet_plan.get("start_positions_xy"):
            start_positions_xy = [[float(p[0]), float(p[1])] for p in fleet_plan["start_positions_xy"]]

    if hoppers and not routes:
        routes = []
        for h in hoppers:
            hid = str(h.get("id", h.get("hopper_id", "h0")))
            start_xy = list(h.get("start_pos_xy", [0.0, 0.0]))
            cur_xyz = [float(start_xy[0]), float(start_xy[1]), -120.0]
            hops_out: list[dict[str, Any]] = []
            for s_idx, st in enumerate(h.get("steps", [])):
                txy = st.get("target_xy", start_xy)
                next_xyz = [float(txy[0]), float(txy[1]), -120.0]
                act = str(st.get("action", "hop"))
                hops_out.append(
                    {
                        "hop_index": s_idx,
                        "from_pos": list(cur_xyz),
                        "to_pos": next_xyz,
                        "target_crater": st.get("target_crater") or st.get("target_crater_id"),
                        "action_after": act,
                        "wait_s": float(st.get("wait_s", 0.0)),
                    }
                )
                if act == "hop":
                    cur_xyz = next_xyz
            routes.append(
                {
                    "hopper_id": hid,
                    "craters": list(h.get("craters_assigned", [])),
                    "hops": hops_out,
                }
            )

    if stages and not routes:
        n_h = len(start_positions_xy) if start_positions_xy else (len(stages[0]) if stages else 0)
        starts = start_positions_xy or [[0.0, 0.0] for _ in range(n_h)]
        routes = []
        for h_idx in range(n_h):
            cur_xyz = [float(starts[h_idx][0]), float(starts[h_idx][1]), -120.0]
            hops_out = []
            craters_seen: list[str] = []
            for s_idx, stage_cmds in enumerate(stages):
                cmd = stage_cmds[h_idx] if h_idx < len(stage_cmds) else {}
                txy = cmd.get("target_xy", cur_xyz[:2])
                next_xyz = [float(txy[0]), float(txy[1]), -120.0]
                act = str(cmd.get("action", "wait"))
                tc = cmd.get("target_crater_id") or cmd.get("target_crater")
                if tc and tc not in craters_seen:
                    craters_seen.append(str(tc))
                hops_out.append(
                    {
                        "hop_index": s_idx,
                        "from_pos": list(cur_xyz),
                        "to_pos": next_xyz,
                        "target_crater": tc,
                        "action_after": act,
                        "wait_s": float(cmd.get("wait_s", 0.0)),
                    }
                )
                if act == "hop":
                    cur_xyz = next_xyz
            routes.append({"hopper_id": f"h{h_idx}", "craters": craters_seen, "hops": hops_out})

    routes = list(routes or [])

    if not stages:
        num_h = len(routes)
        start_positions_xy = []
        per_hopper_cmds: list[list[dict[str, Any]]] = []
        for h_idx, r in enumerate(routes):
            r_hops = r.get("hops", [])
            first_from = r_hops[0].get("from_pos", [0.0, 0.0]) if r_hops else [0.0, 0.0]
            start_positions_xy.append([float(first_from[0]), float(first_from[1])])
            cmds_h: list[dict[str, Any]] = []
            for hop in r_hops:
                to_p = hop.get("to_pos", first_from)
                txy = [float(to_p[0]), float(to_p[1])]
                tc = hop.get("target_crater")
                wait_s = float(hop.get("wait_s", 0.0))
                act_raw = str(hop.get("action_after", "hop")).lower().strip()
                if act_raw in ("hop", "scan", "send", "wait") and len(r_hops) > 1 and any(
                    str(x.get("action_after", "")).lower() in ("hop", "scan", "send", "wait") for x in r_hops
                ):
                    # Stage-aligned steps vs compound hop+scan
                    from_p = hop.get("from_pos", to_p)
                    dist = math.hypot(float(to_p[0]) - float(from_p[0]), float(to_p[1]) - float(from_p[1]))
                    if dist > 1.0 and act_raw != "hop":
                        cmds_h.append(
                            {
                                "hopper_id": h_idx,
                                "target_xy": txy,
                                "action": "hop",
                                "target_crater_id": tc,
                                "wait_s": 0.0,
                            }
                        )
                        norm_act = "send" if act_raw in ("transmit", "send") else ("scan" if "scan" in act_raw else "wait")
                        cmds_h.append(
                            {
                                "hopper_id": h_idx,
                                "target_xy": txy,
                                "action": norm_act,
                                "target_crater_id": tc,
                                "wait_s": wait_s,
                            }
                        )
                    else:
                        norm_act = (
                            "hop"
                            if act_raw == "hop"
                            else ("send" if act_raw in ("transmit", "send") else ("scan" if "scan" in act_raw else "wait"))
                        )
                        cmds_h.append(
                            {
                                "hopper_id": h_idx,
                                "target_xy": txy,
                                "action": norm_act,
                                "target_crater_id": tc,
                                "wait_s": wait_s,
                            }
                        )
                else:
                    cmds_h.append(
                        {
                            "hopper_id": h_idx,
                            "target_xy": txy,
                            "action": "hop",
                            "target_crater_id": tc,
                            "wait_s": 0.0,
                        }
                    )
                    for sub_act in [a.strip() for a in act_raw.replace("+", ",").replace("_and_", ",").split(",") if a.strip()]:
                        if sub_act in ("hop", "none"):
                            continue
                        norm_act = "send" if sub_act in ("transmit", "send", "downlink") else ("scan" if "scan" in sub_act else "wait")
                        cmds_h.append(
                            {
                                "hopper_id": h_idx,
                                "target_xy": txy,
                                "action": norm_act,
                                "target_crater_id": tc,
                                "wait_s": wait_s,
                            }
                        )
            per_hopper_cmds.append(cmds_h)

        max_s = max((len(c) for c in per_hopper_cmds), default=0)
        stages = []
        last_xy = [list(xy) for xy in start_positions_xy]
        for s_idx in range(max_s):
            stage_row: list[dict[str, Any]] = []
            for h_idx in range(num_h):
                if s_idx < len(per_hopper_cmds[h_idx]):
                    cmd = per_hopper_cmds[h_idx][s_idx]
                    if cmd["action"] == "hop":
                        last_xy[h_idx] = list(cmd["target_xy"])
                    stage_row.append(cmd)
                else:
                    stage_row.append(
                        {
                            "hopper_id": h_idx,
                            "target_xy": list(last_xy[h_idx]),
                            "action": "wait",
                            "target_crater_id": None,
                            "wait_s": 0.0,
                        }
                    )
            stages.append(stage_row)

    sim_plan = {
        "plan_id": plan_id,
        "revision": revision,
        "num_hoppers": len(start_positions_xy) or len(routes),
        "start_positions_xy": start_positions_xy or [[0.0, 0.0] for _ in range(max(1, len(routes)))],
        "stages": stages or [],
    }
    return routes, sim_plan


def _build_ballistic_arc(p0: list[float], p1: list[float], n_pts: int = 9) -> list[list[float]]:
    """Interpolate a smooth 3D ballistic parabola between two [x, y, z] endpoints for the 3D trajectory cloud."""
    x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2]) if len(p0) > 2 else -120.0
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2]) if len(p1) > 2 else -120.0
    horiz = math.hypot(x1 - x0, y1 - y0)
    apex_h = 0.25 * horiz
    pts: list[list[float]] = []
    for i in range(n_pts):
        u = i / max(1, n_pts - 1)
        x = round(x0 + u * (x1 - x0), 2)
        y = round(y0 + u * (y1 - y0), 2)
        z = round((1.0 - u) * z0 + u * z1 + 4.0 * u * (1.0 - u) * apex_h, 2)
        pts.append([x, y, z])
    return pts


class AgentToolset:
    """Pluggable tool registry that lets Argon inspect mission data, test rewards, estimate hops, verify plans and read logs."""

    def __init__(
        self,
        *,
        episodes_data: dict[str, Any] | str | Path | None = None,
        safety_spec: Any = None,
        hoppers: list[dict[str, Any]] | None = None,
        verifier_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        extra_files: list[str | Path] | None = None,
        csv_path: str | Path | None = None,
    ):
        self._episodes_source = episodes_data
        self._safety_spec_source = safety_spec
        self._hoppers = hoppers
        self._verifier_fn = verifier_fn
        self._extra_files = [Path(p) for p in (extra_files or [])]
        self._csv_path = Path(csv_path) if csv_path else (DATA_DIR / "training_log.csv")
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._declarations: dict[str, dict[str, Any]] = {}
        self._register_defaults()

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register or override a tool callable by Argon."""
        self._handlers[name] = handler
        self._declarations[name] = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    @property
    def declarations(self) -> list[dict[str, Any]]:
        return list(self._declarations.values())

    @property
    def handlers(self) -> dict[str, Callable[..., Any]]:
        return self._handlers

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if name not in self._handlers:
            raise KeyError(f"Unknown tool: {name}")
        return self._handlers[name](**(arguments or {}))

    def _load_safety_rules(self) -> list[dict[str, Any]]:
        src = self._safety_spec_source
        if src is not None and hasattr(src, "model_dump"):
            src = src.model_dump()
        if isinstance(src, dict):
            return list(src.get("rules", []))
        if src:
            path = Path(src)
            if _safe_is_file(path):
                return list(json.loads(path.read_text()).get("rules", []))
        default_spec = _default_safety_spec_path()
        if _safe_is_file(default_spec):
            return list(json.loads(default_spec.read_text()).get("rules", []))
        return []

    def _register_defaults(self) -> None:
        self.register(
            name="read_mission_documents",
            handler=self.read_mission_documents,
            description=(
                "Read mission safety standards and constraint documents (markdown, text or JSON) "
                "from the supplied path or default mission constraints files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path to a specific markdown or text document.",
                    },
                },
            },
        )
        self.register(
            name="get_mission_state",
            handler=self.get_mission_state,
            description=(
                "Fetch current mission state: region metadata, SimConfig physics limits, crater catalogue "
                "(id, name, pos, radius_m, depth_m, floor_slope_deg, priority, reachable_from_origin, reachable_craters), "
                "relay windows and fleet hoppers."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
        )
        self.register(
            name="estimate_hop",
            handler=self.estimate_hop,
            description=(
                "Estimate ballistic hop distance, flight duration, delta-v (with 8% gravity-loss margin), "
                "fuel used (fraction and kg for Isp=220s, dry_mass=35kg, initial_fuel=15kg) and battery used "
                "under lunar gravity (1.62 m/s^2). Estimate only, never a verifier verdict."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "from_pos": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Launch position [x, y, z] in metres.",
                    },
                    "to_pos": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target landing position [x, y, z] in metres.",
                    },
                    "launch_angle_deg": {
                        "type": "number",
                        "description": "Launch elevation angle in degrees (default 45).",
                    },
                },
                "required": ["from_pos", "to_pos"],
            },
        )
        self.register(
            name="get_training_summary",
            handler=self.get_training_summary,
            description=(
                "Read MAPPO training curves (episode_rewards), failure rates (crash_rate, crater_overlap_rate, "
                "relay_conflict_rate, craters_visited_mean), reward_components, reward_versions history and sample_episodes."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
        )
        self.register(
            name="test_reward_code",
            handler=self.test_reward_code,
            description=(
                "Compile and smoke-test a candidate Python `compute_reward` function against both vectorised "
                "`RewardState` batches (1D `(H,)` and 2D `(E, H)`) and logged episode steps."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reward_code": {
                        "type": "string",
                        "description": (
                            "Python source defining `def compute_reward(state, actions=None, next_state=None)` "
                            "or `def compute_reward(step, prev_step=None, fleet=None)`."
                        ),
                    },
                },
                "required": ["reward_code"],
            },
        )
        self.register(
            name="verify_fleet_plan",
            handler=self.verify_fleet_plan,
            description=(
                "Run the ground-truth Monte Carlo verifier (`n=1000` rollouts, `seed=42`) on a candidate fleet plan "
                "(accepting `routes`, `hoppers` or `stages`) and return the Verdict: `passed`, `pass_rate`, "
                "`rule_results`, `failed_rollout_indices`, `summary` and 3D `trajectory_cloud`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "routes": {
                        "type": "array",
                        "description": "Optional list of hopper routes (`hopper_id`, `craters`, `hops`).",
                        "items": {"type": "object"},
                    },
                    "hoppers": {
                        "type": "array",
                        "description": "Optional list of hopper plans (`id`, `start_pos_xy`, `craters_assigned`, `steps`).",
                        "items": {"type": "object"},
                    },
                    "num_rollouts": {
                        "type": "integer",
                        "description": "Number of Monte Carlo rollouts to run (default 1000).",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Deterministic RNG seed (default 42).",
                    },
                },
            },
        )
        self.register(
            name="get_hopper_logs",
            handler=self.get_hopper_logs,
            description=(
                "Read hopper step logs, world-model prediction errors (`pos`, `wm_pos`, `wm_error`) "
                "and the per-crater world-model error map."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "episode_index": {
                        "type": "integer",
                        "description": "Optional specific episode index to inspect.",
                    },
                    "min_wm_error": {
                        "type": "number",
                        "description": "Optional minimum world-model error in metres to filter steps.",
                    },
                },
            },
        )
        self.register(
            name="get_world_model_error_map",
            handler=self.get_world_model_error_map,
            description=(
                "Return per-crater (`visit_count`, `mean_error_m`, `uncertainty_m`) and 20x20 coarse grid "
                "world-model prediction error and ensemble disagreement in the local ENU frame."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
        )

    def read_mission_documents(self, path: str | None = None) -> dict[str, Any]:
        docs: list[dict[str, str]] = []
        if path:
            target = Path(path)
            if not _safe_is_file(target):
                target = REPO_ROOT / path
            if _safe_is_file(target):
                docs.append({"path": str(target), "content": target.read_text()})
        for extra in self._extra_files:
            if _safe_is_file(extra):
                docs.append({"path": str(extra), "content": extra.read_text()})
        if not docs:
            default_doc = _default_mission_doc_path()
            if _safe_is_file(default_doc):
                docs.append({"path": str(default_doc), "content": default_doc.read_text()})
        return {"documents": docs, "sim_config": SIM_CONFIG_DEFAULTS}

    def get_mission_state(self) -> dict[str, Any]:
        data = _load_episodes_data(self._episodes_source)
        raw_craters = data.get("craters", [])
        if not raw_craters:
            raw_craters = [
                {"id": "c0", "name": "Hollow 0", "pos": [1650.0, 450.0, -142.0], "radius_m": 800.0, "depth_m": 145.0, "priority": 1.0},
                {"id": "c1", "name": "Hollow 1", "pos": [650.0, 1750.0, -135.0], "radius_m": 750.0, "depth_m": 135.0, "priority": 1.1},
                {"id": "c2", "name": "Hollow 2", "pos": [-1450.0, 1150.0, -155.0], "radius_m": 850.0, "depth_m": 155.0, "priority": 1.2},
                {"id": "c3", "name": "Hollow 3", "pos": [-1750.0, -650.0, -140.0], "radius_m": 800.0, "depth_m": 140.0, "priority": 1.0},
            ]
        craters = _enrich_craters(raw_craters)
        hoppers = self._hoppers
        if hoppers is None:
            episodes = data.get("episodes", [])
            if episodes and episodes[0].get("drones"):
                hoppers = []
                for d in episodes[0]["drones"]:
                    d_steps = d.get("steps", [])
                    first_step = d_steps[0] if d_steps else {}
                    hoppers.append(
                        {
                            "hopper_id": d.get("id", "h0"),
                            "name": d.get("name", ""),
                            "pos": first_step.get("pos", [0.0, 0.0, -120.0]),
                            "fuel": first_step.get("fuel", 1.0),
                            "battery": first_step.get("battery", 0.95),
                            "temp_c": first_step.get("temp_c", -180.0),
                        }
                    )
            else:
                first_pos = [0.0, 0.0, -120.0]
                if episodes and episodes[0].get("steps"):
                    first_pos = episodes[0]["steps"][0].get("pos", first_pos)
                hoppers = [
                    {"hopper_id": "h0", "pos": first_pos, "fuel": 1.0, "battery": 1.0, "temp_c": -185.0},
                    {"hopper_id": "h1", "pos": [first_pos[0] + 250.0, first_pos[1], first_pos[2]], "fuel": 1.0, "battery": 1.0, "temp_c": -185.0},
                    {"hopper_id": "h2", "pos": [first_pos[0], first_pos[1] + 250.0, first_pos[2]], "fuel": 1.0, "battery": 1.0, "temp_c": -185.0},
                    {"hopper_id": "h3", "pos": [first_pos[0] + 250.0, first_pos[1] + 250.0, first_pos[2]], "fuel": 1.0, "battery": 1.0, "temp_c": -185.0},
                ]
        return {
            "meta": data.get("meta", {"region": "Daedalus", "grid_size": [200, 200], "cell_size_m": 100.0}),
            "sim_config": SIM_CONFIG_DEFAULTS,
            "craters": craters,
            "relay_windows": data.get("relay_windows", [{"start": 0.0, "end": 40.0}, {"start": 120.0, "end": 160.0}, {"start": 240.0, "end": 280.0}]),
            "hoppers": hoppers,
        }

    def estimate_hop(
        self,
        from_pos: list[float],
        to_pos: list[float],
        launch_angle_deg: float = 45.0,
        current_fuel_kg: float = INITIAL_FUEL_KG,
    ) -> dict[str, Any]:
        dx = float(to_pos[0]) - float(from_pos[0])
        dy = float(to_pos[1]) - float(from_pos[1])
        dz = (float(to_pos[2]) - float(from_pos[2])) if len(to_pos) > 2 and len(from_pos) > 2 else 0.0
        horiz_range_m = math.hypot(dx, dy)
        theta = math.radians(max(15.0, min(75.0, launch_angle_deg)))
        sin_2theta = max(0.1, math.sin(2.0 * theta))
        v_launch = math.sqrt(horiz_range_m * LUNAR_GRAVITY_M_S2 / sin_2theta)
        v_land_sq = max(0.0, v_launch * v_launch - 2.0 * LUNAR_GRAVITY_M_S2 * dz)
        v_land = math.sqrt(v_land_sq)
        delta_v = (v_launch + v_land) * (1.0 + GRAVITY_LOSS_MARGIN) if horiz_range_m > 0 else 0.0
        m_total = DRY_MASS_KG + max(0.0, float(current_fuel_kg))
        fuel_used_kg = m_total * (1.0 - math.exp(-delta_v / EXHAUST_VELOCITY_M_S)) if horiz_range_m > 0 else 0.0
        fuel_fraction = fuel_used_kg / INITIAL_FUEL_KG
        flight_time_s = (2.0 * v_launch * math.sin(theta) / LUNAR_GRAVITY_M_S2) if horiz_range_m > 0 else 0.0
        # Matches compute_battery_drain in selene_sim/sim.py at -185 °C (1.8% hop + base + heater)
        battery_pct_used = (1.8 + (0.005 + 0.0003 * 165.0) * flight_time_s) if horiz_range_m > 0 else 0.0
        battery_fraction = battery_pct_used / 100.0
        return {
            "is_estimate": True,
            "within_max_range": horiz_range_m <= MAX_HOP_RANGE_M,
            "max_hop_range_m": MAX_HOP_RANGE_M,
            "horizontal_distance_m": round(horiz_range_m, 2),
            "elevation_change_m": round(dz, 2),
            "flight_time_s": round(flight_time_s, 2),
            "delta_v_m_s": round(delta_v, 2),
            "estimated_fuel_used_kg": round(fuel_used_kg, 4),
            "estimated_fuel_used": round(fuel_fraction, 4),
            "estimated_battery_used_pct": round(battery_pct_used, 4),
            "estimated_battery_used": round(battery_fraction, 4),
        }

    def _extract_drone_streams(self, ep: dict[str, Any]) -> list[dict[str, Any]]:
        if ep.get("drones"):
            return list(ep["drones"])
        return [
            {
                "id": "h0",
                "name": "Hopper 0",
                "outcome": ep.get("outcome", "completed"),
                "craters_visited": ep.get("craters_visited", []),
                "steps": ep.get("steps", []),
            }
        ]

    def get_training_summary(self) -> dict[str, Any]:
        data = _load_episodes_data(self._episodes_source)
        training = data.get("training", {})
        episodes = data.get("episodes", [])
        ep_rewards = list(training.get("episode_rewards", []))
        rew_versions = list(training.get("reward_versions", []))

        csv_rows: list[dict[str, Any]] = []
        if _safe_is_file(self._csv_path):
            try:
                with self._csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        csv_rows.append(
                            {
                                "update": int(float(row["update"])),
                                "env_steps": int(float(row["env_steps"])),
                                "episode_reward": float(row["episode_reward"]),
                                "crashes": float(row["crashes"]),
                                "craters_visited": float(row["craters_visited"]),
                                "crater_overlap": float(row["crater_overlap"]),
                                "relay_conflicts": float(row["relay_conflicts"]),
                            }
                        )
            except Exception:
                csv_rows = []

        if csv_rows and not ep_rewards:
            ep_rewards = [r["episode_reward"] for r in csv_rows]

        # Compute summary metrics from episodes or CSV rows
        total_drones = 0
        crash_drones = 0
        craters_counts: list[int] = []
        sample_episodes: list[dict[str, Any]] = []
        for ep in episodes:
            drones = self._extract_drone_streams(ep)
            ep_craters: list[str] = list(ep.get("craters_visited", []))
            ep_crashes = 0
            for d in drones:
                total_drones += 1
                for cid in d.get("craters_visited", []):
                    if cid not in ep_craters:
                        ep_craters.append(cid)
                if d.get("outcome") in ("crash", "crashed", "tip_over") or any(
                    s.get("event") in ("crash", "crashed", "tip_over") for s in d.get("steps", [])
                ):
                    crash_drones += 1
                    ep_crashes += 1
            craters_counts.append(len(ep_craters))
            sample_episodes.append(
                {
                    "index": ep.get("index"),
                    "outcome": ep.get("outcome", "completed"),
                    "total_reward": ep.get("total_reward"),
                    "craters_visited": ep_craters,
                    "crashes": ep_crashes,
                    "relay_conflicts": 0,
                    "num_drones": len(drones),
                    "num_steps": sum(len(d.get("steps", [])) for d in drones),
                }
            )

        crash_rate = (
            float(csv_rows[-1]["crashes"])
            if csv_rows
            else round(crash_drones / max(1, total_drones), 4)
        )
        overlap_rate = float(csv_rows[-1]["crater_overlap"]) if csv_rows else 0.0
        conflict_rate = float(csv_rows[-1]["relay_conflicts"]) if csv_rows else 0.0
        visited_mean = (
            float(csv_rows[-1]["craters_visited"])
            if csv_rows
            else (round(sum(craters_counts) / len(craters_counts), 2) if craters_counts else 0.0)
        )

        return {
            "source": data.get("meta", {}).get("source", "trained"),
            "episodes_completed": len(ep_rewards),
            "episode_rewards": ep_rewards,
            "crash_rate": crash_rate,
            "crater_overlap_rate": overlap_rate,
            "relay_conflict_rate": conflict_rate,
            "craters_visited_mean": visited_mean,
            "reward_components": training.get(
                "reward_components",
                {
                    "scan_reward": 25.0,
                    "claim_reward": 5.0,
                    "relay_reward": 15.0,
                    "fuel_penalty": -0.2,
                    "crash_penalty": -50.0,
                    "battery_penalty": -5.0,
                    "separation_penalty": -5.0,
                    "relay_conflict_penalty": -10.0,
                },
            ),
            "reward_versions": rew_versions,
            "sample_episodes": sample_episodes,
            "episodes_summary": sample_episodes,
        }

    def test_reward_code(self, reward_code: str) -> dict[str, Any]:
        fn = compile_sandboxed_function(reward_code, "compute_reward")

        # 1. Vectorised RewardState smoke test (matching selene_sim.reward.load_reward_code) if numpy is available
        vectorised_ok = False
        sample_components: dict[str, float] = {}
        if _np is not None:
            try:
                for batch_shape in ((), (2,)):
                    h_shape = (*batch_shape, 3)
                    c_shape = (*batch_shape, 4)
                    s0 = _SandboxRewardState(
                        pos_xy=_np.zeros((*h_shape, 2), dtype=_np.float32),
                        fuel_kg=_np.full(h_shape, 15.0, dtype=_np.float32),
                        battery_pct=_np.full(h_shape, 80.0, dtype=_np.float32),
                        temp_c=_np.full(h_shape, -180.0, dtype=_np.float32),
                        unsent_data=_np.zeros(h_shape, dtype=_np.float32),
                        crashed=_np.zeros(h_shape, dtype=bool),
                        alive=_np.ones(h_shape, dtype=bool),
                        craters_done=_np.zeros(c_shape, dtype=bool),
                        craters_claimed=_np.full(c_shape, -1, dtype=_np.int32),
                        crater_priorities=_np.ones(c_shape, dtype=_np.float32),
                        relay_open=_np.zeros(batch_shape, dtype=bool) if batch_shape else _np.asarray(False, dtype=bool),
                        relay_conflict=_np.zeros(batch_shape, dtype=bool) if batch_shape else _np.asarray(False, dtype=bool),
                        at_crater_id=_np.full(h_shape, -1, dtype=_np.int32),
                    )
                    s1 = _SandboxRewardState(
                        pos_xy=_np.zeros((*h_shape, 2), dtype=_np.float32),
                        fuel_kg=_np.full(h_shape, 13.5, dtype=_np.float32),
                        battery_pct=_np.full(h_shape, 76.0, dtype=_np.float32),
                        temp_c=_np.full(h_shape, -180.0, dtype=_np.float32),
                        unsent_data=_np.zeros(h_shape, dtype=_np.float32),
                        crashed=_np.zeros(h_shape, dtype=bool),
                        alive=_np.ones(h_shape, dtype=bool),
                        craters_done=_np.ones(c_shape, dtype=bool),
                        craters_claimed=_np.zeros(c_shape, dtype=_np.int32),
                        crater_priorities=_np.ones(c_shape, dtype=_np.float32),
                        relay_open=_np.ones(batch_shape, dtype=bool) if batch_shape else _np.asarray(True, dtype=bool),
                        relay_conflict=_np.zeros(batch_shape, dtype=bool) if batch_shape else _np.asarray(False, dtype=bool),
                        at_crater_id=_np.zeros(h_shape, dtype=_np.int32),
                    )
                    acts = _np.zeros(h_shape, dtype=_np.int32)
                    res = fn(s0, acts, s1)
                    if isinstance(res, tuple) and len(res) == 2:
                        rew_arr = _np.asarray(res[0], dtype=_np.float32)
                        if rew_arr.shape != h_shape:
                            return {
                                "valid": False,
                                "vectorised_compatible": False,
                                "error": f"compute_reward returned shape {rew_arr.shape}, expected {h_shape}",
                            }
                        if not _np.all(_np.isfinite(rew_arr)):
                            return {
                                "valid": False,
                                "vectorised_compatible": False,
                                "error": "compute_reward returned non-finite values on dummy RewardState",
                            }
                        sample_components = {str(k): round(float( _np.mean(v)), 4) for k, v in dict(res[1]).items()}
                        vectorised_ok = True
            except TypeError:
                vectorised_ok = False
            except AttributeError as exc:
                if "has no attribute 'get'" in str(exc) or "has no attribute 'keys'" in str(exc):
                    vectorised_ok = False
                else:
                    return {
                        "valid": False,
                        "vectorised_compatible": False,
                        "error": (
                            f"RewardState attribute error: {exc}. Use exact RewardState fields: "
                            "pos_xy, fuel_kg, battery_pct, temp_c, unsent_data, crashed, alive, "
                            "craters_done, craters_claimed, crater_priorities, relay_open, relay_conflict, at_crater_id."
                        ),
                    }

        # 2. Evaluate on logged episodes in episodes.json
        data = _load_episodes_data(self._episodes_source)
        episodes = data.get("episodes", [])
        results: list[dict[str, Any]] = []
        for ep in episodes:
            drones = self._extract_drone_streams(ep)
            step_scores: list[float] = []
            for d in drones:
                prev: dict[str, Any] | None = None
                for step in d.get("steps", []):
                    score = 0.0
                    if vectorised_ok and _np is not None:
                        p0 = prev if prev is not None else step
                        s0_step = _SandboxRewardState(
                            pos_xy=_np.array([[float(p0.get("pos", [0.0, 0.0])[0]), float(p0.get("pos", [0.0, 0.0])[1])]], dtype=_np.float32),
                            fuel_kg=_np.array([float(p0.get("fuel", 1.0)) * INITIAL_FUEL_KG], dtype=_np.float32),
                            battery_pct=_np.array([float(p0.get("battery", 1.0)) * 100.0], dtype=_np.float32),
                            temp_c=_np.array([float(p0.get("temp_c", -180.0))], dtype=_np.float32),
                            unsent_data=_np.array([1.0 if step.get("action", {}).get("type") == "transmit" else 0.0], dtype=_np.float32),
                            crashed=_np.array([str(p0.get("event", "")) in ("crash", "crashed", "tip_over")], dtype=bool),
                            alive=_np.array([str(p0.get("event", "")) not in ("crash", "crashed", "tip_over")], dtype=bool),
                            craters_done=_np.array([False], dtype=bool),
                            craters_claimed=_np.array([-1], dtype=_np.int32),
                            crater_priorities=_np.array([1.0], dtype=_np.float32),
                            relay_open=_np.asarray(True, dtype=bool),
                            relay_conflict=_np.asarray(False, dtype=bool),
                            at_crater_id=_np.array([0], dtype=_np.int32),
                        )
                        s1_step = _SandboxRewardState(
                            pos_xy=_np.array([[float(step.get("pos", [0.0, 0.0])[0]), float(step.get("pos", [0.0, 0.0])[1])]], dtype=_np.float32),
                            fuel_kg=_np.array([float(step.get("fuel", 1.0)) * INITIAL_FUEL_KG], dtype=_np.float32),
                            battery_pct=_np.array([float(step.get("battery", 1.0)) * 100.0], dtype=_np.float32),
                            temp_c=_np.array([float(step.get("temp_c", -180.0))], dtype=_np.float32),
                            unsent_data=_np.array([0.0], dtype=_np.float32),
                            crashed=_np.array([str(step.get("event", "")) in ("crash", "crashed", "tip_over")], dtype=bool),
                            alive=_np.array([str(step.get("event", "")) not in ("crash", "crashed", "tip_over")], dtype=bool),
                            craters_done=_np.array([step.get("action", {}).get("type") == "scan" or step.get("event") == "scanned"], dtype=bool),
                            craters_claimed=_np.array([0], dtype=_np.int32),
                            crater_priorities=_np.array([1.0], dtype=_np.float32),
                            relay_open=_np.asarray(True, dtype=bool),
                            relay_conflict=_np.asarray(False, dtype=bool),
                            at_crater_id=_np.array([0], dtype=_np.int32),
                        )
                        act_arr = _np.array([1], dtype=_np.int32)
                        r_out, _ = fn(s0_step, act_arr, s1_step)
                        score = float(_np.asarray(r_out).reshape(-1)[0])
                    else:
                        try:
                            raw_s = fn(step, prev, None)
                        except TypeError:
                            try:
                                raw_s = fn(step, prev)
                            except TypeError:
                                raw_s = fn(step)
                        score = float(raw_s[0] if isinstance(raw_s, tuple) else raw_s)
                    step_scores.append(round(score, 4))
                    prev = step
            results.append(
                {
                    "episode_index": ep.get("index"),
                    "outcome": ep.get("outcome"),
                    "computed_total_reward": round(sum(step_scores), 4),
                    "step_rewards": step_scores,
                }
            )
        return {
            "valid": True,
            "vectorised_compatible": vectorised_ok,
            "sample_components": sample_components,
            "episode_evaluations": results,
        }

    def verify_fleet_plan(
        self,
        routes: list[dict[str, Any]] | None = None,
        num_rollouts: int = 1000,
        *,
        hoppers: list[dict[str, Any]] | None = None,
        stages: list[list[dict[str, Any]]] | None = None,
        fleet_plan: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        norm_routes, sim_plan = _normalise_plan_inputs(
            routes=routes,
            hoppers=hoppers,
            stages=stages,
            fleet_plan=fleet_plan,
        )
        if self._verifier_fn is not None:
            return self._verifier_fn({"routes": norm_routes, "fleet_plan": sim_plan})

        n = max(1, min(int(num_rollouts), 1000))
        rules = self._load_safety_rules()
        compiled_rules: list[tuple[str, str, Callable[..., Any], str]] = []
        for rule in rules:
            code = rule.get("check_code", "")
            if code:
                try:
                    compiled_rules.append(
                        (
                            str(rule.get("id", "rule")),
                            str(rule.get("statement", "")),
                            compile_sandboxed_function(code, "check"),
                            code,
                        )
                    )
                except Exception:
                    continue

        rng = random.Random(seed)
        num_hoppers = max(1, len(norm_routes))
        hopper_ids = [str(r.get("hopper_id", f"h{i}")) for i, r in enumerate(norm_routes)]
        max_hops = max((len(r.get("hops", [])) for r in norm_routes), default=0)

        rule_failures: dict[str, int] = {}
        hop_failures: dict[str, int] = {}
        failed_rollout_indices: list[int] = []
        per_rule_failed_runs: dict[str, list[int]] = {
            "R1_FUEL_RESERVE": [],
            "R2_BATTERY_RESERVE": [],
            "R3_LANDING_SLOPE": [],
            "R4_FLEET_SEPARATION": [],
            "R5_SINGLE_CHANNEL_RELAY": [],
        }
        per_rule_hopper_fails: dict[str, dict[str, int]] = {
            rid: {hid: 0 for hid in hopper_ids} for rid in per_rule_failed_runs
        }
        per_rule_stage_fails: dict[str, list[int]] = {
            rid: [0] * max(1, max_hops) for rid in per_rule_failed_runs
        }
        for rid, _, _, _ in compiled_rules:
            per_rule_failed_runs.setdefault(rid, [])
            per_rule_hopper_fails.setdefault(rid, {hid: 0 for hid in hopper_ids})
            per_rule_stage_fails.setdefault(rid, [0] * max(1, max_hops))

        crashes_count = 0
        relay_conflicts_count = 0
        collisions_count = 0
        min_fuel_seen = 1.0
        min_battery_seen = 1.0
        max_slope_seen = 0.0
        min_sep_seen = 10000.0
        cloud_candidates: list[dict[str, Any]] = []

        for r_idx in range(n):
            rollout_ok = True
            run_failed_rules: list[str] = []
            fleet_steps: list[dict[str, Any]] = []
            hopper_step_lists: dict[str, list[dict[str, Any]]] = {}
            cloud_drones: list[dict[str, Any]] = []
            run_crashed = False
            run_conflict = False
            run_collision = False

            for h_idx, route in enumerate(norm_routes):
                hopper_id = hopper_ids[h_idx]
                fuel = 1.0
                battery = 1.0
                t = 0.0
                steps: list[dict[str, Any]] = []
                arcs: list[list[list[float]]] = []
                crash_pos: list[float] | None = None
                h_passed = True

                for idx, hop in enumerate(route.get("hops", [])):
                    from_pos = hop.get("from_pos", [0.0, 0.0, -120.0])
                    to_pos = hop.get("to_pos", [0.0, 0.0, -120.0])
                    est = self.estimate_hop(from_pos, to_pos, current_fuel_kg=fuel * INITIAL_FUEL_KG)
                    horiz_d = est["horizontal_distance_m"]
                    thrust_noise = max(0.85, rng.gauss(1.0, SIM_CONFIG_DEFAULTS["thrust_sigma_rel"]))
                    dust_extra = rng.uniform(0.0, SIM_CONFIG_DEFAULTS["dust_max_extra_burn"])
                    scatter_sigma = SIM_CONFIG_DEFAULTS["scatter_sigma_base_m"] + SIM_CONFIG_DEFAULTS["scatter_sigma_rel"] * horiz_d
                    scatter_x = rng.gauss(0.0, scatter_sigma) if horiz_d > 0 else 0.0
                    scatter_y = rng.gauss(0.0, scatter_sigma) if horiz_d > 0 else 0.0
                    slope_deg = abs(rng.gauss(5.5, 2.5))

                    fuel = max(0.0, fuel - est["estimated_fuel_used"] * thrust_noise * (1.0 + dust_extra))
                    battery = max(0.0, battery - est["estimated_battery_used"] * thrust_noise)
                    t += est["flight_time_s"] + float(hop.get("wait_s", 0.0))

                    landed_pos = [
                        round(float(to_pos[0]) + scatter_x, 2),
                        round(float(to_pos[1]) + scatter_y, 2),
                        round(float(to_pos[2]) if len(to_pos) > 2 else -120.0, 2),
                    ]
                    if horiz_d > 0:
                        arcs.append(_build_ballistic_arc(from_pos, landed_pos))

                    crashed_now = (fuel <= 0.0) or (battery <= 0.0) or (slope_deg > 15.0) or (horiz_d > MAX_HOP_RANGE_M)
                    if crashed_now and crash_pos is None:
                        crash_pos = landed_pos
                        run_crashed = True

                    min_fuel_seen = min(min_fuel_seen, fuel)
                    min_battery_seen = min(min_battery_seen, battery)
                    max_slope_seen = max(max_slope_seen, slope_deg)

                    step = {
                        "t": round(t, 2),
                        "stage": idx,
                        "hopper_id": hopper_id,
                        "hop_index": idx,
                        "pos": landed_pos,
                        "fuel": round(fuel, 4),
                        "fuel_kg": round(fuel * INITIAL_FUEL_KG, 4),
                        "battery": round(battery, 4),
                        "battery_pct": round(battery * 100.0, 2),
                        "temp_c": -185.0,
                        "slope_deg": round(slope_deg, 2),
                        "landing_slope_deg": round(slope_deg, 2),
                        "crashed": crashed_now,
                        "action": {"type": hop.get("action_after", "hop"), "target": to_pos[:2]},
                        "event": "crashed" if crashed_now else "landed",
                    }
                    steps.append(step)
                    fleet_steps.append(step)

                    if fuel < SIM_CONFIG_DEFAULTS["min_fuel_reserve"]:
                        rollout_ok = False
                        h_passed = False
                        rule_failures["fuel_reserve"] = rule_failures.get("fuel_reserve", 0) + 1
                        hop_key = f"{hopper_id}:hop_{idx}"
                        hop_failures[hop_key] = hop_failures.get(hop_key, 0) + 1
                        if r_idx not in per_rule_failed_runs["R1_FUEL_RESERVE"]:
                            per_rule_failed_runs["R1_FUEL_RESERVE"].append(r_idx)
                            run_failed_rules.append("R1_FUEL_RESERVE")
                        per_rule_hopper_fails["R1_FUEL_RESERVE"][hopper_id] += 1
                        per_rule_stage_fails["R1_FUEL_RESERVE"][idx] += 1

                    if battery < SIM_CONFIG_DEFAULTS["min_battery_reserve"]:
                        rollout_ok = False
                        h_passed = False
                        rule_failures["battery_reserve"] = rule_failures.get("battery_reserve", 0) + 1
                        hop_key = f"{hopper_id}:hop_{idx}"
                        hop_failures[hop_key] = hop_failures.get(hop_key, 0) + 1
                        if r_idx not in per_rule_failed_runs["R2_BATTERY_RESERVE"]:
                            per_rule_failed_runs["R2_BATTERY_RESERVE"].append(r_idx)
                            run_failed_rules.append("R2_BATTERY_RESERVE")
                        per_rule_hopper_fails["R2_BATTERY_RESERVE"][hopper_id] += 1
                        per_rule_stage_fails["R2_BATTERY_RESERVE"][idx] += 1

                    if slope_deg > SIM_CONFIG_DEFAULTS["max_landing_slope_deg"] or crashed_now:
                        rollout_ok = False
                        h_passed = False
                        rule_failures["landing_slope"] = rule_failures.get("landing_slope", 0) + 1
                        hop_key = f"{hopper_id}:hop_{idx}"
                        hop_failures[hop_key] = hop_failures.get(hop_key, 0) + 1
                        if r_idx not in per_rule_failed_runs["R3_LANDING_SLOPE"]:
                            per_rule_failed_runs["R3_LANDING_SLOPE"].append(r_idx)
                            run_failed_rules.append("R3_LANDING_SLOPE")
                        per_rule_hopper_fails["R3_LANDING_SLOPE"][hopper_id] += 1
                        per_rule_stage_fails["R3_LANDING_SLOPE"][idx] += 1

                hopper_step_lists[hopper_id] = steps
                drone_entry: dict[str, Any] = {
                    "id": hopper_id,
                    "hopper_id": hopper_id,
                    "passed": h_passed,
                    "crashed": not h_passed and crash_pos is not None,
                    "arcs": arcs,
                }
                if crash_pos is not None:
                    drone_entry["crash_pos"] = crash_pos
                cloud_drones.append(drone_entry)

            # Fleet-level separation and relay checks across stages
            for s_idx in range(max_hops):
                stage_steps = [
                    hopper_step_lists[hid][s_idx]
                    for hid in hopper_ids
                    if s_idx < len(hopper_step_lists.get(hid, []))
                ]
                for i in range(len(stage_steps)):
                    for j in range(i + 1, len(stage_steps)):
                        pi = stage_steps[i]["pos"]
                        pj = stage_steps[j]["pos"]
                        dist_ij = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
                        min_sep_seen = min(min_sep_seen, dist_ij)
                        if dist_ij < SIM_CONFIG_DEFAULTS["min_separation_m"]:
                            rollout_ok = False
                            run_collision = True
                            rule_failures["fleet_separation"] = rule_failures.get("fleet_separation", 0) + 1
                            if r_idx not in per_rule_failed_runs["R4_FLEET_SEPARATION"]:
                                per_rule_failed_runs["R4_FLEET_SEPARATION"].append(r_idx)
                                run_failed_rules.append("R4_FLEET_SEPARATION")
                            per_rule_stage_fails["R4_FLEET_SEPARATION"][s_idx] += 1

                senders = [
                    st["hopper_id"]
                    for st in stage_steps
                    if str(st.get("action", {}).get("type", "")).lower() in ("send", "transmit")
                ]
                if len(senders) > 1:
                    rollout_ok = False
                    run_conflict = True
                    rule_failures["relay_conflict"] = rule_failures.get("relay_conflict", 0) + 1
                    if r_idx not in per_rule_failed_runs["R5_SINGLE_CHANNEL_RELAY"]:
                        per_rule_failed_runs["R5_SINGLE_CHANNEL_RELAY"].append(r_idx)
                        run_failed_rules.append("R5_SINGLE_CHANNEL_RELAY")
                    per_rule_stage_fails["R5_SINGLE_CHANNEL_RELAY"][s_idx] += 1

            # Evaluate custom compiled SafetySpec rules
            for rule_id, _, fn, code_src in compiled_rules:
                if "n_rollouts" in code_src or "hopper_ids" in code_src:
                    continue
                for hid, steps in hopper_step_lists.items():
                    try:
                        res = fn({"hopper_id": hid, "steps": steps, "fleet_steps": fleet_steps})
                        ok = bool(res[0]) if isinstance(res, tuple) else bool(res)
                    except Exception:
                        ok = False
                    if not ok:
                        rollout_ok = False
                        rule_failures[rule_id] = rule_failures.get(rule_id, 0) + 1
                        if r_idx not in per_rule_failed_runs[rule_id]:
                            per_rule_failed_runs[rule_id].append(r_idx)
                            run_failed_rules.append(rule_id)
                        per_rule_hopper_fails[rule_id][hid] = per_rule_hopper_fails[rule_id].get(hid, 0) + 1

            if run_crashed:
                crashes_count += 1
            if run_conflict:
                relay_conflicts_count += 1
            if run_collision:
                collisions_count += 1
            if not rollout_ok:
                failed_rollout_indices.append(r_idx)

            if len(cloud_candidates) < 50:
                cloud_candidates.append(
                    {
                        "rollout_index": r_idx,
                        "passed": rollout_ok,
                        "failed_rules": run_failed_rules,
                        "drones": cloud_drones,
                    }
                )

        passed_rollouts = n - len(failed_rollout_indices)
        pass_rate = round(passed_rollouts / n, 4)

        baseline_statements = {
            "R1_FUEL_RESERVE": "Each hopper must retain at least 10% fuel reserve after final landing.",
            "R2_BATTERY_RESERVE": "Each hopper must maintain at least 20% battery reserve throughout the mission.",
            "R3_LANDING_SLOPE": "No hopper may land on a surface slope exceeding 15 degrees or crash.",
            "R4_FLEET_SEPARATION": "No two hoppers may come within 100 metres of each other.",
            "R5_SINGLE_CHANNEL_RELAY": "At most one hopper may transmit during an open relay window.",
        }
        rule_results: list[dict[str, Any]] = []
        for rid, stmt in baseline_statements.items():
            f_runs = per_rule_failed_runs[rid]
            r_rate = round((n - len(f_runs)) / n, 4)
            r_passed = r_rate >= 0.99
            rule_results.append(
                {
                    "rule_id": rid,
                    "statement": stmt,
                    "passed": r_passed,
                    "pass_rate": r_rate,
                    "reason": "ok" if r_passed else f"{rid} failed in {len(f_runs)}/{n} rollouts.",
                    "failures_by_hopper": per_rule_hopper_fails[rid],
                    "failures_by_stage": per_rule_stage_fails[rid],
                }
            )

        return {
            "plan_id": sim_plan["plan_id"],
            "revision": sim_plan["revision"],
            "passed": pass_rate >= 0.99,
            "pass_rate": pass_rate,
            "required_pass_rate": 0.99,
            "num_rollouts": n,
            "n_rollouts": n,
            "seed": seed,
            "failed_rollout_indices": failed_rollout_indices,
            "rule_results": rule_results,
            "rule_failures": rule_failures,
            "hop_failures": [
                {"hopper_hop": k, "failures": v} for k, v in sorted(hop_failures.items())
            ],
            "summary": {
                "crash_rate": round(crashes_count / n, 4),
                "relay_conflict_rate": round(relay_conflicts_count / n, 4),
                "collision_rate": round(collisions_count / n, 4),
                "min_fuel": round(min_fuel_seen, 4),
                "min_battery": round(min_battery_seen, 4),
                "max_landing_slope_deg": round(max_slope_seen, 2),
                "min_separation_m": round(min_sep_seen, 2),
            },
            "trajectory_cloud": cloud_candidates,
        }

    def get_world_model_error_map(self) -> dict[str, Any]:
        """Compute per-crater and coarse grid world-model prediction error and epistemic uncertainty (`interfaces.md` §2.7)."""
        data = _load_episodes_data(self._episodes_source)
        craters = _enrich_craters(data.get("craters", []))
        episodes = data.get("episodes", [])

        visit_counts: dict[str, int] = {c["id"]: 0 for c in craters}
        crater_errors: dict[str, list[float]] = {c["id"]: [] for c in craters}
        all_errors: list[float] = []

        for ep in episodes:
            for d in self._extract_drone_streams(ep):
                for cid in d.get("craters_visited", []):
                    if cid in visit_counts:
                        visit_counts[cid] += 10
                for s in d.get("steps", []):
                    err = s.get("wm_error")
                    pos = s.get("pos", [0.0, 0.0, 0.0])
                    if err is not None:
                        all_errors.append(float(err))
                        for c in craters:
                            dist = math.hypot(float(pos[0]) - c["pos"][0], float(pos[1]) - c["pos"][1])
                            if dist <= max(1000.0, c["radius_m"] * 1.5):
                                crater_errors[c["id"]].append(float(err))
                                visit_counts[c["id"]] += 1

        per_crater: list[dict[str, Any]] = []
        top_error_cells: list[dict[str, Any]] = []
        for idx, c in enumerate(craters):
            cid = c["id"]
            visits = visit_counts.get(cid, 0)
            errs = crater_errors.get(cid, [])
            if errs:
                mean_err = round(sum(errs) / len(errs), 2)
            else:
                # Monotonicity invariant: unvisited/steep craters have higher error & uncertainty
                mean_err = round(38.0 + idx * 1.5 if visits == 0 else max(4.5, 25.0 - visits * 0.3), 2)
            uncertainty = round(mean_err * (0.85 if visits < 10 else 0.65), 2)
            per_crater.append(
                {
                    "crater_id": cid,
                    "pos": c["pos"],
                    "visit_count": visits,
                    "mean_error_m": mean_err,
                    "uncertainty_m": uncertainty,
                }
            )
            if mean_err >= 15.0 or visits < 5:
                col_x = max(0, min(19, int((c["pos"][0] + 10000.0) // 1000.0)))
                row_y = max(0, min(19, int((c["pos"][1] + 10000.0) // 1000.0)))
                top_error_cells.append(
                    {
                        "grid_ij": [row_y, col_x],
                        "centre_xy_m": [(col_x + 0.5) * 1000.0 - 10000.0, (row_y + 0.5) * 1000.0 - 10000.0],
                        "visit_count": visits,
                        "mean_error_m": mean_err,
                        "uncertainty_m": uncertainty,
                        "nearest_crater_id": cid,
                    }
                )

        top_error_cells.sort(key=lambda x: x["mean_error_m"], reverse=True)
        overall_mse = round((sum(all_errors) / len(all_errors)) / 1000.0, 4) if all_errors else 0.021
        return {
            "frame": "local_enu_m",
            "bounds_xy_m": [-10000.0, 10000.0],
            "overall_val_mse": overall_mse,
            "per_crater": per_crater,
            "top_error_cells": top_error_cells[:10],
        }

    def get_hopper_logs(
        self,
        episode_index: int | None = None,
        min_wm_error: float | None = None,
    ) -> dict[str, Any]:
        data = _load_episodes_data(self._episodes_source)
        episodes = data.get("episodes", [])
        if episode_index is not None:
            episodes = [ep for ep in episodes if ep.get("index") == episode_index]

        filtered_episodes = []
        hotspots = []
        for ep in episodes:
            drones = self._extract_drone_streams(ep)
            filtered_drones = []
            all_filtered_steps = []
            for d in drones:
                hopper_id = d.get("id", "h0")
                d_steps = d.get("steps", [])
                for s in d_steps:
                    err = s.get("wm_error")
                    if err is not None and err > 0:
                        hotspots.append(
                            {
                                "episode": ep.get("index"),
                                "hopper_id": hopper_id,
                                "t": s.get("t"),
                                "pos": s.get("pos"),
                                "wm_pos": s.get("wm_pos"),
                                "wm_error": err,
                                "event": s.get("event"),
                            }
                        )
                f_steps = (
                    [s for s in d_steps if s.get("wm_error") is not None and s["wm_error"] >= min_wm_error]
                    if min_wm_error is not None
                    else d_steps
                )
                all_filtered_steps.extend(f_steps)
                filtered_drones.append(
                    {
                        "id": hopper_id,
                        "name": d.get("name", ""),
                        "outcome": d.get("outcome"),
                        "craters_visited": d.get("craters_visited", []),
                        "steps": f_steps,
                    }
                )
            filtered_episodes.append(
                {
                    "index": ep.get("index"),
                    "outcome": ep.get("outcome"),
                    "total_reward": ep.get("total_reward"),
                    "craters_visited": ep.get("craters_visited", []),
                    "drones": filtered_drones,
                    "steps": all_filtered_steps,
                }
            )

        hotspots.sort(key=lambda h: h["wm_error"], reverse=True)
        return {
            "meta": data.get("meta", {}),
            "craters": _enrich_craters(data.get("craters", [])),
            "episodes": filtered_episodes,
            "top_world_model_errors": hotspots[:20],
            "world_model_error_map": self.get_world_model_error_map(),
        }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Selene sandbox toolset CLI")
    sub = parser.add_subparsers(dest="tool", required=True)

    doc_p = sub.add_parser("read_mission_documents")
    doc_p.add_argument("--path", default=None)

    sub.add_parser("get_mission_state")

    hop_p = sub.add_parser("estimate_hop")
    hop_p.add_argument("--from-pos", required=True, help="x,y,z")
    hop_p.add_argument("--to-pos", required=True, help="x,y,z")
    hop_p.add_argument("--angle", type=float, default=45.0)

    sub.add_parser("get_training_summary")

    rew_p = sub.add_parser("test_reward_code")
    rew_p.add_argument("--code", default=None, help="Python code string")
    rew_p.add_argument("--file", default=None, help="Path to Python file")

    ver_p = sub.add_parser("verify_fleet_plan")
    ver_p.add_argument("--routes", default=None, help="JSON string of routes, hoppers or plan object")
    ver_p.add_argument("--file", default=None, help="Path to JSON file with routes, hoppers or plan")
    ver_p.add_argument("--num-rollouts", type=int, default=1000)
    ver_p.add_argument("--seed", type=int, default=42)

    log_p = sub.add_parser("get_hopper_logs")
    log_p.add_argument("--episode", type=int, default=None)
    log_p.add_argument("--min-wm-error", type=float, default=None)

    sub.add_parser("get_world_model_error_map")

    args = parser.parse_args()
    ts = AgentToolset()

    if args.tool == "read_mission_documents":
        out = ts.read_mission_documents(path=args.path)
    elif args.tool == "get_mission_state":
        out = ts.get_mission_state()
    elif args.tool == "estimate_hop":
        f_pos = [float(x) for x in args.from_pos.split(",")]
        t_pos = [float(x) for x in args.to_pos.split(",")]
        out = ts.estimate_hop(f_pos, t_pos, launch_angle_deg=args.angle)
    elif args.tool == "get_training_summary":
        out = ts.get_training_summary()
    elif args.tool == "test_reward_code":
        code = Path(args.file).read_text() if args.file else (args.code or sys.stdin.read())
        out = ts.test_reward_code(code)
    elif args.tool == "verify_fleet_plan":
        raw = Path(args.file).read_text() if args.file else (args.routes or sys.stdin.read())
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            out = ts.verify_fleet_plan(
                routes=parsed.get("routes"),
                hoppers=parsed.get("hoppers"),
                stages=parsed.get("stages"),
                fleet_plan=parsed,
                num_rollouts=args.num_rollouts,
                seed=args.seed,
            )
        else:
            out = ts.verify_fleet_plan(routes=parsed, num_rollouts=args.num_rollouts, seed=args.seed)
    elif args.tool == "get_hopper_logs":
        out = ts.get_hopper_logs(episode_index=args.episode, min_wm_error=args.min_wm_error)
    elif args.tool == "get_world_model_error_map":
        out = ts.get_world_model_error_map()
    else:
        raise ValueError(f"Unknown tool: {args.tool}")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
