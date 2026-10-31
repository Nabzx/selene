from __future__ import annotations

import ast
import math
import typing
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]

SourceType = Literal["live", "cached", "scripted", "trained"]
INITIAL_FUEL_KG = 15.0


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


def _validate_python_function(code: str, expected_name: str) -> str:
    if not code or not code.strip():
        return ""
    tree = ast.parse(code)
    fn_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if expected_name not in fn_names:
        raise ValueError(f"Expected function 'def {expected_name}(...)' in code, found {fn_names}")
    return code


def _load_function(code: str, fn_name: str):
    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "Any": Any,
    }
    if _np is not None:
        namespace["np"] = _np
        namespace["numpy"] = _np
    exec(compile(code, f"<{fn_name}>", "exec"), namespace)
    return namespace[fn_name]


def normalise_trace(trace: Any) -> dict[str, Any]:
    """Normalise a RolloutTraces dataclass, interfaces.md trace dict, or single-rollout dict into a unified dict."""
    if isinstance(trace, dict) and "_normalised_trace" in trace:
        return trace

    raw: dict[str, Any] = {}
    if isinstance(trace, dict):
        raw.update(trace)
    else:
        for attr in (
            "positions_xyz",
            "pos",
            "fuel_kg",
            "fuel",
            "battery_pct",
            "battery",
            "landing_slope_deg",
            "hop_durations_s",
            "hop_time_s",
            "min_separation_m",
            "crashed",
            "crash_reason",
            "relay_conflict",
            "in_relay_window",
            "data_downlinked",
            "hopper_ids",
            "n_rollouts",
        ):
            if hasattr(trace, attr):
                raw[attr] = getattr(trace, attr)

    # Case 1: Batch trace from selene_sim.verify.simulate_plan or interfaces.md
    if any(k in raw for k in ("positions_xyz", "pos", "fuel_kg", "fuel")) and not (
        "steps" in raw and "fuel" not in raw and "fuel_kg" not in raw
    ):
        pos = raw.get("pos") if "pos" in raw else raw.get("positions_xyz")
        fuel_kg = raw.get("fuel_kg")
        fuel_frac = raw.get("fuel")
        if fuel_frac is None and fuel_kg is not None:
            fuel_frac = (
                fuel_kg / INITIAL_FUEL_KG
                if _np is not None and isinstance(fuel_kg, _np.ndarray)
                else [[ [ float(v) / INITIAL_FUEL_KG for v in stage ] for stage in run ] for run in fuel_kg]
            )
        elif fuel_kg is None and fuel_frac is not None:
            fuel_kg = (
                fuel_frac * INITIAL_FUEL_KG
                if _np is not None and isinstance(fuel_frac, _np.ndarray)
                else [[ [ float(v) * INITIAL_FUEL_KG for v in stage ] for stage in run ] for run in fuel_frac]
            )

        battery_pct = raw.get("battery_pct")
        battery_frac = raw.get("battery")
        if battery_frac is None and battery_pct is not None:
            battery_frac = (
                battery_pct / 100.0
                if _np is not None and isinstance(battery_pct, _np.ndarray)
                else [[ [ float(v) / 100.0 for v in stage ] for stage in run ] for run in battery_pct]
            )
        elif battery_pct is None and battery_frac is not None:
            battery_pct = (
                battery_frac * 100.0
                if _np is not None and isinstance(battery_frac, _np.ndarray)
                else [[ [ float(v) * 100.0 for v in stage ] for stage in run ] for run in battery_frac]
            )

        hop_time = raw.get("hop_time_s") if "hop_time_s" in raw else raw.get("hop_durations_s")
        slope_deg = raw.get("landing_slope_deg")
        crashed = raw.get("crashed")
        min_sep = raw.get("min_separation_m")
        relay_conflict = raw.get("relay_conflict")

        n_rollouts = int(raw.get("n_rollouts") or raw.get("num_runs") or (len(fuel_frac) if fuel_frac is not None else 1))
        n_hoppers = 1
        n_stages = 0
        if fuel_frac is not None and len(fuel_frac) > 0:
            n_stages = max(0, len(fuel_frac[0]) - 1)
            if len(fuel_frac[0]) > 0:
                n_hoppers = len(fuel_frac[0][0])

        hopper_ids = list(raw.get("hopper_ids") or [f"h{i}" for i in range(n_hoppers)])

        out = dict(raw)
        out.update(
            {
                "_normalised_trace": True,
                "_is_batch": True,
                "n_rollouts": n_rollouts,
                "num_runs": n_rollouts,
                "num_stages": n_stages,
                "num_hoppers": n_hoppers,
                "hopper_ids": hopper_ids,
                "pos": pos,
                "positions_xyz": pos,
                "fuel": fuel_frac,
                "fuel_kg": fuel_kg,
                "battery": battery_frac,
                "battery_pct": battery_pct,
                "landing_slope_deg": slope_deg,
                "hop_time_s": hop_time,
                "hop_durations_s": hop_time,
                "min_separation_m": min_sep,
                "crashed": crashed,
                "relay_conflict": relay_conflict,
            }
        )
        return out

    # Case 2: Single-rollout dict with `steps` and/or `fleet_steps`
    steps = list(raw.get("steps") or [])
    fleet_steps = list(raw.get("fleet_steps") or steps)
    hopper_id = str(raw.get("hopper_id") or "h0")
    out = dict(raw)
    out["_normalised_trace"] = True
    out["_is_batch"] = False
    out["steps"] = steps
    out["fleet_steps"] = fleet_steps
    out["hopper_id"] = hopper_id
    out["n_rollouts"] = 1
    out["hopper_ids"] = [hopper_id]

    if _np is not None and steps:
        s_len = len(steps)
        f_arr = _np.array([float(s.get("fuel", 1.0)) for s in steps], dtype=_np.float32).reshape(1, s_len, 1)
        b_arr = _np.array([float(s.get("battery", 1.0)) for s in steps], dtype=_np.float32).reshape(1, s_len, 1)
        sl_arr = _np.array([float(s.get("slope_deg", s.get("landing_slope_deg", 0.0))) for s in steps], dtype=_np.float32).reshape(1, s_len, 1)
        cr_arr = _np.array([str(s.get("event", "")) in ("crash", "crashed", "tip_over") or bool(s.get("crashed", False)) for s in steps], dtype=bool).reshape(1, s_len, 1)
        p_arr = _np.array([list(s.get("pos", [0.0, 0.0, 0.0]))[:3] for s in steps], dtype=_np.float32).reshape(1, s_len, 1, 3)
        sep_arr = _np.array([float(s.get("min_separation_m", 250.0)) for s in steps], dtype=_np.float32).reshape(1, s_len)
        rc_arr = _np.array([bool(s.get("relay_conflict", False)) for s in steps], dtype=bool).reshape(1, s_len)
        out.setdefault("fuel", f_arr)
        out.setdefault("fuel_kg", f_arr * INITIAL_FUEL_KG)
        out.setdefault("battery", b_arr)
        out.setdefault("battery_pct", b_arr * 100.0)
        out.setdefault("landing_slope_deg", sl_arr)
        out.setdefault("crashed", cr_arr)
        out.setdefault("pos", p_arr)
        out.setdefault("positions_xyz", p_arr)
        out.setdefault("min_separation_m", sep_arr)
        out.setdefault("relay_conflict", rc_arr)

    return out


def _single_rollout_from_batch(norm_trace: dict[str, Any], run_idx: int) -> dict[str, Any]:
    """Extract a single-rollout step dict from a batch trace for single-rollout `check(rollout)` rules."""
    n_stages = int(norm_trace.get("num_stages", 0))
    n_hoppers = int(norm_trace.get("num_hoppers", 1))
    hopper_ids = norm_trace.get("hopper_ids", [f"h{i}" for i in range(n_hoppers)])
    fuel = norm_trace.get("fuel")
    battery = norm_trace.get("battery")
    pos = norm_trace.get("pos")
    slope = norm_trace.get("landing_slope_deg")
    crashed = norm_trace.get("crashed")
    min_sep = norm_trace.get("min_separation_m")
    relay_conflict = norm_trace.get("relay_conflict")
    hop_time = norm_trace.get("hop_time_s")

    fleet_steps: list[dict[str, Any]] = []
    hopper_steps_map: dict[str, list[dict[str, Any]]] = {hid: [] for hid in hopper_ids}
    t_cum = 0.0
    for s in range(n_stages):
        dt = float(hop_time[run_idx][s][0]) if hop_time is not None else 10.0
        t_cum += dt
        for h_idx, hid in enumerate(hopper_ids):
            f_val = float(fuel[run_idx][s + 1][h_idx]) if fuel is not None else 1.0
            b_val = float(battery[run_idx][s + 1][h_idx]) if battery is not None else 1.0
            p_val = [float(x) for x in pos[run_idx][s + 1][h_idx]] if pos is not None else [0.0, 0.0, 0.0]
            sl_val = float(slope[run_idx][s][h_idx]) if slope is not None else 0.0
            cr_val = bool(crashed[run_idx][s][h_idx]) if crashed is not None else False
            sep_val = float(min_sep[run_idx][s]) if min_sep is not None else 250.0
            rc_val = bool(relay_conflict[run_idx][s]) if relay_conflict is not None else False
            step_dict = {
                "t": round(t_cum, 2),
                "stage": s,
                "hop_index": s,
                "hopper_id": hid,
                "pos": p_val,
                "fuel": f_val,
                "fuel_kg": f_val * INITIAL_FUEL_KG,
                "battery": b_val,
                "battery_pct": b_val * 100.0,
                "slope_deg": sl_val,
                "landing_slope_deg": sl_val,
                "crashed": cr_val,
                "min_separation_m": sep_val,
                "relay_conflict": rc_val,
                "event": "crashed" if cr_val else "landed",
            }
            hopper_steps_map[hid].append(step_dict)
            fleet_steps.append(step_dict)

    return {
        "hopper_id": hopper_ids[0] if hopper_ids else "h0",
        "steps": fleet_steps,
        "fleet_steps": fleet_steps,
        "hopper_steps": hopper_steps_map,
    }


class RuleResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_id: str
    statement: str = ""
    passed: bool
    pass_rate: float
    reason: str = "ok"
    failed_rollouts: list[int] = Field(default_factory=list)
    failures_by_hopper: dict[str, int] = Field(default_factory=dict)
    failures_by_stage: list[int] = Field(default_factory=list)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="allow")

    plan_id: str = "plan_v1"
    revision: int = 0
    passed: bool
    pass_rate: float
    required_pass_rate: float = 0.99
    approved_by_human: bool = False
    failed_rollout_indices: list[int] = Field(default_factory=list)
    rule_results: list[RuleResult] = Field(default_factory=list)


class SafetyRule(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str = ""
    scope: Literal["step", "hop", "route", "fleet"] = "hop"
    severity: str = "hard"
    statement: str
    source_excerpt: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    check_code: str = ""

    @field_validator("check_code")
    @classmethod
    def validate_check_code(cls, value: str) -> str:
        return _validate_python_function(value, "check")

    def evaluate_trace(
        self,
        trace: Any,
        *,
        required_pass_rate: float = 0.99,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Evaluate this rule across a batch trace or single rollout, returning (passed, reason, stats)."""
        if not self.check_code:
            raise ValueError(f"Rule {self.id} has no check_code")
        fn = _load_function(self.check_code, "check")
        norm = normalise_trace(trace)
        n_rollouts = int(norm.get("n_rollouts", 1))
        hopper_ids = list(norm.get("hopper_ids", ["h0"]))
        n_stages = int(norm.get("num_stages", 0))

        uses_single_steps = ("steps" in self.check_code or "fleet_steps" in self.check_code) and (
            "n_rollouts" not in self.check_code and "hopper_ids" not in self.check_code
        )

        if norm.get("_is_batch") and uses_single_steps:
            failed_rollouts: list[int] = []
            for r in range(n_rollouts):
                single_r = _single_rollout_from_batch(norm, r)
                try:
                    res = fn(single_r)
                    ok = bool(res[0]) if isinstance(res, tuple) else bool(res)
                except Exception:
                    ok = False
                if not ok:
                    failed_rollouts.append(r)
            pass_rate = float((n_rollouts - len(failed_rollouts)) / max(1, n_rollouts))
            passed = pass_rate >= required_pass_rate
            reason = (
                "ok"
                if passed
                else f"Rule {self.id} failed in {len(failed_rollouts)}/{n_rollouts} rollouts."
            )
            stats = {
                "rule_id": self.id,
                "pass_rate": round(pass_rate, 4),
                "failed_rollouts": failed_rollouts,
                "failures_by_hopper": {hid: 0 for hid in hopper_ids},
                "failures_by_stage": [0] * n_stages,
            }
            return passed, reason, stats

        raw_out = fn(norm)
        if isinstance(raw_out, tuple) and len(raw_out) == 3:
            passed_val, reason_val, stats_val = raw_out
            stats_dict = dict(stats_val) if isinstance(stats_val, dict) else {}
            pass_rate = float(stats_dict.get("pass_rate", 1.0 if passed_val else 0.0))
            passed = bool(passed_val) and (pass_rate >= required_pass_rate)
            stats_dict.setdefault("rule_id", self.id)
            stats_dict["pass_rate"] = round(pass_rate, 4)
            stats_dict.setdefault("failed_rollouts", [])
            stats_dict.setdefault("failures_by_hopper", {hid: 0 for hid in hopper_ids})
            stats_dict.setdefault("failures_by_stage", [0] * n_stages)
            return passed, str(reason_val), stats_dict

        if _np is not None and isinstance(raw_out, _np.ndarray) and raw_out.ndim >= 1:
            run_ok = raw_out.astype(bool).reshape(n_rollouts, -1).all(axis=1)
            failed_rollouts = _np.flatnonzero(~run_ok).astype(int).tolist()
            pass_rate = float(_np.mean(run_ok))
            passed = pass_rate >= required_pass_rate
            reason = "ok" if passed else f"Rule {self.id} failed in {len(failed_rollouts)}/{n_rollouts} rollouts."
            return passed, reason, {
                "rule_id": self.id,
                "pass_rate": round(pass_rate, 4),
                "failed_rollouts": failed_rollouts,
                "failures_by_hopper": {hid: 0 for hid in hopper_ids},
                "failures_by_stage": [0] * n_stages,
            }

        ok_bool = bool(raw_out)
        pass_rate = 1.0 if ok_bool else 0.0
        failed = [] if ok_bool else list(range(n_rollouts))
        reason = "ok" if ok_bool else f"Rule {self.id} violated."
        return ok_bool, reason, {
            "rule_id": self.id,
            "pass_rate": pass_rate,
            "failed_rollouts": failed,
            "failures_by_hopper": {hid: (0 if ok_bool else 1) for hid in hopper_ids},
            "failures_by_stage": [0] * n_stages,
        }

    def check(self, rollout: Any) -> bool:
        """Return True if `rollout` (or batch `trace`) satisfies this safety rule."""
        passed, _, _ = self.evaluate_trace(rollout)
        return bool(passed)


class SafetySpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    summary: str = ""
    rules: list[SafetyRule] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    source: SourceType = "live"
    interaction_id: str | None = None
    environment_id: str | None = None

    def verify_rollout(self, rollout: Any) -> dict[str, bool]:
        return {rule.id: rule.check(rollout) for rule in self.rules if rule.check_code}

    def evaluate_traces(
        self,
        trace: Any,
        *,
        plan_id: str = "plan_v1",
        revision: int = 0,
        required_pass_rate: float = 0.99,
    ) -> Verdict:
        """Evaluate all safety rules across a rollout trace batch and return a structured Verdict."""
        norm = normalise_trace(trace)
        n_rollouts = int(norm.get("n_rollouts", 1))
        all_failed_set: set[int] = set()
        rule_results: list[RuleResult] = []

        for rule in self.rules:
            if not rule.check_code:
                continue
            passed, reason, stats = rule.evaluate_trace(norm, required_pass_rate=required_pass_rate)
            failed_indices = [int(i) for i in stats.get("failed_rollouts", [])]
            all_failed_set.update(failed_indices)
            rule_results.append(
                RuleResult(
                    rule_id=rule.id,
                    statement=rule.statement,
                    passed=passed,
                    pass_rate=float(stats.get("pass_rate", 1.0 if passed else 0.0)),
                    reason=reason,
                    failed_rollouts=failed_indices,
                    failures_by_hopper=dict(stats.get("failures_by_hopper", {})),
                    failures_by_stage=list(stats.get("failures_by_stage", [])),
                )
            )

        failed_sorted = sorted(all_failed_set)
        joint_pass_rate = round(float((n_rollouts - len(failed_sorted)) / max(1, n_rollouts)), 4)
        overall_passed = (joint_pass_rate >= required_pass_rate) and all(r.passed for r in rule_results)
        return Verdict(
            plan_id=plan_id,
            revision=revision,
            passed=overall_passed,
            pass_rate=joint_pass_rate,
            required_pass_rate=required_pass_rate,
            approved_by_human=False,
            failed_rollout_indices=failed_sorted,
            rule_results=rule_results,
        )


class RewardTerm(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    sign: Literal["+", "-"] = "+"
    weight: float = 1.0
    expression: str = ""
    description: str = ""


class RewardSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    episode: int = 1
    summary: str
    diagnosis: str = ""
    terms: list[RewardTerm] = Field(default_factory=list)
    reward_code: str = ""
    missing_data: list[str] = Field(default_factory=list)
    source: SourceType = "live"
    interaction_id: str | None = None
    environment_id: str | None = None

    @field_validator("reward_code")
    @classmethod
    def validate_reward_code(cls, value: str) -> str:
        return _validate_python_function(value, "compute_reward")

    def evaluate(
        self,
        state_or_step: Any,
        actions_or_prev: Any = None,
        next_state_or_fleet: Any = None,
    ) -> Any:
        """Evaluate `compute_reward` on either vectorised (state, actions, next_state) or step dicts."""
        if not self.reward_code:
            raise ValueError("RewardSpec has no reward_code")
        fn = _load_function(self.reward_code, "compute_reward")
        res = fn(state_or_step, actions_or_prev, next_state_or_fleet)
        if isinstance(res, tuple) and len(res) == 2:
            if isinstance(state_or_step, dict):
                rewards_val, _ = res
                if _np is not None and isinstance(rewards_val, _np.ndarray):
                    return float(_np.sum(rewards_val))
                if isinstance(rewards_val, (list, tuple)):
                    return float(sum(rewards_val))
                return float(rewards_val)
            return res
        return float(res)

    def to_reward_version(self) -> dict[str, Any]:
        return {"episode": self.episode, "summary": self.summary}


class StageStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    stage: int = 0
    action: str = "hop"  # "hop" | "scan" | "send" | "wait"
    target_crater: str | None = None
    target_xy: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    from_xy: list[float] | None = None
    to_xy: list[float] | None = None
    wait_s: float = 0.0

    @model_validator(mode="after")
    def _sync_xy(self) -> StageStep:
        if self.to_xy and len(self.to_xy) >= 2 and self.target_xy == [0.0, 0.0]:
            self.target_xy = [float(self.to_xy[0]), float(self.to_xy[1])]
        elif self.to_xy is None and len(self.target_xy) >= 2:
            self.to_xy = [float(self.target_xy[0]), float(self.target_xy[1])]
        return self


HopStep = StageStep


class HopperPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    start_pos_xy: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    craters_assigned: list[str] = Field(default_factory=list)
    steps: list[StageStep] = Field(default_factory=list)


HopperPlanEntry = HopperPlan


class PlannedHop(BaseModel):
    model_config = ConfigDict(extra="allow")

    hop_index: int = 0
    from_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, -120.0])
    to_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, -120.0])
    target_crater: str | None = None
    t_start: float = 0.0
    action_after: str = "scan"
    wait_s: float = 0.0
    relay_window_index: int | None = None
    estimated_fuel_used: float | None = None
    estimated_battery_used: float | None = None
    notes: str = ""


class HopperRoute(BaseModel):
    model_config = ConfigDict(extra="allow")

    hopper_id: str
    craters: list[str] = Field(default_factory=list)
    hops: list[PlannedHop] = Field(default_factory=list)


class FleetPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    plan_id: str = "plan_v1"
    revision: int = 0
    rationale: str = ""
    previous_verdict_pass_rate: float | None = None
    summary: str = ""
    hoppers: list[HopperPlan] = Field(default_factory=list)
    num_hoppers: int = 0
    start_positions_xy: list[list[float]] = Field(default_factory=list)
    stages: list[list[dict[str, Any]]] = Field(default_factory=list)
    routes: list[HopperRoute] = Field(default_factory=list)
    unassigned_craters: list[str] = Field(default_factory=list)
    revision_summary: str = ""
    changes: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    source: SourceType = "live"
    interaction_id: str | None = None
    environment_id: str | None = None

    @model_validator(mode="after")
    def _sync_representations(self) -> FleetPlan:
        if not self.summary and self.rationale:
            self.summary = self.rationale
        elif not self.rationale and self.summary:
            self.rationale = self.summary
        if self.revision_summary and self.rationale == self.summary:
            self.rationale = self.revision_summary

        # 1. If `hoppers` (interfaces.md) is missing but `routes` is present, build `hoppers` from `routes`
        if not self.hoppers and self.routes:
            built_hoppers: list[HopperPlan] = []
            for r_idx, route in enumerate(self.routes):
                first_from = route.hops[0].from_pos[:2] if route.hops else [0.0, 0.0]
                stage_steps: list[StageStep] = []
                s_counter = 0
                for hop in route.hops:
                    txy = [float(hop.to_pos[0]), float(hop.to_pos[1])] if len(hop.to_pos) >= 2 else [0.0, 0.0]
                    act_raw = (hop.action_after or "hop").lower().strip()
                    if act_raw in ("hop", "none", ""):
                        stage_steps.append(
                            StageStep(
                                stage=s_counter,
                                action="hop",
                                target_crater=hop.target_crater,
                                target_xy=txy,
                                wait_s=hop.wait_s,
                            )
                        )
                        s_counter += 1
                    else:
                        stage_steps.append(
                            StageStep(
                                stage=s_counter,
                                action="hop",
                                target_crater=hop.target_crater,
                                target_xy=txy,
                                wait_s=0.0,
                            )
                        )
                        s_counter += 1
                        for sub_act in [a.strip() for a in act_raw.replace("+", ",").replace("_and_", ",").split(",") if a.strip()]:
                            norm_act = "send" if sub_act in ("transmit", "send", "downlink") else ("scan" if "scan" in sub_act else "wait")
                            stage_steps.append(
                                StageStep(
                                    stage=s_counter,
                                    action=norm_act,
                                    target_crater=hop.target_crater,
                                    target_xy=txy,
                                    wait_s=hop.wait_s,
                                )
                            )
                            s_counter += 1
                built_hoppers.append(
                    HopperPlan(
                        id=route.hopper_id or f"h{r_idx}",
                        start_pos_xy=[float(first_from[0]), float(first_from[1])],
                        craters_assigned=list(route.craters),
                        steps=stage_steps,
                    )
                )
            self.hoppers = built_hoppers

        # 2. If `hoppers` is missing but `stages` (selene_sim.verify.FleetPlan) is present, build `hoppers`
        if not self.hoppers and self.stages:
            n_h = self.num_hoppers or (len(self.start_positions_xy) if self.start_positions_xy else len(self.stages[0]))
            starts = self.start_positions_xy or [[0.0, 0.0] for _ in range(n_h)]
            built_hoppers = []
            for h_idx in range(n_h):
                h_steps: list[StageStep] = []
                craters_seen: list[str] = []
                for s_idx, stage_cmds in enumerate(self.stages):
                    cmd = stage_cmds[h_idx] if h_idx < len(stage_cmds) else {}
                    tc = cmd.get("target_crater_id") or cmd.get("target_crater")
                    if tc and tc not in craters_seen:
                        craters_seen.append(str(tc))
                    txy = cmd.get("target_xy", starts[h_idx] if h_idx < len(starts) else [0.0, 0.0])
                    h_steps.append(
                        StageStep(
                            stage=s_idx,
                            action=str(cmd.get("action", "wait")),
                            target_crater=tc,
                            target_xy=[float(txy[0]), float(txy[1])],
                            wait_s=float(cmd.get("wait_s", 0.0)),
                        )
                    )
                built_hoppers.append(
                    HopperPlan(
                        id=f"h{h_idx}",
                        start_pos_xy=[float(starts[h_idx][0]), float(starts[h_idx][1])] if h_idx < len(starts) else [0.0, 0.0],
                        craters_assigned=craters_seen,
                        steps=h_steps,
                    )
                )
            self.hoppers = built_hoppers

        # 3. Ensure `routes` is populated from `hoppers` if `routes` was omitted
        if not self.routes and self.hoppers:
            built_routes: list[HopperRoute] = []
            for h in self.hoppers:
                cur_xy = list(h.start_pos_xy[:2]) + [-120.0]
                hops_list: list[PlannedHop] = []
                for idx, st in enumerate(h.steps):
                    next_xyz = [float(st.target_xy[0]), float(st.target_xy[1]), -120.0]
                    hops_list.append(
                        PlannedHop(
                            hop_index=idx,
                            from_pos=list(cur_xy),
                            to_pos=next_xyz,
                            target_crater=st.target_crater,
                            action_after=st.action,
                            wait_s=st.wait_s,
                        )
                    )
                    if st.action == "hop":
                        cur_xy = next_xyz
                built_routes.append(
                    HopperRoute(
                        hopper_id=h.id,
                        craters=list(h.craters_assigned),
                        hops=hops_list,
                    )
                )
            self.routes = built_routes

        # 4. Ensure `num_hoppers`, `start_positions_xy`, and `stages` are populated from `hoppers`
        if self.hoppers and not self.stages:
            self.num_hoppers = len(self.hoppers)
            self.start_positions_xy = [[float(h.start_pos_xy[0]), float(h.start_pos_xy[1])] for h in self.hoppers]
            max_stages = max((len(h.steps) for h in self.hoppers), default=0)
            stages_out: list[list[dict[str, Any]]] = []
            last_xy = [list(xy) for xy in self.start_positions_xy]
            for s_idx in range(max_stages):
                stage_row: list[dict[str, Any]] = []
                for h_idx, h in enumerate(self.hoppers):
                    if s_idx < len(h.steps):
                        st = h.steps[s_idx]
                        txy = [float(st.target_xy[0]), float(st.target_xy[1])]
                        if st.action == "hop":
                            last_xy[h_idx] = txy
                        stage_row.append(
                            {
                                "hopper_id": h_idx,
                                "target_xy": txy,
                                "action": st.action,
                                "target_crater_id": st.target_crater,
                                "wait_s": float(st.wait_s),
                            }
                        )
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
                stages_out.append(stage_row)
            self.stages = stages_out

        return self

    def to_sim_plan(self) -> dict[str, Any]:
        """Return dict matching `selene_sim.verify.FleetPlan.from_any`."""
        return {
            "num_hoppers": self.num_hoppers or len(self.hoppers),
            "start_positions_xy": self.start_positions_xy,
            "stages": self.stages,
        }

    def to_interfaces_dict(self) -> dict[str, Any]:
        """Return dict matching `docs/interfaces.md` §2.3 & §3.2."""
        out: dict[str, Any] = {
            "plan_id": self.plan_id,
            "revision": self.revision,
            "rationale": self.rationale or self.summary,
            "hoppers": [h.model_dump() for h in self.hoppers],
        }
        if self.previous_verdict_pass_rate is not None:
            out["previous_verdict_pass_rate"] = self.previous_verdict_pass_rate
        return out


class HopperIncident(BaseModel):
    model_config = ConfigDict(extra="allow")

    hopper_id: str = "unknown"
    episode: int | None = None
    t: float | None = None
    event: str
    explanation: str
    fix: str = ""


class WorldModelHotspot(BaseModel):
    model_config = ConfigDict(extra="allow")

    location: str
    crater_id: str | None = None
    t: float | None = None
    wm_error_m: float | None = None
    uncertainty_m: float | None = None
    visit_count: int | None = None
    predicted_pos: list[float] | None = None
    actual_pos: list[float] | None = None
    explanation: str
    trust_recommendation: str = ""


class LogReview(BaseModel):
    model_config = ConfigDict(extra="allow")

    summary: str
    hopper_incidents: list[HopperIncident] = Field(default_factory=list)
    world_model_hotspots: list[WorldModelHotspot] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    source: SourceType = "live"
    interaction_id: str | None = None
    environment_id: str | None = None
