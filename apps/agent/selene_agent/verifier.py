"""Verdicts for fleet plans: simulator rollouts checked against the safety spec. No model runs here."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from selene_sim.sim import SimConfig, TerrainGrid
from selene_sim.verify import FleetPlan as SimFleetPlan
from selene_sim.verify import RolloutTraces, sample_trajectory_cloud, simulate_plan

from selene_agent.schemas import RuleResult, SafetySpec

REQUIRED_PASS_RATE = 0.99
CRASH_RULE_ID = "SIM_NO_CRASH"


def passes(verdict: dict[str, Any], required_pass_rate: float = REQUIRED_PASS_RATE) -> bool:
    return bool(verdict.get("passed", False)) and float(verdict.get("pass_rate", 0.0)) >= required_pass_rate


def verify_sim_plan(
    sim_plan: dict[str, Any],
    spec: SafetySpec,
    *,
    n: int = 1000,
    seed: int = 42,
    terrain: TerrainGrid | None = None,
    required_pass_rate: float = REQUIRED_PASS_RATE,
    max_cloud_runs: int = 25,
) -> dict[str, Any]:
    """Replay a plan in the simulator and return the verdict. A plan that can't be simulated fails."""
    if not spec.rules:
        raise ValueError("Safety spec has no rules, so a verdict would mean nothing")

    plan_id = str(sim_plan.get("plan_id", "plan_v1"))
    revision = int(sim_plan.get("revision", 0))
    try:
        plan = SimFleetPlan.from_any(sim_plan)
        if not any(cmd.action == "hop" for stage in plan.stages for cmd in stage):
            raise ValueError("plan has no hops")
        traces = simulate_plan(plan, n=n, seed=seed, terrain=terrain)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return _rejected_verdict(plan_id, revision, n, seed, required_pass_rate, str(exc))

    verdict = spec.evaluate_traces(
        traces, plan_id=plan_id, revision=revision, required_pass_rate=required_pass_rate
    )
    # Argon writes the rules, so a crash fails a rollout whether or not a rule covers it.
    crashed_runs = np.flatnonzero(np.any(traces.crashed, axis=(1, 2))).astype(int).tolist()
    crash_rate = round((n - len(crashed_runs)) / n, 4)
    verdict.rule_results.append(
        RuleResult(
            rule_id=CRASH_RULE_ID,
            statement="No hopper crashes in the simulator.",
            passed=crash_rate >= required_pass_rate,
            pass_rate=crash_rate,
            reason="ok" if crash_rate >= required_pass_rate else f"Crashes in {len(crashed_runs)}/{n} rollouts.",
            failed_rollouts=crashed_runs,
        )
    )
    failed = sorted(set(verdict.failed_rollout_indices) | set(crashed_runs))
    pass_rate = min(verdict.pass_rate, round((n - len(failed)) / n, 4))
    failing_rules = [r for r in verdict.rule_results if not r.passed]

    return {
        **verdict.model_dump(),
        "passed": pass_rate >= required_pass_rate and not failing_rules,
        "pass_rate": pass_rate,
        "failed_rollout_indices": failed,
        "source": "simulator",
        "terrain": terrain.region_name if terrain is not None else "synthetic",
        "seed": seed,
        "rollout_count": n,
        "n_rollouts": n,
        "rule_failures": {r.rule_id: round((1.0 - r.pass_rate) * n) for r in failing_rules},
        "worst_failure": min(
            ({"rule_id": r.rule_id, "pass_rate": r.pass_rate, "reason": r.reason} for r in failing_rules),
            key=lambda r: r["pass_rate"],
            default=None,
        ),
        "failing_hops": _failing_hops(traces, plan),
        "stage_conflicts": _stage_conflicts(traces),
        "resources": _resources(traces),
        "trajectory_cloud": sample_trajectory_cloud(traces, max_runs=max_cloud_runs),
    }


def verdict_for_revision(verdict: dict[str, Any]) -> dict[str, Any]:
    """The failure statistics Argon needs, without per-rollout lists or the trajectory cloud."""
    compact = {
        key: value
        for key, value in verdict.items()
        if key not in ("trajectory_cloud", "arcs", "failed_rollout_indices", "rule_results", "history")
    }
    compact["failed_rollout_count"] = len(verdict.get("failed_rollout_indices", []))
    compact["rule_results"] = [
        {key: rule[key] for key in ("rule_id", "statement", "passed", "pass_rate", "reason") if key in rule}
        for rule in verdict.get("rule_results", [])
    ]
    return compact


def _rejected_verdict(
    plan_id: str, revision: int, n: int, seed: int, required_pass_rate: float, error: str
) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "revision": revision,
        "passed": False,
        "pass_rate": 0.0,
        "required_pass_rate": required_pass_rate,
        "approved_by_human": False,
        "source": "simulator",
        "seed": seed,
        "rollout_count": n,
        "n_rollouts": n,
        "error": f"The simulator rejected this plan: {error}",
        "failed_rollout_indices": list(range(n)),
        "rule_results": [],
        "rule_failures": {},
        "trajectory_cloud": [],
    }


def _failing_hops(traces: RolloutTraces, plan: SimFleetPlan) -> list[dict[str, Any]]:
    """Per hop, how often it crashed or landed too steep, so Argon knows which hop to move."""
    max_slope = SimConfig().max_landing_slope_deg
    newly_crashed = traces.crashed.copy()
    newly_crashed[:, 1:, :] &= ~traces.crashed[:, :-1, :]
    hops = []
    for stage, commands in enumerate(plan.stages):
        for hopper, command in enumerate(commands):
            if command.action != "hop":
                continue
            crashes = newly_crashed[:, stage, hopper]
            slopes = traces.landing_slope_deg[:, stage, hopper]
            steep = int(np.sum(slopes > max_slope))
            if crashes.any() or steep:
                hops.append(
                    {
                        "hopper_id": f"h{hopper}",
                        "stage": stage,
                        "target_xy": list(command.target_xy),
                        "crash_rollouts": int(crashes.sum()),
                        "crash_reasons": dict(Counter(str(r) for r in traces.crash_reason[:, stage, hopper][crashes])),
                        "steep_landing_rollouts": steep,
                        "max_landing_slope_deg": round(float(slopes.max()), 2),
                    }
                )
    return hops


def _stage_conflicts(traces: RolloutTraces) -> list[dict[str, Any]]:
    conflicts = []
    for stage in range(traces.relay_conflict.shape[1]):
        relay = int(np.sum(traces.relay_conflict[:, stage]))
        too_close = int(np.sum(traces.min_separation_m[:, stage] < SimConfig().min_separation_m))
        if relay or too_close:
            conflicts.append(
                {
                    "stage": stage,
                    "relay_conflict_rollouts": relay,
                    "too_close_rollouts": too_close,
                    "min_separation_m": round(float(traces.min_separation_m[:, stage].min()), 1),
                }
            )
    return conflicts


def _resources(traces: RolloutTraces) -> list[dict[str, Any]]:
    cfg = SimConfig()
    return [
        {
            "hopper_id": f"h{hopper}",
            "min_fuel_fraction": round(float(traces.fuel_kg[:, :, hopper].min()) / cfg.initial_fuel_kg, 3),
            "min_battery_fraction": round(float(traces.battery_pct[:, :, hopper].min()) / 100.0, 3),
        }
        for hopper in range(traces.fuel_kg.shape[2])
    ]
