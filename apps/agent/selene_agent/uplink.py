"""Human signoff before uplink. A package exists only after a passing verdict, an open relay window and a person."""

from __future__ import annotations

import getpass
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from selene_sim.sim import is_relay_open, time_to_next_relay

from selene_agent.schemas import FleetPlan
from selene_agent.verifier import passes, verdict_for_revision

APPROVAL_WORD = "APPROVE"


class SignoffRefused(Exception):
    """No uplink package may be produced."""


def sign_off(
    plan: FleetPlan,
    verdict: dict[str, Any],
    mission_time_s: float,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Ask a person to approve a verified plan and return the uplink package, or raise SignoffRefused."""
    if verdict.get("source") != "simulator":
        raise SignoffRefused("The verdict did not come from the simulator")
    if not passes(verdict):
        raise SignoffRefused(f"The verifier did not pass this plan (pass rate {verdict.get('pass_rate')})")
    if not bool(is_relay_open(mission_time_s)):
        wait_s = float(time_to_next_relay(mission_time_s))
        raise SignoffRefused(f"No relay window is open at t={mission_time_s} s. The next opens in {wait_s:.0f} s")

    output_fn(_summary(plan, verdict, mission_time_s))
    try:
        answer = input_fn(f"Type {APPROVAL_WORD} to sign off this uplink: ")
    except EOFError as exc:
        raise SignoffRefused("No operator answered the signoff prompt") from exc
    if answer.strip() != APPROVAL_WORD:
        raise SignoffRefused("The operator did not approve this plan")

    sim_plan = plan.to_sim_plan()
    return {
        "plan_id": plan.plan_id,
        "revision": plan.revision,
        "plan_source": plan.source,
        "approved_by_human": True,
        "approved_by": getpass.getuser(),
        "approved_at": datetime.now(UTC).isoformat(),
        "mission_time_s": mission_time_s,
        "plan_sha256": hashlib.sha256(json.dumps(sim_plan, sort_keys=True).encode()).hexdigest(),
        "verdict": {**verdict_for_revision(verdict), "approved_by_human": True},
        "fleet_plan": plan.to_interfaces_dict(),
        "sim_plan": sim_plan,
    }


def _summary(plan: FleetPlan, verdict: dict[str, Any], mission_time_s: float) -> str:
    lines = [
        f"Fleet plan {plan.plan_id}, revision {plan.revision} (source: {plan.source})",
        plan.summary,
        f"Verifier: pass rate {verdict['pass_rate']:.3f} over {verdict['rollout_count']} rollouts "
        f"on {verdict['terrain']} terrain, seed {verdict['seed']}",
    ]
    lines += [f"  {r['rule_id']}: {r['pass_rate']:.3f}" for r in verdict.get("rule_results", [])]
    lines.append(f"Relay window open at t={mission_time_s} s")
    return "\n".join(lines)
