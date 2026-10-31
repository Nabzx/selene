import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from selene_sim.sim import TerrainGrid

from selene_agent import orchestrator
from selene_agent.schemas import FleetPlan, SafetyRule, SafetySpec
from selene_agent.tools import AgentToolset
from selene_agent.verifier import passes, verify_sim_plan

STARTS = [[-300.0, -300.0], [300.0, -300.0], [-300.0, 300.0], [300.0, 300.0]]


def flat_target(terrain, start):
    """A cell near `start` whose surroundings are flat enough to survive landing scatter."""
    xs, ys = np.meshgrid(np.arange(-1200, 1201, 100.0), np.arange(-1200, 1201, 100.0))
    xs, ys = xs.ravel() + start[0], ys.ravel() + start[1]
    worst = np.max(
        [terrain.slope_at(xs + dx, ys + dy) for dx in (-60, 0, 60) for dy in (-60, 0, 60)], axis=0
    )
    distance = np.hypot(xs - start[0], ys - start[1])
    candidates = np.flatnonzero((distance > 200) & (worst < 8.0))
    best = candidates[np.argmin(distance[candidates])]
    return [float(xs[best]), float(ys[best])]


def plan_payload(targets, revision=0):
    return {
        "plan_id": "plan_v1",
        "revision": revision,
        "summary": f"Revision {revision}",
        "num_hoppers": 4,
        "start_positions_xy": STARTS,
        "stages": [
            [
                {"hopper_id": i, "action": "hop", "target_xy": target}
                for i, target in enumerate(targets)
            ]
        ],
    }


def interaction(payload, interaction_id):
    return SimpleNamespace(
        id=interaction_id,
        environment_id="env-1",
        status="completed",
        output_text=json.dumps(payload),
        steps=[],
        usage=SimpleNamespace(total_tokens=1),
    )


@pytest.fixture
def spec():
    return orchestrator.stub_argon_plan_and_spec(seed=42)[0]


@pytest.fixture
def terrain():
    return TerrainGrid.synthetic(seed=42)


@pytest.fixture
def good_targets(terrain):
    return [flat_target(terrain, start) for start in STARTS]


@pytest.fixture
def failing_plan():
    return orchestrator.stub_argon_plan_and_spec(seed=42)[1]


def test_loop_feeds_simulator_failures_to_argon_until_pass_rate_reaches_bar(
    spec, terrain, good_targets, failing_plan
):
    failing_plan.interaction_id = "i-plan"
    failing_plan.environment_id = "env-1"
    client = MagicMock()
    client.interactions.create.return_value = interaction(plan_payload(good_targets, 1), "i-rev-1")

    plan, verdict = orchestrator.verify_and_revise(client, failing_plan, spec=spec, terrain=terrain)

    assert verdict["source"] == "simulator"
    assert passes(verdict) and verdict["pass_rate"] >= 0.99
    assert verdict["approved_by_human"] is False
    assert [h["revision"] for h in verdict["history"]] == [failing_plan.revision, 1]
    assert verdict["history"][0]["pass_rate"] < 0.99
    assert plan.previous_verdict_pass_rate == verdict["history"][0]["pass_rate"]

    kwargs = client.interactions.create.call_args.kwargs
    assert kwargs["previous_interaction_id"] == "i-plan"
    assert kwargs["environment"] == "env-1"
    prompt = kwargs["input"]
    assert "R3_landing_slope" in prompt
    assert "failing_hops" in prompt
    assert "steep_landing_rollouts" in prompt
    assert "trajectory_cloud" not in prompt.split("## Input data")[1]


def test_each_revision_continues_the_previous_interaction(spec, terrain, good_targets, failing_plan):
    still_bad = plan_payload([[9000.0, 9000.0]] * 4, 1)
    client = MagicMock()
    client.interactions.create.side_effect = [
        interaction(still_bad, "i-rev-1"),
        interaction(plan_payload(good_targets, 2), "i-rev-2"),
    ]

    plan, verdict = orchestrator.verify_and_revise(client, failing_plan, spec=spec, terrain=terrain)

    assert passes(verdict)
    assert plan.revision == 2
    assert client.interactions.create.call_args_list[1].kwargs["previous_interaction_id"] == "i-rev-1"


def test_loop_stops_at_cap_and_returns_the_failing_verdict(spec, terrain, failing_plan):
    client = MagicMock()
    client.interactions.create.side_effect = lambda **_: interaction(
        plan_payload([[9000.0, 9000.0]] * 4, 1), "i-rev"
    )

    _, verdict = orchestrator.verify_and_revise(
        client, failing_plan, spec=spec, terrain=terrain, max_revisions=2
    )

    assert client.interactions.create.call_count == 2
    assert not passes(verdict)
    assert len(verdict["history"]) == 3


def test_passed_flag_alone_does_not_stop_the_loop(failing_plan, good_targets):
    client = MagicMock()
    client.interactions.create.return_value = interaction(plan_payload(good_targets, 1), "i-rev")
    verdicts = [{"passed": True, "pass_rate": 0.95}, {"passed": True, "pass_rate": 0.995}]

    _, verdict = orchestrator.verify_and_revise(client, failing_plan, verifier_fn=lambda _: verdicts.pop(0))

    assert client.interactions.create.call_count == 1
    assert verdict["pass_rate"] == 0.995


def test_argon_verify_tool_uses_the_simulator_during_the_loop(spec, terrain, good_targets, failing_plan):
    toolset = AgentToolset()
    client = MagicMock()
    client.interactions.create.return_value = interaction(plan_payload(good_targets, 1), "i-rev")

    orchestrator.verify_and_revise(client, failing_plan, spec=spec, terrain=terrain, toolset=toolset)

    result = toolset.verify_fleet_plan(fleet_plan=plan_payload(good_targets))
    assert result["source"] == "simulator"
    assert "trajectory_cloud" not in result


def test_hop_beyond_range_fails(spec, terrain, good_targets):
    targets = [*good_targets[:3], [good_targets[3][0] + 6000.0, good_targets[3][1]]]

    verdict = verify_sim_plan(plan_payload(targets), spec, terrain=terrain)

    assert not passes(verdict)
    assert "R6_max_hop_range" in verdict["rule_failures"]


def test_fleet_rules_fire_on_batch_traces(spec, terrain, good_targets):
    same_target = [good_targets[0]] * 4

    verdict = verify_sim_plan(plan_payload(same_target), spec, terrain=terrain)

    assert not passes(verdict)
    assert "R4_fleet_separation" in verdict["rule_failures"]
    assert verdict["stage_conflicts"][0]["too_close_rollouts"] > 0


def test_single_rollout_rule_still_runs_per_rollout(terrain, good_targets):
    spec = SafetySpec(
        rules=[
            SafetyRule(
                id="steps_only",
                name="Steps only",
                statement="Fuel never drops below 0.99.",
                check_code=(
                    "def check(rollout):\n"
                    "    return all(s.get('fuel', 1.0) >= 0.99 for s in rollout.get('steps', []))\n"
                ),
            )
        ]
    )

    verdict = verify_sim_plan(plan_payload(good_targets), spec, n=20, terrain=terrain)

    assert not passes(verdict)
    assert "steps_only" in verdict["rule_failures"]


@pytest.mark.parametrize(
    "plan",
    [
        {"num_hoppers": 1, "start_positions_xy": [[0.0, 0.0]], "stages": []},
        {"num_hoppers": 1, "start_positions_xy": [[0.0, 0.0]], "stages": [[{"action": "wait"}]]},
        {"stages": []},
    ],
)
def test_plan_without_hops_fails_and_reports_why(spec, terrain, plan):
    verdict = verify_sim_plan(plan, spec, terrain=terrain)

    assert not passes(verdict)
    assert verdict["pass_rate"] == 0.0
    assert "rejected" in verdict["error"]


def test_empty_safety_spec_is_refused(terrain, good_targets):
    with pytest.raises(ValueError, match="no rules"):
        verify_sim_plan(plan_payload(good_targets), SafetySpec(rules=[]), terrain=terrain)


def test_crash_fails_even_if_no_rule_covers_it(failing_plan, terrain):
    lenient = SafetySpec(
        rules=[
            SafetyRule(
                id="always_true",
                name="Always true",
                statement="Always true.",
                check_code="def check(trace):\n    return True\n",
            )
        ]
    )

    verdict = verify_sim_plan(failing_plan.to_sim_plan(), lenient, terrain=terrain)

    assert not passes(verdict)
    assert "SIM_NO_CRASH" in verdict["rule_failures"]
    assert verdict["failing_hops"]


def test_rule_that_raises_fails_rather_than_crashing_the_loop(terrain, good_targets):
    broken = SafetySpec(
        rules=[
            SafetyRule(
                id="broken",
                name="Broken",
                statement="Broken.",
                check_code="def check(trace):\n    return 1 / 0\n",
            )
        ]
    )

    verdict = verify_sim_plan(plan_payload(good_targets), broken, terrain=terrain)

    assert not passes(verdict)
    assert "broken" in verdict["rule_failures"]


def test_rules_reporting_only_per_rollout_ok_lower_the_joint_rate(terrain, good_targets):
    code = (
        "def check(trace):\n"
        "    ok = [i != {bad} for i in range(trace['n_rollouts'])]\n"
        "    return (True, 'ok', {{'pass_rate': sum(ok) / len(ok), 'per_rollout_ok': ok}})\n"
    )
    spec = SafetySpec(
        rules=[
            SafetyRule(id=f"r{bad}", name="r", statement="r", check_code=code.format(bad=bad))
            for bad in (0, 1)
        ]
    )

    verdict = verify_sim_plan(plan_payload(good_targets), spec, n=100, terrain=terrain)

    assert verdict["pass_rate"] == 0.98
    assert verdict["failed_rollout_indices"] == [0, 1]
    assert not passes(verdict)


def test_sandbox_sources_stay_under_the_inline_limits(tmp_path):
    episodes = tmp_path / "episodes.json"
    episodes.write_text(json.dumps({"craters": [], "episodes": [{"index": i, "pad": "x" * 400_000} for i in range(4)]}))

    sources = orchestrator.build_cloud_sources(episodes_path=episodes)

    sizes = {s["target"]: len(s["content"].encode()) for s in sources}
    assert max(sizes.values()) <= 1_000_000
    assert sum(sizes.values()) <= 2_000_000
    mounted = json.loads(next(s["content"] for s in sources if s["target"].endswith("episodes.json")))
    assert [e["index"] for e in mounted["episodes"]] == [2, 3]
    assert mounted["episodes_trimmed_for_sandbox"] is True


def test_verification_block_never_marks_human_approval(spec, terrain, good_targets):
    plan = FleetPlan.model_validate(plan_payload(good_targets))
    verdict = verify_sim_plan({**plan.to_sim_plan(), "plan_id": plan.plan_id}, spec, terrain=terrain)

    block = orchestrator.build_verification_block(plan, verdict)

    assert block["passed"] is True
    assert block["approved_by_human"] is False
    assert block["rollout_count"] == 1000
    assert block["trajectory_cloud"]
