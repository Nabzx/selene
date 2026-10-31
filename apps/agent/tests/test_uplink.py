import json
import sys

import pytest

from selene_agent import orchestrator
from selene_agent.schemas import FleetPlan
from selene_agent.uplink import SignoffRefused, sign_off

OPEN_T = 10.0
CLOSED_T = 60.0


def plan():
    return FleetPlan.model_validate(
        {
            "plan_id": "plan_v1",
            "revision": 2,
            "summary": "Two hops",
            "num_hoppers": 1,
            "start_positions_xy": [[0.0, 0.0]],
            "stages": [[{"hopper_id": 0, "action": "hop", "target_xy": [500.0, 0.0]}]],
        }
    )


def verdict(**overrides):
    return {
        "source": "simulator",
        "passed": True,
        "pass_rate": 0.995,
        "rollout_count": 1000,
        "terrain": "Daedalus",
        "seed": 42,
        "rule_results": [{"rule_id": "R1_fuel_reserve", "pass_rate": 1.0, "passed": True}],
        "trajectory_cloud": [{"large": True}],
        **overrides,
    }


def never_asked(_):
    raise AssertionError("The operator must not be asked")


def test_approval_produces_package_with_human_flag():
    prompts = []
    shown = []

    package = sign_off(
        plan(),
        verdict(),
        OPEN_T,
        input_fn=lambda p: prompts.append(p) or "APPROVE",
        output_fn=shown.append,
    )

    assert package["approved_by_human"] is True
    assert package["verdict"]["approved_by_human"] is True
    assert package["approved_by"]
    assert package["plan_sha256"]
    assert package["sim_plan"]["stages"]
    assert "trajectory_cloud" not in package["verdict"]
    assert "APPROVE" in prompts[0]
    assert "0.995" in shown[0]


@pytest.mark.parametrize("answer", ["", "n", "yes", "approve", "APPROVED"])
def test_anything_but_the_approval_word_refuses(answer):
    with pytest.raises(SignoffRefused, match="did not approve"):
        sign_off(plan(), verdict(), OPEN_T, input_fn=lambda _: answer, output_fn=lambda _: None)


def test_no_operator_refuses():
    def closed_stdin(_):
        raise EOFError

    with pytest.raises(SignoffRefused, match="No operator"):
        sign_off(plan(), verdict(), OPEN_T, input_fn=closed_stdin, output_fn=lambda _: None)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"passed": False}, "did not pass"),
        ({"pass_rate": 0.98}, "did not pass"),
        ({"source": "sandbox_estimate"}, "simulator"),
    ],
)
def test_operator_is_not_asked_without_a_real_pass(bad, message):
    with pytest.raises(SignoffRefused, match=message):
        sign_off(plan(), verdict(**bad), OPEN_T, input_fn=never_asked)


def test_operator_is_not_asked_outside_a_relay_window():
    with pytest.raises(SignoffRefused, match="next opens in 60 s"):
        sign_off(plan(), verdict(), CLOSED_T, input_fn=never_asked)


def test_cli_refusal_writes_no_package(tmp_path, monkeypatch):
    failing = orchestrator.stub_argon_plan_and_spec(seed=42)[1]
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(failing.model_dump_json())
    out = tmp_path / "uplink.json"
    monkeypatch.setattr("builtins.input", never_asked)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchestrator", "signoff", str(plan_file), "--mission-time", str(OPEN_T), "-o", str(out)],
    )

    with pytest.raises(SystemExit, match="Uplink refused"):
        orchestrator.main()

    assert not out.exists()
    assert json.loads(plan_file.read_text())["plan_id"]
