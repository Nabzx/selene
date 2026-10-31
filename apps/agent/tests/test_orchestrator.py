import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from selene_agent import AgentToolset, Orchestrator, orchestrator


def interaction(status, output_text="", interaction_id="i-1", environment_id="env-1", steps=None):
    return SimpleNamespace(
        id=interaction_id,
        environment_id=environment_id,
        status=status,
        output_text=output_text,
        steps=steps or [],
        usage=SimpleNamespace(total_tokens=42),
    )


def function_call_step(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(type="function_call", id=call_id, name=name, arguments=arguments)


def test_create_agent_replaces_agent_on_argon_with_network_disabled():
    client = MagicMock()

    orchestrator.create_agent(client)

    client.agents.delete.assert_called_once_with(id="selene-orchestrator")
    client.agents.create.assert_called_once_with(
        id="selene-orchestrator",
        base_agent="antigravity-preview-09-2026",
        agent_config={"type": "antigravity", "model": "gemini-4-argon"},
        system_instruction=orchestrator.PROMPT.read_text(),
        base_environment={"type": "remote", "network": "disabled"},
    )


def test_ask_polls_in_background_until_completed(monkeypatch):
    monkeypatch.setattr(orchestrator.time, "sleep", lambda _: None)
    client = MagicMock()
    client.interactions.create.return_value = interaction("queued")
    client.interactions.get.side_effect = [interaction("in_progress"), interaction("completed")]

    result = orchestrator.ask(client, "hello")

    assert result.status == "completed"
    assert client.interactions.get.call_count == 2
    client.interactions.create.assert_called_once_with(
        agent="selene-orchestrator",
        input="hello",
        environment="remote",
        background=True,
    )


@pytest.mark.parametrize("status", ["failed", "cancelled", "requires_action", "incomplete"])
def test_ask_raises_unless_completed(status):
    client = MagicMock()
    client.interactions.create.return_value = interaction(status)

    with pytest.raises(RuntimeError, match=status):
        orchestrator.ask(client, "hello")


def test_write_safety_spec_from_arbitrary_markdown_and_execute_checks(tmp_path):
    md_file = tmp_path / "custom_standards.md"
    md_file.write_text(
        "# Far-Side Night Rules\n"
        "- Fuel must never drop below 15% at any step.\n"
        "- Maximum landing slope is 12 degrees.\n"
    )

    payload = {
        "summary": "Extracted two rules from custom_standards.md",
        "rules": [
            {
                "id": "fuel_reserve",
                "name": "Minimum fuel reserve",
                "scope": "step",
                "statement": "Fuel must stay at or above 0.15 at every step.",
                "source_excerpt": "Fuel must never drop below 15% at any step.",
                "parameters": {"min_fuel": 0.15},
                "check_code": (
                    "def check(rollout: dict) -> bool:\n"
                    "    return all(s.get('fuel', 1.0) >= 0.15 for s in rollout.get('steps', []))\n"
                ),
            },
            {
                "id": "landing_slope",
                "name": "Maximum landing slope",
                "scope": "hop",
                "statement": "Landing slope must not exceed 12 degrees.",
                "source_excerpt": "Maximum landing slope is 12 degrees.",
                "parameters": {"max_slope_deg": 12.0},
                "check_code": (
                    "def check(rollout: dict) -> bool:\n"
                    "    return all(s.get('slope_deg', 0.0) <= 12.0 for s in rollout.get('steps', []))\n"
                ),
            },
        ],
        "missing_data": [],
    }
    client = MagicMock()
    client.interactions.create.return_value = interaction(
        "completed", output_text=f"```json\n{json.dumps(payload)}\n```"
    )
    cache_file = tmp_path / "cached_spec.json"

    spec = orchestrator.write_safety_spec(client, md_file, cache_path=cache_file)

    assert spec.source == "live"
    assert len(spec.rules) == 2
    assert spec.verify_rollout({"steps": [{"fuel": 0.25, "slope_deg": 8.0}]}) == {
        "fuel_reserve": True,
        "landing_slope": True,
    }
    assert spec.verify_rollout({"steps": [{"fuel": 0.10, "slope_deg": 15.0}]}) == {
        "fuel_reserve": False,
        "landing_slope": False,
    }

    cached = orchestrator.write_safety_spec(None, md_file, cache_path=cache_file, use_cache=True)
    assert cached.source == "cached"
    assert len(cached.rules) == 2


def test_reward_design_calls_training_and_test_tools():
    candidate_code = (
        "def compute_reward(step: dict, prev_step: dict | None = None, fleet: list[dict] | None = None) -> float:\n"
        "    r = 5.0 if step.get('action', {}).get('type') == 'scan' else 0.0\n"
        "    if step.get('battery', 1.0) < 0.2:\n"
        "        r -= 10.0\n"
        "    return r\n"
    )
    reward_payload = {
        "episode": 200,
        "summary": "Added battery reserve term after the policy learned to strand itself",
        "diagnosis": "Hoppers drained battery below 0.20 in night cold.",
        "terms": [
            {"name": "scan_bonus", "sign": "+", "weight": 5.0, "description": "Scanning a crater"},
            {"name": "low_battery", "sign": "-", "weight": 10.0, "description": "Battery < 0.2"},
        ],
        "reward_code": candidate_code,
        "missing_data": [],
    }
    client = MagicMock()
    client.interactions.create.side_effect = [
        interaction(
            "requires_action",
            interaction_id="i-rew-1",
            environment_id="env-rew",
            steps=[
                function_call_step("c1", "get_training_summary", {}),
                function_call_step("c2", "test_reward_code", {"reward_code": candidate_code}),
            ],
        ),
        interaction(
            "completed",
            output_text=json.dumps(reward_payload),
            interaction_id="i-rew-2",
            environment_id="env-rew",
        ),
    ]

    spec = orchestrator.rewrite_reward(client, episode=200)

    assert client.interactions.create.call_count == 2
    tool_results = client.interactions.create.call_args_list[1].kwargs["input"]
    assert len(tool_results) == 2
    assert tool_results[0]["name"] == "get_training_summary"
    assert json.loads(tool_results[1]["result"])["valid"] is True
    assert spec.to_reward_version() == {
        "episode": 200,
        "summary": "Added battery reserve term after the policy learned to strand itself",
    }
    assert spec.evaluate({"action": {"type": "scan"}, "battery": 0.8}) == 5.0


def test_plan_fleet_uses_mission_state_hop_estimator_and_verifier_tools():
    final_plan_payload = {
        "summary": "Planned verified route for h1 from c1 to c2",
        "routes": [
            {
                "hopper_id": "h1",
                "craters": ["c1", "c2"],
                "hops": [
                    {
                        "hop_index": 0,
                        "from_pos": [0, 0, -120],
                        "to_pos": [1800, 600, -80],
                        "target_crater": "c2",
                        "estimated_fuel_used": 0.05,
                    }
                ],
            }
        ],
    }
    client = MagicMock()
    client.interactions.create.side_effect = [
        interaction(
            "requires_action",
            interaction_id="i-plan-1",
            environment_id="env-plan",
            steps=[
                function_call_step("c1", "get_mission_state", {}),
                function_call_step(
                    "c2", "estimate_hop", {"from_pos": [0, 0, -120], "to_pos": [1800, 600, -80]}
                ),
                function_call_step(
                    "c3",
                    "verify_fleet_plan",
                    {
                        "routes": [
                            {
                                "hopper_id": "h1",
                                "craters": ["c1", "c2"],
                                "hops": [{"from_pos": [0, 0, -120], "to_pos": [1800, 600, -80]}],
                            }
                        ],
                        "num_rollouts": 20,
                    },
                ),
            ],
        ),
        interaction(
            "completed",
            output_text=json.dumps(final_plan_payload),
            interaction_id="i-plan-2",
            environment_id="env-plan",
        ),
    ]

    plan = orchestrator.plan_fleet(client)

    assert len(plan.routes) == 1
    assert plan.routes[0].craters == ["c1", "c2"]
    tool_results = client.interactions.create.call_args_list[1].kwargs["input"]
    assert [r["name"] for r in tool_results] == [
        "get_mission_state",
        "estimate_hop",
        "verify_fleet_plan",
    ]
    hop_estimate = json.loads(tool_results[1]["result"])
    assert hop_estimate["is_estimate"] is True
    assert hop_estimate["horizontal_distance_m"] > 1800


def test_plan_and_verify_revision_loop_preserves_interaction_context():
    initial_payload = {
        "summary": "Initial fleet plan across c1 and c2",
        "routes": [
            {
                "hopper_id": "h1",
                "craters": ["c1", "c2"],
                "hops": [{"hop_index": 0, "from_pos": [0, 0, -120], "to_pos": [1800, 600, -80]}],
            }
        ],
    }
    revised_payload = {
        "summary": "Revised fleet plan avoiding steep rim on c2",
        "revision_summary": "Moved h1 hop 0 target inside rim to pass slope rule",
        "changes": ["Shifted h1 landing target from [1800, 600, -80] to [1750, 580, -82]"],
        "routes": [
            {
                "hopper_id": "h1",
                "craters": ["c1", "c2"],
                "hops": [{"hop_index": 0, "from_pos": [0, 0, -120], "to_pos": [1750, 580, -82]}],
            }
        ],
    }
    client = MagicMock()
    client.interactions.create.side_effect = [
        interaction("completed", json.dumps(initial_payload), interaction_id="i-plan", environment_id="env-9"),
        interaction("completed", json.dumps(revised_payload), interaction_id="i-rev", environment_id="env-9"),
    ]

    plan = orchestrator.plan_fleet(client, "Craters c1 and c2, hopper h1 at origin")
    verdicts = [
        {"passed": False, "pass_rate": 0.84, "rule_failures": {"landing_slope": 160}},
        {"passed": True, "pass_rate": 0.995, "rule_failures": {}},
    ]
    final_plan, final_verdict = orchestrator.verify_and_revise(
        client, plan, verifier_fn=lambda _: verdicts.pop(0)
    )

    assert final_verdict["passed"] is True
    assert final_plan.routes[0].hops[0].to_pos == [1750.0, 580.0, -82.0]
    second_call_kwargs = client.interactions.create.call_args_list[1].kwargs
    assert second_call_kwargs["previous_interaction_id"] == "i-plan"
    assert second_call_kwargs["environment"] == "env-9"


def test_review_logs_uses_get_hopper_logs_tool():
    review_payload = {
        "summary": "Hopper completed route but world model had 36.1 m error at Hollow 2.",
        "hopper_incidents": [
            {
                "hopper_id": "h1",
                "episode": 500,
                "t": 18.0,
                "event": "landed",
                "explanation": "Landed 14 m from target with 36.1 m world-model error.",
                "fix": "Add transition samples near Hollow 2 rim.",
            }
        ],
        "world_model_hotspots": [
            {
                "location": "Hollow 2 (c2)",
                "t": 18.0,
                "wm_error_m": 36.1,
                "predicted_pos": [1760, 590, -78],
                "actual_pos": [1790, 610, -80],
                "explanation": "Unvisited crater rim slope caused landing scatter.",
                "trust_recommendation": "Do not trust world model near c2 until updated.",
            }
        ],
        "recommendations": ["Retrain world model on Episode 500 transitions."],
    }
    client = MagicMock()
    client.interactions.create.side_effect = [
        interaction(
            "requires_action",
            interaction_id="i-log-1",
            environment_id="env-3",
            steps=[function_call_step("c-log", "get_hopper_logs", {"min_wm_error": 10.0})],
        ),
        interaction(
            "completed",
            output_text=json.dumps(review_payload),
            interaction_id="i-log-2",
            environment_id="env-3",
        ),
    ]
    example_path = Path(__file__).resolve().parents[3] / "data" / "episodes.example.json"
    toolset = AgentToolset(episodes_data=example_path)

    orch = Orchestrator(client=client, toolset=toolset)
    review = orch.review_logs()

    assert review.world_model_hotspots[0].wm_error_m == 36.1
    assert orch.last_interaction_id == "i-log-2"
    assert orch.environment_id == "env-3"
    log_tool_output = json.loads(client.interactions.create.call_args_list[1].kwargs["input"][0]["result"])
    assert log_tool_output["top_world_model_errors"][0]["wm_error"] == 36.1


def test_create_agent_mounts_mission_doc_and_toolset_reads_it(tmp_path):
    doc_file = tmp_path / "mission.md"
    doc_file.write_text("# Mission\nMinimum fuel: 15%")
    client = MagicMock()

    orchestrator.create_agent(client, mission_doc=doc_file)

    call_kwargs = client.agents.create.call_args.kwargs
    assert call_kwargs["base_environment"]["sources"] == [
        {
            "type": "inline",
            "target": "/workspace/mission_constraints.md",
            "content": "# Mission\nMinimum fuel: 15%",
        }
    ]
    toolset = AgentToolset(extra_files=[doc_file])
    docs = toolset.call("read_mission_documents")
    assert docs["documents"][0]["content"] == "# Mission\nMinimum fuel: 15%"


def test_safety_spec_evaluates_selene_sim_rollout_traces_and_interfaces_dict():
    from selene_sim.sim import SimConfig, TerrainGrid
    from selene_sim.verify import FleetPlan as SimFleetPlan, simulate_plan

    spec, plan = orchestrator.stub_argon_plan_and_spec(seed=42)
    assert spec.source == "scripted"
    assert plan.source == "scripted"
    assert len(spec.rules) == 6

    terrain = TerrainGrid.synthetic(seed=42)
    sim_plan = SimFleetPlan.from_any(plan.to_sim_plan())
    traces = simulate_plan(sim_plan, n=40, seed=42, terrain=terrain, config=SimConfig())

    verdict = spec.evaluate_traces(traces, plan_id=plan.plan_id, revision=plan.revision)
    assert verdict.approved_by_human is False
    assert len(verdict.rule_results) == 6
    assert 0.0 <= verdict.pass_rate <= 1.0

    # Also test single-rollout step dict compatibility on the same rules
    single_ok = spec.verify_rollout(
        {"steps": [{"fuel": 0.50, "battery": 0.60, "slope_deg": 6.0}]}
    )
    assert all(single_ok.values())
    single_fail = spec.verify_rollout(
        {"steps": [{"fuel": 0.05, "battery": 0.10, "slope_deg": 22.0}]}
    )
    assert single_fail["R1_fuel_reserve"] is False
    assert single_fail["R2_battery_reserve"] is False
    assert single_fail["R3_landing_slope"] is False


def test_reward_spec_hot_swap_with_selene_sim_and_self_correction_retry():
    bad_code = (
        "import numpy as np\n"
        "def compute_reward(state, actions=None, next_state=None):\n"
        "    return state.fuel * 0.1\n"
    )
    good_code = (
        "import numpy as np\n"
        "def compute_reward(state, actions=None, next_state=None):\n"
        "    if isinstance(state, dict):\n"
        "        return 5.0 if state.get('action', {}).get('type') == 'scan' else 0.0\n"
        "    s1 = next_state if next_state is not None else state\n"
        "    fuel = np.asarray(s1.fuel_kg, dtype=np.float32)\n"
        "    bat = np.asarray(s1.battery_pct, dtype=np.float32)\n"
        "    crashed = np.asarray(s1.crashed, dtype=bool)\n"
        "    rew = np.where(crashed, -50.0, 1.0).astype(np.float32)\n"
        "    rew = rew - np.where(fuel < 1.5, 10.0, 0.0).astype(np.float32)\n"
        "    rew = rew - np.where(bat < 20.0, 10.0, 0.0).astype(np.float32)\n"
        "    return rew, {'survival': float(np.mean(rew))}\n"
    )
    first_payload = {
        "episode": 100,
        "summary": "Broken reward using state.fuel",
        "diagnosis": "Initial attempt",
        "terms": [],
        "reward_code": bad_code,
    }
    second_payload = {
        "episode": 100,
        "summary": "Fixed vectorised + step-dict reward",
        "diagnosis": "Uses fuel_kg and battery_pct on RewardState",
        "terms": [{"name": "survival", "sign": "+", "weight": 1.0}],
        "reward_code": good_code,
    }
    client = MagicMock()
    client.interactions.create.side_effect = [
        interaction("completed", json.dumps(first_payload), interaction_id="i-bad", environment_id="env-r"),
        interaction("completed", json.dumps(second_payload), interaction_id="i-good", environment_id="env-r"),
    ]

    spec = orchestrator.write_reward(client, "Design reward", episode=100, hot_swap=True)

    assert client.interactions.create.call_count == 2
    retry_kwargs = client.interactions.create.call_args_list[1].kwargs
    assert retry_kwargs["previous_interaction_id"] == "i-bad"
    assert "fuel_kg" in retry_kwargs["input"]
    assert spec.summary == "Fixed vectorised + step-dict reward"
    assert spec.evaluate({"action": {"type": "scan"}}) == 5.0


def test_fleet_plan_three_way_sync_and_verify_fleet_plan_with_trajectory_cloud():
    from selene_agent.schemas import FleetPlan

    plan_from_interfaces = FleetPlan.model_validate(
        {
            "plan_id": "plan-sync-test",
            "revision": 2,
            "rationale": "Test 3-way sync between hoppers, routes and stages",
            "hoppers": [
                {
                    "id": "h0",
                    "start_pos_xy": [-200.0, -200.0],
                    "craters_assigned": ["c1"],
                    "steps": [
                        {
                            "stage": 0,
                            "action": "hop",
                            "target_crater": "c1",
                            "target_xy": [-800.0, -600.0],
                        },
                        {
                            "stage": 1,
                            "action": "scan",
                            "target_crater": "c1",
                            "target_xy": [-800.0, -600.0],
                        },
                    ],
                },
                {
                    "id": "h1",
                    "start_pos_xy": [200.0, 200.0],
                    "craters_assigned": ["c2"],
                    "steps": [
                        {
                            "stage": 0,
                            "action": "hop",
                            "target_crater": "c2",
                            "target_xy": [900.0, 700.0],
                        },
                        {
                            "stage": 1,
                            "action": "send",
                            "target_crater": "c2",
                            "target_xy": [900.0, 700.0],
                        },
                    ],
                },
            ],
        }
    )
    assert len(plan_from_interfaces.routes) == 2
    assert len(plan_from_interfaces.stages) == 2
    assert plan_from_interfaces.num_hoppers == 2
    sim_dict = plan_from_interfaces.to_sim_plan()
    assert sim_dict["num_hoppers"] == 2
    assert len(sim_dict["stages"]) == 2

    toolset = AgentToolset()
    verdict = toolset.verify_fleet_plan(fleet_plan=plan_from_interfaces.model_dump(), num_rollouts=30, seed=42)
    assert verdict["rollout_count"] == 30
    assert "rule_results" in verdict
    assert "trajectory_cloud" in verdict
    assert len(verdict["trajectory_cloud"]) > 0

    block = orchestrator.build_verification_block(plan_from_interfaces, verdict)
    assert block["approved_by_human"] is False
    assert block["plan_id"] == "plan-sync-test"
    assert block["revision"] == 2
    assert len(block["trajectory_cloud"]) > 0


def test_should_rewrite_reward_triggers_and_training_log_csv(tmp_path):
    csv_file = tmp_path / "training_log.csv"
    csv_file.write_text(
        "episode,mean_reward,craters_visited,data_downlinked,crash_rate,fuel_remaining_pct,crater_overlap_rate,relay_conflict_rate\n"
        "0,10.0,1.2,0.5,0.25,40.0,0.35,0.20\n"
        "50,12.0,1.4,0.6,0.22,38.0,0.30,0.18\n"
        "100,12.1,1.5,0.7,0.20,37.0,0.28,0.16\n"
        "150,12.2,1.5,0.7,0.18,36.0,0.27,0.16\n"
    )
    example_path = Path(__file__).resolve().parents[3] / "data" / "episodes.example.json"
    toolset = AgentToolset(episodes_data=example_path, extra_files=[csv_file])
    summary = toolset.get_training_summary()

    should_rewrite, reasons = orchestrator.should_rewrite_reward(summary)
    assert should_rewrite is True
    assert len(reasons) >= 3


def test_get_world_model_error_map():
    example_path = Path(__file__).resolve().parents[3] / "data" / "episodes.example.json"
    toolset = AgentToolset(episodes_data=example_path)
    err_map = toolset.get_world_model_error_map(min_wm_error=5.0)
    assert err_map["hotspot_count"] >= 1
    assert err_map["hotspots"][0]["error_m"] == 36.1
    assert "craters" in err_map


