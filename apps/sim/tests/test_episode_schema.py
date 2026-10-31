import copy
import json
from pathlib import Path
import pytest

EXAMPLE = Path(__file__).resolve().parents[3] / "data" / "episodes.example.json"

DRONE_FIELDS = {"id", "name", "outcome", "craters_visited", "steps"}
STEP_FIELDS = {
    "t", "pos", "yaw", "phase", "fuel", "battery", "temp_c", "action",
    "reward", "wm_pos", "wm_error", "event",
}
OUTCOMES = {"completed", "crashed", "battery", "timeout"}
PHASES = {"ground", "ascent", "coast", "descent"}
ACTION_TYPES = {"hop", "scan", "wait", "send"}
EVENTS = {None, "landed", "crashed", "scan", "relay_contact"}


def validate_episode_doc(doc: dict) -> None:
    assert doc["schema_version"] == 1
    assert doc["meta"]["source"] in {"trained", "scripted"}
    fleet_size = doc["meta"]["fleet_size"]
    assert isinstance(fleet_size, int) and 1 <= fleet_size <= 5
    crater_ids = {c["id"] for c in doc["craters"]}
    for ep in doc["episodes"]:
        assert ep["outcome"] in OUTCOMES
        assert set(ep["craters_visited"]) <= crater_ids
        assert len(ep["drones"]) == fleet_size
        seen_ids = set()
        for drone in ep["drones"]:
            assert set(drone) == DRONE_FIELDS
            assert drone["id"] not in seen_ids
            seen_ids.add(drone["id"])
            assert drone["outcome"] in OUTCOMES
            assert set(drone["craters_visited"]) <= crater_ids
            assert len(drone["steps"]) >= 1
            for step in drone["steps"]:
                assert set(step) == STEP_FIELDS
                assert step["phase"] in PHASES
                assert len(step["pos"]) == 3
                assert 0.0 <= step["fuel"] <= 1.0
                assert 0.0 <= step["battery"] <= 1.0
                act = step["action"]
                assert act["type"] in ACTION_TYPES
                if act["type"] == "hop":
                    assert len(act["target"]) == 2
                if step["wm_pos"] is not None:
                    assert len(step["wm_pos"]) == 3
                    assert isinstance(step["wm_error"], (int, float))
                assert step["event"] in EVENTS


def test_example_matches_schema():
    doc = json.loads(EXAMPLE.read_text())
    validate_episode_doc(doc)


def test_schema_rejects_mismatched_fleet_size():
    doc = json.loads(EXAMPLE.read_text())
    bad = copy.deepcopy(doc)
    bad["meta"]["fleet_size"] = 5
    with pytest.raises(AssertionError):
        validate_episode_doc(bad)


def test_schema_rejects_invalid_action_type():
    doc = json.loads(EXAMPLE.read_text())
    bad = copy.deepcopy(doc)
    bad["episodes"][0]["drones"][0]["steps"][0]["action"]["type"] = "hover"
    with pytest.raises(AssertionError):
        validate_episode_doc(bad)
