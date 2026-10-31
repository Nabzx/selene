import json
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[3] / "data" / "episodes.example.json"

STEP_FIELDS = {"t", "pos", "yaw", "phase", "fuel", "battery", "temp_c", "action",
               "reward", "wm_pos", "wm_error", "event"}


def test_example_matches_schema():
    doc = json.loads(EXAMPLE.read_text())
    assert doc["schema_version"] == 1
    assert doc["meta"]["source"] in {"trained", "scripted"}
    crater_ids = {c["id"] for c in doc["craters"]}
    for ep in doc["episodes"]:
        assert ep["outcome"] in {"completed", "crashed", "battery", "timeout"}
        assert set(ep["craters_visited"]) <= crater_ids
        for step in ep["steps"]:
            assert set(step) == STEP_FIELDS
            assert step["phase"] in {"ground", "ascent", "coast", "descent"}
            assert len(step["pos"]) == 3
