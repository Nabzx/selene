"""Unit tests for the hackathon demo bundle exporter (scripts/export_demo_bundle.py)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.export_demo_bundle import generate_demo_bundle, main
from tests.test_episode_schema import (
    validate_episode_doc as validate_episodes_document,
)


def test_generate_demo_bundle_creates_all_artefacts_and_matches_schema(
    tmp_path: Path,
) -> None:
    """Verify bundle files exist, JSON schemas validate, and episode progression holds."""
    summary = generate_demo_bundle(tmp_path, seed=42, quick=True)

    episodes_path = tmp_path / "episodes.json"
    verify_path = tmp_path / "verification_cloud.json"
    wm_path = tmp_path / "world_model_map.json"

    assert episodes_path.is_file()
    assert verify_path.is_file()
    assert wm_path.is_file()

    # 1. Validate episodes.json against the canonical episode schema
    episodes_doc = json.loads(episodes_path.read_text(encoding="utf-8"))
    validate_episodes_document(episodes_doc)

    assert episodes_doc["meta"]["region"] == "Daedalus"
    assert episodes_doc["meta"]["fleet_size"] == 4
    episodes = episodes_doc["episodes"]
    assert len(episodes) == 3

    ep0, ep1, ep2 = episodes[0], episodes[1], episodes[2]
    assert ep0["phase"] == "early_training"
    assert ep1["phase"] == "mid_training"
    assert ep2["phase"] == "final_policy"

    # Early training suffers at least one steep-slope crash and one relay conflict
    assert ep0["metrics"]["crashes"] >= 1
    assert ep0["metrics"]["relay_conflicts"] >= 1
    assert ep0["outcome"] == "crashed"

    # Mid training after Argon Eureka reward update eliminates crashes and visits more craters
    assert ep1["metrics"]["crashes"] == 0
    assert ep1["metrics"]["relay_conflicts"] == 0
    assert len(ep1["craters_visited"]) > len(ep0["craters_visited"])
    assert ep1["total_reward"] > ep0["total_reward"]

    # Final policy splits craters cleanly across the 4-hopper fleet with zero crashes/conflicts
    assert ep2["metrics"]["crashes"] == 0
    assert ep2["metrics"]["relay_conflicts"] == 0
    assert ep2["metrics"]["crater_overlap"] == 0
    assert len(ep2["craters_visited"]) >= 4
    assert len(ep2["craters_visited"]) > len(ep0["craters_visited"])
    assert ep2["total_reward"] > ep0["total_reward"]
    assert ep2["total_reward"] > ep1["total_reward"]

    # Verify 0.5 s parabolic keyframe interpolation is present on flight steps
    h0_steps = ep2["drones"][0]["steps"]
    assert len(h0_steps) > 10
    phases_seen = {step["phase"] for step in h0_steps}
    assert {"ascent", "coast", "descent", "ground"} <= phases_seen

    # 2. Honest source tagging check: heuristic_fallback when no checkpoint is loaded
    assert summary["source"] == "heuristic_fallback"
    for ep in episodes:
        assert ep["source"] == "heuristic_fallback"

    # 3. Validate verification_cloud.json (1,000 Monte Carlo rollouts + 50-run 3D trajectory fan)
    verify_doc = json.loads(verify_path.read_text(encoding="utf-8"))
    assert verify_doc["n_runs"] == 1000
    assert verify_doc["num_runs"] == 1000
    assert verify_doc["pass_rate"] >= 0.99
    assert verify_doc["pass"] is True
    assert verify_doc["source"] == "heuristic_fallback"

    cloud = verify_doc["trajectory_cloud"]
    assert isinstance(cloud, list)
    assert len(cloud) == 50
    assert len(cloud[0]["hoppers"]) == 4
    assert len(cloud[0]["hoppers"][0]["path_xyz"]) >= 2
    assert len(cloud[0]["hoppers"][0]["path_xyz"][0]) == 3

    # 4. Validate world_model_map.json (5-MLP ensemble error and epistemic uncertainty grid)
    wm_doc = json.loads(wm_path.read_text(encoding="utf-8"))
    assert isinstance(wm_doc["overall_val_mse"], float)
    assert len(wm_doc["crater_metrics"]) == 6
    assert len(wm_doc["uncertainty_grid"]) == 20
    assert len(wm_doc["uncertainty_grid"][0]) == 20
    assert len(wm_doc["mean_error_grid"]) == 20
    assert len(wm_doc["visit_count_grid"]) == 20
    assert len(wm_doc["top_error_cells"]) == 5

    visited_unc = [
        c["epistemic_uncertainty"]
        for c in wm_doc["crater_metrics"]
        if c["visit_count"] > 0
    ]
    unvisited_unc = [
        c["epistemic_uncertainty"]
        for c in wm_doc["crater_metrics"]
        if c["visit_count"] == 0
    ]
    assert visited_unc and unvisited_unc
    assert sum(unvisited_unc) / len(unvisited_unc) > sum(visited_unc) / len(visited_unc)


def test_honest_source_tagging_with_checkpoint_and_cli(tmp_path: Path) -> None:
    """Verify source is tagged 'learned' only when a valid checkpoint is loaded, and test CLI."""
    ckpt_path = tmp_path / "policy_ckpt.pt"
    torch.save({"actor_state_dict": {"weight": torch.ones(4, 4)}, "update": 20}, ckpt_path)

    learned_dir = tmp_path / "bundle_learned"
    summary_learned = generate_demo_bundle(
        learned_dir,
        seed=42,
        quick=True,
        checkpoint_path=ckpt_path,
    )
    assert summary_learned["source"] == "learned"
    episodes_learned = json.loads(
        (learned_dir / "episodes.json").read_text(encoding="utf-8")
    )
    for ep in episodes_learned["episodes"]:
        assert ep["source"] == "learned"

    cli_dir = tmp_path / "bundle_cli"
    rc = main(["--output-dir", str(cli_dir), "--seed", "42", "--quick"])
    assert rc == 0
    assert (cli_dir / "episodes.json").is_file()
    assert (cli_dir / "verification_cloud.json").is_file()
    assert (cli_dir / "world_model_map.json").is_file()
