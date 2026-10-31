"""Tests for CleanRL-style MAPPO trainer, MBPO imagination rollouts, and CSV summary CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from selene_sim.env import VectorFleetEnv
from selene_sim.mappo import (
    Actor,
    Critic,
    MAPPOConfig,
    export_episodes_json,
    imagine_rollout,
    train_mappo,
    training_summary,
)
from selene_sim.world_model import TransitionBuffer, WorldModelEnsemble
from tests.test_episode_schema import validate_episode_doc

validate_episodes_document = validate_episode_doc

EXPECTED_CSV_COLUMNS = [
    "update",
    "env_steps",
    "wall_time_s",
    "episode_reward",
    "crashes",
    "craters_visited",
    "crater_overlap",
    "relay_conflicts",
    "entropy",
    "value_loss",
    "wm_error",
]


def test_mappo_config_and_actor_critic_masking() -> None:
    """Verify MAPPOConfig quick/full defaults and invalid-action masking in Actor."""
    quick_cfg = MAPPOConfig(quick=True)
    assert quick_cfg.n_hoppers == 2
    assert quick_cfg.n_craters == 6
    assert quick_cfg.max_steps == 20
    assert quick_cfg.num_envs == 64
    assert quick_cfg.rollout_steps == 20
    assert quick_cfg.hidden_dim == 64
    assert quick_cfg.total_updates == 40
    assert quick_cfg.time_budget_s == 120.0

    full_cfg = MAPPOConfig(quick=False)
    assert full_cfg.n_hoppers == 4
    assert full_cfg.n_craters == 12
    assert full_cfg.max_steps == 30
    assert full_cfg.num_envs == 128
    assert full_cfg.rollout_steps == 30
    assert full_cfg.hidden_dim == 128
    assert full_cfg.total_updates == 200
    assert full_cfg.time_budget_s == 1200.0

    obs_dim = 31
    state_dim = 28
    n_actions = 9
    actor = Actor(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=64)
    critic = Critic(global_state_dim=state_dim, hidden_dim=64)

    obs = torch.randn(4, obs_dim)
    state = torch.randn(4, state_dim)
    mask = torch.zeros(4, n_actions, dtype=torch.bool)
    mask[:, 7] = True  # Only wait action is valid

    logits = actor(obs, mask)
    assert logits.shape == (4, n_actions)
    assert torch.all(logits[:, :7] <= -1e7)
    assert torch.all(logits[:, 8:] <= -1e7)

    actions, log_prob, entropy = actor.get_action(obs, mask)
    assert torch.all(actions == 7)
    assert torch.all(torch.isfinite(log_prob))
    assert torch.all(torch.isfinite(entropy))

    values = critic(state)
    assert values.shape == (4,)
    assert torch.all(torch.isfinite(values))


def test_stage2_quick_learning_and_stage3_summary(tmp_path: Path) -> None:
    """Stage 2/3 check: --quick MAPPO improves reward by > +15, keeps crashes low, and logs CSV."""
    csv_path = tmp_path / "training_log.csv"
    ckpt_dir = tmp_path / "checkpoints"

    cfg = MAPPOConfig(
        quick=True,
        total_updates=35,
        log_interval=1,
        checkpoint_interval=10,
        seed=42,
    )

    t0 = time.perf_counter()
    result = train_mappo(config=cfg, csv_path=csv_path, checkpoint_dir=ckpt_dir)
    elapsed_s = time.perf_counter() - t0

    assert elapsed_s < 20.0, f"Expected --quick 35 updates in < 20s, took {elapsed_s:.2f}s"
    assert csv_path.exists()
    assert (ckpt_dir / "actor.pt").exists()
    assert (ckpt_dir / "critic.pt").exists()
    assert (ckpt_dir / "config.json").exists()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == EXPECTED_CSV_COLUMNS
        rows = list(reader)

    assert len(rows) == 35
    first_3 = rows[:3]
    last_3 = rows[-3:]

    initial_reward = float(np.mean([float(r["episode_reward"]) for r in first_3]))
    final_reward = float(np.mean([float(r["episode_reward"]) for r in last_3]))
    reward_gain = final_reward - initial_reward

    initial_crashes = float(np.mean([float(r["crashes"]) for r in first_3]))
    final_crashes = float(np.mean([float(r["crashes"]) for r in last_3]))

    initial_overlap = float(np.mean([float(r["crater_overlap"]) for r in first_3]))
    final_overlap = float(np.mean([float(r["crater_overlap"]) for r in last_3]))

    initial_wm = float(np.mean([float(r["wm_error"]) for r in first_3]))
    final_wm = float(np.mean([float(r["wm_error"]) for r in last_3]))

    print(
        f"\n[Stage 2 --quick] elapsed={elapsed_s:.2f}s | "
        f"initial_reward={initial_reward:.2f} -> final_reward={final_reward:.2f} "
        f"(+{reward_gain:.2f}) | crashes={initial_crashes:.2f} -> {final_crashes:.2f} | "
        f"overlap={initial_overlap:.2f} -> {final_overlap:.2f} | "
        f"wm_error={initial_wm:.4f} -> {final_wm:.4f}"
    )

    assert reward_gain > 15.0, (
        f"Expected > +15.0 reward improvement in --quick mode, got +{reward_gain:.2f} "
        f"({initial_reward:.2f} -> {final_reward:.2f})"
    )
    assert final_crashes <= 0.15
    assert final_crashes <= initial_crashes + 0.05
    assert final_overlap < initial_overlap
    assert final_wm < initial_wm

    # Verify training_summary() Python helper for Argon orchestrator
    summary = training_summary(csv_path)
    assert summary["num_updates"] == 35
    assert len(summary["episode_rewards"]) == 35
    assert summary["reward_improvement"] > 15.0
    assert summary["stage2_pass"] is True

    # Verify scripts/training_summary.py CLI output
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "training_summary.py"
    proc = subprocess.run(
        [sys.executable, str(script_path), str(csv_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = proc.stdout
    print("\n" + stdout)
    assert "episode_reward" in stdout
    assert "crashes" in stdout
    assert "craters_visited" in stdout
    assert "crater_overlap" in stdout
    assert "relay_conflicts" in stdout
    assert "wm_error" in stdout
    assert "PASS" in stdout

    # Verify Episode Export Schema Check using the trained actor
    episodes_path = tmp_path / "episodes.json"
    doc = export_episodes_json(
        actor=result["actor"],
        config=cfg,
        path=episodes_path,
        source="learned",
        world_model=result["world_model"],
        episode_rewards=summary["episode_rewards"],
    )
    assert episodes_path.exists()
    loaded_doc = json.loads(episodes_path.read_text(encoding="utf-8"))
    validate_episodes_document(doc)
    validate_episodes_document(loaded_doc)


def test_issue13_mbpo_imagine_rollout_uncertainty_gating() -> None:
    """Issue #13 check: imagine_rollout generates k-step rollouts and masks high-uncertainty branches."""
    env = VectorFleetEnv(num_envs=8, quick=True, seed=42, record_transitions=True)
    obs, info = env.reset(seed=42)

    for _ in range(15):
        env.step(env.sample_random_actions())
    trans = env.get_transitions(clear=True)

    buf = TransitionBuffer(max_size=5000)
    buf.add(
        obs=trans["obs"],
        actions=trans["actions"],
        next_obs=trans["next_obs"],
        rewards=trans["rewards"],
        crashed=trans["crashed"],
    )

    wm = WorldModelEnsemble(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden_dim=64,
        device="cpu",
        seed=42,
    )
    wm.train_epochs(buf, epochs=8, batch_size=256, seed=42)

    actor = Actor(obs_dim=env.obs_dim, n_actions=env.n_actions, hidden_dim=64)
    start_obs = obs.reshape(-1, env.obs_dim).copy()
    start_masks = info["action_masks"].reshape(-1, env.n_actions).copy()
    start_gs = np.repeat(info["global_state"], env.n_hoppers, axis=0)

    # Corrupt half of the starting observations into extreme out-of-distribution values
    half = start_obs.shape[0] // 2
    start_obs[half:] = 25.0

    # First run with a very permissive threshold so all branches stay active at step 0
    rollout_all = imagine_rollout(
        world_model=wm,
        actor=actor,
        start_obs=start_obs,
        start_global_state=start_gs,
        action_masks=start_masks,
        k=3,
        uncertainty_threshold=100.0,
    )
    assert rollout_all["obs"].shape == (3, start_obs.shape[0], env.obs_dim)
    assert rollout_all["next_obs"].shape == (3, start_obs.shape[0], env.obs_dim)
    assert rollout_all["actions"].shape == (3, start_obs.shape[0])
    assert rollout_all["rewards"].shape == (3, start_obs.shape[0])
    assert rollout_all["uncertainties"].shape == (3, start_obs.shape[0])
    assert rollout_all["valid_mask"].shape == (3, start_obs.shape[0])
    assert np.all(rollout_all["valid_mask"][0])

    # Set threshold between in-distribution uncertainty and OOD uncertainty
    in_dist_unc = float(np.median(rollout_all["uncertainties"][0, :half]))
    ood_unc = float(np.median(rollout_all["uncertainties"][0, half:]))
    assert ood_unc > in_dist_unc

    threshold = 0.5 * (in_dist_unc + ood_unc)
    rollout_gated = imagine_rollout(
        world_model=wm,
        actor=actor,
        start_obs=start_obs,
        start_global_state=start_gs,
        action_masks=start_masks,
        k=3,
        uncertainty_threshold=threshold,
    )

    # Branches where uncertainty > threshold must be masked out, and remain masked on later steps
    exceeded_step0 = rollout_gated["uncertainties"][0] > threshold
    assert np.any(exceeded_step0)
    assert np.any(~exceeded_step0)
    assert not np.any(rollout_gated["valid_mask"][0, exceeded_step0])
    assert not np.any(rollout_gated["valid_mask"][1, exceeded_step0])
    assert not np.any(rollout_gated["valid_mask"][2, exceeded_step0])
