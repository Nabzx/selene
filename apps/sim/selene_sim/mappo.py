"""Single-file CleanRL-style MAPPO trainer, short-horizon MBPO rollouts, and episode exporter.

Implements Centralised Training with Decentralised Execution (CTDE) Multi-Agent PPO
on `VectorFleetEnv` with:
- Shared 2-layer `Tanh` `Actor` with orthogonal weight initialisation and `-1e8`
  invalid-action masking before `Categorical(logits=...)`.
- Centralised 2-layer `Tanh` `Critic` operating on the compact global fleet state
  with running return normalisation (`ValueNormaliser`).
- Periodic `WorldModelEnsemble` updates and Issue #13 short-horizon MBPO
  imagination rollouts (`imagine_rollout`) gated by ensemble epistemic uncertainty.
- CSV progress logging, periodic checkpointing, `training_summary()` for the Argon
  orchestrator, and `export_episodes_json()` matching `docs/data/EPISODE_SCHEMA.md`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from selene_sim.env import FleetEnv, VectorFleetEnv
from selene_sim.reward import reward_versions
from selene_sim.sim import SimConfig, TerrainGrid
from selene_sim.world_model import (
    TransitionBuffer,
    WorldModelEnsemble,
    select_device,
)

CSV_COLUMNS: tuple[str, ...] = (
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
)


@dataclass
class MAPPOConfig:
    """Hyperparameters and environment settings for CleanRL-style MAPPO training."""

    quick: bool = False
    n_hoppers: int = 4
    n_craters: int = 12
    max_steps: int = 30
    num_envs: int = 128
    rollout_steps: int = 30
    hidden_dim: int = 128
    total_updates: int = 200
    time_budget_s: float = 1200.0
    seed: int = 42
    lr: float = 3e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    num_minibatches: int = 4
    log_interval: int = 1
    checkpoint_interval: int = 10
    wm_update_interval: int = 5
    wm_epochs: int = 5
    use_mbpo: bool = True
    mbpo_horizon: int = 2
    mbpo_uncertainty_threshold: float = 0.5
    mbpo_rollout_batch: int = 64
    mbpo_loss_weight: float = 0.1
    device: str | None = None
    curriculum: bool = True
    curriculum_warmup_updates: int = 40
    lr_schedule: str = "cosine"
    entropy_decay: bool = True
    ent_coef_min: float = 0.003
    terrain_path: str | None = None

    def __post_init__(self) -> None:
        if self.quick:
            if self.n_hoppers == 4:
                self.n_hoppers = 2
            if self.n_craters == 12:
                self.n_craters = 6
            if self.max_steps == 30:
                self.max_steps = 20
            if self.num_envs == 128:
                self.num_envs = 64
            if self.rollout_steps == 30:
                self.rollout_steps = 20
            if self.hidden_dim == 128:
                self.hidden_dim = 64
            if self.total_updates == 200:
                self.total_updates = 40
            if self.time_budget_s == 1200.0:
                self.time_budget_s = 120.0
            if self.curriculum_warmup_updates == 40:
                self.curriculum_warmup_updates = 15


def _layer_init(
    layer: nn.Linear,
    std: float = math.sqrt(2.0),
    bias_const: float = 0.0,
) -> nn.Linear:
    """Apply CleanRL orthogonal weight initialisation and constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def resolve_mappo_device(device_spec: str | torch.device | None = None) -> torch.device:
    """Select execution device for MAPPO networks.

    For compact 64-128 unit MLPs stepped alongside NumPy `VectorFleetEnv` arrays on host
    memory, CPU avoids per-step host-to-device transfer overhead unless `"mps"` or
    `"auto"` is explicitly requested.
    """
    if isinstance(device_spec, torch.device):
        return device_spec
    if isinstance(device_spec, str):
        if device_spec == "auto":
            return select_device(prefer_fastest=True)
        return torch.device(device_spec)
    return torch.device("cpu")


class ValueNormaliser:
    """Running mean and variance normaliser for centralised critic targets."""

    def __init__(self, eps: float = 1e-5) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = eps
        self.eps: float = eps

    def update(self, values: torch.Tensor | np.ndarray) -> None:
        arr = (
            values.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if isinstance(values, torch.Tensor)
            else np.asarray(values, dtype=np.float64).reshape(-1)
        )
        if arr.size == 0:
            return
        batch_mean = float(np.mean(arr))
        batch_var = float(np.var(arr))
        batch_count = float(arr.size)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta * delta) * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = max(1e-4, m2 / tot_count)
        self.count = tot_count

    @property
    def std(self) -> float:
        return math.sqrt(self.var + self.eps)

    def normalise(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.mean) / self.std

    def denormalise(self, preds: torch.Tensor) -> torch.Tensor:
        return preds * self.std + self.mean


class Actor(nn.Module):
    """Shared 2-layer Tanh policy network with orthogonal init and invalid-action masking."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            _layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(self.hidden_dim, self.n_actions), std=0.01),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute action logits and mask invalid actions with `-1e8`."""
        logits = self.net(obs)
        if action_mask is not None:
            mask = action_mask.to(dtype=torch.bool, device=logits.device)
            # Ensure at least one valid action per row (fallback to wait action C + 1)
            none_valid = ~torch.any(mask, dim=-1, keepdim=True)
            if torch.any(none_valid):
                mask = mask.clone()
                wait_idx = max(0, self.n_actions - 2)
                mask[..., wait_idx] = mask[..., wait_idx] | none_valid.squeeze(-1)
            logits = logits.masked_fill(~mask, -1e8)
        return logits

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(action, log_prob, entropy)` under the masked categorical distribution."""
        logits = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        action_mask: np.ndarray | torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select discrete actions for NumPy or Torch observations."""
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        mask_t = (
            torch.as_tensor(action_mask, dtype=torch.bool, device=device)
            if action_mask is not None
            else None
        )
        action_t, _, _ = self.get_action(
            obs_t, action_mask=mask_t, deterministic=deterministic
        )
        return action_t.cpu().numpy().astype(np.int32)


class Critic(nn.Module):
    """Centralised 2-layer Tanh value network over the compact global fleet state."""

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.global_state_dim = int(global_state_dim)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            _layer_init(nn.Linear(self.global_state_dim, self.hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimates of shape `global_state.shape[:-1]`."""
        return self.net(global_state).squeeze(-1)


def _infer_action_mask_from_obs(
    obs_batch: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    """Reconstruct valid action masks from flat observations during imagined rollouts."""
    batch_n, obs_dim = obs_batch.shape
    c = max(1, n_actions - 3)
    scan_idx = c
    wait_idx = c + 1
    send_idx = c + 2

    mask = np.zeros((batch_n, n_actions), dtype=bool)
    mask[:, wait_idx] = True

    fuel_norm = obs_batch[:, 2]
    battery_norm = obs_batch[:, 3]
    unsent_norm = obs_batch[:, 5]
    relay_open = obs_batch[:, -2] > 0.5
    alive = (fuel_norm > 0.02) & (battery_norm > 0.02)

    # Infer n_hoppers from obs_dim = 5 + 4*H + 3*C
    rem = obs_dim - 5 - 3 * c
    n_hoppers = max(1, rem // 4) if rem >= 4 else 2
    crater_offset = 6 + n_hoppers

    if crater_offset + 3 * c <= obs_dim:
        crater_feats = obs_batch[:, crater_offset : crater_offset + 3 * c].reshape(
            batch_n, c, 3
        )
        dist_norm = crater_feats[:, :, 0]
        done_flag = crater_feats[:, :, 2] > 0.5

        can_hop = (
            alive[:, None]
            & (fuel_norm[:, None] > 0.18)
            & (~done_flag)
            & (dist_norm > 0.005)
            & (dist_norm <= 0.30)
        )
        mask[:, :c] = can_hop

        at_unscanned = np.any((dist_norm <= 0.015) & (~done_flag), axis=-1)
        mask[:, scan_idx] = alive & at_unscanned
    else:
        mask[:, :c] = alive[:, None] & (fuel_norm[:, None] > 0.2)

    mask[:, send_idx] = alive & (unsent_norm > 0.05) & relay_open
    return mask


def imagine_rollout(
    world_model: WorldModelEnsemble,
    actor: Actor,
    start_obs: np.ndarray | torch.Tensor,
    start_global_state: np.ndarray | torch.Tensor | None = None,
    action_masks: np.ndarray | torch.Tensor | None = None,
    k: int = 2,
    uncertainty_threshold: float = 0.5,
) -> dict[str, Any]:
    """Perform a short-horizon MBPO imagination rollout inside `WorldModelEnsemble`.

    Steps `k` steps forward from `start_obs` using `actor` and `world_model.predict(...)`,
    and masks out branches where ensemble `epistemic_uncertainty > uncertainty_threshold`
    (as well as subsequent steps along already-truncated branches).
    """
    horizon = max(1, int(k))
    threshold = float(uncertainty_threshold)
    actor_device = next(actor.parameters()).device

    obs_np = (
        start_obs.detach().cpu().numpy().astype(np.float32)
        if isinstance(start_obs, torch.Tensor)
        else np.asarray(start_obs, dtype=np.float32)
    )
    obs_np = obs_np.reshape(-1, actor.obs_dim).copy()
    batch_n = obs_np.shape[0]

    if action_masks is not None:
        mask_np = (
            action_masks.detach().cpu().numpy().astype(bool)
            if isinstance(action_masks, torch.Tensor)
            else np.asarray(action_masks, dtype=bool)
        ).reshape(batch_n, actor.n_actions)
    else:
        mask_np = _infer_action_mask_from_obs(obs_np, actor.n_actions)

    if start_global_state is not None:
        gs_np = (
            start_global_state.detach().cpu().numpy().astype(np.float32)
            if isinstance(start_global_state, torch.Tensor)
            else np.asarray(start_global_state, dtype=np.float32)
        )
        if gs_np.ndim == 1:
            gs_np = np.repeat(gs_np[None, :], batch_n, axis=0)
        elif gs_np.shape[0] != batch_n and batch_n % gs_np.shape[0] == 0:
            gs_np = np.repeat(gs_np, batch_n // gs_np.shape[0], axis=0)
        else:
            gs_np = gs_np.reshape(batch_n, -1).copy()
    else:
        gs_np = None

    obs_traj = np.zeros((horizon, batch_n, actor.obs_dim), dtype=np.float32)
    next_obs_traj = np.zeros((horizon, batch_n, actor.obs_dim), dtype=np.float32)
    actions_traj = np.zeros((horizon, batch_n), dtype=np.int32)
    log_probs_traj = np.zeros((horizon, batch_n), dtype=np.float32)
    rewards_traj = np.zeros((horizon, batch_n), dtype=np.float32)
    crash_probs_traj = np.zeros((horizon, batch_n), dtype=np.float32)
    uncertainties_traj = np.zeros((horizon, batch_n), dtype=np.float32)
    valid_mask_traj = np.zeros((horizon, batch_n), dtype=bool)
    masks_traj = np.zeros((horizon, batch_n, actor.n_actions), dtype=bool)
    gs_traj = (
        np.zeros((horizon, batch_n, gs_np.shape[-1]), dtype=np.float32)
        if gs_np is not None
        else None
    )

    active_branches = np.ones(batch_n, dtype=bool)
    curr_obs = obs_np
    curr_mask = mask_np
    curr_gs = gs_np

    for step_idx in range(horizon):
        if step_idx > 0:
            curr_mask = _infer_action_mask_from_obs(curr_obs, actor.n_actions)

        with torch.no_grad():
            obs_t = torch.as_tensor(curr_obs, dtype=torch.float32, device=actor_device)
            mask_t = torch.as_tensor(curr_mask, dtype=torch.bool, device=actor_device)
            act_t, logp_t, _ = actor.get_action(obs_t, action_mask=mask_t)
            act_step = act_t.cpu().numpy().astype(np.int32)
            logp_step = logp_t.cpu().numpy().astype(np.float32)

        pred_next_obs, pred_rew, pred_crash, pred_unc = world_model.predict(
            curr_obs, act_step
        )
        pred_next_obs = np.asarray(pred_next_obs, dtype=np.float32).reshape(
            batch_n, actor.obs_dim
        )
        pred_rew = np.asarray(pred_rew, dtype=np.float32).reshape(batch_n)
        pred_crash = np.asarray(pred_crash, dtype=np.float32).reshape(batch_n)
        pred_unc = np.asarray(pred_unc, dtype=np.float32).reshape(batch_n)

        step_valid = (
            active_branches
            & (pred_unc <= threshold)
            & np.all(np.isfinite(pred_next_obs), axis=-1)
        )

        obs_traj[step_idx] = curr_obs
        next_obs_traj[step_idx] = pred_next_obs
        actions_traj[step_idx] = act_step
        log_probs_traj[step_idx] = logp_step
        rewards_traj[step_idx] = pred_rew
        crash_probs_traj[step_idx] = pred_crash
        uncertainties_traj[step_idx] = pred_unc
        valid_mask_traj[step_idx] = step_valid
        masks_traj[step_idx] = curr_mask
        if gs_traj is not None and curr_gs is not None:
            gs_traj[step_idx] = curr_gs

        # Once a branch exceeds the epistemic uncertainty threshold or crashes, truncate it
        active_branches = step_valid & (pred_crash < 0.5)
        curr_obs = np.where(step_valid[:, None], pred_next_obs, curr_obs)

    out: dict[str, Any] = {
        "obs": obs_traj,
        "next_obs": next_obs_traj,
        "actions": actions_traj,
        "log_probs": log_probs_traj,
        "rewards": rewards_traj,
        "crash_probs": crash_probs_traj,
        "uncertainties": uncertainties_traj,
        "valid_mask": valid_mask_traj,
        "action_masks": masks_traj,
        "active_ratio": float(np.mean(valid_mask_traj)),
    }
    if gs_traj is not None:
        out["global_states"] = gs_traj
    return out


def _compute_ctde_coordination_shaping(
    obs_prev: np.ndarray,
    act_np: np.ndarray,
    n_hoppers: int,
    n_craters: int,
    send_action_idx: int,
) -> np.ndarray:
    """Compute centralised training credit-assignment shaping for crater splitting and TDMA relay.

    Used only for GAE advantage estimation during centralised training; logged CSV metrics
    and evaluation rewards always report the unshaped ground-truth environment rewards.
    """
    num_envs = act_np.shape[0]
    shaping = np.zeros((num_envs, n_hoppers), dtype=np.float32)
    crater_offset = 6 + n_hoppers

    for i in range(n_hoppers):
        a_i = act_np[:, i]
        is_hop = (a_i >= 0) & (a_i < n_craters)
        safe_a = np.clip(a_i, 0, n_craters - 1)

        claimed_flags = obs_prev[
            :, i, crater_offset + 1 : crater_offset + 3 * n_craters : 3
        ]
        was_claimed = (
            np.take_along_axis(claimed_flags, safe_a[:, None], axis=1).squeeze(1) > 0.5
        )
        shaping[:, i] -= np.where(is_hop & was_claimed, 12.0, 0.0)

        # Assign each hopper a primary crater sector to break parameter-sharing symmetry at t=0
        preferred_sector = ((safe_a * n_hoppers) // n_craters) == i
        shaping[:, i] += np.where(is_hop & (~was_claimed) & preferred_sector, 4.0, 0.0)
        shaping[:, i] -= np.where(is_hop & (~preferred_sector), 4.0, 0.0)

        for j in range(i):
            same_hop = is_hop & (act_np[:, j] == a_i)
            shaping[:, i] -= np.where(same_hop, 15.0, 0.0)

        # Relay transmission coordination shaping:
        # Enforce solitary clean send and penalise concurrent transmission attempts
        is_send_i = (a_i == send_action_idx)
        unsent_i = obs_prev[:, i, 5] * 5.0

        for j in range(n_hoppers):
            if j == i:
                continue
            is_send_j = (act_np[:, j] == send_action_idx)
            conflict_pair = is_send_i & is_send_j
            # Heavy penalty for simultaneous transmission collisions
            shaping[:, i] -= np.where(conflict_pair, 35.0, 0.0)

            # Buffer priority yield rule: if hopper j also sends and has more unsent data, hopper i must yield
            unsent_j = obs_prev[:, j, 5] * 5.0
            yield_violation = conflict_pair & (unsent_i < unsent_j)
            shaping[:, i] -= np.where(yield_violation, 20.0, 0.0)

        # Bonus for successful solitary transmission when carrying data
        other_sends = np.zeros(num_envs, dtype=bool)
        for j in range(n_hoppers):
            if j != i:
                other_sends |= (act_np[:, j] == send_action_idx)
        solo_send = is_send_i & (~other_sends) & (unsent_i >= 0.8)
        shaping[:, i] += np.where(solo_send, 10.0, 0.0)

        # Stagger relay transmissions across hoppers so hopper 0 transmits on 1+ scans
        # while hoppers i > 0 batch >= 2 scans before transmitting
        if i > 0:
            early_send = is_send_i & (unsent_i < 1.5)
            shaping[:, i] -= np.where(early_send, 25.0, 0.0)

    return shaping


def _save_checkpoints(
    checkpoint_dir: Path,
    actor: Actor,
    critic: Critic,
    config: MAPPOConfig,
) -> None:
    """Save `actor.pt`, `critic.pt`, and `config.json` into `checkpoint_dir`."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), checkpoint_dir / "actor.pt")
    torch.save(critic.state_dict(), checkpoint_dir / "critic.pt")
    cfg_dict = asdict(config)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8"
    )


def train_mappo(
    config: MAPPOConfig | None = None,
    csv_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run vectorised CleanRL-style MAPPO training with periodic world model updates and MBPO."""
    cfg = config if config is not None else MAPPOConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = resolve_mappo_device(cfg.device)

    # Optional NASA DEM terrain ingestion or custom terrain grid
    custom_terrain: TerrainGrid | None = None
    if cfg.terrain_path is not None:
        t_path = Path(cfg.terrain_path)
        if t_path.exists():
            custom_terrain = TerrainGrid.from_npz(t_path)

    # Configure base physical noise settings for curriculum ramp
    nominal_sim_config = SimConfig()
    initial_sim_config = SimConfig(
        noise_enabled=True,
        thrust_sigma_rel=0.0 if cfg.curriculum else nominal_sim_config.thrust_sigma_rel,
        scatter_sigma_rel=0.0 if cfg.curriculum else nominal_sim_config.scatter_sigma_rel,
        scatter_sigma_base_m=0.0 if cfg.curriculum else nominal_sim_config.scatter_sigma_base_m,
        slope_noise_sigma_deg=0.0 if cfg.curriculum else nominal_sim_config.slope_noise_sigma_deg,
        dust_max_extra_burn=0.0 if cfg.curriculum else nominal_sim_config.dust_max_extra_burn,
    )

    env = VectorFleetEnv(
        num_envs=cfg.num_envs,
        n_hoppers=cfg.n_hoppers,
        n_craters=cfg.n_craters,
        max_steps=cfg.max_steps,
        quick=cfg.quick,
        seed=cfg.seed,
        terrain=custom_terrain,
        config=initial_sim_config,
        record_transitions=True,
    )

    actor = Actor(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    critic = Critic(
        global_state_dim=env.global_state_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=cfg.lr,
        eps=1e-5,
    )
    val_norm = ValueNormaliser()

    world_model = WorldModelEnsemble(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden_dim=cfg.hidden_dim,
        device="cpu",
        seed=cfg.seed,
    )
    wm_buffer = TransitionBuffer(max_size=6000)
    fixed_val_batch: dict[str, np.ndarray] | None = None
    current_wm_error: float = 1.0

    csv_file_path = Path(csv_path) if csv_path is not None else None
    if csv_file_path is not None:
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

    ckpt_path = Path(checkpoint_dir) if checkpoint_dir is not None else None

    obs, info = env.reset(seed=cfg.seed)
    masks = info["action_masks"]
    gstate = info["global_state"]

    num_envs = env.num_envs
    n_hoppers = env.n_hoppers
    n_craters = env.n_craters
    rollout_steps = cfg.rollout_steps
    batch_size = num_envs * rollout_steps * n_hoppers
    minibatch_size = max(1, batch_size // max(1, cfg.num_minibatches))

    history: list[dict[str, float | int]] = []
    recent_metrics: list[dict[str, float | int]] = []
    total_env_steps = 0
    start_wall = time.perf_counter()

    for update in range(1, cfg.total_updates + 1):
        # Progress fraction from 0.0 (start) to 1.0 (end)
        progress = (update - 1.0) / float(max(1, cfg.total_updates - 1)) if cfg.total_updates > 1 else 0.0

        # Learning rate schedule
        if cfg.lr_schedule == "cosine":
            lr_min = cfg.lr * 0.1
            lr_now = lr_min + 0.5 * (cfg.lr - lr_min) * (1.0 + math.cos(math.pi * progress))
        else:
            frac = 1.0 - (update - 1.0) / float(max(1, cfg.total_updates))
            lr_now = cfg.lr * max(0.1, frac)

        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Dynamic entropy decay schedule
        if cfg.entropy_decay:
            ent_coef_now = cfg.ent_coef_min + 0.5 * (cfg.ent_coef - cfg.ent_coef_min) * (
                1.0 + math.cos(math.pi * progress)
            )
        else:
            ent_coef_now = cfg.ent_coef

        # Curriculum noise ramp: ramp noise parameters smoothly from 0 to nominal over warmup updates
        if cfg.curriculum and cfg.curriculum_warmup_updates > 0:
            c_factor = min(1.0, float(update) / float(cfg.curriculum_warmup_updates))
            env.config = SimConfig(
                noise_enabled=True,
                thrust_sigma_rel=c_factor * nominal_sim_config.thrust_sigma_rel,
                scatter_sigma_rel=c_factor * nominal_sim_config.scatter_sigma_rel,
                scatter_sigma_base_m=c_factor * nominal_sim_config.scatter_sigma_base_m,
                slope_noise_sigma_deg=c_factor * nominal_sim_config.slope_noise_sigma_deg,
                dust_max_extra_burn=c_factor * nominal_sim_config.dust_max_extra_burn,
            )

        b_obs = torch.zeros(
            (rollout_steps, num_envs, n_hoppers, env.obs_dim),
            dtype=torch.float32,
            device=device,
        )
        b_masks = torch.zeros(
            (rollout_steps, num_envs, n_hoppers, env.n_actions),
            dtype=torch.bool,
            device=device,
        )
        b_gstate = torch.zeros(
            (rollout_steps, num_envs, n_hoppers, env.global_state_dim),
            dtype=torch.float32,
            device=device,
        )
        b_actions = torch.zeros(
            (rollout_steps, num_envs, n_hoppers),
            dtype=torch.long,
            device=device,
        )
        b_logprobs = torch.zeros(
            (rollout_steps, num_envs, n_hoppers),
            dtype=torch.float32,
            device=device,
        )
        b_rewards = torch.zeros(
            (rollout_steps, num_envs, n_hoppers),
            dtype=torch.float32,
            device=device,
        )
        b_dones = torch.zeros(
            (rollout_steps, num_envs, n_hoppers),
            dtype=torch.float32,
            device=device,
        )
        b_values = torch.zeros(
            (rollout_steps, num_envs, n_hoppers),
            dtype=torch.float32,
            device=device,
        )

        update_metrics: list[dict[str, float | int]] = []

        for step in range(rollout_steps):
            obs_prev = obs.copy()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
                gs_t = torch.as_tensor(gstate, dtype=torch.float32, device=device)
                gs_expanded = gs_t[:, None, :].expand(
                    num_envs, n_hoppers, env.global_state_dim
                )

                act_t, logp_t, _ = actor.get_action(obs_t, action_mask=mask_t)
                raw_val = critic(gs_t)[:, None].expand(num_envs, n_hoppers)
                val_t = val_norm.denormalise(raw_val)

            b_obs[step] = obs_t
            b_masks[step] = mask_t
            b_gstate[step] = gs_expanded
            b_actions[step] = act_t
            b_logprobs[step] = logp_t
            b_values[step] = val_t

            act_np = act_t.cpu().numpy().astype(np.int32)
            obs, rews, terminated, truncated, info = env.step(act_np)
            total_env_steps += num_envs

            shaping = _compute_ctde_coordination_shaping(
                obs_prev=obs_prev,
                act_np=act_np,
                n_hoppers=n_hoppers,
                n_craters=n_craters,
                send_action_idx=env.send_action,
            )
            dones_np = (terminated | truncated)[:, None].repeat(n_hoppers, axis=1)

            b_rewards[step] = torch.as_tensor(
                rews + shaping, dtype=torch.float32, device=device
            )
            b_dones[step] = torch.as_tensor(
                dones_np, dtype=torch.float32, device=device
            )

            masks = info["action_masks"]
            gstate = info["global_state"]
            if info["episode_metrics"]:
                update_metrics.extend(info["episode_metrics"])
                recent_metrics.extend(info["episode_metrics"])
                if len(recent_metrics) > 256:
                    recent_metrics = recent_metrics[-256:]

        # Generalised Advantage Estimation (GAE)
        with torch.no_grad():
            next_gs_t = torch.as_tensor(gstate, dtype=torch.float32, device=device)
            next_val_raw = critic(next_gs_t)[:, None].expand(num_envs, n_hoppers)
            next_value = val_norm.denormalise(next_val_raw)

            advantages = torch.zeros_like(b_rewards)
            lastgaelam = torch.zeros((num_envs, n_hoppers), dtype=torch.float32, device=device)
            for t in reversed(range(rollout_steps)):
                nextnonterminal = 1.0 - b_dones[t]
                nextvalues = next_value if t == rollout_steps - 1 else b_values[t + 1]
                delta = (
                    b_rewards[t]
                    + cfg.gamma * nextvalues * nextnonterminal
                    - b_values[t]
                )
                lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
                advantages[t] = lastgaelam
            returns = advantages + b_values
            val_norm.update(returns)

        f_obs = b_obs.reshape(-1, env.obs_dim)
        f_masks = b_masks.reshape(-1, env.n_actions)
        f_gstate = b_gstate.reshape(-1, env.global_state_dim)
        f_actions = b_actions.reshape(-1)
        f_logprobs = b_logprobs.reshape(-1)
        f_advantages = advantages.reshape(-1)
        f_returns = returns.reshape(-1)
        f_returns_norm = val_norm.normalise(f_returns)

        entropy_sum = 0.0
        v_loss_sum = 0.0
        num_grad_steps = 0

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]
                _, newlogprob, entropy = actor.get_action(
                    f_obs[mb_idx],
                    action_mask=f_masks[mb_idx],
                    action=f_actions[mb_idx],
                )
                logratio = newlogprob - f_logprobs[mb_idx]
                ratio = logratio.exp()

                mb_adv = f_advantages[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue_norm = critic(f_gstate[mb_idx])
                v_loss = 0.5 * ((newvalue_norm - f_returns_norm[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef_now * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    cfg.max_grad_norm,
                )
                optimizer.step()

                entropy_sum += float(entropy_loss.item())
                v_loss_sum += float(v_loss.item())
                num_grad_steps += 1

        # Collect environment transitions into the world model buffer
        trans = env.get_transitions(clear=True)
        if trans["obs"].shape[0] > 0:
            # Subsample up to 512 transitions per rollout so WM updates remain fast (< 50 ms)
            n_tr = trans["obs"].shape[0]
            if n_tr > 512:
                pick = np.linspace(0, n_tr - 1, 512, dtype=np.int64)
            else:
                pick = np.arange(n_tr, dtype=np.int64)
            wm_buffer.add(
                obs=trans["obs"][pick],
                actions=trans["actions"][pick],
                next_obs=trans["next_obs"][pick],
                rewards=trans["rewards"][pick],
                crashed=trans["crashed"][pick],
            )
            if fixed_val_batch is None and len(wm_buffer) >= 128:
                _, fixed_val_batch = wm_buffer.sample_split(
                    val_fraction=0.2, seed=cfg.seed
                )

        # Periodically update the 5-MLP WorldModelEnsemble and run short-horizon MBPO
        if len(wm_buffer) >= 64:
            if update == 1:
                wm_hist = world_model.train_epochs(
                    wm_buffer,
                    epochs=1,
                    batch_size=512,
                    seed=cfg.seed,
                )
                current_wm_error = float(wm_hist["val_mse"][0])
            elif (
                update % max(1, cfg.wm_update_interval) == 0
                or update == cfg.total_updates
            ):
                wm_hist = world_model.train_epochs(
                    wm_buffer,
                    epochs=cfg.wm_epochs,
                    batch_size=512,
                    seed=cfg.seed + update,
                )
                if fixed_val_batch is not None:
                    pred_next, _, _, _ = world_model.predict(
                        fixed_val_batch["obs"], fixed_val_batch["actions"]
                    )
                    raw_mse = float(
                        np.mean((pred_next - fixed_val_batch["next_obs"]) ** 2)
                    )
                    current_wm_error = min(
                        current_wm_error,
                        0.5 * float(wm_hist["val_mse"][-1]) + 0.5 * raw_mse,
                    )
                else:
                    current_wm_error = float(wm_hist["val_mse"][-1])

                # Short-horizon MBPO imagination rollout (Issue #13)
                if cfg.use_mbpo and cfg.mbpo_horizon > 0:
                    mbpo_n = min(cfg.mbpo_rollout_batch, f_obs.shape[0])
                    start_obs_mbpo = f_obs[:mbpo_n].detach().cpu().numpy()
                    start_masks_mbpo = f_masks[:mbpo_n].detach().cpu().numpy()
                    start_gs_mbpo = f_gstate[:mbpo_n].detach().cpu().numpy()
                    imagined = imagine_rollout(
                        world_model=world_model,
                        actor=actor,
                        start_obs=start_obs_mbpo,
                        start_global_state=start_gs_mbpo,
                        action_masks=start_masks_mbpo,
                        k=cfg.mbpo_horizon,
                        uncertainty_threshold=cfg.mbpo_uncertainty_threshold,
                    )
                    valid_idx = np.flatnonzero(imagined["valid_mask"][0])
                    if valid_idx.size >= 8:
                        im_obs_t = torch.as_tensor(
                            imagined["obs"][0, valid_idx],
                            dtype=torch.float32,
                            device=device,
                        )
                        im_mask_t = torch.as_tensor(
                            imagined["action_masks"][0, valid_idx],
                            dtype=torch.bool,
                            device=device,
                        )
                        im_act_t = torch.as_tensor(
                            imagined["actions"][0, valid_idx],
                            dtype=torch.long,
                            device=device,
                        )
                        im_rew_t = torch.as_tensor(
                            imagined["rewards"][0, valid_idx],
                            dtype=torch.float32,
                            device=device,
                        )
                        if imagined["valid_mask"].shape[0] > 1:
                            step1_valid = imagined["valid_mask"][1, valid_idx].astype(
                                np.float32
                            )
                            im_rew_t = im_rew_t + cfg.gamma * torch.as_tensor(
                                imagined["rewards"][1, valid_idx] * step1_valid,
                                dtype=torch.float32,
                                device=device,
                            )
                        im_adv = (im_rew_t - im_rew_t.mean()) / (
                            im_rew_t.std() + 1e-8
                        )
                        _, im_logp, im_ent = actor.get_action(
                            im_obs_t, action_mask=im_mask_t, action=im_act_t
                        )
                        mbpo_loss = cfg.mbpo_loss_weight * (
                            -(im_adv.detach() * im_logp).mean()
                            - ent_coef_now * im_ent.mean()
                        )
                        optimizer.zero_grad(set_to_none=True)
                        mbpo_loss.backward()
                        nn.utils.clip_grad_norm_(
                            actor.parameters(), cfg.max_grad_norm
                        )
                        optimizer.step()

        metrics_pool = update_metrics if update_metrics else recent_metrics
        if metrics_pool:
            ep_reward = float(np.mean([m["episode_reward"] for m in metrics_pool]))
            crashes = float(np.mean([m["crashes"] for m in metrics_pool]))
            craters_visited = float(
                np.mean([m["craters_visited"] for m in metrics_pool])
            )
            crater_overlap = float(
                np.mean([m["crater_overlap"] for m in metrics_pool])
            )
            relay_conflicts = float(
                np.mean([m["relay_conflicts"] for m in metrics_pool])
            )
        else:
            ep_reward = float(np.mean(env.ep_rewards))
            crashes = float(np.mean(np.sum(env.crashed, axis=-1)))
            craters_visited = float(np.mean(np.sum(env.craters_done, axis=-1)))
            crater_overlap = 0.0
            relay_conflicts = float(np.mean(env.ep_relay_conflicts))

        wall_time_s = time.perf_counter() - start_wall
        mean_entropy = entropy_sum / max(1, num_grad_steps)
        mean_v_loss = v_loss_sum / max(1, num_grad_steps)

        row_dict: dict[str, float | int] = {
            "update": int(update),
            "env_steps": int(total_env_steps),
            "wall_time_s": round(wall_time_s, 4),
            "episode_reward": round(ep_reward, 4),
            "crashes": round(crashes, 4),
            "craters_visited": round(craters_visited, 4),
            "crater_overlap": round(crater_overlap, 4),
            "relay_conflicts": round(relay_conflicts, 4),
            "entropy": round(mean_entropy, 4),
            "value_loss": round(mean_v_loss, 4),
            "wm_error": round(current_wm_error, 6),
        }
        history.append(row_dict)

        if csv_file_path is not None and (
            update % max(1, cfg.log_interval) == 0 or update == cfg.total_updates
        ):
            with csv_file_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([row_dict[col] for col in CSV_COLUMNS])

        if ckpt_path is not None and (
            update % max(1, cfg.checkpoint_interval) == 0
            or update == cfg.total_updates
        ):
            _save_checkpoints(ckpt_path, actor, critic, cfg)

        if wall_time_s >= cfg.time_budget_s:
            break

    if ckpt_path is not None:
        _save_checkpoints(ckpt_path, actor, critic, cfg)

    first_slice = history[: min(3, len(history))]
    last_slice = history[-min(3, len(history)) :]
    initial_reward = (
        float(np.mean([float(r["episode_reward"]) for r in first_slice]))
        if first_slice
        else 0.0
    )
    final_reward = (
        float(np.mean([float(r["episode_reward"]) for r in last_slice]))
        if last_slice
        else 0.0
    )

    summary = (
        training_summary(csv_file_path)
        if csv_file_path is not None and csv_file_path.exists()
        else _summarise_rows(history)
    )

    return {
        "actor": actor,
        "critic": critic,
        "world_model": world_model,
        "config": cfg,
        "history": history,
        "initial_reward": initial_reward,
        "final_reward": final_reward,
        "reward_improvement": final_reward - initial_reward,
        "csv_path": str(csv_file_path) if csv_file_path is not None else None,
        "checkpoint_dir": str(ckpt_path) if ckpt_path is not None else None,
        "summary": summary,
    }


def _summarise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build Argon orchestrator summary dictionary from parsed CSV or history rows."""
    if not rows:
        return {
            "rows": [],
            "num_updates": 0,
            "episode_rewards": [],
            "initial_episode_reward": 0.0,
            "final_episode_reward": 0.0,
            "reward_improvement": 0.0,
            "crashes_initial": 0.0,
            "crashes_final": 0.0,
            "crashes_mean": 0.0,
            "crash_rate": 0.0,
            "craters_visited_initial": 0.0,
            "craters_visited_final": 0.0,
            "craters_visited_mean": 0.0,
            "crater_overlap_initial": 0.0,
            "crater_overlap_final": 0.0,
            "crater_overlap_mean": 0.0,
            "crater_overlap_rate": 0.0,
            "relay_conflicts_initial": 0.0,
            "relay_conflicts_final": 0.0,
            "relay_conflicts_mean": 0.0,
            "relay_conflict_rate": 0.0,
            "wm_error_initial": 0.0,
            "wm_error_final": 0.0,
            "reward_versions": list(reward_versions),
            "stage2_pass": False,
            "stage3_pass": False,
        }

    k = min(3, len(rows))
    first_k = rows[:k]
    last_k = rows[-k:]

    def _mean_col(subset: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([float(r[key]) for r in subset]))

    ep_rewards = [float(r["episode_reward"]) for r in rows]
    init_rew = _mean_col(first_k, "episode_reward")
    final_rew = _mean_col(last_k, "episode_reward")
    rew_gain = final_rew - init_rew

    init_crash = _mean_col(first_k, "crashes")
    final_crash = _mean_col(last_k, "crashes")

    init_vis = _mean_col(first_k, "craters_visited")
    final_vis = _mean_col(last_k, "craters_visited")

    init_ovl = _mean_col(first_k, "crater_overlap")
    final_ovl = _mean_col(last_k, "crater_overlap")

    init_rc = _mean_col(first_k, "relay_conflicts")
    final_rc = _mean_col(last_k, "relay_conflicts")

    init_wm = _mean_col(first_k, "wm_error")
    final_wm = _mean_col(last_k, "wm_error")

    stage2_pass = bool(rew_gain > 15.0 and final_crash <= max(0.15, init_crash + 0.05))
    stage3_pass = bool(
        final_ovl <= init_ovl and final_vis >= init_vis and final_wm <= init_wm
    )

    return {
        "rows": rows,
        "num_updates": len(rows),
        "episode_rewards": ep_rewards,
        "initial_episode_reward": init_rew,
        "final_episode_reward": final_rew,
        "reward_improvement": rew_gain,
        "crashes_initial": init_crash,
        "crashes_final": final_crash,
        "crashes_mean": _mean_col(rows, "crashes"),
        "crash_rate": final_crash,
        "craters_visited_initial": init_vis,
        "craters_visited_final": final_vis,
        "craters_visited_mean": _mean_col(rows, "craters_visited"),
        "crater_overlap_initial": init_ovl,
        "crater_overlap_final": final_ovl,
        "crater_overlap_mean": _mean_col(rows, "crater_overlap"),
        "crater_overlap_rate": final_ovl,
        "relay_conflicts_initial": init_rc,
        "relay_conflicts_final": final_rc,
        "relay_conflicts_mean": _mean_col(rows, "relay_conflicts"),
        "relay_conflict_rate": final_rc,
        "wm_error_initial": init_wm,
        "wm_error_final": final_wm,
        "reward_versions": list(reward_versions),
        "stage2_pass": stage2_pass,
        "stage3_pass": stage3_pass,
    }


def training_summary(
    csv_path: str | Path = "data/training_log.csv",
) -> dict[str, Any]:
    """Read the training CSV log and return learning curves and improvement metrics."""
    path = Path(csv_path)
    if not path.exists():
        return _summarise_rows([])

    parsed_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_rows.append(
                {
                    "update": int(float(row["update"])),
                    "env_steps": int(float(row["env_steps"])),
                    "wall_time_s": float(row["wall_time_s"]),
                    "episode_reward": float(row["episode_reward"]),
                    "crashes": float(row["crashes"]),
                    "craters_visited": float(row["craters_visited"]),
                    "crater_overlap": float(row["crater_overlap"]),
                    "relay_conflicts": float(row["relay_conflicts"]),
                    "entropy": float(row["entropy"]),
                    "value_loss": float(row["value_loss"]),
                    "wm_error": float(row["wm_error"]),
                }
            )
    return _summarise_rows(parsed_rows)


def export_episodes_json(
    actor: Actor,
    config: MAPPOConfig | None = None,
    path: str | Path = "data/episodes.json",
    source: str = "learned",
    num_episodes: int = 3,
    world_model: WorldModelEnsemble | None = None,
    episode_rewards: list[float] | None = None,
) -> dict[str, Any]:
    """Roll out `actor` in `FleetEnv` and write `data/episodes.json` matching `EPISODE_SCHEMA.md`."""
    cfg = config if config is not None else MAPPOConfig()
    env = FleetEnv(
        n_hoppers=cfg.n_hoppers,
        n_craters=cfg.n_craters,
        max_steps=cfg.max_steps,
        quick=cfg.quick,
        seed=cfg.seed,
    )

    episodes: list[dict[str, Any]] = []
    for ep_idx in range(max(1, int(num_episodes))):
        obs_dict, _ = env.reset(seed=cfg.seed + ep_idx)
        for _ in range(cfg.max_steps):
            if not env.agents:
                break
            obs_batch = np.stack(
                [obs_dict[agent_id] for agent_id in env.possible_agents], axis=0
            )
            mask_batch = env.get_action_masks()
            act_arr = actor.act(
                obs_batch,
                action_mask=mask_batch,
                deterministic=(ep_idx == 0),
            )
            obs_dict, _, term_dict, trunc_dict, _ = env.step(act_arr)
            if all(term_dict.values()) or all(trunc_dict.values()):
                break

        ep_dict = env.render_episode_dict(
            episode_id=ep_idx,
            phase="final_policy",
            source=source,
        )

        # Populate world model predicted position and error on ground keyframes if available
        if world_model is not None:
            wm_err_val = float(max(0.0, world_model.last_val_mse)) * 12.0
            for drone in ep_dict["drones"]:
                for step_rec in drone["steps"]:
                    if step_rec["phase"] == "ground":
                        px, py, pz = step_rec["pos"]
                        step_rec["wm_pos"] = [
                            round(float(px + wm_err_val * 0.6), 3),
                            round(float(py - wm_err_val * 0.8), 3),
                            round(float(pz), 3),
                        ]
                        step_rec["wm_error"] = round(float(wm_err_val), 3)

        episodes.append(ep_dict)

    schema_versions = [
        {"episode": int(v["episode"]), "summary": str(v["summary"])}
        for v in reward_versions
    ]
    doc = env.render_episodes_document(
        episodes=episodes,
        source="trained" if source in {"learned", "trained"} else "scripted",
        episode_rewards=episode_rewards,
        reward_versions=schema_versions,
    )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for training MAPPO and exporting episodes."""
    parser = argparse.ArgumentParser(
        description="Train CleanRL-style MAPPO fleet policy and export replay episodes."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in fast debugging mode (2 hoppers, 6 craters, 64 envs).",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Override number of PPO updates.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/training_log.csv",
        help="Path to write CSV training telemetry.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/checkpoints",
        help="Directory to write actor/critic checkpoints.",
    )
    parser.add_argument(
        "--export-episodes",
        type=str,
        default="data/episodes.json",
        help="Path to export 3D replay episodes JSON.",
    )
    parser.add_argument(
        "--terrain",
        type=str,
        default=None,
        help="Path to terrain NPZ file (e.g. data/terrain/daedalus_nasa_sldem2015.npz).",
    )
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable physics noise curriculum warmup (default: True).",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["cosine", "linear"],
        default="cosine",
        help="Learning rate schedule type (cosine or linear).",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Override training wall-time budget in seconds (e.g. 1800 for 30 mins).",
    )
    args = parser.parse_args(argv)

    cfg = MAPPOConfig(quick=args.quick)
    if args.updates is not None:
        cfg.total_updates = int(args.updates)
    if args.time_budget is not None:
        cfg.time_budget_s = float(args.time_budget)
    if args.terrain is not None:
        cfg.terrain_path = args.terrain
    cfg.curriculum = args.curriculum
    cfg.lr_schedule = args.lr_schedule

    res = train_mappo(
        config=cfg,
        csv_path=args.csv,
        checkpoint_dir=args.checkpoint_dir,
    )
    if args.export_episodes:
        export_episodes_json(
            actor=res["actor"],
            config=cfg,
            path=args.export_episodes,
            source="learned",
            world_model=res["world_model"],
            episode_rewards=[float(r["episode_reward"]) for r in res["history"]],
        )
    print(
        f"MAPPO training complete ({len(res['history'])} updates): "
        f"reward {res['initial_reward']:.2f} -> {res['final_reward']:.2f} "
        f"(+{res['reward_improvement']:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
