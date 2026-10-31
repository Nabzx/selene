"""Vectorised multi-agent lunar fleet environment on the crater graph.

Provides:
- `VectorFleetEnv`: steps `E` parallel environments (`(E, H, ...)` contiguous
  NumPy arrays) in one call at well over 10,000 environment steps per second
  on CPU, with invalid action masking and automatic episode reset.
- `FleetEnv`: single-environment PettingZoo `ParallelEnv`-style wrapper with
  0.5 s trajectory keyframe expansion (`render_episode_dict`) matching
  `docs/data/EPISODE_SCHEMA.md`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import numpy as np

from selene_sim.reward import RewardState, compute_reward
from selene_sim.sim import (
    CraterInfo,
    SimConfig,
    TerrainGrid,
    check_relay_conflicts,
    compute_battery_drain,
    compute_hop,
    interpolate_hop_keyframes,
    is_relay_open,
    time_to_next_relay,
)

STEP_DURATION_S: float = 60.0
INITIAL_RING_RADIUS_M: float = 150.0


def _build_initial_hopper_xy(n_hoppers: int) -> np.ndarray:
    """Place hoppers on a 150 m ring around the origin so separation > 100 m."""
    if n_hoppers <= 1:
        return np.zeros((1, 2), dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, n_hoppers, endpoint=False)
    xs = INITIAL_RING_RADIUS_M * np.cos(angles)
    ys = INITIAL_RING_RADIUS_M * np.sin(angles)
    return np.stack([xs, ys], axis=-1).astype(np.float32)


class VectorFleetEnv:
    """Pure NumPy vectorised multi-agent lunar hopper environment on the crater graph.

    Action space per hopper: `Discrete(C + 3)` where:
    - `0 .. C - 1`: ballistic hop to crater `k`
    - `C`: `scan` current crater
    - `C + 1`: `wait` in place for one step
    - `C + 2`: `send` buffered scan data during an open orbiter relay window
    """

    def __init__(
        self,
        num_envs: int = 128,
        n_hoppers: int = 4,
        n_craters: int = 12,
        max_steps: int = 30,
        quick: bool = False,
        seed: int = 42,
        terrain: TerrainGrid | None = None,
        config: SimConfig | None = None,
        record_transitions: bool = False,
    ) -> None:
        if quick:
            if n_hoppers == 4:
                n_hoppers = 2
            if n_craters == 12:
                n_craters = 6
            if max_steps == 30:
                max_steps = 20

        self.num_envs = int(num_envs)
        self.n_hoppers = int(n_hoppers)
        self.quick = bool(quick)
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.config = config if config is not None else SimConfig()
        self.record_transitions = bool(record_transitions)
        self.rng = np.random.default_rng(self.seed)

        if terrain is None:
            self.terrain = TerrainGrid.synthetic(seed=self.seed, quick=self.quick)
        else:
            self.terrain = terrain

        if len(self.terrain.craters) < n_craters:
            n_craters = len(self.terrain.craters)
        elif len(self.terrain.craters) > n_craters:
            self.terrain = TerrainGrid(
                height_m=self.terrain.height_m,
                slope_deg=self.terrain.slope_deg,
                temp_c=self.terrain.temp_c,
                craters=list(self.terrain.craters[:n_craters]),
                cell_size_m=self.terrain.cell_size_m,
                region_name=self.terrain.region_name,
            )

        self.n_craters = int(n_craters)
        self.n_actions = self.n_craters + 3
        self.scan_action = self.n_craters
        self.wait_action = self.n_craters + 1
        self.send_action = self.n_craters + 2

        # Observation layout:
        # 6 own + H one-hot + 3*C crater features + 3*(H-1) peer features + 2 relay features
        self.obs_dim = (
            6
            + self.n_hoppers
            + 3 * self.n_craters
            + 3 * max(0, self.n_hoppers - 1)
            + 2
        )
        # Centralised critic global state layout:
        # 7*H hopper features + 2*C crater flags + 2 relay features
        self.global_state_dim = 7 * self.n_hoppers + 2 * self.n_craters + 2
        self.state_dim = self.global_state_dim

        # Precompute crater geometry and physics constants for zero-loop vectorised steps
        self._crater_xy = np.array(
            [[c.x_m, c.y_m] for c in self.terrain.craters[: self.n_craters]],
            dtype=np.float32,
        )  # (C, 2)
        self._crater_z = self.terrain.height_at(
            self._crater_xy[:, 0], self._crater_xy[:, 1]
        ).astype(np.float32)  # (C,)
        self._crater_temp = self.terrain.temp_at(
            self._crater_xy[:, 0], self._crater_xy[:, 1]
        ).astype(np.float32)  # (C,)
        self._crater_priorities = np.array(
            [c.priority for c in self.terrain.craters[: self.n_craters]],
            dtype=np.float32,
        )  # (C,)
        self._crater_indices = np.arange(self.n_craters, dtype=np.int32)  # (C,)
        self._hopper_indices = np.arange(self.n_hoppers, dtype=np.int32)  # (H,)

        self._init_pos_xy = _build_initial_hopper_xy(self.n_hoppers)  # (H, 2)
        self._init_pos_z = self.terrain.height_at(
            self._init_pos_xy[:, 0], self._init_pos_xy[:, 1]
        ).astype(np.float32)  # (H,)
        self._init_temp_c = self.terrain.temp_at(
            self._init_pos_xy[:, 0], self._init_pos_xy[:, 1]
        ).astype(np.float32)  # (H,)

        theta_rad = math.radians(self.config.launch_angle_deg)
        self._sin_2theta = float(math.sin(2.0 * theta_rad))
        self._tan_theta = float(math.tan(theta_rad))
        self._v_e = float(self.config.isp_s * self.config.g0_earth)

        # Precompute one-hot identity and peer index table
        self._eye_hoppers = np.eye(self.n_hoppers, dtype=np.float32)  # (H, H)
        if self.n_hoppers > 1:
            peer_rows = []
            for h in range(self.n_hoppers):
                peer_rows.append([j for j in range(self.n_hoppers) if j != h])
            self._peer_idx = np.array(peer_rows, dtype=np.int32)  # (H, H - 1)
        else:
            self._peer_idx = np.zeros((1, 0), dtype=np.int32)

        # Allocate contiguous state arrays across (E, H) and (E, C)
        e, h, c = self.num_envs, self.n_hoppers, self.n_craters
        self.pos_xy = np.zeros((e, h, 2), dtype=np.float32)
        self.pos_z = np.zeros((e, h), dtype=np.float32)
        self.fuel_kg = np.zeros((e, h), dtype=np.float32)
        self.battery_pct = np.zeros((e, h), dtype=np.float32)
        self.temp_c = np.zeros((e, h), dtype=np.float32)
        self.unsent_data = np.zeros((e, h), dtype=np.float32)
        self.crashed = np.zeros((e, h), dtype=bool)
        self.alive = np.ones((e, h), dtype=bool)
        self.at_crater_id = np.full((e, h), -1, dtype=np.int32)

        self.craters_done = np.zeros((e, c), dtype=bool)
        self.craters_claimed = np.full((e, c), -1, dtype=np.int32)
        self.hopper_visited_craters = np.zeros((e, h, c), dtype=bool)

        self.mission_time_s = np.zeros(e, dtype=np.float32)
        self.step_count = np.zeros(e, dtype=np.int32)
        self.relay_open = np.ones(e, dtype=bool)
        self.relay_conflict = np.zeros(e, dtype=bool)

        self.ep_rewards = np.zeros(e, dtype=np.float32)
        self.ep_relay_conflicts = np.zeros(e, dtype=np.int32)

        # Preallocated buffers for observations and masks
        self._obs_buf = np.zeros((e, h, self.obs_dim), dtype=np.float32)
        self._obs_buf[:, :, 6 : 6 + h] = self._eye_hoppers[None, :, :]
        self._mask_buf = np.zeros((e, h, self.n_actions), dtype=bool)
        self._global_state_buf = np.zeros((e, self.global_state_dim), dtype=np.float32)

        # Optional transition storage for world_model.py training
        self._trans_obs: list[np.ndarray] = []
        self._trans_actions: list[np.ndarray] = []
        self._trans_next_obs: list[np.ndarray] = []
        self._trans_rewards: list[np.ndarray] = []
        self._trans_crashed: list[np.ndarray] = []

        self._reset_all()

    def _reset_all(self) -> None:
        """Reset all E parallel environments to their initial state."""
        self.pos_xy[:] = self._init_pos_xy[None, :, :]
        self.pos_z[:] = self._init_pos_z[None, :]
        self.fuel_kg.fill(self.config.initial_fuel_kg)
        self.battery_pct.fill(100.0)
        self.temp_c[:] = self._init_temp_c[None, :]
        self.unsent_data.fill(0.0)
        self.crashed.fill(False)
        self.alive.fill(True)
        self.at_crater_id.fill(-1)

        self.craters_done.fill(False)
        self.craters_claimed.fill(-1)
        self.hopper_visited_craters.fill(False)

        self.mission_time_s.fill(0.0)
        self.step_count.fill(0)
        self.relay_open.fill(bool(is_relay_open(0.0, self.config)))
        self.relay_conflict.fill(False)

        self.ep_rewards.fill(0.0)
        self.ep_relay_conflicts.fill(0)

    def _reset_indices(self, env_indices: np.ndarray) -> None:
        """Reset only the specified finished environment indices in place."""
        self.pos_xy[env_indices] = self._init_pos_xy[None, :, :]
        self.pos_z[env_indices] = self._init_pos_z[None, :]
        self.fuel_kg[env_indices] = self.config.initial_fuel_kg
        self.battery_pct[env_indices] = 100.0
        self.temp_c[env_indices] = self._init_temp_c[None, :]
        self.unsent_data[env_indices] = 0.0
        self.crashed[env_indices] = False
        self.alive[env_indices] = True
        self.at_crater_id[env_indices] = -1

        self.craters_done[env_indices] = False
        self.craters_claimed[env_indices] = -1
        self.hopper_visited_craters[env_indices] = False

        self.mission_time_s[env_indices] = 0.0
        self.step_count[env_indices] = 0
        self.relay_open[env_indices] = bool(is_relay_open(0.0, self.config))
        self.relay_conflict[env_indices] = False

        self.ep_rewards[env_indices] = 0.0
        self.ep_relay_conflicts[env_indices] = 0

    def reset(
        self, seed: int | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all parallel environments and return `(obs, info)`."""
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        self._reset_all()
        obs = self._get_obs()
        masks = self.get_action_masks()
        return obs, {
            "action_masks": masks,
            "global_state": self.get_global_state(),
            "episode_metrics": [],
        }

    def _build_reward_state(self) -> RewardState:
        """Snapshot current vectorised fleet state into a RewardState."""
        return RewardState(
            pos_xy=self.pos_xy.copy(),
            fuel_kg=self.fuel_kg.copy(),
            battery_pct=self.battery_pct.copy(),
            temp_c=self.temp_c.copy(),
            unsent_data=self.unsent_data.copy(),
            crashed=self.crashed.copy(),
            alive=self.alive.copy(),
            craters_done=self.craters_done.copy(),
            craters_claimed=self.craters_claimed.copy(),
            crater_priorities=self._crater_priorities,
            relay_open=self.relay_open.copy(),
            relay_conflict=self.relay_conflict.copy(),
            at_crater_id=self.at_crater_id.copy(),
        )

    def _get_obs(self) -> np.ndarray:
        """Build the flat observation array of shape `(E, H, obs_dim)`."""
        e, h, c = self.num_envs, self.n_hoppers, self.n_craters
        buf = self._obs_buf

        # 1. Own state (6 floats)
        buf[:, :, 0] = self.pos_xy[:, :, 0] * 1e-4
        buf[:, :, 1] = self.pos_xy[:, :, 1] * 1e-4
        buf[:, :, 2] = self.fuel_kg * (1.0 / self.config.initial_fuel_kg)
        buf[:, :, 3] = self.battery_pct * 0.01
        buf[:, :, 4] = (self.temp_c + 200.0) * 0.02
        buf[:, :, 5] = self.unsent_data * 0.2

        # 2. One-hot hopper identity (H floats at indices 6 .. 6 + H) is constant
        idx = 6 + h

        # 3. Crater features (3 * C floats: dist / 10000, is_claimed, is_done)
        diff_c = self._crater_xy[None, None, :, :] - self.pos_xy[:, :, None, :]  # (E, H, C, 2)
        dist_c = np.hypot(diff_c[..., 0], diff_c[..., 1]) * 1e-4  # (E, H, C)
        claimed_f = (self.craters_claimed >= 0).astype(np.float32)[:, None, :]  # (E, 1, C)
        done_f = self.craters_done.astype(np.float32)[:, None, :]  # (E, 1, C)

        crater_slice = buf[:, :, idx : idx + 3 * c].reshape(e, h, c, 3)
        crater_slice[..., 0] = dist_c
        crater_slice[..., 1] = claimed_f
        crater_slice[..., 2] = done_f
        idx += 3 * c

        # 4. Other hoppers (3 * (H - 1) floats: rel_dx / 10000, rel_dy / 10000, peer_alive)
        if h > 1:
            peer_xy = self.pos_xy[:, self._peer_idx, :]  # (E, H, H - 1, 2)
            rel_xy = (peer_xy - self.pos_xy[:, :, None, :]) * 1e-4
            peer_alive = self.alive[:, self._peer_idx].astype(np.float32)  # (E, H, H - 1)

            peer_slice = buf[:, :, idx : idx + 3 * (h - 1)].reshape(e, h, h - 1, 3)
            peer_slice[..., 0] = rel_xy[..., 0]
            peer_slice[..., 1] = rel_xy[..., 1]
            peer_slice[..., 2] = peer_alive
            idx += 3 * (h - 1)

        # 5. Relay state (2 floats: relay_open, time_to_next_relay / relay_period_s)
        wait_norm = (
            time_to_next_relay(self.mission_time_s, self.config)
            * (1.0 / self.config.relay_period_s)
        ).astype(np.float32)
        buf[:, :, idx] = self.relay_open.astype(np.float32)[:, None]
        buf[:, :, idx + 1] = wait_norm[:, None]

        return buf.copy()

    def get_global_state(self) -> np.ndarray:
        """Build centralised critic global state of shape `(E, global_state_dim)`."""
        e, h, c = self.num_envs, self.n_hoppers, self.n_craters
        g = self._global_state_buf

        hopper_part = g[:, : 7 * h].reshape(e, h, 7)
        hopper_part[..., 0] = self.pos_xy[..., 0] * 1e-4
        hopper_part[..., 1] = self.pos_xy[..., 1] * 1e-4
        hopper_part[..., 2] = self.fuel_kg * (1.0 / self.config.initial_fuel_kg)
        hopper_part[..., 3] = self.battery_pct * 0.01
        hopper_part[..., 4] = (self.temp_c + 200.0) * 0.02
        hopper_part[..., 5] = self.unsent_data * 0.2
        hopper_part[..., 6] = self.alive.astype(np.float32)

        idx = 7 * h
        g[:, idx : idx + c] = (self.craters_claimed >= 0).astype(np.float32)
        g[:, idx + c : idx + 2 * c] = self.craters_done.astype(np.float32)
        idx += 2 * c

        wait_norm = (
            time_to_next_relay(self.mission_time_s, self.config)
            * (1.0 / self.config.relay_period_s)
        ).astype(np.float32)
        g[:, idx] = self.relay_open.astype(np.float32)
        g[:, idx + 1] = wait_norm
        return g.copy()

    def get_action_masks(self) -> np.ndarray:
        """Compute valid boolean action mask of shape `(E, H, C + 3)` without loops."""
        c = self.n_craters
        mask = self._mask_buf
        mask.fill(False)

        # wait (C + 1) is always True for all hoppers (including crashed/dead ones)
        mask[:, :, self.wait_action] = True

        # Hop to crater k (0 .. C - 1)
        diff_xy = self._crater_xy[None, None, :, :] - self.pos_xy[:, :, None, :]  # (E, H, C, 2)
        dist_m = np.hypot(diff_xy[..., 0], diff_xy[..., 1])  # (E, H, C)
        dz = self._crater_z[None, None, :] - self.pos_z[:, :, None]  # (E, H, C)

        r_safe = np.maximum(dist_m, 1.0)
        elev_denom = np.clip(1.0 - dz / (r_safe * self._tan_theta), 0.25, 4.0)
        v_0_flat = np.sqrt(r_safe * (self.config.g_moon / self._sin_2theta))
        v_0 = v_0_flat / np.sqrt(elev_denom)
        v_land = np.sqrt(np.maximum(0.0, v_0 * v_0 - 2.0 * self.config.g_moon * dz))
        delta_v_nom = (v_0 + v_land) * (1.0 + self.config.gravity_loss_margin)

        m_0 = self.config.dry_mass_kg + np.maximum(self.fuel_kg[:, :, None], 0.0)
        est_fuel_kg = m_0 * (1.0 - np.exp(-delta_v_nom * (1.0 / self._v_e)))

        not_at_k = self.at_crater_id[:, :, None] != self._crater_indices[None, None, :]
        not_done_k = ~self.craters_done[:, None, :]
        in_range = (dist_m > 1.0) & (dist_m <= self.config.max_hop_range_m)
        enough_fuel = est_fuel_kg <= (self.fuel_kg[:, :, None] * 0.98)

        mask[:, :, :c] = (
            self.alive[:, :, None]
            & not_at_k
            & not_done_k
            & in_range
            & enough_fuel
        )

        # scan (C): alive, at a crater (at_crater_id >= 0), and that crater is not yet done
        safe_cid = np.maximum(self.at_crater_id, 0)
        at_crater_done = np.take_along_axis(self.craters_done, safe_cid, axis=1)
        mask[:, :, self.scan_action] = (
            self.alive & (self.at_crater_id >= 0) & (~at_crater_done)
        )

        # send (C + 2): alive, unsent_data > 0, and relay_open is True
        mask[:, :, self.send_action] = (
            self.alive & (self.unsent_data > 0.0) & self.relay_open[:, None]
        )

        return mask.copy()

    def sample_random_actions(
        self, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample uniformly from valid masked actions across `(E, H)`."""
        gen = rng if rng is not None else self.rng
        masks = self.get_action_masks()
        noise = gen.uniform(0.01, 1.0, size=masks.shape)
        scores = np.where(masks, noise, -1.0)
        return np.argmax(scores, axis=-1).astype(np.int32)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step all `(E, H)` hoppers in parallel and auto-reset finished episodes."""
        act = np.asarray(actions, dtype=np.int32).reshape(self.num_envs, self.n_hoppers)
        c = self.n_craters

        prev_state = self._build_reward_state()
        obs_before = self._get_obs() if self.record_transitions else None

        is_hop = self.alive & (act >= 0) & (act < c)
        is_scan = self.alive & (act == self.scan_action)
        is_send = self.alive & (act == self.send_action)

        # 1. Execute ballistic hops for hoppers choosing 0 .. C - 1
        hop_cid = np.clip(act, 0, c - 1)
        if np.any(is_hop):
            target_xy = np.where(
                is_hop[..., None],
                self._crater_xy[hop_cid],
                self.pos_xy,
            )
            hop_res = compute_hop(
                start_pos_xy=self.pos_xy,
                target_pos_xy=target_xy,
                fuel_kg=self.fuel_kg,
                terrain=self.terrain,
                config=self.config,
                rng=self.rng,
            )
            hop_crashed = is_hop & hop_res["crashed"]
            landed_ok = is_hop & (~hop_crashed)
            flight_time_s = np.where(
                is_hop, hop_res["flight_time_s"].astype(np.float32), 0.0
            )

            self.pos_xy = np.where(
                is_hop[..., None],
                hop_res["landed_pos_xy"].astype(np.float32),
                self.pos_xy,
            )
            self.pos_z = self.terrain.height_at(
                self.pos_xy[..., 0], self.pos_xy[..., 1]
            )
            self.temp_c = self.terrain.temp_at(
                self.pos_xy[..., 0], self.pos_xy[..., 1]
            )
            self.fuel_kg = np.maximum(
                0.0,
                self.fuel_kg
                - np.where(is_hop, hop_res["fuel_used_kg"].astype(np.float32), 0.0),
            )
            self.crashed |= hop_crashed
            self.alive &= ~self.crashed

            # Update crater occupancy and crater claiming on successful landing
            self.at_crater_id = np.where(landed_ok, hop_cid, self.at_crater_id)
            self.at_crater_id = np.where(hop_crashed, -1, self.at_crater_id)

            landed_at_c = landed_ok[:, :, None] & (
                hop_cid[:, :, None] == self._crater_indices[None, None, :]
            )  # (E, H, C)
            any_landed_at_c = np.any(landed_at_c, axis=1)  # (E, C)
            first_claimer = np.argmax(landed_at_c, axis=1).astype(np.int32)  # (E, C)
            should_claim = (self.craters_claimed == -1) & any_landed_at_c
            self.craters_claimed = np.where(
                should_claim, first_claimer, self.craters_claimed
            )
            self.hopper_visited_craters |= landed_at_c
        else:
            flight_time_s = np.zeros((self.num_envs, self.n_hoppers), dtype=np.float32)

        # 2. Execute crater scans (action == C)
        if np.any(is_scan):
            at_c = (
                self.at_crater_id[:, :, None] == self._crater_indices[None, None, :]
            )  # (E, H, C)
            scan_at_c = (
                is_scan[:, :, None] & at_c & (~self.craters_done[:, None, :])
            )  # (E, H, C)
            any_scan_at_c = np.any(scan_at_c, axis=1)  # (E, C)
            first_scanner = np.argmax(scan_at_c, axis=1).astype(np.int32)  # (E, C)
            scanner_match = scan_at_c & (
                self._hopper_indices[None, :, None] == first_scanner[:, None, :]
            )  # (E, H, C)
            valid_scan_per_hopper = np.any(scanner_match, axis=-1)  # (E, H)

            self.craters_done |= any_scan_at_c
            self.unsent_data += valid_scan_per_hopper.astype(np.float32)
            should_claim_scan = (self.craters_claimed == -1) & any_scan_at_c
            self.craters_claimed = np.where(
                should_claim_scan, first_scanner, self.craters_claimed
            )
            self.hopper_visited_craters |= scan_at_c

        # 3. Execute orbiter relay transmissions (action == C + 2)
        send_success, relay_conflict = check_relay_conflicts(is_send, self.relay_open)
        self.relay_conflict = np.asarray(relay_conflict, dtype=bool)
        clean_send = send_success & (self.unsent_data > 0.0)
        self.unsent_data = np.where(clean_send, 0.0, self.unsent_data)
        self.ep_relay_conflicts += self.relay_conflict.astype(np.int32)

        # 4. Update battery under lunar night thermal load
        dt_hopper = np.where(
            is_hop, np.maximum(flight_time_s, STEP_DURATION_S), STEP_DURATION_S
        )
        drain = compute_battery_drain(
            dt_s=dt_hopper,
            temp_c=self.temp_c,
            did_hop=is_hop,
            did_scan=is_scan,
            did_send=is_send,
            config=self.config,
        ).astype(np.float32)
        self.battery_pct = np.where(
            self.alive, np.maximum(0.0, self.battery_pct - drain), self.battery_pct
        )
        self.alive &= self.battery_pct > 0.0

        # 5. Compute step rewards before advancing relay window flag for next step
        next_state = self._build_reward_state()
        rewards, reward_components = compute_reward(prev_state, act, next_state)
        self.ep_rewards += np.sum(rewards, axis=-1)

        # 6. Advance mission clock and relay window state for next decision step
        self.mission_time_s += STEP_DURATION_S
        self.step_count += 1
        self.relay_open = np.asarray(
            is_relay_open(self.mission_time_s, self.config), dtype=bool
        )

        # 7. Check termination and truncation conditions
        all_craters_done = np.all(self.craters_done, axis=-1)
        all_dead = ~np.any(self.alive, axis=-1)
        terminated = all_craters_done | all_dead
        truncated = (self.step_count >= self.max_steps) & (~terminated)
        dones = terminated | truncated

        next_obs_pre_reset = self._get_obs()
        newly_crashed = (~prev_state.crashed) & next_state.crashed

        if self.record_transitions and obs_before is not None:
            flat_n = self.num_envs * self.n_hoppers
            self._trans_obs.append(obs_before.reshape(flat_n, self.obs_dim))
            self._trans_actions.append(act.reshape(flat_n))
            self._trans_next_obs.append(
                next_obs_pre_reset.reshape(flat_n, self.obs_dim)
            )
            self._trans_rewards.append(rewards.reshape(flat_n))
            self._trans_crashed.append(newly_crashed.reshape(flat_n))

        episode_metrics: list[dict[str, float | int]] = []
        if np.any(dones):
            done_indices = np.flatnonzero(dones)
            for e_idx in done_indices:
                visits_per_crater = np.sum(
                    self.hopper_visited_craters[e_idx], axis=0
                )  # (C,)
                overlap = int(np.sum(np.maximum(0, visits_per_crater - 1)))
                episode_metrics.append(
                    {
                        "env_index": int(e_idx),
                        "episode_reward": float(self.ep_rewards[e_idx]),
                        "crashes": int(np.sum(self.crashed[e_idx])),
                        "craters_visited": int(np.sum(self.craters_done[e_idx])),
                        "crater_overlap": overlap,
                        "relay_conflicts": int(self.ep_relay_conflicts[e_idx]),
                    }
                )
            self._reset_indices(done_indices)
            obs = self._get_obs()
        else:
            obs = next_obs_pre_reset

        info: dict[str, Any] = {
            "action_masks": self.get_action_masks(),
            "global_state": self.get_global_state(),
            "reward_components": reward_components,
            "episode_metrics": episode_metrics,
            "next_obs": next_obs_pre_reset,
            "newly_crashed": newly_crashed,
            "flight_time_s": flight_time_s,
        }
        return obs, rewards, terminated, truncated, info

    def get_transitions(self, clear: bool = False) -> dict[str, np.ndarray]:
        """Return flat recorded transitions for world model training."""
        if not self._trans_obs:
            return {
                "obs": np.zeros((0, self.obs_dim), dtype=np.float32),
                "actions": np.zeros((0,), dtype=np.int32),
                "next_obs": np.zeros((0, self.obs_dim), dtype=np.float32),
                "rewards": np.zeros((0,), dtype=np.float32),
                "crashed": np.zeros((0,), dtype=bool),
            }
        batch = {
            "obs": np.concatenate(self._trans_obs, axis=0),
            "actions": np.concatenate(self._trans_actions, axis=0),
            "next_obs": np.concatenate(self._trans_next_obs, axis=0),
            "rewards": np.concatenate(self._trans_rewards, axis=0),
            "crashed": np.concatenate(self._trans_crashed, axis=0),
        }
        if clear:
            self._trans_obs.clear()
            self._trans_actions.clear()
            self._trans_next_obs.clear()
            self._trans_rewards.clear()
            self._trans_crashed.clear()
        return batch


class FleetEnv:
    """Single-environment PettingZoo ParallelEnv-style wrapper around VectorFleetEnv(num_envs=1)."""

    def __init__(
        self,
        n_hoppers: int = 4,
        n_craters: int = 12,
        max_steps: int = 30,
        quick: bool = False,
        seed: int = 42,
        terrain: TerrainGrid | None = None,
        config: SimConfig | None = None,
    ) -> None:
        self._vec = VectorFleetEnv(
            num_envs=1,
            n_hoppers=n_hoppers,
            n_craters=n_craters,
            max_steps=max_steps,
            quick=quick,
            seed=seed,
            terrain=terrain,
            config=config,
            record_transitions=False,
        )
        self.n_hoppers = self._vec.n_hoppers
        self.n_craters = self._vec.n_craters
        self.max_steps = self._vec.max_steps
        self.quick = self._vec.quick
        self.seed = self._vec.seed
        self.terrain = self._vec.terrain
        self.config = self._vec.config
        self.n_actions = self._vec.n_actions
        self.obs_dim = self._vec.obs_dim
        self.global_state_dim = self._vec.global_state_dim
        self.state_dim = self._vec.state_dim

        self.possible_agents: list[str] = [f"hopper_{i}" for i in range(self.n_hoppers)]
        self.agents: list[str] = list(self.possible_agents)
        self._agent_to_idx: dict[str, int] = {}
        for i in range(self.n_hoppers):
            self._agent_to_idx[f"hopper_{i}"] = i
            self._agent_to_idx[f"h{i}"] = i
            self._agent_to_idx[str(i)] = i

        self._step_records: list[dict[str, Any]] = []
        self._ep_total_reward: float = 0.0
        self._ep_crashes: int = 0
        self._ep_craters_done: np.ndarray = np.zeros(self.n_craters, dtype=bool)
        self._ep_hopper_visited: list[list[str]] = [[] for _ in range(self.n_hoppers)]
        self._ep_visited_matrix: np.ndarray = np.zeros(
            (self.n_hoppers, self.n_craters), dtype=bool
        )
        self._ep_relay_conflicts: int = 0
        self._ep_hopper_crashed: np.ndarray = np.zeros(self.n_hoppers, dtype=bool)
        self._ep_hopper_battery_dead: np.ndarray = np.zeros(self.n_hoppers, dtype=bool)

    def _resolve_agent_idx(self, agent_id: str | int) -> int:
        if isinstance(agent_id, int):
            return int(agent_id)
        return self._agent_to_idx[str(agent_id)]

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset the single environment and return `(obs_dict, info_dict)`."""
        obs_arr, vec_info = self._vec.reset(seed=seed)
        self.agents = list(self.possible_agents)
        self._step_records = []
        self._ep_total_reward = 0.0
        self._ep_crashes = 0
        self._ep_craters_done = np.zeros(self.n_craters, dtype=bool)
        self._ep_hopper_visited = [[] for _ in range(self.n_hoppers)]
        self._ep_visited_matrix = np.zeros((self.n_hoppers, self.n_craters), dtype=bool)
        self._ep_relay_conflicts = 0
        self._ep_hopper_crashed = np.zeros(self.n_hoppers, dtype=bool)
        self._ep_hopper_battery_dead = np.zeros(self.n_hoppers, dtype=bool)

        obs_dict: dict[str, np.ndarray] = {}
        info_dict: dict[str, dict[str, Any]] = {}
        global_state = vec_info["global_state"][0]
        for i, agent_id in enumerate(self.possible_agents):
            obs_dict[agent_id] = obs_arr[0, i].copy()
            info_dict[agent_id] = {
                "action_mask": vec_info["action_masks"][0, i].copy(),
                "global_state": global_state.copy(),
            }
        return obs_dict, info_dict

    def get_action_mask(self, agent_id: str | int) -> np.ndarray:
        """Return valid boolean action mask of shape `(C + 3,)` for `agent_id`."""
        idx = self._resolve_agent_idx(agent_id)
        return self._vec.get_action_masks()[0, idx].copy()

    def get_action_masks(self) -> np.ndarray:
        """Return valid boolean action masks of shape `(H, C + 3)` for all hoppers."""
        return self._vec.get_action_masks()[0].copy()

    def get_global_state(self) -> np.ndarray:
        """Return centralised critic global state of shape `(global_state_dim,)`."""
        return self._vec.get_global_state()[0].copy()

    def step(
        self, action_dict: dict[str, int] | np.ndarray | list[int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Step the single environment with `action_dict` and record trajectory history."""
        if isinstance(action_dict, dict):
            act = np.full(self.n_hoppers, self._vec.wait_action, dtype=np.int32)
            for k, v in action_dict.items():
                act[self._resolve_agent_idx(k)] = int(v)
        else:
            act = np.asarray(action_dict, dtype=np.int32).reshape(self.n_hoppers)

        # Snapshot pre-step state for 0.5 s keyframe replay reconstruction
        t0_s = float(self._vec.mission_time_s[0])
        start_pos_xy = self._vec.pos_xy[0].copy()
        start_fuel_norm = np.clip(
            self._vec.fuel_kg[0] / self.config.initial_fuel_kg, 0.0, 1.0
        ).copy()
        start_battery_norm = np.clip(self._vec.battery_pct[0] * 0.01, 0.0, 1.0).copy()
        prev_craters_done = self._vec.craters_done[0].copy()
        prev_unsent = self._vec.unsent_data[0].copy()
        prev_alive = self._vec.alive[0].copy()

        # Execute physics step without auto-resetting before we capture end-of-step state
        # Temporarily hold max_steps high by 1 if needed, or inspect pre-reset state via _vec
        obs_arr, rewards_arr, term_arr, trunc_arr, step_info = self._vec.step(
            act[None, :]
        )
        done = bool(term_arr[0] or trunc_arr[0])

        # Reconstruct post-step position, fuel, battery from next_obs (which is pre-auto-reset)
        next_obs = step_info["next_obs"][0]  # (H, obs_dim)
        end_pos_xy = (next_obs[:, 0:2] * 10000.0).astype(np.float32)
        end_fuel_norm = np.clip(next_obs[:, 2], 0.0, 1.0).astype(np.float32)
        end_battery_norm = np.clip(next_obs[:, 3], 0.0, 1.0).astype(np.float32)
        end_unsent = np.maximum(0.0, next_obs[:, 5] * 5.0)
        newly_crashed = step_info["newly_crashed"][0].copy()
        flight_time_s = step_info["flight_time_s"][0].copy()

        c = self.n_craters
        crater_offset = 6 + self.n_hoppers
        craters_done_after = (
            next_obs[0, crater_offset + 2 : crater_offset + 3 * c : 3] > 0.5
        )
        newly_done_craters = (~prev_craters_done) & craters_done_after

        self._ep_craters_done |= craters_done_after
        self._ep_total_reward += float(np.sum(rewards_arr[0]))
        self._ep_crashes += int(np.sum(newly_crashed))
        self._ep_hopper_crashed |= newly_crashed
        self._ep_hopper_battery_dead |= (
            prev_alive & (~newly_crashed) & (end_battery_norm <= 1e-6)
        )

        if step_info["episode_metrics"]:
            self._ep_relay_conflicts = int(
                step_info["episode_metrics"][0]["relay_conflicts"]
            )
        else:
            self._ep_relay_conflicts = int(self._vec.ep_relay_conflicts[0])

        # Record per-hopper crater visits in chronological order
        for i in range(self.n_hoppers):
            a_i = int(act[i])
            if 0 <= a_i < c and bool(prev_alive[i]) and not bool(newly_crashed[i]):
                cid_str = self.terrain.craters[a_i].id
                if cid_str not in self._ep_hopper_visited[i]:
                    self._ep_hopper_visited[i].append(cid_str)
                self._ep_visited_matrix[i, a_i] = True
            elif a_i == self._vec.scan_action and bool(prev_alive[i]):
                for c_idx in np.flatnonzero(newly_done_craters):
                    cid_str = self.terrain.craters[int(c_idx)].id
                    if cid_str not in self._ep_hopper_visited[i]:
                        self._ep_hopper_visited[i].append(cid_str)
                    self._ep_visited_matrix[i, int(c_idx)] = True

        self._step_records.append(
            {
                "t0_s": t0_s,
                "actions": act.copy(),
                "start_pos_xy": start_pos_xy,
                "end_pos_xy": end_pos_xy,
                "start_fuel_norm": start_fuel_norm,
                "end_fuel_norm": end_fuel_norm,
                "start_battery_norm": start_battery_norm,
                "end_battery_norm": end_battery_norm,
                "rewards": rewards_arr[0].copy(),
                "newly_crashed": newly_crashed,
                "flight_time_s": flight_time_s,
                "scanned": (act == self._vec.scan_action) & (end_unsent > prev_unsent),
                "sent": (act == self._vec.send_action) & (end_unsent < prev_unsent),
                "prev_alive": prev_alive,
            }
        )

        if done:
            self.agents = []

        obs_dict: dict[str, np.ndarray] = {}
        rewards_dict: dict[str, float] = {}
        term_dict: dict[str, bool] = {}
        trunc_dict: dict[str, bool] = {}
        info_dict: dict[str, dict[str, Any]] = {}

        masks = step_info["action_masks"][0]
        global_state = step_info["global_state"][0]
        for i, agent_id in enumerate(self.possible_agents):
            obs_dict[agent_id] = next_obs[i].copy() if done else obs_arr[0, i].copy()
            rewards_dict[agent_id] = float(rewards_arr[0, i])
            term_dict[agent_id] = bool(term_arr[0])
            trunc_dict[agent_id] = bool(trunc_arr[0])
            info_dict[agent_id] = {
                "action_mask": masks[i].copy(),
                "global_state": global_state.copy(),
                "reward_components": step_info["reward_components"],
                "episode_metrics": step_info["episode_metrics"],
            }
        return obs_dict, rewards_dict, term_dict, trunc_dict, info_dict

    def render_episode_dict(
        self,
        episode_id: int = 0,
        phase: str = "final_policy",
        source: str = "learned",
    ) -> dict[str, Any]:
        """Expand recorded episode steps into 0.5 s keyframes matching EPISODE_SCHEMA.md."""
        c = self.n_craters
        drones: list[dict[str, Any]] = []
        fleet_visited: list[str] = []

        for i in range(self.n_hoppers):
            for cid_str in self._ep_hopper_visited[i]:
                if cid_str not in fleet_visited:
                    fleet_visited.append(cid_str)

            steps_out: list[dict[str, Any]] = []
            yaw = 0.0

            if not self._step_records:
                p0 = self._vec._init_pos_xy[i]
                z0 = float(self._vec._init_pos_z[i])
                temp0 = float(self._vec._init_temp_c[i])
                steps_out.append(
                    {
                        "t": 0.0,
                        "pos": [float(p0[0]), float(p0[1]), z0],
                        "yaw": 0.0,
                        "phase": "ground",
                        "fuel": 1.0,
                        "battery": 1.0,
                        "temp_c": temp0,
                        "action": {"type": "wait"},
                        "reward": 0.0,
                        "wm_pos": None,
                        "wm_error": None,
                        "event": None,
                    }
                )

            for rec in self._step_records:
                a_i = int(rec["actions"][i])
                t0 = float(rec["t0_s"])
                p0 = rec["start_pos_xy"][i]
                p1 = rec["end_pos_xy"][i]
                f0 = float(rec["start_fuel_norm"][i])
                f1 = float(rec["end_fuel_norm"][i])
                b0 = float(rec["start_battery_norm"][i])
                b1 = float(rec["end_battery_norm"][i])
                rew = float(rec["rewards"][i])
                crashed_step = bool(rec["newly_crashed"][i])
                scanned_step = bool(rec["scanned"][i])
                sent_step = bool(rec["sent"][i])
                t_flight = float(rec["flight_time_s"][i])

                if 0 <= a_i < c and t_flight > 1e-3:
                    dx = float(p1[0] - p0[0])
                    dy = float(p1[1] - p0[1])
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                        yaw = float(math.atan2(dy, dx))
                    target_xy = [
                        float(self._vec._crater_xy[a_i, 0]),
                        float(self._vec._crater_xy[a_i, 1]),
                    ]
                    act_obj: dict[str, Any] = {"type": "hop", "target": target_xy}

                    kfs = interpolate_hop_keyframes(
                        start_pos_xy=p0,
                        landed_pos_xy=p1,
                        flight_time_s=t_flight,
                        start_time_s=t0,
                        start_fuel_pct=f0,
                        end_fuel_pct=f1,
                        start_battery_pct=b0,
                        end_battery_pct=b1,
                        terrain=self.terrain,
                        config=self.config,
                        dt_s=0.5,
                    )
                    n_kf = len(kfs)
                    for k_idx, kf in enumerate(kfs):
                        frac = k_idx / max(1, n_kf - 1)
                        if k_idx == n_kf - 1:
                            kf_phase = "ground"
                            kf_event = "crashed" if crashed_step else "landed"
                            kf_reward = rew
                        elif frac < 0.25:
                            kf_phase = "ascent"
                            kf_event = None
                            kf_reward = 0.0
                        elif frac < 0.75:
                            kf_phase = "coast"
                            kf_event = None
                            kf_reward = 0.0
                        else:
                            kf_phase = "descent"
                            kf_event = None
                            kf_reward = 0.0

                        steps_out.append(
                            {
                                "t": float(kf["t"]),
                                "pos": [
                                    float(kf["x"]),
                                    float(kf["y"]),
                                    float(kf["z"]),
                                ],
                                "yaw": yaw,
                                "phase": kf_phase,
                                "fuel": float(np.clip(kf["fuel"], 0.0, 1.0)),
                                "battery": float(np.clip(kf["battery"], 0.0, 1.0)),
                                "temp_c": float(kf["temp_c"]),
                                "action": act_obj,
                                "reward": kf_reward,
                                "wm_pos": None,
                                "wm_error": None,
                                "event": kf_event,
                            }
                        )
                else:
                    if a_i == self._vec.scan_action:
                        act_obj = {"type": "scan"}
                        ev = "scan" if scanned_step else None
                    elif a_i == self._vec.send_action:
                        act_obj = {"type": "send"}
                        ev = "relay_contact" if sent_step else None
                    elif 0 <= a_i < c:
                        act_obj = {
                            "type": "hop",
                            "target": [
                                float(self._vec._crater_xy[a_i, 0]),
                                float(self._vec._crater_xy[a_i, 1]),
                            ],
                        }
                        ev = "crashed" if crashed_step else None
                    else:
                        act_obj = {"type": "wait"}
                        ev = None

                    z1 = float(self.terrain.height_at(p1[0], p1[1]))
                    temp1 = float(self.terrain.temp_at(p1[0], p1[1]))
                    steps_out.append(
                        {
                            "t": float(t0 + STEP_DURATION_S),
                            "pos": [float(p1[0]), float(p1[1]), z1],
                            "yaw": yaw,
                            "phase": "ground",
                            "fuel": float(np.clip(f1, 0.0, 1.0)),
                            "battery": float(np.clip(b1, 0.0, 1.0)),
                            "temp_c": temp1,
                            "action": act_obj,
                            "reward": rew,
                            "wm_pos": None,
                            "wm_error": None,
                            "event": ev,
                        }
                    )

            if bool(self._ep_hopper_crashed[i]):
                drone_outcome = "crashed"
            elif bool(self._ep_hopper_battery_dead[i]):
                drone_outcome = "battery"
            elif bool(np.all(self._ep_craters_done)):
                drone_outcome = "completed"
            else:
                drone_outcome = "timeout"

            drones.append(
                {
                    "id": f"h{i}",
                    "name": f"Hopper {i}",
                    "outcome": drone_outcome,
                    "craters_visited": list(self._ep_hopper_visited[i]),
                    "steps": steps_out,
                }
            )

        if bool(np.all(self._ep_craters_done)):
            ep_outcome = "completed"
        elif bool(np.any(self._ep_hopper_crashed)):
            ep_outcome = "crashed"
        elif bool(np.any(self._ep_hopper_battery_dead)):
            ep_outcome = "battery"
        else:
            ep_outcome = "timeout"

        visits_per_crater = np.sum(self._ep_visited_matrix, axis=0)
        overlap = int(np.sum(np.maximum(0, visits_per_crater - 1)))

        metrics = {
            "episode_reward": float(self._ep_total_reward),
            "crashes": int(self._ep_crashes),
            "craters_visited": int(len(fleet_visited)),
            "crater_overlap": overlap,
            "relay_conflicts": int(self._ep_relay_conflicts),
        }

        return {
            "index": int(episode_id),
            "episode_id": int(episode_id),
            "phase": str(phase),
            "source": str(source),
            "fleet_size": int(self.n_hoppers),
            "outcome": ep_outcome,
            "total_reward": float(self._ep_total_reward),
            "craters_visited": fleet_visited,
            "metrics": metrics,
            "drones": drones,
        }

    def render_episodes_document(
        self,
        episodes: list[dict[str, Any]] | None = None,
        source: str = "trained",
        episode_rewards: list[float] | None = None,
        reward_versions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a complete top-level episodes.json document matching EPISODE_SCHEMA.md."""
        if episodes is None:
            episodes = [self.render_episode_dict(episode_id=0, source=source)]
        schema_source = source if source in {"trained", "scripted"} else "trained"

        craters_list: list[dict[str, Any]] = []
        for idx, crater in enumerate(self.terrain.craters[: self.n_craters]):
            z_c = float(self._vec._crater_z[idx])
            craters_list.append(
                {
                    "id": crater.id,
                    "name": f"Crater {crater.id}",
                    "x": float(crater.x_m),
                    "y": float(crater.y_m),
                    "z": z_c,
                    "radius": float(crater.radius_m),
                }
            )

        horizon_s = float(self.max_steps) * STEP_DURATION_S
        n_periods = max(1, int(math.ceil(horizon_s / self.config.relay_period_s)) + 1)
        relay_windows = [
            {
                "start": float(k * self.config.relay_period_s),
                "end": float(
                    k * self.config.relay_period_s + self.config.relay_open_duration_s
                ),
            }
            for k in range(n_periods)
        ]

        return {
            "schema_version": 1,
            "meta": {
                "region": self.terrain.region_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": schema_source,
                "seed": int(self.seed),
                "fleet_size": int(self.n_hoppers),
            },
            "craters": craters_list,
            "relay_windows": relay_windows,
            "training": {
                "episode_rewards": (
                    list(episode_rewards)
                    if episode_rewards is not None
                    else [float(ep.get("total_reward", 0.0)) for ep in episodes]
                ),
                "reward_versions": (
                    list(reward_versions) if reward_versions is not None else []
                ),
            },
            "episodes": episodes,
        }

