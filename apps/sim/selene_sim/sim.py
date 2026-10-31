"""Ground-truth lunar physics simulator in pure NumPy.

Provides the 200x200 Daedalus terrain grid, closed-form vectorised ballistic
hop kinematics, Tsiolkovsky propellant burn, lunar night thermal battery drain,
single-channel orbiter relay conflict checks, and 0.5 s trajectory keyframe
interpolation for 3D replay export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimConfig:
    """Physical constants and stochastic noise parameters for lunar hopper simulation."""

    g_moon: float = 1.62
    g0_earth: float = 9.80665
    isp_s: float = 220.0
    dry_mass_kg: float = 35.0
    initial_fuel_kg: float = 15.0
    max_hop_range_m: float = 3000.0
    launch_angle_deg: float = 45.0
    gravity_loss_margin: float = 0.08
    max_landing_slope_deg: float = 15.0
    min_separation_m: float = 100.0
    battery_base_pct_per_s: float = 0.005
    battery_heater_coeff: float = 0.0003
    battery_hop_pct: float = 1.8
    battery_scan_pct: float = 1.2
    battery_send_pct: float = 0.8
    relay_period_s: float = 120.0
    relay_open_duration_s: float = 40.0
    noise_enabled: bool = True
    thrust_sigma_rel: float = 0.02
    scatter_sigma_rel: float = 0.015
    scatter_sigma_base_m: float = 8.0
    slope_noise_sigma_deg: float = 1.0
    dust_max_extra_burn: float = 0.03


@dataclass(frozen=True)
class CraterInfo:
    """Catalogue entry for a lunar impact crater on the local East-North-Up grid."""

    id: str
    x_m: float
    y_m: float
    radius_m: float
    depth_m: float
    priority: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise crater metadata to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "radius_m": float(self.radius_m),
            "depth_m": float(self.depth_m),
            "priority": float(self.priority),
        }


# Base Daedalus crater catalogue: 6 inner-ring craters (<= 2400 m from origin)
# followed by 6 outer-ring craters (each within <= 2400 m of the inner ring),
# so hops of <= 3000 m connect the entire graph from (0, 0).
_BASE_CRATER_SPECS: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("daedalus_a", 1650.0, 450.0, 800.0, 145.0, 1.0),
    ("daedalus_b", 650.0, 1750.0, 750.0, 135.0, 1.1),
    ("daedalus_c", -1450.0, 1150.0, 850.0, 155.0, 1.2),
    ("daedalus_d", -1750.0, -650.0, 800.0, 140.0, 1.0),
    ("daedalus_e", -450.0, -1850.0, 750.0, 130.0, 1.1),
    ("daedalus_f", 1450.0, -1350.0, 850.0, 150.0, 1.2),
    ("daedalus_g", 3650.0, 1050.0, 950.0, 165.0, 1.4),
    ("daedalus_h", 1450.0, 3650.0, 900.0, 160.0, 1.3),
    ("daedalus_i", -2950.0, 2650.0, 950.0, 170.0, 1.5),
    ("daedalus_j", -3750.0, -1250.0, 900.0, 160.0, 1.3),
    ("daedalus_k", -1150.0, -3750.0, 950.0, 165.0, 1.4),
    ("daedalus_l", 3150.0, -2750.0, 900.0, 155.0, 1.3),
)


@dataclass
class TerrainGrid:
    """200x200 lunar terrain grid with elevation, slope, night temperature, and craters."""

    height_m: np.ndarray
    slope_deg: np.ndarray
    temp_c: np.ndarray
    craters: list[CraterInfo]
    cell_size_m: float = 100.0
    region_name: str = "Daedalus"

    @property
    def half_extent_m(self) -> float:
        """Half-width of the square terrain grid in metres (10,000 m for 200x200 at 100 m)."""
        return 0.5 * float(self.height_m.shape[1]) * float(self.cell_size_m)

    @classmethod
    def synthetic(cls, seed: int = 42, quick: bool = False) -> TerrainGrid:
        """Generate a deterministic 200x200 Daedalus-profile lunar terrain grid."""
        rng = np.random.default_rng(seed)
        grid_n = 200
        cell_size_m = 100.0
        half_extent = 0.5 * grid_n * cell_size_m

        coords = (np.arange(grid_n, dtype=np.float64) + 0.5) * cell_size_m - half_extent
        xx, yy = np.meshgrid(coords, coords, indexing="xy")

        num_craters = 6 if quick else 12
        craters: list[CraterInfo] = []
        for idx in range(num_craters):
            cid, bx, by, brad, bdepth, bprio = _BASE_CRATER_SPECS[idx]
            # Snap small seed jitter to cell centres so crater centres sit cleanly on grid cells
            jx = float(rng.integers(-1, 2)) * cell_size_m
            jy = float(rng.integers(-1, 2)) * cell_size_m
            rad = float(brad + rng.uniform(-25.0, 25.0))
            depth = float(bdepth + rng.uniform(-8.0, 8.0))
            craters.append(
                CraterInfo(
                    id=cid,
                    x_m=bx + jx,
                    y_m=by + jy,
                    radius_m=rad,
                    depth_m=depth,
                    priority=bprio,
                )
            )

        # Gentle long-wavelength background highlands undulation
        phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
        phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
        bg_height = (
            12.0 * np.sin(xx / 2800.0 + phase1) * np.cos(yy / 2800.0 + phase2)
            + 6.0 * np.cos((xx + yy) / 3600.0 - phase1)
        )

        crater_height = np.zeros_like(xx)
        floor_mask_smooth = np.ones_like(xx)
        temp_drop = np.zeros_like(xx)

        for crater in craters:
            dx = xx - crater.x_m
            dy = yy - crater.y_m
            r = np.hypot(dx, dy)
            u = r / crater.radius_m

            # Flat/gentle bowl floor for u <= 0.55, steep wall rising to raised rim at u = 1.0,
            # and smooth outer ejecta decay for 1.0 < u <= 1.45.
            profile = np.zeros_like(u)

            in_floor = u <= 0.55
            u_floor = u[in_floor] / 0.55
            profile[in_floor] = -crater.depth_m * (1.0 - 0.08 * (u_floor**2))

            in_wall = (u > 0.55) & (u <= 1.0)
            s_wall = (u[in_wall] - 0.55) / 0.45
            # Raised rim reaches +0.20 * depth_m
            profile[in_wall] = crater.depth_m * (
                -0.92 + 1.12 * (0.5 - 0.5 * np.cos(np.pi * s_wall))
            )

            in_ejecta = (u > 1.0) & (u <= 1.45)
            s_ejecta = (u[in_ejecta] - 1.0) / 0.45
            profile[in_ejecta] = (
                0.20 * crater.depth_m * (0.5 + 0.5 * np.cos(np.pi * s_ejecta))
            )

            crater_height += profile

            # Damp background undulation inside crater floors so centres are flat (< 5 deg)
            floor_weight = np.clip((u - 0.45) / 0.25, 0.0, 1.0)
            floor_mask_smooth = np.minimum(floor_mask_smooth, floor_weight)

            # Night temperature drops sharply inside shaded crater bowls (-170 C plain -> -225 C floor)
            cold_bowl = np.clip(1.0 - (u / 0.95) ** 2, 0.0, 1.0)
            temp_drop = np.maximum(temp_drop, (38.0 + 0.11 * crater.depth_m) * cold_bowl)

        height_m = (bg_height * floor_mask_smooth + crater_height).astype(np.float32)

        gy, gx = np.gradient(height_m.astype(np.float64), cell_size_m)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

        base_temp = (
            -170.0
            + 2.5 * np.sin(xx / 4500.0 + phase1)
            + 1.5 * np.cos(yy / 4500.0 + phase2)
        )
        temp_c = np.clip(base_temp - temp_drop, -225.0, -155.0).astype(np.float32)

        return cls(
            height_m=height_m,
            slope_deg=slope_deg,
            temp_c=temp_c,
            craters=craters,
            cell_size_m=cell_size_m,
            region_name="Daedalus",
        )

    def to_npz(self, path: str | Path) -> None:
        """Save terrain arrays and crater catalogue to an uncompressed .npz archive."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crater_ids = np.array([c.id for c in self.craters], dtype="<U64")
        crater_numeric = np.array(
            [[c.x_m, c.y_m, c.radius_m, c.depth_m, c.priority] for c in self.craters],
            dtype=np.float64,
        )
        np.savez(
            out_path,
            height_m=self.height_m.astype(np.float32),
            slope_deg=self.slope_deg.astype(np.float32),
            temp_c=self.temp_c.astype(np.float32),
            cell_size_m=np.array(self.cell_size_m, dtype=np.float64),
            region_name=np.array(self.region_name, dtype="<U64"),
            crater_ids=crater_ids,
            crater_numeric=crater_numeric,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> TerrainGrid:
        """Load a TerrainGrid from a .npz archive."""
        with np.load(Path(path), allow_pickle=False) as data:
            height_m = np.asarray(data["height_m"], dtype=np.float32)
            slope_deg = np.asarray(data["slope_deg"], dtype=np.float32)
            temp_c = np.asarray(data["temp_c"], dtype=np.float32)
            cell_size_m = float(data["cell_size_m"])
            region_name = str(data["region_name"])
            crater_ids = data["crater_ids"]
            crater_numeric = data["crater_numeric"]
            craters = [
                CraterInfo(
                    id=str(cid),
                    x_m=float(row[0]),
                    y_m=float(row[1]),
                    radius_m=float(row[2]),
                    depth_m=float(row[3]),
                    priority=float(row[4]),
                )
                for cid, row in zip(crater_ids, crater_numeric, strict=True)
            ]
        return cls(
            height_m=height_m,
            slope_deg=slope_deg,
            temp_c=temp_c,
            craters=craters,
            cell_size_m=cell_size_m,
            region_name=region_name,
        )

    def _xy_to_indices(
        self, x_m: np.ndarray | float, y_m: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        x_arr, y_arr = np.broadcast_arrays(
            np.asarray(x_m, dtype=np.float64),
            np.asarray(y_m, dtype=np.float64),
        )
        h_cells, w_cells = self.height_m.shape
        half_w = 0.5 * w_cells * self.cell_size_m
        half_h = 0.5 * h_cells * self.cell_size_m
        ix = np.clip(
            np.floor((x_arr + half_w) / self.cell_size_m).astype(np.int64),
            0,
            w_cells - 1,
        )
        iy = np.clip(
            np.floor((y_arr + half_h) / self.cell_size_m).astype(np.int64),
            0,
            h_cells - 1,
        )
        return iy, ix, x_arr.shape

    def height_at(self, x_m: np.ndarray | float, y_m: np.ndarray | float) -> np.ndarray:
        """Vectorised terrain elevation lookup in metres at (x_m, y_m)."""
        iy, ix, shape = self._xy_to_indices(x_m, y_m)
        return np.asarray(self.height_m[iy, ix], dtype=np.float32).reshape(shape)

    def slope_at(self, x_m: np.ndarray | float, y_m: np.ndarray | float) -> np.ndarray:
        """Vectorised terrain slope lookup in degrees at (x_m, y_m)."""
        iy, ix, shape = self._xy_to_indices(x_m, y_m)
        return np.asarray(self.slope_deg[iy, ix], dtype=np.float32).reshape(shape)

    def temp_at(self, x_m: np.ndarray | float, y_m: np.ndarray | float) -> np.ndarray:
        """Vectorised lunar night surface temperature lookup in Celsius at (x_m, y_m)."""
        iy, ix, shape = self._xy_to_indices(x_m, y_m)
        return np.asarray(self.temp_c[iy, ix], dtype=np.float32).reshape(shape)

    def crater_at(self, x_m: np.ndarray | float, y_m: np.ndarray | float) -> np.ndarray:
        """Return crater index 0..C-1 if within 0.85 * radius_m of a crater centre, else -1."""
        x_arr, y_arr = np.broadcast_arrays(
            np.asarray(x_m, dtype=np.float64),
            np.asarray(y_m, dtype=np.float64),
        )
        if not self.craters:
            return np.full(x_arr.shape, -1, dtype=np.int32)

        cx = np.array([c.x_m for c in self.craters], dtype=np.float64)
        cy = np.array([c.y_m for c in self.craters], dtype=np.float64)
        cr = np.array([0.85 * c.radius_m for c in self.craters], dtype=np.float64)

        dist = np.hypot(x_arr[..., None] - cx, y_arr[..., None] - cy)
        inside = dist <= cr
        any_inside = np.any(inside, axis=-1)
        masked_dist = np.where(inside, dist, np.inf)
        best_idx = np.argmin(masked_dist, axis=-1).astype(np.int32)
        return np.where(any_inside, best_idx, np.int32(-1)).reshape(x_arr.shape)


def compute_hop(
    start_pos_xy: np.ndarray,
    target_pos_xy: np.ndarray,
    fuel_kg: np.ndarray,
    terrain: TerrainGrid,
    config: SimConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate ballistic hop kinematics, propellant burn, and landing safety in closed form.

    Supports arbitrary batch shapes `(..., 2)` for positions and `(...)` for `fuel_kg`.
    Crash reason codes: `0 = ok`, `1 = out_of_fuel`, `2 = steep_slope`, `3 = out_of_bounds`.
    """
    cfg = config if config is not None else SimConfig()
    if cfg.noise_enabled and rng is None:
        rng = np.random.default_rng()

    start_xy = np.asarray(start_pos_xy, dtype=np.float64)
    target_xy = np.asarray(target_pos_xy, dtype=np.float64)
    fuel = np.asarray(fuel_kg, dtype=np.float64)
    batch_shape = start_xy.shape[:-1]

    delta_xy = target_xy - start_xy
    r_cmd = np.linalg.norm(delta_xy, axis=-1)
    zero_hop = r_cmd <= 1e-9

    if cfg.noise_enabled and rng is not None:
        sigma_scatter = cfg.scatter_sigma_base_m + cfg.scatter_sigma_rel * r_cmd
        scatter_xy = rng.normal(0.0, 1.0, size=target_xy.shape) * sigma_scatter[..., None]
        landed_xy = np.where(zero_hop[..., None], start_xy, target_xy + scatter_xy)
    else:
        landed_xy = np.where(zero_hop[..., None], start_xy, target_xy.copy())

    z_start = terrain.height_at(start_xy[..., 0], start_xy[..., 1]).astype(np.float64)
    z_end = terrain.height_at(landed_xy[..., 0], landed_xy[..., 1]).astype(np.float64)
    dz = np.where(zero_hop, 0.0, z_end - z_start)

    theta_rad = np.radians(cfg.launch_angle_deg)
    sin_2theta = np.sin(2.0 * theta_rad)
    sin_theta = np.sin(theta_rad)
    tan_theta = np.tan(theta_rad)

    r_safe = np.where(zero_hop, 1.0, r_cmd)
    # Elevation adjustment factor (1 - dz / (R * tan(theta))) reduces identically to 1 when dz == 0
    elev_denom = np.clip(1.0 - dz / (r_safe * tan_theta), 0.25, 4.0)
    v_0_flat = np.sqrt(r_safe * cfg.g_moon / sin_2theta)
    v_0 = np.where(zero_hop, 0.0, v_0_flat / np.sqrt(elev_denom))

    flight_time_s = np.where(
        zero_hop,
        0.0,
        (2.0 * v_0_flat * sin_theta / cfg.g_moon) * np.sqrt(elev_denom),
    )

    v_land = np.where(
        zero_hop,
        0.0,
        np.sqrt(np.maximum(0.0, v_0**2 - 2.0 * cfg.g_moon * dz)),
    )
    delta_v_nom = (v_0 + v_land) * (1.0 + cfg.gravity_loss_margin)

    if cfg.noise_enabled and rng is not None:
        thrust_err = np.abs(rng.normal(0.0, cfg.thrust_sigma_rel, size=batch_shape))
        dust_extra = rng.uniform(0.0, cfg.dust_max_extra_burn, size=batch_shape)
        delta_v_ms = np.where(
            zero_hop, 0.0, delta_v_nom * (1.0 + thrust_err + dust_extra)
        )
    else:
        delta_v_ms = np.where(zero_hop, 0.0, delta_v_nom)

    v_e = cfg.isp_s * cfg.g0_earth
    m_0 = cfg.dry_mass_kg + np.maximum(fuel, 0.0)
    fuel_used_kg = np.where(zero_hop, 0.0, m_0 * (1.0 - np.exp(-delta_v_ms / v_e)))

    base_slope = terrain.slope_at(landed_xy[..., 0], landed_xy[..., 1]).astype(np.float64)
    if cfg.noise_enabled and rng is not None:
        slope_noise = rng.normal(0.0, cfg.slope_noise_sigma_deg, size=batch_shape)
        landing_slope_deg = np.clip(
            base_slope + np.where(zero_hop, 0.0, slope_noise), 0.0, 90.0
        )
    else:
        landing_slope_deg = base_slope

    half_w = 0.5 * terrain.height_m.shape[1] * terrain.cell_size_m
    half_h = 0.5 * terrain.height_m.shape[0] * terrain.cell_size_m

    out_of_fuel = (~zero_hop) & (fuel_used_kg > fuel)
    steep_slope = (~zero_hop) & (landing_slope_deg > cfg.max_landing_slope_deg)
    out_of_bounds = (np.abs(landed_xy[..., 0]) > half_w) | (
        np.abs(landed_xy[..., 1]) > half_h
    )

    crashed = out_of_fuel | steep_slope | out_of_bounds
    crash_reason_code = np.zeros(batch_shape, dtype=np.int32)
    crash_reason_code = np.where(out_of_bounds, np.int32(3), crash_reason_code)
    crash_reason_code = np.where(steep_slope, np.int32(2), crash_reason_code)
    crash_reason_code = np.where(out_of_fuel, np.int32(1), crash_reason_code)

    return {
        "landed_pos_xy": landed_xy,
        "flight_time_s": flight_time_s,
        "delta_v_ms": delta_v_ms,
        "fuel_used_kg": fuel_used_kg,
        "landing_slope_deg": landing_slope_deg,
        "crashed": crashed,
        "crash_reason_code": crash_reason_code,
    }


def compute_battery_drain(
    dt_s: np.ndarray | float,
    temp_c: np.ndarray | float,
    did_hop: np.ndarray | bool,
    did_scan: np.ndarray | bool,
    did_send: np.ndarray | bool,
    config: SimConfig | None = None,
) -> np.ndarray:
    """Compute percentage battery drain over dt_s under lunar night thermal load."""
    cfg = config if config is not None else SimConfig()
    dt, temp, hop, scan, send = np.broadcast_arrays(
        np.asarray(dt_s, dtype=np.float64),
        np.asarray(temp_c, dtype=np.float64),
        np.asarray(did_hop, dtype=bool),
        np.asarray(did_scan, dtype=bool),
        np.asarray(did_send, dtype=bool),
    )
    # Heater load grows linearly as ambient night temperature drops below -150 C
    cold_excess_c = np.maximum(0.0, -150.0 - temp)
    continuous_drain = (
        cfg.battery_base_pct_per_s + cfg.battery_heater_coeff * cold_excess_c
    ) * np.maximum(0.0, dt)
    action_drain = (
        np.where(hop, cfg.battery_hop_pct, 0.0)
        + np.where(scan, cfg.battery_scan_pct, 0.0)
        + np.where(send, cfg.battery_send_pct, 0.0)
    )
    return continuous_drain + action_drain


def is_relay_open(
    t_s: np.ndarray | float, config: SimConfig | None = None
) -> np.ndarray | bool:
    """Return True where the orbiter relay window is open at mission time t_s."""
    cfg = config if config is not None else SimConfig()
    t_arr = np.asarray(t_s, dtype=np.float64)
    phase = np.mod(t_arr, cfg.relay_period_s)
    open_mask = phase < cfg.relay_open_duration_s
    if np.ndim(t_s) == 0:
        return bool(open_mask)
    return open_mask


def time_to_next_relay(
    t_s: np.ndarray | float, config: SimConfig | None = None
) -> np.ndarray | float:
    """Return seconds until the next orbiter relay window (0.0 if currently open)."""
    cfg = config if config is not None else SimConfig()
    t_arr = np.asarray(t_s, dtype=np.float64)
    phase = np.mod(t_arr, cfg.relay_period_s)
    wait_s = np.where(phase < cfg.relay_open_duration_s, 0.0, cfg.relay_period_s - phase)
    if np.ndim(t_s) == 0:
        return float(wait_s)
    return wait_s


def check_relay_conflicts(
    send_mask: np.ndarray, relay_open: np.ndarray | bool
) -> tuple[np.ndarray, np.ndarray]:
    """Check single-channel orbiter relay transmissions for contention or out-of-window sends.

    Args:
        send_mask: Boolean array of shape `(..., N)` indicating which hoppers attempt `send`.
        relay_open: Boolean array of shape `(...)` or scalar indicating if relay is open.

    Returns:
        send_success: Boolean array of shape `(..., N)`, True only for a sole sender in an open window.
        conflict: Boolean array of shape `(...)`, True if >=2 hoppers send at once or any sends when closed.
    """
    send_arr = np.asarray(send_mask, dtype=bool)
    open_arr = np.asarray(relay_open, dtype=bool)
    send_count = np.sum(send_arr, axis=-1)

    single_valid = open_arr & (send_count == 1)
    send_success = send_arr & single_valid[..., None]
    conflict = (send_count > 1) | ((send_count >= 1) & (~open_arr))
    return send_success, conflict


def interpolate_hop_keyframes(
    start_pos_xy: np.ndarray,
    landed_pos_xy: np.ndarray,
    flight_time_s: float,
    start_time_s: float,
    start_fuel_pct: float,
    end_fuel_pct: float,
    start_battery_pct: float,
    end_battery_pct: float,
    terrain: TerrainGrid,
    config: SimConfig | None = None,
    dt_s: float = 0.5,
) -> list[dict[str, float]]:
    """Interpolate a ballistic hop into 0.5 s keyframes for 3D replay export."""
    cfg = config if config is not None else SimConfig()
    p0 = np.asarray(start_pos_xy, dtype=np.float64).reshape(2)
    p1 = np.asarray(landed_pos_xy, dtype=np.float64).reshape(2)

    z0 = float(terrain.height_at(p0[0], p0[1]))
    z1 = float(terrain.height_at(p1[0], p1[1]))

    total_t = max(float(flight_time_s), 0.0)
    if total_t <= 1e-6:
        temp0 = float(terrain.temp_at(p0[0], p0[1]))
        return [
            {
                "t": float(start_time_s),
                "x": float(p0[0]),
                "y": float(p0[1]),
                "z": z0,
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "fuel": float(end_fuel_pct),
                "battery": float(end_battery_pct),
                "temp_c": temp0,
            }
        ]

    n_steps = max(1, int(round(total_t / dt_s)))
    u = np.linspace(0.0, 1.0, n_steps + 1)
    tau = u * total_t

    xs = (1.0 - u) * p0[0] + u * p1[0]
    ys = (1.0 - u) * p0[1] + u * p1[1]
    zs = (1.0 - u) * z0 + u * z1 + 0.5 * cfg.g_moon * tau * (total_t - tau)

    vx = float((p1[0] - p0[0]) / total_t)
    vy = float((p1[1] - p0[1]) / total_t)
    vzs = (z1 - z0) / total_t + 0.5 * cfg.g_moon * (total_t - 2.0 * tau)

    fuels = (1.0 - u) * float(start_fuel_pct) + u * float(end_fuel_pct)
    batteries = (1.0 - u) * float(start_battery_pct) + u * float(end_battery_pct)
    temps = terrain.temp_at(xs, ys).astype(np.float64)

    keyframes: list[dict[str, float]] = []
    for i in range(n_steps + 1):
        keyframes.append(
            {
                "t": float(start_time_s + tau[i]),
                "x": float(xs[i]),
                "y": float(ys[i]),
                "z": float(zs[i]),
                "vx": vx,
                "vy": vy,
                "vz": float(vzs[i]),
                "fuel": float(fuels[i]),
                "battery": float(batteries[i]),
                "temp_c": float(temps[i]),
            }
        )
    return keyframes
