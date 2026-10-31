"""Build the 200x200 Daedalus crater terrain crop and export to NPZ.

Supports either ingesting an external raw elevation array / GeoTIFF-exported
`.npy` via `--input-dem` with 2D bilinear resampling, or generating the
calibrated Daedalus (5.9 deg S, 179.4 deg E) 20x20 km terrain crop (100 m/cell)
with a Robbins-style crater catalogue, Horn 3x3 finite-difference surface
slopes, and Diviner-calibrated lunar night temperatures (-170 C to -225 C).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Ensure selene_sim is importable when executed directly as a script
_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIM_ROOT))

from selene_sim.sim import _BASE_CRATER_SPECS, CraterInfo, TerrainGrid  # noqa: E402

R_MOON_M: float = 1_737_400.0
DAEDALUS_LAT_DEG: float = -5.9
DAEDALUS_LON_DEG: float = 179.4


def local_xy_to_latlon_deg(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    lat0_deg: float = DAEDALUS_LAT_DEG,
    lon0_deg: float = DAEDALUS_LON_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Project local East-North-Up metres (centred at 0, 0) to lunar cylindrical degrees."""
    x_arr = np.asarray(x_m, dtype=np.float64)
    y_arr = np.asarray(y_m, dtype=np.float64)
    s_y = R_MOON_M * (math.pi / 180.0)
    s_x = R_MOON_M * math.cos(math.radians(lat0_deg)) * (math.pi / 180.0)
    lat_deg = lat0_deg + y_arr / s_y
    lon_deg = lon0_deg + x_arr / s_x
    return lat_deg, lon_deg


def latlon_to_local_xy_m(
    lat_deg: np.ndarray | float,
    lon_deg: np.ndarray | float,
    lat0_deg: float = DAEDALUS_LAT_DEG,
    lon0_deg: float = DAEDALUS_LON_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lunar cylindrical degrees to local East-North-Up metres around Daedalus centre."""
    lat_arr = np.asarray(lat_deg, dtype=np.float64)
    lon_arr = np.asarray(lon_deg, dtype=np.float64)
    s_y = R_MOON_M * (math.pi / 180.0)
    s_x = R_MOON_M * math.cos(math.radians(lat0_deg)) * (math.pi / 180.0)
    x_m = (lon_arr - lon0_deg) * s_x
    y_m = (lat_arr - lat0_deg) * s_y
    return x_m, y_m


def resample_bilinear_2d(
    raw_dem: np.ndarray, target_shape: tuple[int, int] = (200, 200)
) -> np.ndarray:
    """Resample a 2D elevation raster onto target_shape using vectorised bilinear interpolation."""
    src = np.asarray(raw_dem, dtype=np.float64)
    if src.ndim != 2:
        raise ValueError(f"Expected 2D elevation array, got shape {src.shape}")
    src_h, src_w = src.shape
    dst_h, dst_w = target_shape
    if (src_h, src_w) == (dst_h, dst_w):
        return src.astype(np.float32)

    v = np.linspace(0.0, max(src_h - 1, 0), dst_h, dtype=np.float64)
    u = np.linspace(0.0, max(src_w - 1, 0), dst_w, dtype=np.float64)

    v0 = np.floor(v).astype(np.int64)
    u0 = np.floor(u).astype(np.int64)
    v1 = np.minimum(v0 + 1, src_h - 1)
    u1 = np.minimum(u0 + 1, src_w - 1)

    beta = (v - v0)[:, None]
    alpha = (u - u0)[None, :]

    h00 = src[np.ix_(v0, u0)]
    h10 = src[np.ix_(v0, u1)]
    h01 = src[np.ix_(v1, u0)]
    h11 = src[np.ix_(v1, u1)]

    interp = (
        (1.0 - alpha) * (1.0 - beta) * h00
        + alpha * (1.0 - beta) * h10
        + (1.0 - alpha) * beta * h01
        + alpha * beta * h11
    )
    return interp.astype(np.float32)


def compute_horn_slope_deg(
    height_m: np.ndarray, cell_size_m: float = 100.0
) -> np.ndarray:
    """Compute surface slope angle in degrees using Horn's 3x3 weighted finite-difference stencil."""
    z = np.asarray(height_m, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"Expected 2D height_m array, got shape {z.shape}")

    # Pad borders by edge replication (standard planetary GIS / GDAL boundary convention)
    zp = np.pad(z, pad_width=1, mode="edge")

    dz_dx = (
        (zp[:-2, 2:] + 2.0 * zp[1:-1, 2:] + zp[2:, 2:])
        - (zp[:-2, :-2] + 2.0 * zp[1:-1, :-2] + zp[2:, :-2])
    ) / (8.0 * float(cell_size_m))

    dz_dy = (
        (zp[2:, :-2] + 2.0 * zp[2:, 1:-1] + zp[2:, 2:])
        - (zp[:-2, :-2] + 2.0 * zp[:-2, 1:-1] + zp[:-2, 2:])
    ) / (8.0 * float(cell_size_m))

    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    return np.degrees(slope_rad).astype(np.float32)


def _load_external_dem(input_dem_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Load an external raw DEM from .npy or .npz and resample to target_shape."""
    if input_dem_path.suffix.lower() == ".npz":
        with np.load(input_dem_path, allow_pickle=False) as data:
            key = "height_m" if "height_m" in data else list(data.keys())[0]
            raw = np.asarray(data[key], dtype=np.float64)
    else:
        raw = np.asarray(np.load(input_dem_path, allow_pickle=False), dtype=np.float64)
    return resample_bilinear_2d(raw, target_shape=target_shape)


def build_daedalus_crop(
    seed: int = 42,
    quick: bool = False,
    input_dem_path: str | Path | None = None,
    cell_size_m: float = 100.0,
    grid_n: int = 200,
) -> TerrainGrid:
    """Build the 200x200 Daedalus terrain crop with Horn slopes and Diviner night temperatures."""
    rng = np.random.default_rng(seed)
    half_extent = 0.5 * grid_n * cell_size_m

    coords = (np.arange(grid_n, dtype=np.float64) + 0.5) * cell_size_m - half_extent
    xx, yy = np.meshgrid(coords, coords, indexing="xy")

    num_craters = 6 if quick else len(_BASE_CRATER_SPECS)
    craters: list[CraterInfo] = []
    for idx in range(num_craters):
        cid, bx, by, brad, bdepth, bprio = _BASE_CRATER_SPECS[idx]
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

    phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * np.pi))

    if input_dem_path is not None:
        bg_height = _load_external_dem(Path(input_dem_path), target_shape=(grid_n, grid_n)).astype(
            np.float64
        )
        bg_height = bg_height - float(np.mean(bg_height))
    else:
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

        profile = np.zeros_like(u)

        in_floor = u <= 0.55
        u_floor = u[in_floor] / 0.55
        profile[in_floor] = -crater.depth_m * (1.0 - 0.08 * (u_floor**2))

        in_wall = (u > 0.55) & (u <= 1.0)
        s_wall = (u[in_wall] - 0.55) / 0.45
        profile[in_wall] = crater.depth_m * (
            -0.92 + 1.12 * (0.5 - 0.5 * np.cos(np.pi * s_wall))
        )

        in_ejecta = (u > 1.0) & (u <= 1.45)
        s_ejecta = (u[in_ejecta] - 1.0) / 0.45
        profile[in_ejecta] = (
            0.20 * crater.depth_m * (0.5 + 0.5 * np.cos(np.pi * s_ejecta))
        )

        crater_height += profile

        floor_weight = np.clip((u - 0.45) / 0.25, 0.0, 1.0)
        floor_mask_smooth = np.minimum(floor_mask_smooth, floor_weight)

        cold_bowl = np.clip(1.0 - (u / 0.95) ** 2, 0.0, 1.0)
        temp_drop = np.maximum(temp_drop, (38.0 + 0.11 * crater.depth_m) * cold_bowl)

    height_m = (bg_height * floor_mask_smooth + crater_height).astype(np.float32)
    slope_deg = compute_horn_slope_deg(height_m, cell_size_m=cell_size_m)

    base_temp = (
        -170.0
        + 2.5 * np.sin(xx / 4500.0 + phase1)
        + 1.5 * np.cos(yy / 4500.0 + phase2)
    )
    temp_c = np.clip(base_temp - temp_drop, -225.0, -166.0).astype(np.float32)

    return TerrainGrid(
        height_m=height_m,
        slope_deg=slope_deg,
        temp_c=temp_c,
        craters=craters,
        cell_size_m=cell_size_m,
        region_name="Daedalus",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point to build and export the Daedalus 200x200 terrain crop."""
    default_out = Path(__file__).resolve().parents[3] / "data" / "terrain" / "daedalus_200x200.npz"
    parser = argparse.ArgumentParser(
        description="Build Daedalus 200x200 terrain crop and export to NPZ."
    )
    parser.add_argument(
        "--input-dem",
        type=Path,
        default=None,
        help="Optional path to an external raw DEM (.npy or .npz) to resample onto the 200x200 grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="Destination path for the exported .npz terrain bundle.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed for crater catalogue calibration.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate quick 6-crater catalogue instead of full 12-crater catalogue.",
    )
    args = parser.parse_args(argv)

    terrain = build_daedalus_crop(
        seed=args.seed,
        quick=args.quick,
        input_dem_path=args.input_dem,
    )
    terrain.to_npz(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
