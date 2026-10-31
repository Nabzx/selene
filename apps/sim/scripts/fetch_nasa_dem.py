"""Fetch real NASA LRO LOLA + Kaguya SLDEM2015 elevation data for Daedalus crater.

Directly streams the required bounding box via HTTP Range requests from the
NASA Planetary Data System (PDS) Geosciences Node without downloading multi-gigabyte
global rasters, resamples onto the 200x200 (100 m/cell) local grid, and builds the
terrain crop with Horn slopes and Diviner temperatures.
"""

from __future__ import annotations

import argparse
import math
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np

# Ensure selene_sim is importable
_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIM_ROOT))

from selene_sim.sim import TerrainGrid  # noqa: E402
from scripts.build_terrain_crop import (  # noqa: E402
    DAEDALUS_LAT_DEG,
    DAEDALUS_LON_DEG,
    R_MOON_M,
    build_daedalus_crop,
    compute_horn_slope_deg,
    resample_bilinear_2d,
)

# NASA PDS Geosciences Node SLDEM2015 tile details
PDS_TILE_URL = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/"
    "sldem2015/tiles/float_img/sldem2015_256_60s_0s_120_240_float.img"
)
PIXELS_PER_DEGREE = 256
BYTES_PER_SAMPLE = 4
SAMPLES_PER_LINE = 30720
BYTES_PER_LINE = SAMPLES_PER_LINE * BYTES_PER_SAMPLE
MIN_LAT = -60.0
MAX_LAT = 0.0
MIN_LON = 120.0
MAX_LON = 240.0


def fetch_pds_bounding_box(
    center_lat: float = DAEDALUS_LAT_DEG,
    center_lon: float = DAEDALUS_LON_DEG,
    extent_m: float = 20_000.0,
    timeout: float = 30.0,
) -> np.ndarray:
    """Fetch raw elevation array (metres) for the specified bounding box from NASA PDS."""
    half_extent = extent_m / 2.0
    s_y = R_MOON_M * (math.pi / 180.0)
    s_x = R_MOON_M * math.cos(math.radians(center_lat)) * (math.pi / 180.0)

    delta_lat = half_extent / s_y
    delta_lon = half_extent / s_x

    lat_max = center_lat + delta_lat
    lat_min = center_lat - delta_lat
    lon_min = center_lon - delta_lon
    lon_max = center_lon + delta_lon

    # Convert lat/lon to line/sample indices (Simple Cylindrical from 0 to -60 deg)
    line_start = int(math.floor((MAX_LAT - lat_max) * PIXELS_PER_DEGREE))
    line_end = int(math.ceil((MAX_LAT - lat_min) * PIXELS_PER_DEGREE))

    sample_start = int(math.floor((lon_min - MIN_LON) * PIXELS_PER_DEGREE))
    sample_end = int(math.ceil((lon_max - MIN_LON) * PIXELS_PER_DEGREE))

    # Add 2-pixel margin for interpolation
    line_start = max(0, line_start - 2)
    line_end = min(15360, line_end + 2)
    sample_start = max(0, sample_start - 2)
    sample_end = min(SAMPLES_PER_LINE, sample_end + 2)

    byte_start = line_start * BYTES_PER_LINE
    byte_end = line_end * BYTES_PER_LINE - 1

    req = urllib.request.Request(
        PDS_TILE_URL,
        headers={
            "Range": f"bytes={byte_start}-{byte_end}",
            "User-Agent": "Selene-NASA-DEM-Ingester/1.0",
        },
    )

    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        if resp.status not in (200, 206):
            raise RuntimeError(f"NASA PDS returned HTTP status {resp.status}")
        raw_bytes = resp.read()

    num_lines = line_end - line_start
    full_slice = np.frombuffer(raw_bytes, dtype="<f4").reshape(num_lines, SAMPLES_PER_LINE)
    crop_km = full_slice[:, sample_start:sample_end]

    # Convert km relative to IAU reference sphere to metres elevation
    crop_m = crop_km * 1000.0
    return crop_m


def build_nasa_daedalus_crop(
    output_path: Path,
    seed: int = 42,
    quick: bool = False,
) -> TerrainGrid:
    """Ingest NASA SLDEM2015 elevation, resample to 200x200, compute slopes and export."""
    raw_m = fetch_pds_bounding_box(
        center_lat=DAEDALUS_LAT_DEG,
        center_lon=DAEDALUS_LON_DEG,
        extent_m=20_000.0,
    )
    resampled_m = resample_bilinear_2d(raw_m, target_shape=(200, 200))
    # Normalise so minimum basin elevation is zero
    resampled_m = resampled_m - float(np.min(resampled_m))

    # Save temporary npy for build_daedalus_crop ingestion
    temp_npy = output_path.parent / "_temp_nasa_dem.npy"
    np.save(temp_npy, resampled_m)

    try:
        terrain = build_daedalus_crop(
            seed=seed,
            quick=quick,
            input_dem_path=temp_npy,
        )
        terrain.to_npz(output_path)
    finally:
        if temp_npy.exists():
            temp_npy.unlink()

    return terrain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch NASA LRO LOLA SLDEM2015 elevation and build Daedalus terrain crop."
    )
    default_out = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "terrain"
        / "daedalus_nasa_sldem2015.npz"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="Target output path for the exported .npz file.",
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
        help="Generate 6-crater catalogue instead of full 12-crater catalogue.",
    )
    args = parser.parse_args(argv)

    print(f"Streaming NASA PDS SLDEM2015 tile for Daedalus crater...")
    terrain = build_nasa_daedalus_crop(
        output_path=args.output,
        seed=args.seed,
        quick=args.quick,
    )
    h, w = terrain.height_m.shape
    print(
        f"Exported NASA Daedalus terrain crop: {args.output} "
        f"({h}x{w}, {len(terrain.craters)} craters, "
        f"elev range {np.min(terrain.height_m):.1f}m to {np.max(terrain.height_m):.1f}m, "
        f"max slope {np.max(terrain.slope_deg):.1f} deg)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
