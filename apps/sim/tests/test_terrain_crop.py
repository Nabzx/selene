"""Unit tests for the Daedalus terrain crop generator and NPZ exporter."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.build_terrain_crop import (
    DAEDALUS_LAT_DEG,
    DAEDALUS_LON_DEG,
    R_MOON_M,
    build_daedalus_crop,
    compute_horn_slope_deg,
    local_xy_to_latlon_deg,
    main,
    resample_bilinear_2d,
)
from selene_sim.sim import TerrainGrid


def test_horn_slope_stencil_accuracy() -> None:
    """Verify Horn 3x3 finite-difference slope calculation against an analytical inclined plane."""
    cell_size_m = 100.0
    grid_n = 200
    coords = (np.arange(grid_n, dtype=np.float64) + 0.5) * cell_size_m
    xx, yy = np.meshgrid(coords, coords, indexing="xy")

    # Plane with dz/dx = 0.15 and dz/dy = -0.20 -> |grad z| = 0.25
    gx_true = 0.15
    gy_true = -0.20
    expected_slope_deg = math.degrees(math.atan(math.hypot(gx_true, gy_true)))

    height_plane = (gx_true * xx + gy_true * yy).astype(np.float32)
    slope_deg = compute_horn_slope_deg(height_plane, cell_size_m=cell_size_m)

    assert slope_deg.shape == (grid_n, grid_n)
    assert slope_deg.dtype == np.float32
    # Interior cells (excluding 1-cell padded border) must match analytical slope closely
    interior = slope_deg[1:-1, 1:-1]
    np.testing.assert_allclose(interior, expected_slope_deg, atol=1e-4)


def test_cylindrical_projection_and_bilinear_resampling() -> None:
    """Verify lunar cylindrical coordinate conversion and 2D bilinear DEM resampling."""
    lat, lon = local_xy_to_latlon_deg(0.0, 0.0)
    assert float(lat) == pytest.approx(DAEDALUS_LAT_DEG, abs=1e-6)
    assert float(lon) == pytest.approx(DAEDALUS_LON_DEG, abs=1e-6)
    assert R_MOON_M == pytest.approx(1_737_400.0)

    # Resample a linear 64x64 ramp to 200x200 and verify smooth preservation of bounds
    raw_y = np.linspace(0.0, 100.0, 64, dtype=np.float32)
    raw_x = np.linspace(-50.0, 50.0, 64, dtype=np.float32)
    raw_xx, raw_yy = np.meshgrid(raw_x, raw_y, indexing="xy")
    raw_dem = (raw_xx + raw_yy).astype(np.float32)

    resampled = resample_bilinear_2d(raw_dem, target_shape=(200, 200))
    assert resampled.shape == (200, 200)
    assert resampled.dtype == np.float32
    assert float(resampled[0, 0]) == pytest.approx(-50.0, abs=1e-4)
    assert float(resampled[-1, -1]) == pytest.approx(150.0, abs=1e-4)


def test_build_daedalus_crop_synthetic_and_external_dem(tmp_path: Path) -> None:
    """Verify build_daedalus_crop in both synthetic mode and external DEM ingest mode."""
    terrain = build_daedalus_crop(seed=42, quick=False)
    assert isinstance(terrain, TerrainGrid)
    assert terrain.height_m.shape == (200, 200)
    assert terrain.slope_deg.shape == (200, 200)
    assert terrain.temp_c.shape == (200, 200)
    assert terrain.cell_size_m == pytest.approx(100.0)
    assert terrain.region_name == "Daedalus"
    assert 12 <= len(terrain.craters) <= 15

    # Check lunar night temperature bounds [-230 C, -165 C]
    assert float(np.min(terrain.temp_c)) >= -230.0
    assert float(np.max(terrain.temp_c)) <= -165.0
    # Crater bowls must be significantly colder than inter-crater plains
    assert float(np.min(terrain.temp_c)) < -205.0
    assert float(np.max(terrain.temp_c)) > -175.0

    # Check Horn slope matches height_m
    expected_horn = compute_horn_slope_deg(terrain.height_m, cell_size_m=terrain.cell_size_m)
    np.testing.assert_allclose(terrain.slope_deg, expected_horn, atol=1e-5)

    # Test ingesting an external raw height array (.npy) via --input-dem
    raw_dem_path = tmp_path / "external_dem.npy"
    rng = np.random.default_rng(7)
    external_dem = rng.normal(0.0, 5.0, size=(120, 120)).astype(np.float32)
    np.save(raw_dem_path, external_dem)

    out_npz = tmp_path / "custom_daedalus.npz"
    rc = main(["--input-dem", str(raw_dem_path), "--output", str(out_npz), "--seed", "42"])
    assert rc == 0
    assert out_npz.exists()

    loaded_custom = TerrainGrid.from_npz(out_npz)
    assert loaded_custom.height_m.shape == (200, 200)
    assert 12 <= len(loaded_custom.craters) <= 15
    assert float(np.min(loaded_custom.temp_c)) >= -230.0
    assert float(np.max(loaded_custom.temp_c)) <= -165.0


def test_generated_daedalus_npz_bundle_out_of_the_box() -> None:
    """Verify data/terrain/daedalus_200x200.npz loads cleanly via TerrainGrid.from_npz."""
    repo_root = Path(__file__).resolve().parents[3]
    npz_path = repo_root / "data" / "terrain" / "daedalus_200x200.npz"
    assert npz_path.exists(), f"Expected pre-built terrain bundle at {npz_path}"

    terrain = TerrainGrid.from_npz(npz_path)
    assert terrain.height_m.shape == (200, 200)
    assert terrain.slope_deg.shape == (200, 200)
    assert terrain.temp_c.shape == (200, 200)
    assert terrain.cell_size_m == pytest.approx(100.0)
    assert terrain.half_extent_m == pytest.approx(10_000.0)
    assert terrain.region_name == "Daedalus"
    assert 12 <= len(terrain.craters) <= 15

    # Verify lunar night temperature bounds [-230 C, -165 C]
    assert float(np.min(terrain.temp_c)) >= -230.0
    assert float(np.max(terrain.temp_c)) <= -165.0

    # Verify Horn 3x3 finite-difference slope accuracy on the stored height_m grid
    horn_slope = compute_horn_slope_deg(terrain.height_m, cell_size_m=terrain.cell_size_m)
    np.testing.assert_allclose(terrain.slope_deg, horn_slope, atol=1e-5)

    # Verify crater centres are flat landing zones (< 10 deg) and cold bowls (< -195 C)
    for crater in terrain.craters:
        c_slope = float(terrain.slope_at(crater.x_m, crater.y_m))
        c_temp = float(terrain.temp_at(crater.x_m, crater.y_m))
        assert c_slope < 10.0
        assert c_temp < -195.0
