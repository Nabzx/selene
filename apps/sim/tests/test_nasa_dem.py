"""Unit tests for NASA SLDEM2015 ingestion and terrain building."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.fetch_nasa_dem import (
    PDS_TILE_URL,
    SAMPLES_PER_LINE,
    build_nasa_daedalus_crop,
    fetch_pds_bounding_box,
    main,
)
from selene_sim.sim import TerrainGrid


def test_fetch_pds_bounding_box_mocked() -> None:
    """Verify HTTP range calculation and byte parsing without network call."""
    mock_lines = 174
    mock_samples = SAMPLES_PER_LINE
    # Generate mock float32 raster
    synthetic_slice = np.linspace(-0.2, 1.1, mock_lines * mock_samples, dtype="<f4")
    raw_bytes = synthetic_slice.tobytes()

    mock_resp = MagicMock()
    mock_resp.status = 206
    mock_resp.read.return_value = raw_bytes
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        crop_m = fetch_pds_bounding_box(extent_m=20_000.0)

        assert crop_m.ndim == 2
        # Extent of ~20km yields ~170x170 cells
        assert 160 <= crop_m.shape[0] <= 180
        assert 160 <= crop_m.shape[1] <= 180
        # Elevation converted to metres
        assert np.max(crop_m) > 500.0
        assert mock_urlopen.called


def test_build_nasa_daedalus_crop(tmp_path: Path) -> None:
    """Test full NASA Daedalus crop building with mocked PDS response."""
    mock_lines = 174
    mock_samples = SAMPLES_PER_LINE
    synthetic_slice = np.full((mock_lines, mock_samples), 0.5, dtype="<f4")
    raw_bytes = synthetic_slice.tobytes()

    mock_resp = MagicMock()
    mock_resp.status = 206
    mock_resp.read.return_value = raw_bytes
    mock_resp.__enter__.return_value = mock_resp

    out_file = tmp_path / "nasa_test_crop.npz"

    with patch("urllib.request.urlopen", return_value=mock_resp):
        terrain = build_nasa_daedalus_crop(output_path=out_file, quick=True)

        assert out_file.exists()
        assert isinstance(terrain, TerrainGrid)
        assert terrain.height_m.shape == (200, 200)
        assert terrain.slope_deg.shape == (200, 200)
        assert terrain.temp_c.shape == (200, 200)
        assert len(terrain.craters) == 6

        # Reload from saved file
        loaded = TerrainGrid.from_npz(out_file)
        assert loaded.height_m.shape == (200, 200)
        assert len(loaded.craters) == 6


def test_main_cli(tmp_path: Path) -> None:
    """Test CLI invocation of fetch_nasa_dem script."""
    mock_lines = 174
    mock_samples = SAMPLES_PER_LINE
    synthetic_slice = np.full((mock_lines, mock_samples), 0.2, dtype="<f4")
    raw_bytes = synthetic_slice.tobytes()

    mock_resp = MagicMock()
    mock_resp.status = 206
    mock_resp.read.return_value = raw_bytes
    mock_resp.__enter__.return_value = mock_resp

    out_file = tmp_path / "cli_crop.npz"

    with patch("urllib.request.urlopen", return_value=mock_resp):
        rc = main(["--output", str(out_file), "--quick"])
        assert rc == 0
        assert out_file.exists()
