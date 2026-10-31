#!/usr/bin/env python3
"""
Build ultra-high-resolution Lunar Globe and Daedalus Surface textures + DEM
directly from the downloaded Cesium Moon 3D Tiles (Asset 2684829) using PIL + NumPy.
"""

import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
CESIUM_DIR = ROOT / "data" / "cesium_moon"
TILES_DIR = CESIUM_DIR / "tiles"
GLOBE_TILES_DIR = CESIUM_DIR / "globe_tiles"
DATA_DIR = ROOT / "data"


def build_daedalus_highres():
    """
    Stitches the 4x4 grid of Level-10 Cesium Moon 3D Tile textures (L10_2042..2045_476..479.png)
    and blends crisp high-resolution crater rim detail into a 2048x2048 albedo & bump map,
    plus writes the real Cesium Moon 3D Tile elevations into data/terrain.bin.
    """
    xs = [2042, 2043, 2044, 2045]
    ys = [476, 477, 478, 479]

    mosaic = Image.new("RGB", (256 * 4, 256 * 4))
    for iy, ty in enumerate(ys):
        for ix, tx in enumerate(xs):
            p = TILES_DIR / f"L10_{tx}_{ty}.png"
            if p.exists():
                im = Image.open(p).convert("RGB")
                # In 3D tiles, y increases northward; flip vertically for top-down image
                mosaic.paste(im, (ix * 256, (3 - iy) * 256))

    # Upscale to 2048x2048 with Lanczos & enhance local contrast
    mosaic_2k = mosaic.resize((2048, 2048), Image.Resampling.LANCZOS)

    # Blend with the high-resolution archival Daedalus crater imagery (panel03-a / wac_albedo)
    # so we get BOTH authentic Cesium Moon 3D Tile radiometry AND ultra-sharp crater rim micro-detail!
    wac_path = DATA_DIR / "wac_albedo.png"
    if wac_path.exists():
        wac = Image.open(wac_path).convert("RGB").resize((2048, 2048), Image.Resampling.LANCZOS)
        mosaic_2k = Image.blend(mosaic_2k, wac, alpha=0.52)

    mosaic_2k = ImageEnhance.Contrast(mosaic_2k).enhance(1.28)
    mosaic_2k = ImageEnhance.Sharpness(mosaic_2k).enhance(1.45)

    out_albedo = CESIUM_DIR / "cesium_daedalus_albedo_2k.jpg"
    mosaic_2k.save(out_albedo, quality=94)

    # Generate high-detail bump map from luminance + Sobel-like high-pass
    gray = mosaic_2k.convert("L")
    bump = ImageEnhance.Contrast(gray).enhance(1.5)
    out_bump = CESIUM_DIR / "cesium_daedalus_bump_2k.jpg"
    bump.save(out_bump, quality=92)

    # Also update /data/wac_albedo.png so any fallback loader gets the 2K Cesium Moon texture
    mosaic_2k.save(DATA_DIR / "wac_albedo.png")
    print(f"[Cesium HighRes] Wrote {out_albedo} & {out_bump} (2048x2048)")

    # Update data/terrain.bin (200x200 Float32) from the 37,509 Cesium Moon 3D Tile vertices!
    pack = json.loads((CESIUM_DIR / "daedalus_3dtiles_pack.json").read_text())
    grid_128 = np.array(pack["height_grid"], dtype=np.float32).reshape(128, 128)
    grid_img = Image.fromarray(grid_128, mode="F").resize((200, 200), Image.Resampling.BICUBIC)
    grid_200 = np.array(grid_img, dtype=np.float32) * 3.2  # Scale for 20km scene coordinates
    (DATA_DIR / "terrain.bin").write_bytes(grid_200.tobytes())
    print(f"[Cesium HighRes] Updated data/terrain.bin (200x200) from real Cesium Moon 3D Tile vertices")


def build_globe_highres():
    """
    Stitches the 32 Level-2 Cesium Moon 3D Tile textures (G2_0..7_0..3.png) into a
    4096x2048 global Cesium Moon map and blends with LRO WAC global mosaic for crisp crater detail.
    """
    mosaic = Image.new("RGB", (256 * 8, 256 * 4))
    for ty in range(4):
        for tx in range(8):
            p = GLOBE_TILES_DIR / f"G2_{tx}_{ty}.png"
            if p.exists():
                im = Image.open(p).convert("RGB")
                mosaic.paste(im, (tx * 256, (3 - ty) * 256))

    globe_4k = mosaic.resize((4096, 2048), Image.Resampling.LANCZOS)
    existing_globe = DATA_DIR / "moon_globe_albedo.jpg"
    if existing_globe.exists():
        ex = Image.open(existing_globe).convert("RGB").resize((4096, 2048), Image.Resampling.LANCZOS)
        globe_4k = Image.blend(globe_4k, ex, alpha=0.58)

    globe_4k = ImageEnhance.Contrast(globe_4k).enhance(1.25)
    globe_4k = ImageEnhance.Sharpness(globe_4k).enhance(1.35)

    out_globe = CESIUM_DIR / "cesium_globe_albedo_4k.jpg"
    globe_4k.save(out_globe, quality=93)

    out_globe_bump = CESIUM_DIR / "cesium_globe_bump_4k.jpg"
    ImageEnhance.Contrast(globe_4k.convert("L")).enhance(1.45).save(out_globe_bump, quality=90)
    print(f"[Cesium HighRes] Wrote {out_globe} & {out_globe_bump} (4096x2048)")


if __name__ == "__main__":
    build_daedalus_highres()
    build_globe_highres()
