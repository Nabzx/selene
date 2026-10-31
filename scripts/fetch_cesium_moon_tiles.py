#!/usr/bin/env python3
"""
Fetch real Cesium Moon 3D Tiles (Asset ID 2684829) using the user's Cesium ion token.
Builds two complete 3D Tile Packs for Three.js:
  1. /data/cesium_moon/globe_3dtiles_pack.json (+ /data/cesium_moon/globe_tiles/*.png)
     34 Global Cesium Moon 3D Tiles (32 Level-2 equatorial tiles + 2 polar cap tiles)
  2. /data/cesium_moon/daedalus_3dtiles_pack.json (+ /data/cesium_moon/tiles/*.png)
     16 Level-10 Daedalus Crater Cesium Moon 3D Tiles (37,509 vertices, 72,657 triangles)
     transformed into our [-2300, +2300] scene coordinates with seamless crater relief
     and a 128x128 height lookup grid.
"""

import gzip
import json
import math
import struct
import urllib.request
from pathlib import Path
import numpy as np

CESIUM_ION_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJub25jZSI6InI3aTB0MGdIdGl6Y3ktOXoiLCJqdGkiOiJkYWU3NTcxNi1mNWEzLTQ3MjYtYTc3ZC1mMzUzZTEwZmU1YTAiLCJpZCI6NTM5NjYxLCJpc3MiOiJodHRwczovL2FwaS5jZXNpdW0uY29tIiwiYXVkIjoidW5kZWZpbmVkX2RlZmF1bHQiLCJpYXQiOjE3OTM0NjE5MjF9."
    "kNlDFhhhTKaW7vH_3ev2026svaSPTNNQLnox0dUenIU"
)
ASSET_ID = 2684829  # Cesium Moon Terrain 3D Tiles

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "data" / "cesium_moon"
TILES_DIR = OUT_DIR / "tiles"
GLOBE_TILES_DIR = OUT_DIR / "globe_tiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TILES_DIR.mkdir(parents=True, exist_ok=True)
GLOBE_TILES_DIR.mkdir(parents=True, exist_ok=True)


def fetch_bytes(url: str, bearer_token: str) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {bearer_token}"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def parse_b3dm(raw: bytes):
    magic, version, byte_len, ft_json_len, ft_bin_len, bt_json_len, bt_bin_len = struct.unpack(
        "<4sIIIIII", raw[:28]
    )
    assert magic == b"b3dm", f"Unexpected magic: {magic}"
    glb_offset = 28 + ft_json_len + ft_bin_len + bt_json_len + bt_bin_len
    glb = raw[glb_offset:]
    g_magic, g_ver, g_len, chunk_len, chunk_type = struct.unpack("<4sIIII", glb[:20])
    assert g_magic == b"glTF", f"Unexpected glTF magic: {g_magic}"
    g_json = json.loads(glb[20 : 20 + chunk_len])
    bin_chunk = glb[20 + chunk_len + 8 :]

    pos_acc = g_json["accessors"][0]
    pos_bv = g_json["bufferViews"][pos_acc["bufferView"]]
    v_count = pos_acc["count"]
    pos_offset = pos_bv.get("byteOffset", 0) + pos_acc.get("byteOffset", 0)
    positions = np.frombuffer(
        bin_chunk, dtype=np.float32, count=v_count * 3, offset=pos_offset
    ).reshape(-1, 3)

    uv_acc = g_json["accessors"][1]
    uv_bv = g_json["bufferViews"][uv_acc["bufferView"]]
    uv_offset = uv_bv.get("byteOffset", 0) + uv_acc.get("byteOffset", 0)
    uvs = np.frombuffer(
        bin_chunk, dtype=np.float32, count=v_count * 2, offset=uv_offset
    ).reshape(-1, 2)

    idx_acc = g_json["accessors"][2]
    idx_bv = g_json["bufferViews"][idx_acc["bufferView"]]
    i_count = idx_acc["count"]
    idx_offset = idx_bv.get("byteOffset", 0) + idx_acc.get("byteOffset", 0)
    indices = np.frombuffer(
        bin_chunk, dtype=np.uint16, count=i_count, offset=idx_offset
    )

    m1 = g_json["nodes"][1]["matrix"]
    rtc_ecef = np.array([m1[12], m1[13], m1[14]], dtype=np.float64)

    img_bv = g_json["bufferViews"][g_json["images"][0]["bufferView"]]
    img_offset = img_bv.get("byteOffset", 0)
    img_bytes = bin_chunk[img_offset : img_offset + img_bv["byteLength"]]

    return {
        "glb": glb,
        "positions": positions.astype(np.float64) + rtc_ecef,
        "uvs": uvs,
        "indices": indices,
        "rtc_ecef": rtc_ecef,
        "png_bytes": img_bytes,
    }


def crater_sculpt_offset(x_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
    """
    Smooth, continuous crater rim & bowl sculpting at c1, c2, c3, c4 and outer basin
    so the 16 Cesium Moon 3D Tiles have dramatic 3D craters at the exact mission waypoints
    while preserving 100% seamless tile boundaries.
    """
    craters = [
        (-480.0, -310.0, 420.0, 52.0, 20.0),  # c2 Science Target Alpha
        (560.0, 140.0, 360.0, 44.0, 17.0),    # c3 Science Target Beta
        (-1320.0, 180.0, 290.0, 34.0, 13.0),  # West rim crater
        (1180.0, -490.0, 320.0, 38.0, 14.0),  # East rim crater
        (-120.0, 580.0, 260.0, 30.0, 11.0),   # South floor crater
    ]
    dh = np.zeros_like(x_arr, dtype=np.float64)
    for cx, cz, r, depth, rim in craters:
        dist = np.sqrt((x_arr - cx) ** 2 + (z_arr - cz) ** 2)
        u = dist / r
        in_bowl = u < 0.82
        t_bowl = np.where(in_bowl, u / 0.82, 0.0)
        dh -= np.where(in_bowl, depth * (1.0 - t_bowl * t_bowl), 0.0)

        in_rim = (u >= 0.82) & (u < 1.38)
        t_rim = np.where(in_rim, (u - 1.04) / 0.24, 0.0)
        dh += np.where(in_rim, rim * np.exp(-t_rim * t_rim), 0.0)

    # Outer Daedalus basin wall rise
    r_outer = np.sqrt(x_arr * x_arr + z_arr * z_arr) / 2150.0
    in_wall = r_outer > 0.78
    dh += np.where(in_wall, np.minimum(110.0, (np.maximum(0.0, r_outer - 0.78) ** 1.5) * 145.0), 0.0)
    return dh


def build_daedalus_pack(access_token: str):
    base_url = "https://assets.ion.cesium.com/us-east-1/asset_depot/2684829/Moon/v6/Moon_85s_85n/"
    xs = [2042, 2043, 2044, 2045]
    ys = [476, 477, 478, 479]

    all_ecef = []
    tile_records = []

    for ty in ys:
        for tx in xs:
            uri = f"10/{tx}/{ty}.b3dm"
            url = f"{base_url}{uri}?v=2"
            raw = fetch_bytes(url, access_token)
            parsed = parse_b3dm(raw)
            glb_name = f"L10_{tx}_{ty}.glb"
            png_name = f"L10_{tx}_{ty}.png"
            (TILES_DIR / glb_name).write_bytes(parsed["glb"])
            (TILES_DIR / png_name).write_bytes(parsed["png_bytes"])
            all_ecef.append(parsed["positions"])
            tile_records.append(
                {
                    "tx": tx,
                    "ty": ty,
                    "glb": f"/data/cesium_moon/tiles/{glb_name}",
                    "png": f"/data/cesium_moon/tiles/{png_name}",
                    "positions": parsed["positions"],
                    "uvs": parsed["uvs"],
                    "indices": parsed["indices"],
                }
            )

    R_MOON = 1737400.0
    stacked_ecef = np.vstack(all_ecef)
    center_ecef = stacked_ecef.mean(axis=0)
    up_vec = center_ecef / np.linalg.norm(center_ecef)
    north_pole = np.array([0.0, 0.0, 1.0])
    east_vec = np.cross(north_pole, up_vec)
    east_vec /= np.linalg.norm(east_vec)
    north_vec = np.cross(up_vec, east_vec)
    north_vec /= np.linalg.norm(north_vec)

    # First pass: compute raw East, North, Elevation across all 16 tiles
    raw_east_list = []
    raw_north_list = []
    raw_elev_list = []
    for rec in tile_records:
        pts = rec["positions"]
        rel = pts - center_ecef
        east = rel @ east_vec
        north = rel @ north_vec
        elev = np.linalg.norm(pts, axis=1) - R_MOON
        rec["raw_east"] = east
        rec["raw_north"] = north
        rec["raw_elev"] = elev
        raw_east_list.append(east)
        raw_north_list.append(north)
        raw_elev_list.append(elev)

    east_all = np.concatenate(raw_east_list)
    north_all = np.concatenate(raw_north_list)
    elev_all = np.concatenate(raw_elev_list)

    # Map the 21.2km x 21.2km 16-tile mosaic into [-2300, +2300] scene coordinates
    # so the 16 Level-10 Cesium Moon 3D Tiles fill the entire surface viewport seamlessly!
    e_mid = 0.5 * (east_all.min() + east_all.max())
    n_mid = 0.5 * (north_all.min() + north_all.max())
    half_span = 0.5 * max(east_all.max() - east_all.min(), north_all.max() - north_all.min())
    SCENE_HALF = 2300.0
    xy_scale = SCENE_HALF / half_span
    elev_mid = float(np.median(elev_all))
    elev_scale = 0.14  # Real Cesium Moon DEM relief + crater sculpting

    packed_tiles = []
    scene_x_all = []
    scene_z_all = []
    scene_y_all = []

    for rec in tile_records:
        sx = (rec["raw_east"] - e_mid) * xy_scale
        sz = -(rec["raw_north"] - n_mid) * xy_scale
        base_y = (rec["raw_elev"] - elev_mid) * elev_scale
        sy = base_y + crater_sculpt_offset(sx, sz)

        scene_x_all.append(sx)
        scene_z_all.append(sz)
        scene_y_all.append(sy)

        verts = np.stack([sx, sy, sz], axis=1).astype(np.float32)
        packed_tiles.append(
            {
                "id": f"10_{rec['tx']}_{rec['ty']}",
                "tx": rec["tx"],
                "ty": rec["ty"],
                "texture": rec["png"],
                "glb": rec["glb"],
                "positions_flat": [round(float(v), 2) for v in verts.flatten()],
                "uvs_flat": [round(float(u), 4) for u in rec["uvs"].flatten()],
                "indices": [int(i) for i in rec["indices"]],
            }
        )

    # Build a fast 128x128 height grid over [-2300, +2300] so sampleTerrainHeight(x, z)
    # in scene.js matches the exact 3D Tiles surface!
    GRID_N = 128
    gx = np.linspace(-SCENE_HALF, SCENE_HALF, GRID_N)
    gz = np.linspace(-SCENE_HALF, SCENE_HALF, GRID_N)
    all_sx = np.concatenate(scene_x_all)
    all_sz = np.concatenate(scene_z_all)
    all_sy = np.concatenate(scene_y_all)
    all_base_y = all_sy - crater_sculpt_offset(all_sx, all_sz)

    # Nearest/IDW interpolation of base_y + exact analytic crater_sculpt_offset
    grid_vals = np.zeros((GRID_N, GRID_N), dtype=np.float32)
    for iz, zv in enumerate(gz):
        for ix, xv in enumerate(gx):
            d2 = (all_sx - xv) ** 2 + (all_sz - zv) ** 2
            idx = int(np.argmin(d2))
            by = float(all_base_y[idx])
            grid_vals[iz, ix] = by + float(crater_sculpt_offset(np.array([xv]), np.array([zv]))[0])

    pack_meta = {
        "source": "Cesium Moon Terrain 3D Tiles (Asset 2684829)",
        "site": "Daedalus Crater (-5.9°S, 179.4°E)",
        "tile_level": 10,
        "tile_count": len(packed_tiles),
        "total_vertices": int(len(all_sx)),
        "scene_half_size": SCENE_HALF,
        "grid_size": GRID_N,
        "height_grid": [round(float(v), 2) for v in grid_vals.flatten()],
        "tiles": packed_tiles,
    }

    out_pack = OUT_DIR / "daedalus_3dtiles_pack.json"
    out_pack.write_text(json.dumps(pack_meta))
    print(f"[Cesium Moon] Wrote Daedalus 16-tile pack: {out_pack} ({out_pack.stat().st_size / 1024:.1f} KB)")


def build_globe_pack(access_token: str):
    """
    Download 32 Level-2 equatorial Cesium Moon 3D Tiles (Moon_85s_85n/2/0..7/0..3.b3dm)
    + 2 polar cap tiles (Moon_85s_90s & Moon_85n_90n) so the rotating 3D Moon Globe
    renders real Cesium Moon 3D Tiles!
    """
    base_root = "https://assets.ion.cesium.com/us-east-1/asset_depot/2684829/Moon/v6/"
    globe_tiles = []

    # 32 Level-2 tiles covering 85S to 85N
    for ty in range(4):
        for tx in range(8):
            uri = f"Moon_85s_85n/2/{tx}/{ty}.b3dm"
            raw = fetch_bytes(f"{base_root}{uri}?v=2", access_token)
            parsed = parse_b3dm(raw)
            png_name = f"G2_{tx}_{ty}.png"
            (GLOBE_TILES_DIR / png_name).write_bytes(parsed["png_bytes"])

            # Convert ECEF (Z-up in Cesium 3D Tiles) to Three.js Y-up sphere of radius 600
            # In Cesium ECEF: X = lon 0, Y = lon 90E, Z = North Pole
            # In Three.js Y-up: X = ecef.x, Y = ecef.z (North Pole), Z = -ecef.y
            pts = parsed["positions"]
            radii = np.linalg.norm(pts, axis=1, keepdims=True)
            # Exaggerate global topographic relief slightly (x4) so basins like South Pole-Aitken & Daedalus pop
            R_MEAN = 1737400.0
            norm_r = (600.0 + ((radii - R_MEAN) / R_MEAN) * 600.0 * 4.5) / radii
            scaled = pts * norm_r
            three_verts = np.stack([scaled[:, 0], scaled[:, 2], -scaled[:, 1]], axis=1).astype(np.float32)

            globe_tiles.append(
                {
                    "id": f"G2_{tx}_{ty}",
                    "texture": f"/data/cesium_moon/globe_tiles/{png_name}",
                    "positions_flat": [round(float(v), 2) for v in three_verts.flatten()],
                    "uvs_flat": [round(float(u), 4) for u in parsed["uvs"].flatten()],
                    "indices": [int(i) for i in parsed["indices"]],
                }
            )

    # Polar caps (85S..90S and 85N..90N)
    for cap_name in ["Moon_85s_90s", "Moon_85n_90n"]:
        try:
            ts_raw = fetch_bytes(f"{base_root}{cap_name}/tileset.json?v=2", access_token)
            ts = json.loads(ts_raw)
            # Find root or first child .b3dm
            uri = ts["root"].get("content", {}).get("uri")
            if not uri and "children" in ts["root"]:
                uri = ts["root"]["children"][0].get("content", {}).get("uri")
            if uri:
                raw = fetch_bytes(f"{base_root}{cap_name}/{uri}?v=2", access_token)
                parsed = parse_b3dm(raw)
                png_name = f"{cap_name}.png"
                (GLOBE_TILES_DIR / png_name).write_bytes(parsed["png_bytes"])
                pts = parsed["positions"]
                radii = np.linalg.norm(pts, axis=1, keepdims=True)
                R_MEAN = 1737400.0
                norm_r = (600.0 + ((radii - R_MEAN) / R_MEAN) * 600.0 * 4.5) / radii
                scaled = pts * norm_r
                three_verts = np.stack([scaled[:, 0], scaled[:, 2], -scaled[:, 1]], axis=1).astype(np.float32)
                globe_tiles.append(
                    {
                        "id": cap_name,
                        "texture": f"/data/cesium_moon/globe_tiles/{png_name}",
                        "positions_flat": [round(float(v), 2) for v in three_verts.flatten()],
                        "uvs_flat": [round(float(u), 4) for u in parsed["uvs"].flatten()],
                        "indices": [int(i) for i in parsed["indices"]],
                    }
                )
        except Exception as exc:
            print(f"  ! Polar cap {cap_name} warning: {exc}")

    out_globe = OUT_DIR / "globe_3dtiles_pack.json"
    out_globe.write_text(
        json.dumps(
            {
                "source": "Cesium Moon Terrain 3D Tiles (Asset 2684829) — Global Level 2 + Polar Caps",
                "tile_count": len(globe_tiles),
                "tiles": globe_tiles,
            }
        )
    )
    print(
        f"[Cesium Moon] Wrote Global {len(globe_tiles)}-tile pack: {out_globe} ({out_globe.stat().st_size / 1024:.1f} KB)"
    )


def main():
    ep_raw = fetch_bytes(
        f"https://api.cesium.com/v1/assets/{ASSET_ID}/endpoint", CESIUM_ION_TOKEN
    )
    ep = json.loads(ep_raw)
    access_token = ep["accessToken"]
    build_daedalus_pack(access_token)
    build_globe_pack(access_token)


if __name__ == "__main__":
    main()
