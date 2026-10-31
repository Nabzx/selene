"""Render the README concept GIF.

A labelled concept render, not live output: synthetic far-side terrain at
lunar night, a trajectory cloud with a red failure cluster at hop 3, Argon's
revision turning the cloud green, then the approved flight and a crater scan.

Run from the repo root:
    uv run --with numpy --with matplotlib --with pillow scripts/concept_gif.py
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

NIGHT = LinearSegmentedColormap.from_list("night", ["#05030a", "#1c1030", "#4a2a63", "#b07a5a", "#f2c49b"])

OUT = Path(__file__).resolve().parent.parent / "docs" / "images" / "demo.gif"
W, H = 320, 180
rng = np.random.default_rng(7)

# terrain: craters with raised rims plus fractal noise
y, x = np.mgrid[0:H, 0:W].astype(float)
z = np.zeros((H, W))
craters = [(220, 95, 52, 1.0)] + [
    (rng.uniform(0, W), rng.uniform(0, H), rng.uniform(5, 22), rng.uniform(0.3, 0.7))
    for _ in range(28)
]
for cx, cy, r, d in craters:
    rr = np.hypot(x - cx, y - cy) / r
    z += d * r * (np.where(rr < 1, rr**2 - 1, 0) * 0.6 + 0.25 * np.exp(-((rr - 1) ** 2) / 0.04))
for scale in (40, 20, 10, 5):
    n = rng.normal(size=(H // scale + 2, W // scale + 2)).astype(np.float32)
    up = Image.fromarray(n, mode="F").resize((W + 2 * scale, H + 2 * scale), Image.BICUBIC)
    z += np.asarray(up)[:H, :W] * scale * 0.012

gy, gx = np.gradient(z)
slope = np.hypot(gx, gy)

# low sun from the west: hillshade plus cast shadows
alt = np.radians(9)
shade = np.clip(0.5 + 1.6 * (gx * -1 + 0.3 * gy) / np.sqrt(1 + gx**2 + gy**2), 0.05, 1)
shadow = np.zeros_like(z, dtype=bool)
for step in range(1, 160):
    src = np.roll(z, step, axis=1)
    src[:, :step] = -1e9
    shadow |= src - step * np.tan(alt) > z
light = np.where(shadow, shade * 0.12, shade)
psr = shadow & (np.hypot(x - 220, y - 95) < 44)

start = np.array([30.0, 120.0])
target = np.array([222.0, 100.0])
plan_bad = np.array([start, [85, 95], [135, 105], [172, 92], target])
plan_good = np.array([start, [85, 95], [135, 105], [166, 122], target])


def steep(p):
    i, j = int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))
    return slope[i, j] > np.percentile(slope, 91)


def rollouts(plan, n, seed):
    r = np.random.default_rng(seed)
    runs = []
    for _ in range(n):
        pts, drift, ok = [plan[0]], np.zeros(2), True
        for k, wp in enumerate(plan[1:], start=1):
            drift = drift * 0.5 + r.normal(0, 2.2 + 0.8 * k, 2)
            p = wp + drift
            if k < len(plan) - 1 and steep(p):
                ok = False
            pts.append(p)
        runs.append((np.array(pts), ok))
    return runs


def arc(a, b, lift):
    t = np.linspace(0, 1, 18)[:, None]
    p = a + (b - a) * t
    p[:, 1] -= lift * np.sin(np.pi * t[:, 0]) * np.linalg.norm(b - a) / 6
    return p


bad = rollouts(plan_bad, 160, 1)
good = rollouts(plan_good, 160, 2)
# force the story: the red cluster sits at hop 3 on the steep rim
bad = [(p, i % 9 != 0) for i, (p, _) in enumerate(bad)]
for p, ok in bad:
    if not ok:
        p[3] += (7, -9)
good = [(p, True) for p, _ in good]
stars_x, stars_y = rng.uniform(0, W, 120), rng.uniform(0, H, 120)
ice = np.exp(-(((x - 228) ** 2) / 90 + ((y - 102) ** 2) / 60)) * (np.hypot(x - 220, y - 95) < 44)

frames = []


def frame(cloud, shown, verdict, button, hop_t=None, scan=0.0, banner=None):
    fig = plt.figure(figsize=(8, 4.5), dpi=90, facecolor="#05060a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(light, cmap=NIGHT, vmin=0, vmax=1.05, interpolation="bilinear")
    if scan:
        ax.imshow(np.ma.masked_less(ice * scan, 0.05), cmap="autumn", alpha=0.7, vmin=0, vmax=1)
    for p, ok in cloud[:shown]:
        for a, b in zip(p[:-1], p[1:]):
            q = arc(a, b, 1)
            ax.plot(q[:, 0], q[:, 1], color="#3ddc84" if ok else "#ff4d4d",
                    lw=0.5, alpha=0.32 if ok else 0.8)
    if hop_t is not None:
        path = np.vstack([arc(a, b, 1) for a, b in zip(plan_good[:-1], plan_good[1:])])
        k = int(hop_t * (len(path) - 1))
        ax.plot(path[: k + 1, 0], path[: k + 1, 1], color="#ff8c2a", lw=1.6)
        ax.scatter(*path[k], s=40, color="#ff8c2a", zorder=5)
    ax.scatter(*start, s=22, color="#ffd166", marker="^", zorder=5)
    ax.text(6, 10, "SELENE  ·  far side, lunar night  ·  Daedalus", color="#c9d1e0", fontsize=10,
            family="monospace", weight="bold")
    ax.text(6, 21, "concept render, not live output", color="#c9d1e0", fontsize=7,
            family="monospace", bbox=dict(facecolor="#05060a", edgecolor="none", alpha=0.7, pad=2))
    ax.scatter(stars_x, stars_y, s=0.6, color="#e8dff5", alpha=0.5)
    if verdict:
        colour = "#3ddc84" if "PASS" in verdict else "#ff4d4d" if "FAIL" in verdict else "#c9d1e0"
        ax.text(6, 168, verdict, color=colour, fontsize=9, family="monospace", weight="bold")
    if banner:
        ax.text(160, 150, banner, color="#c9d1e0", fontsize=9, family="monospace",
                ha="center", bbox=dict(facecolor="#11141c", edgecolor="#3a4258", pad=5))
    on = button == "on"
    ax.text(314, 168, "APPROVE UPLINK" if button != "sent" else "UPLINKED",
            color="#05060a" if on else "#7d8799", fontsize=9, family="monospace",
            weight="bold", ha="right",
            bbox=dict(facecolor="#3ddc84" if on else "#1a1f2b", edgecolor="none", pad=4))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128))


for i in range(1, 15):
    frame(bad, i * 12, "verifying hop plan v1 ...", "off")
for _ in range(6):
    frame(bad, len(bad), "FAIL  96.2% pass  ·  hop 3 slope > 15°", "off")
for _ in range(6):
    frame(bad, len(bad), "FAIL  96.2% pass  ·  hop 3 slope > 15°", "off",
          banner="Argon: moving hop 3 off the rim")
for i in range(1, 13):
    frame(good, i * 14, "verifying hop plan v2 ...", "off")
for _ in range(6):
    frame(good, len(good), "PASS  100% of 1000 rollouts", "on")
for i in range(16):
    frame(good[:40], 40, "PASS  ·  uplink approved", "sent", hop_t=i / 15)
for i in range(1, 11):
    frame(good[:40], 40, "scanning Hollow 3  ·  payload (simulated)", "sent", hop_t=1.0, scan=i / 10)
for _ in range(8):
    frame(good[:40], 40, "scanning Hollow 3  ·  payload (simulated)", "sent", hop_t=1.0, scan=1.0)

OUT.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=False)
print(OUT, len(frames), "frames", OUT.stat().st_size // 1024, "KB")
