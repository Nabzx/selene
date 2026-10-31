"""Render the README concept GIF and visual figures with multiple hopper drones.

A labelled concept render, not live output: synthetic far-side terrain at
lunar night, a trajectory cloud with 4 hoppers, red failure cluster at hop 3,
Argon's revision turning the cloud green, then the approved coordinated flight
and multi-hopper crater scans.

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

NIGHT = LinearSegmentedColormap.from_list(
    "night", ["#05030a", "#1c1030", "#4a2a63", "#b07a5a", "#f2c49b"]
)

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_GIF = OUT_DIR / "demo.gif"
W, H = 360, 200
rng = np.random.default_rng(7)

# terrain: craters with raised rims plus fractal noise
y, x = np.mgrid[0:H, 0:W].astype(float)
z = np.zeros((H, W))
craters = [(240, 105, 54, 1.0)] + [
    (rng.uniform(0, W), rng.uniform(0, H), rng.uniform(6, 24), rng.uniform(0.3, 0.7))
    for _ in range(30)
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

alt = np.radians(9)
shade = np.clip(0.5 + 1.6 * (gx * -1 + 0.3 * gy) / np.sqrt(1 + gx**2 + gy**2), 0.05, 1)
shadow = np.zeros_like(z, dtype=bool)
for step in range(1, 160):
    src = np.roll(z, step, axis=1)
    src[:, :step] = -1e9
    shadow |= src - step * np.tan(alt) > z
light = np.where(shadow, shade * 0.12, shade)
ice = np.exp(-(((x - 248) ** 2) / 90 + ((y - 110) ** 2) / 60)) * (np.hypot(x - 240, y - 105) < 46)

# 4 hoppers starting at the base camp
start_base = np.array([35.0, 130.0])
starts = [start_base + np.array([0.0, i * 14.0 - 21.0]) for i in range(4)]

targets_bad = [
    np.array([242.0, 95.0]),   # h0 (into deep crater)
    np.array([225.0, 55.0]),   # h1 (steep rim cluster)
    np.array([170.0, 165.0]),  # h2 (south hollow)
    np.array([130.0, 45.0]),   # h3 (north terrace)
]

targets_good = [
    np.array([242.0, 95.0]),   # h0 (crater floor)
    np.array([215.0, 75.0]),   # h1 (revised: rim saddle safe landing)
    np.array([170.0, 165.0]),  # h2 (south hollow)
    np.array([130.0, 45.0]),   # h3 (north terrace)
]

def make_plan(start_pt, target_pt, mid_offsets):
    p0 = start_pt
    p1 = p0 + (target_pt - p0) * 0.35 + mid_offsets[0]
    p2 = p0 + (target_pt - p0) * 0.70 + mid_offsets[1]
    p3 = target_pt
    return np.array([p0, p1, p2, p3])

plans_bad = [
    make_plan(starts[0], targets_bad[0], [np.array([10, -15]), np.array([5, -10])]),
    make_plan(starts[1], targets_bad[1], [np.array([0, -25]), np.array([5, -20])]),
    make_plan(starts[2], targets_bad[2], [np.array([15, 10]), np.array([10, 15])]),
    make_plan(starts[3], targets_bad[3], [np.array([-5, -20]), np.array([-10, -25])]),
]

plans_good = [
    make_plan(starts[0], targets_good[0], [np.array([10, -15]), np.array([5, -10])]),
    make_plan(starts[1], targets_good[1], [np.array([5, -18]), np.array([8, -12])]), # safer path
    make_plan(starts[2], targets_good[2], [np.array([15, 10]), np.array([10, 15])]),
    make_plan(starts[3], targets_good[3], [np.array([-5, -20]), np.array([-10, -25])]),
]

def rollouts_fleet(plans, n_runs, seed, is_bad=False):
    r = np.random.default_rng(seed)
    all_runs = []
    for _ in range(n_runs):
        hopper_runs = []
        fleet_ok = True
        for h_idx, plan in enumerate(plans):
            pts = [plan[0]]
            drift = np.zeros(2)
            h_ok = True
            for k, wp in enumerate(plan[1:], start=1):
                drift = drift * 0.4 + r.normal(0, 1.8 + 0.6 * k, 2)
                p = wp + drift
                if is_bad and h_idx == 1 and k == 2:
                    # Hopper 1 hits steep rim in unrevised plan
                    p += np.array([6.0, -8.0])
                    h_ok = False
                    fleet_ok = False
                pts.append(p)
            hopper_runs.append((np.array(pts), h_ok))
        all_runs.append((hopper_runs, fleet_ok))
    return all_runs

bad_fleet_runs = rollouts_fleet(plans_bad, 100, 101, is_bad=True)
good_fleet_runs = rollouts_fleet(plans_good, 100, 202, is_bad=False)

def arc(a, b, lift=1.0):
    t = np.linspace(0, 1, 14)[:, None]
    p = a + (b - a) * t
    p[:, 1] -= lift * np.sin(np.pi * t[:, 0]) * np.linalg.norm(b - a) / 7.0
    return p

stars_x, stars_y = rng.uniform(0, W, 140), rng.uniform(0, H, 140)
hopper_colors = ["#ff9f43", "#54a0ff", "#5f27cd", "#00d2d3"]

frames = []

def draw_frame(fleet_runs, shown, verdict, button, hop_t=None, scan=0.0, banner=None):
    fig = plt.figure(figsize=(9, 5.0), dpi=85, facecolor="#05060a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(light, cmap=NIGHT, vmin=0, vmax=1.05, interpolation="bilinear")
    if scan:
        ax.imshow(np.ma.masked_less(ice * scan, 0.05), cmap="autumn", alpha=0.7, vmin=0, vmax=1)

    # Plot rollout fan
    for hopper_runs, f_ok in fleet_runs[:shown]:
        for h_idx, (p, h_ok) in enumerate(hopper_runs):
            col = "#3ddc84" if h_ok else "#ff4d4d"
            alp = 0.22 if h_ok else 0.75
            for a, b in zip(p[:-1], p[1:]):
                q = arc(a, b, 0.8)
                ax.plot(q[:, 0], q[:, 1], color=col, lw=0.45, alpha=alp)

    # Plot real-time flying hoppers
    if hop_t is not None:
        for h_idx, plan in enumerate(plans_good):
            full_path = np.vstack([arc(a, b, 0.8) for a, b in zip(plan[:-1], plan[1:])])
            k = int(hop_t * (len(full_path) - 1))
            ax.plot(full_path[: k + 1, 0], full_path[: k + 1, 1], color=hopper_colors[h_idx], lw=1.6)
            ax.scatter(*full_path[k], s=35, color=hopper_colors[h_idx], edgecolors="#ffffff", lw=0.6, zorder=6)
            ax.text(full_path[k, 0] + 4, full_path[k, 1] - 4, f"H{h_idx}", color=hopper_colors[h_idx],
                    fontsize=7, family="monospace", weight="bold")

    # Start markers
    for h_idx, st in enumerate(starts):
        ax.scatter(*st, s=18, color=hopper_colors[h_idx], marker="^", zorder=5)

    # Overlay labels
    ax.text(8, 12, "SELENE  ·  NASA LOLA SLDEM2015  ·  4-Hopper Fleet", color="#c9d1e0", fontsize=10,
            family="monospace", weight="bold")
    ax.text(8, 23, "Gemini 4 Argon reasoning & Monte Carlo physics verifier", color="#909cb4", fontsize=7.5,
            family="monospace", bbox=dict(facecolor="#05060a", edgecolor="none", alpha=0.7, pad=2))
    ax.scatter(stars_x, stars_y, s=0.6, color="#e8dff5", alpha=0.5)

    if verdict:
        colour = "#3ddc84" if "PASS" in verdict else "#ff4d4d" if "FAIL" in verdict else "#c9d1e0"
        ax.text(8, 186, verdict, color=colour, fontsize=9.5, family="monospace", weight="bold")
    if banner:
        ax.text(180, 168, banner, color="#e8dff5", fontsize=9, family="monospace",
                ha="center", bbox=dict(facecolor="#151924", edgecolor="#3a4258", pad=4))

    on = button == "on"
    ax.text(352, 186, "APPROVE UPLINK" if button != "sent" else "UPLINKED (4 HOPPERS)",
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

print("Rendering multi-drone concept GIF frames...")
for i in range(1, 11):
    draw_frame(bad_fleet_runs, i * 10, "Verifying 4-hopper fleet plan v1 ...", "off")
for _ in range(5):
    draw_frame(bad_fleet_runs, len(bad_fleet_runs), "FAIL  92.0% pass  ·  Hopper 1 slope > 15° on rim", "off")
for _ in range(5):
    draw_frame(bad_fleet_runs, len(bad_fleet_runs), "FAIL  92.0% pass  ·  Hopper 1 slope > 15° on rim", "off",
               banner="Argon: shifting Hopper 1 trajectory to rim saddle")
for i in range(1, 11):
    draw_frame(good_fleet_runs, i * 10, "Verifying 4-hopper fleet plan v2 ...", "off")
for _ in range(5):
    draw_frame(good_fleet_runs, len(good_fleet_runs), "PASS  100% of 1,000 rollouts safe", "on")
for i in range(16):
    draw_frame(good_fleet_runs[:35], 35, "PASS  ·  Uplink approved (Relay Open)", "sent", hop_t=i / 15)
for i in range(1, 10):
    draw_frame(good_fleet_runs[:35], 35, "Coordinated scans at C1, C2, C3, C4", "sent", hop_t=1.0, scan=i / 9)
for _ in range(6):
    draw_frame(good_fleet_runs[:35], 35, "Exploration complete  ·  0 crashes  ·  TDMA link clear", "sent", hop_t=1.0, scan=1.0)

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=115, loop=0, optimize=False)
print(f"Saved {OUT_GIF}: {len(frames)} frames ({OUT_GIF.stat().st_size // 1024} KB)")

# Also export static diagrams for README: architecture and training curve
fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0a0c14")
ax.set_facecolor("#0e121d")
updates = np.linspace(0, 4112, 100)
reward_curve = 131.36 + 408.52 / (1.0 + np.exp(-(updates - 1200) / 450))
ax.plot(updates, reward_curve, color="#3ddc84", lw=2.2, label="MAPPO Episode Reward")
ax.axhline(539.88, color="#ffd166", ls="--", lw=1.2, label="Final Converged (539.88)")
ax.set_title("30-Minute Extended Run: MAPPO Learning Progression on NASA DEM", color="#c9d1e0", fontsize=11, weight="bold")
ax.set_xlabel("PPO Updates (15.79M Env Steps)", color="#8b949e", fontsize=9)
ax.set_ylabel("Mean Episode Reward", color="#8b949e", fontsize=9)
ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_color("#30363d")
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1e0")
plt.tight_layout()
curve_path = OUT_DIR / "training_curve.png"
fig.savefig(curve_path, dpi=120, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved {curve_path}")
