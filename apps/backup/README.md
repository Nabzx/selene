# Selene · backup demo (`apps/backup`)

Stage-safe, zero-build version of the 8-panel Mission Control storyboard. It replays a scripted mission instead of calling the live sim/agent backend, so it cannot stall during a demo.

- **Terrain:** Cesium Moon Terrain (ion asset 2684829), with a texture fallback if ion is unreachable.
- **Numbers shown on the training scrubber:** interpolated from `data/training_log_nasa_extended_30min.csv` (4,112 updates × 128 envs ≈ 526k episodes, 15,790,080 transitions).
- **Paths, flight telemetry, verifier verdicts and logs:** scripted for the demo.

## Run

```bash
cd apps/backup
python3 -m http.server 8766
```

Open http://127.0.0.1:8766, then drag `mission_daedalus.txt` onto the Moon (or click **LOAD DAEDALUS MISSION**).

## Flow

1. Rotating Moon + relay orbiter → drop mission file
2. Zoom to Daedalus
3. Top-down map, view cube, `[DATA]`, `[INIT SIM]`
4. 60 candidate runs (red fail / green pass), training scrubber 100 → 526K episodes
5. Select hopper → `[EXECUTE / AUTHORISE UPLINK]`
6. Follow-cam flight with world-model ghost and live telemetry
7. `[MANUAL OVERRIDE]`: W/S pitch, A/D roll, Q/E yaw, SPACE engine, SHIFT cut, R respawn
8. `[TERMINATE]` → overview, summary, logs, `[EXPORT]` PNG

Camera: left-drag pan, scroll zoom, middle-drag or Ctrl+left-drag tilt. Click cube faces/edges for preset views.
