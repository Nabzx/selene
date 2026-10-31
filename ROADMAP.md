# Roadmap

Built in one hackathon day. Phases are time-boxed, not feature-boxed: when the box ends, cut scope, don't extend it.

| Phase | Box | Goal | Done when |
|---|---|---|---|
| 0 | Before kickoff | Scaffold, docs, data downloaded | Repo public, terrain crop and PSR map on disk, first Argon call returns |
| 1 | 2 h | Grid and simulator | A hop plan runs on the real grid with fuel, slope, shadow and line-of-sight checks |
| 2 | 2 h | Argon loop | Argon writes the safety spec from documents, proposes a plan, reads a verdict, revises it |
| 3 | 1.5 h | Verifier | 1000 noisy rollouts per plan in under 10 s, verdict with per-rule failure stats |
| 4 | 2 h | 3D view | Real terrain, low sun, trajectory cloud in green and red, approve button gated on a pass |
| 5 | 1 h | RL and reward design | Argon writes the reward, a small policy trains in under 30 s, Argon reads the curves |
| 6 | 1 h | Demo hardening | 3-minute script rehearsed, cached fallbacks labelled, backup recording made |

Phase 5 is the first cut if time runs short. The demo still holds without it.

## Cut order

1. RL policy (Argon plans directly instead)
2. Nano Banana textures
3. Flash-Lite guardrail
4. Lunar day slider
