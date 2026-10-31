<div align="center">

# Selene

### A lunar hopper drone that proves every plan before it flies

**Selene plans prospecting hops into the permanently shadowed craters of the lunar south pole, on real NASA terrain. Gemini 4 Argon writes the safety rules from mission documents, designs the reward a policy learns from, and proposes each hop plan. A verifier stress-tests the plan a thousand times. Nothing is uplinked until every check passes and a person approves it.**

[![CI](https://github.com/Nabzx/selene/actions/workflows/ci.yml/badge.svg)](https://github.com/Nabzx/selene/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue?style=flat-square)](LICENSE)
[![Gemini 4 Argon](https://img.shields.io/badge/model-gemini--4--argon-4285F4?style=flat-square)](https://aistudio.google.com/learn/gemini-4-argon-developer-guide)

![Concept render: a fan of a thousand simulated hop trajectories over cratered terrain, a red cluster where one hop crosses a steep rim, then the revised plan turning fully green](docs/images/demo.gif)

<sub>Concept render, not live output. Replaced by a capture of the running app once it exists.</sub>

</div>

---

## About

Water ice at the lunar south pole sits in craters that never see sunlight. Rovers can't safely get in and out, and a commanded drone can't be flown by joystick: every plan crosses a communications link, so it has to be right before it leaves the ground.

Selene treats that as a verification problem.

1. **Argon reads the mission.** Gemini 4 Argon takes NASA safety standards, mission constraints and the terrain data, and writes a short, machine-checkable safety spec: fuel reserve, maximum landing slope, time in shadow, line of sight to Earth.
2. **Argon designs the learning problem.** It writes the reward function for a small reinforcement learning policy, reads the training curves, and rewrites the reward until the policy behaves.
3. **Argon proposes a plan.** A sequence of hops from a sunlit ridge into a shadowed crater and back.
4. **The verifier tries to break it.** The simulator replays the plan about a thousand times with thrust error, landing scatter, dust and sensor noise. Every run is checked against the spec. In the 3D view, each run is a line: green if it holds, red if it breaks a rule.
5. **Argon fixes what failed.** Failure statistics go back to Argon, which revises the plan. The cloud re-runs.
6. **A person approves the uplink.** The approval button only unlocks once the plan passes. Then the hopper flies it.

Everything runs on real data: LOLA terrain, permanently shadowed region maps, illumination and Diviner temperature maps. See [`docs/data/DATASETS.md`](docs/data/DATASETS.md).

The full project summary is in [`docs/OVERVIEW.md`](docs/OVERVIEW.md).

## Track

Built for the Gemini 4 Argon one-day hackathon, **autonomous systems that reason and verify**.

## Models

| Job | Model |
|---|---|
| Safety spec, reward design, hop planning, failure analysis | `gemini-4-argon` |
| Parallel scenario evaluation and run summaries | `gemini-3.8-flash` |
| Action guardrail on every proposed hop | `gemini-3.5-flash-lite` |
| Terrain textures for the 3D scene | `gemini-nano-banana-2.1` |

All model calls use the Interactions API through `google-genai`.

## Run it locally

Needs [uv](https://docs.astral.sh/uv/), Python 3.12+ and a Gemini API key.

```
git clone https://github.com/Nabzx/selene.git
cd selene
cp .env.example .env        # add GEMINI_API_KEY

# the simulation, verifier and Argon backend
cd apps/sim
uv run uvicorn selene_sim.api:app --reload

# the 3D view, in a second terminal, from the repo root
python3 -m http.server 5173 --directory apps/web
```

Open [http://localhost:5173](http://localhost:5173).

Status: scaffold. The run commands land with Phase 1, see [`ROADMAP.md`](ROADMAP.md).

## Structure

```
apps/sim     FastAPI backend: simulator, RL training, verifier, Argon calls
apps/web     no-build Three.js view, served as static files
data/        terrain crops and derived grids (raw downloads are gitignored)
docs/        overview, ADRs, datasets, demo script, research notes
scripts/     one-off tools, including the concept GIF renderer
```

This repository follows a phase-and-wayfinder structure. See [`AGENTS.md`](AGENTS.md) and [`ROADMAP.md`](ROADMAP.md).

## Licence

[MIT](LICENSE).
