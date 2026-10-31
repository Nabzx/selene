<div align="center">

# Selene

### A fleet of hopper drones that learns to explore the lunar far side at night

**Selene trains a team of small rocket-hopping drones to explore the far side of the Moon at night on their own, since Earth is never in view and nobody can fly them by hand. Gemini 4 Argon is the single orchestrator: it reads the mission documents, plans every trajectory, checks the drones' logs and runs the verifier. Every plan is tested a thousand times in simulation before a person approves it.**

[![CI](https://github.com/Nabzx/selene/actions/workflows/ci.yml/badge.svg)](https://github.com/Nabzx/selene/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue?style=flat-square)](LICENSE)
[![Gemini 4 Argon](https://img.shields.io/badge/model-gemini--4--argon-4285F4?style=flat-square)](https://aistudio.google.com/learn/gemini-4-argon-developer-guide)

![Concept render: a fan of simulated hop trajectories across cratered far-side terrain at lunar night, a red failure cluster on a crater rim, then the revised plan turning fully green](docs/images/demo.gif)

<sub>Concept render, not live output.</sub>

</div>

---

## The idea

The far side of the Moon never faces Earth, so nothing there can be flown by hand, and during the two-week lunar night it drops below -170 °C with no solar power. Selene trains a fleet of four to five hoppers to work there alone, crater to crater, with Gemini 4 Argon as the mission commander that plans, checks and explains, and a person who approves every uplink.

## How it works

**Physics simulator.** A hand-coded copy of the Moon built on real NASA terrain, which calculates each hop under lunar gravity (1.62 m/s²), fuel use, battery drain, the night cold and when the relay orbiter is overhead. It is the ground truth: whatever happens in the simulator is what would really happen.

**World model.** A small neural network the fleet learns by watching the simulator, which predicts "if I do this hop, I'll probably end up here". All hoppers feed one shared world model, so terrain one hopper explores helps the others imagine outcomes before they act.

**Reinforcement learning.** The hoppers learn by trial and error, earning points for reaching craters and collecting data and losing points for crashes, collisions or running out of power. They practise thousands of times inside the world model with MAPPO (one shared policy, a centralised critic during training), and Argon rewrites the scoring when learning stalls.

**Multi-agent system.** The fleet is the multi-agent system: four to five hoppers that split up the craters, keep clear of each other and take turns on the single relay link. Argon is the single orchestrator above them, the only part that plans, reads documents and judges results.

**Argon, the orchestrator.**

| Job | What Argon does |
|---|---|
| Mission documents | Reads safety standards and mission constraints, writes machine-checkable safety rules |
| Reward design | Writes the reward the fleet learns from, reads training curves, rewrites it |
| Trajectory planning | Assigns craters to hoppers and plans every hop for the whole fleet |
| Verification | Sends the fleet plan to the verifier, reads the failures, revises the plan |
| Drone logs | Reads each hopper's log and the world model's prediction errors, explains what went wrong |

**Verifier.** Before anything flies, the simulator replays the fleet's plan about a thousand times with random thrust error, landing scatter and sensor noise, and checks every run against the safety rules: fuel reserve, landing slope, collisions, relay conflicts. The verifier is plain code, not a model. A model proposes, code checks, a person decides.

The full write-up is in [`docs/SUMMARY.md`](docs/SUMMARY.md).

## Track

Gemini 4 Argon one-day hackathon, **autonomous systems that reason and verify**.

## Run it locally

Needs [uv](https://docs.astral.sh/uv/), Python 3.12+ and a Gemini API key.

```
git clone https://github.com/Nabzx/selene.git
cd selene
cp .env.example .env        # add GEMINI_API_KEY
cd apps/sim
uv run uvicorn selene_sim.api:app --reload
```

Status: scaffold. See [`ROADMAP.md`](ROADMAP.md).

## Structure

```
apps/sim     simulator, world model, MAPPO training, verifier, Argon orchestrator
data/        terrain crop and logged episodes (raw downloads are gitignored)
docs/        summary, ADRs, datasets, research notes
scripts/     one-off tools
```

This repository follows a phase-and-wayfinder structure. See [`AGENTS.md`](AGENTS.md) and [`ROADMAP.md`](ROADMAP.md).

## Licence

[MIT](LICENSE).
