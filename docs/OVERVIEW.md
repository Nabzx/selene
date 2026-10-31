# Selene: project overview

One page that says what Selene is, who it's for, how it works and how it's demoed.

**Tags:** `lunar` `far-side` `space` `world-models` `reinforcement-learning` `autonomous-systems` `verification` `monte-carlo` `human-in-the-loop` `gemini-4-argon` `threejs` `fastapi` `hackathon` `halloween`

**Track:** autonomous systems that reason and verify.

**Theme:** Halloween, eerie but credible. Lunar night, orange and violet palette, drifting dust, glowing scan beams. The science stays real.

---

## In one line

A hopper drone that learns to work alone on the far side of the Moon at night, and proves every route safe before it flies.

## The problem

Nothing on the far side can be flown by hand. Earth is never in view, so every command waits for a relay orbiter. At night there is no solar power and the surface drops below -170 °C. A machine there has to plan, act and recover on its own, and a bad plan can't be corrected mid-flight.

Training that autonomy is slow and risky, and checking a plan against mission rules is manual work spread across long standards documents.

## Who it's for

A surface operations lead on a far-side mission, the person who signs off each uplink. The same problem exists underground on Earth, where Kessel's machines work offline: learn the environment, plan inside a model of it, prove the plan before the link drops.

## How it works

```
mission docs ──► Argon ──► safety spec ─────────────────────────┐
                   │                                            │
                   ├──► reward ──► RL training in the simulator │
                   │        ▲              │                    │
                   │        └── curves ────┤                    │
                   │                       ▼                    │
                   │              learned world model           │
                   │                       │                    │
                   │   prediction errors ◄─┘                    │
                   ├──◄ (Argon explains them, proposes fixes)   │
                   │                                            ▼
                   └──► route ──► verifier: ~1000 noisy rollouts vs spec
                          ▲               │
                          └── failures ◄──┤ fail
                                          │ pass
                                          ▼
                         person approves in a relay window ──► uplink

training episodes ──► data/episodes.json ──► 3D view (replay)
```

| Job | What happens | Model |
|---|---|---|
| Safety spec | Mission documents become rules with executable checks | `gemini-4-argon`, high |
| Reward design | Argon writes the reward, reads training curves, rewrites it | `gemini-4-argon`, medium |
| Route planning | Argon picks craters and order, given battery, temperature and relay windows | `gemini-4-argon`, medium |
| Verification | About 1000 noisy rollouts, each checked against the spec | none, deterministic code |
| World-model review | Argon reads the error map and explains where the model is wrong and why | `gemini-4-argon`, medium |
| Mission log | Short narrated log entries, eerie in tone | `gemini-3.8-flash` |
| Guardrail | Every proposed hop screened for obviously unsafe actions | `gemini-3.5-flash-lite` |
| Approval | A person presses approve. Locked until the verdict passes and a relay window is open | none |

A model proposes, code checks, a person decides.

## The world model

The simulator is ground truth. The hopper learns a separate, small world model (a few-layer network) from simulator transitions: state and action in, next state out. The policy plans inside the world model, Dreamer-style, so it can try hops without spending fuel.

The world model is wrong in places, especially on terrain it hasn't visited. Those errors are the interesting part: the 3D view can show predicted and actual landings side by side, and Argon reads the error map and says where the model is unreliable and why.

## Why it needs Argon

- **Long documents in, exact rules out.** Mission and safety documents fit in the 1M-token context, and the rules must be precise enough to execute.
- **Reward code written and rewritten from training curves** (the Eureka approach).
- **Reasoning about a model's errors**, not just reporting them.
- **Route planning** under several interacting constraints.

Swap Argon for a small model and the spec gets vague, the reward revisions get random, and the demo breaks.

## Data

| Layer | Source | Real or simulated |
|---|---|---|
| Terrain, slopes | LOLA global DEM, SLDEM2015 for higher resolution below 60° latitude | Real |
| Night temperature | Diviner global temperature maps | Real |
| Relay windows | Relay orbit model | Simulated |
| Hopper flight, battery, thermal | Ballistic and simple energy models | Simulated |
| Payload readings | Depends on the payload decision | Simulated |

Details in [`data/DATASETS.md`](data/DATASETS.md).

## The 3D view

Built by a teammate. It plays `data/episodes.json` and nothing else (schema in [`data/EPISODE_SCHEMA.md`](data/EPISODE_SCHEMA.md)).

- Real far-side terrain at night. Starlight, the relay orbiter crossing the sky, no Earth.
- Training replay: early episodes crash and wander, later ones hop cleanly crater to crater. An episode counter and reward curve show progress.
- World-model ghost: a faint predicted hopper flies beside the real one. Where they diverge, the terrain glows.
- Trajectory cloud for the verifier: green and red lines, then all green after Argon's revision.
- Approval button that unlocks only on a pass inside a relay window.

## Demo, under 3 minutes

| Time | Beat |
|---|---|
| 0:00 | The far side at night. Why nothing here can be flown by hand |
| 0:20 | Argon read the mission documents. Show the safety spec it wrote |
| 0:35 | Training replay, episode 1 to 500. Reward curve, Argon's reward rewrite at the plateau |
| 1:05 | World-model ghost diverges on a crater rim. Argon explains why |
| 1:25 | Live: Argon plans a new route. The verifier runs, red cluster, Argon revises, all green |
| 2:15 | Relay window opens. Approve. The hopper flies |
| 2:45 | Close: learn the world, plan inside it, prove it, then act |

Precomputed and labelled: the training run and its episodes, a baseline spec. Live: route planning, verification, the revision, approval.

## Open decisions

- Payload: what the hopper collects at each crater. GitHub issue.
- Verifier depth: Monte Carlo only, or with Z3 for strict rules (ADR-0003).
- Which far-side region (candidates: Daedalus crater, Tsiolkovskiy, Apollo basin).

## Risks

| Risk | Mitigation |
|---|---|
| World model doesn't learn in time | Small grid, small network, logged episodes from a morning run |
| Argon slow on stage | Background mode started early, cached fallback labelled cached |
| 3D and Python drift apart | One episode schema, validated by a test |
| Venue wifi | All data on disk before kickoff |
