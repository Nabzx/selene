# Selene: project overview

One page that says what Selene is, who it's for, how it works and how it's demoed.

**Tags:** `lunar` `space` `autonomous-systems` `verification` `reinforcement-learning` `monte-carlo` `human-in-the-loop` `digital-twin` `gemini-4-argon` `threejs` `webgpu` `fastapi` `hackathon`

**Track:** autonomous systems that reason and verify.

---

## In one line

A lunar hopper drone that won't fly a plan until it has proved the plan safe, a thousand times over, on real NASA terrain.

## The problem

The water ice at the lunar south pole is inside permanently shadowed craters: below 40 K, no solar power, steep rims. Rovers can't safely get in and out. A hopper can, but nobody can fly it by hand across a communications link, so every plan must be right before it is sent.

Today that checking is slow and manual: engineers assemble the constraints from long standards documents, then hand-check a plan against them.

## Who it's for

A surface operations lead on a lunar prospecting mission, the person who signs off each command batch before uplink. The same problem exists underground on Earth, where Kessel's machines work offline: a plan has to be proven before the link drops.

## How it works

```
NASA docs + mission constraints ──► Argon ──► safety spec (rules + checks)
LOLA DEM, PSR, illumination,    ──► grid + simulator
Diviner temperature                    │
                                       ▼
                     Argon ──► reward function ──► RL policy trains (~30 s)
                       ▲                                 │
                       │                                 ▼
                 failure stats ◄── verifier ◄── hop plan (Argon + policy)
                       │          1000 noisy rollouts
                       │          checked against the spec
                       ▼
             pass? ──► person approves ──► uplink ──► hopper flies it in 3D
```

| Step | What happens | Model |
|---|---|---|
| Read the mission | Argon turns standards and constraints into a safety spec: rule id, statement, check | `gemini-4-argon`, high |
| Design the learning | Argon writes the reward, reads training curves, rewrites it | `gemini-4-argon`, medium |
| Plan | Argon proposes a hop plan, using the policy's suggestions | `gemini-4-argon`, medium |
| Verify | Simulator runs about 1000 rollouts with noise, checks each against the spec | none, deterministic code |
| Explain | Short per-run summaries for the log | `gemini-3.8-flash` |
| Guard | Every hop is screened for obviously unsafe actions | `gemini-3.5-flash-lite` |
| Revise | Argon reads failure stats and fixes the plan | `gemini-4-argon`, medium |
| Approve | A person presses approve. The button is locked until the verdict passes | none |

The verifier is plain code, not a model. A model proposes, code checks, a person decides.

## Why it needs Argon

- **Long documents in, precise rules out.** Safety standards and mission documents fit in the 1M-token context, and the rules have to be exact enough to execute.
- **Writing and revising reward code** from training curves, the Eureka approach.
- **Reasoning over failure statistics** to change the right hop, not just retry.
- **Multi-file code generation** for the simulator and checks, verified against the test suite.

Swap Argon for a small model and the spec gets vague, the revisions get random, and the demo breaks. That's the test.

## Data

| Layer | Source | Real or simulated |
|---|---|---|
| Terrain, slopes | LOLA 20 m south pole DEM | Real |
| Permanent shadow | LOLA PSR map | Real |
| Sunlight | Mazarico polar illumination maps | Real |
| Temperature | Diviner Polar Resource Product | Real |
| Ice candidates | Lemelin et al. 2021 PSR ice database, LAMP | Real |
| Ice abundance per cell | Sampled from the above plus noise | Simulated |
| Hopper fuel and flight | Ballistic model, 1.62 m/s² | Simulated |

Details and links in [`data/DATASETS.md`](data/DATASETS.md).

## The 3D view: verification you can see

- Real Shackleton terrain under a sun a few degrees above the horizon, long shadows, PSRs pitch black, Earth low on the horizon with a line-of-sight indicator.
- When a plan is checked, all of its rollouts appear as a trajectory cloud. Green passes, red fails.
- Argon's revision re-runs the cloud. The approve button unlocks only on a full pass.
- On uplink the hopper flies the plan, drops into the crater, and the scan fills in an ice heatmap. A comms delay countdown runs in the corner.

## Demo, under 3 minutes

| Time | Beat |
|---|---|
| 0:00 | The problem in one sentence, Shackleton on screen |
| 0:20 | Argon has read NASA's standards. Show the safety spec it wrote |
| 0:40 | Argon proposes a hop plan. The verifier runs: a red cluster at hop 3, a rim slope too steep |
| 1:10 | Argon reads the failure stats and revises. The cloud re-runs, fully green |
| 1:40 | Live input: a new constraint (a dust storm, a lower fuel reserve). Argon re-plans |
| 2:20 | Approve. The hopper flies into the crater, the ice heatmap fills in |
| 2:45 | Close: a model proposes, code checks, a person decides |

Precomputed in the morning and labelled as cached: the safety spec, a baseline plan, a trained policy, a backup screen recording. Live on stage: the new constraint, Argon's re-plan, the verifier run, the approval.

## Open decisions

- Verifier depth: Monte Carlo with rule checks, or Monte Carlo plus Z3 proofs for the strict rules. See ADR-0003.

## Risks

| Risk | Mitigation |
|---|---|
| Argon is slow on stage | Start the call in background mode at the top of the demo, cached fallback labelled as cached |
| Venue wifi | All data and textures on disk before kickoff |
| RL doesn't converge in time | Cut it, Argon plans directly (ROADMAP cut order) |
| 3D view is slow on the demo laptop | Grid at 100 m for the sim, decimated mesh for the view |
