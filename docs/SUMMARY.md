# Selene: in-depth summary

What Selene is, why it exists, and how each part works.

**Tags:** `lunar` `far-side` `space` `multi-agent` `reinforcement-learning` `mappo` `world-models` `physics-simulation` `autonomous-systems` `verification` `monte-carlo` `human-in-the-loop` `gemini-4-argon` `fastapi` `hackathon`

**Track:** autonomous systems that reason and verify.

---

## 1. The idea

Selene trains a fleet of four to five small rocket-hopping drones to explore the far side of the Moon at night on their own. Gemini 4 Argon is the single orchestrator: it reads the mission documents, writes the safety rules and the reward, plans every trajectory, checks the drones' logs and drives the verifier. Every plan is replayed about a thousand times in a physics simulator before a person approves it.

In one line: learn the world, plan inside it, prove the plan, then act.

## 2. The problem

- **No line of sight to Earth.** The far side never faces Earth. Every command goes through a relay orbiter that is overhead only part of the time, so nothing can be flown by hand and a bad plan can't be corrected mid-flight.
- **Lunar night.** About 14.75 Earth days without sunlight. The surface falls below -170 °C, there is no solar charging, and keeping electronics warm drains the battery.
- **Terrain.** Crater rims are steep and rough. A hopper that lands on a slope tips over.
- **Several machines, one link.** A fleet covers more ground than one hopper, but the hoppers must not collide, must not chase the same crater, and must share one relay link.
- **Checking plans is slow.** The constraints are spread across long safety standards and mission documents, and checking a plan against them is manual work.

## 3. Who it's for

A surface operations lead on a far-side mission, the person who signs off each uplink. The same problem exists underground on Earth, where Kessel's machines work offline: learn an unknown environment, plan inside a model of it, and prove the plan before the link drops.

## 4. System overview

```
                         ┌────────────────────────────────────────────┐
mission documents ─────► │           Gemini 4 Argon (orchestrator)    │
                         │  safety rules · reward · fleet trajectories │
                         │  log review · verification loop            │
                         └───┬───────────────┬──────────────────▲─────┘
                             │ reward        │ fleet plan       │ verdict, logs,
                             ▼               ▼                  │ world-model errors
                   ┌──────────────────┐  ┌──────────────────────┴──┐
                   │  MAPPO training  │  │  Verifier               │
                   │  4-5 hoppers,    │  │  ~1000 noisy rollouts   │
                   │  shared policy   │  │  in the simulator,      │
                   └────────┬─────────┘  │  checked against rules  │
                            │ imagines   └──────────▲──────────────┘
                            ▼                       │
                   ┌──────────────────┐             │
                   │  World model     │◄── learns ──┤
                   │  (shared, neural)│             │
                   └──────────────────┘   ┌─────────┴──────────┐
                                          │ Physics simulator  │
                                          │ real LOLA terrain  │
                                          └────────────────────┘

         verdict passes ──► person approves in a relay window ──► uplink
```

## 5. Physics simulator

A hand-coded copy of the Moon built on real NASA terrain. It is the ground truth: whatever happens in the simulator is what would really happen.

**Terrain.** A crop of about 20 × 20 km of far-side terrain from SLDEM2015, resampled to about 100 m per cell (roughly 200 × 200 cells). Each cell stores height, slope, roughness and night temperature. Crater positions and radii come from the Robbins crater catalogue.

**Hop physics.** Each hop is a ballistic arc under lunar gravity, g = 1.62 m/s², with no air drag. For a hop of horizontal range R at launch angle θ, the launch speed is v = √(R·g / sin 2θ). Fuel use follows the rocket equation, Δm = m·(1 − e^(−Δv / vₑ)), with Δv covering ascent and the landing burn.

**Energy and thermal.** Battery drains with a fixed base load, an extra heater load that grows as temperature drops, and a charge cost per hop. There is no solar charging at night.

**Relay windows.** A simple orbit model gives the times when the relay orbiter is above each hopper's horizon. Only then can a hopper send data or receive an uplink, and only one hopper can use the link at a time.

**Noise.** Each hop can be run with sampled thrust error, landing scatter, sensor noise and dust. The verifier uses this. Training can too.

**Determinism.** Given a seed, the simulator always produces the same result, so a failure can be replayed exactly.

## 6. World model

A small neural network the fleet learns by watching the simulator. Given a hopper's state and an action, it predicts the next state: "if I do this hop, I'll probably end up here, with this much fuel and battery".

**Shared.** All hoppers feed one world model with their experience, so terrain explored by one hopper improves predictions for all of them.

**Inputs and outputs.** Input: position, fuel, battery, temperature, the local terrain patch around the hopper, and the chosen action. Output: predicted next position, fuel, battery and temperature, plus whether the landing is likely to fail.

**Why have one.** The policy can practise thousands of hops inside the world model without spending simulated fuel or time (the Dreamer approach). That makes training fast and is how a real machine would learn an environment it can't afford to make mistakes in.

**Its errors matter.** The world model is wrong in places, especially on terrain no hopper has visited. Every step logs the predicted and actual outcome. Argon reads the error map and explains where the model can't be trusted and why. The verifier never uses the world model, only the simulator.

## 7. Reinforcement learning

The hoppers learn by trial and error. Good behaviour earns points, bad behaviour loses them.

**Algorithm.** MAPPO, the multi-agent version of PPO. All hoppers share one policy network (parameter sharing), and each acts only on its own observations. During training a centralised critic sees the whole fleet's state, which helps the hoppers learn to cooperate. At run time each hopper acts alone.

**Observations per hopper.** Its own state, the local terrain patch, nearby hoppers' positions, which craters are claimed or done, time to the next relay window.

**Actions.** Hop to a target cell, scan the current crater, wait, or send data in a relay window.

**Reward.** Written by Argon. A starting version:

| Term | Sign |
|---|---|
| Reaching an unclaimed crater | + |
| Collecting data at a crater | + |
| Sending data during a relay window | + |
| Fuel used | − |
| Crash or tip-over | large − |
| Battery below reserve | − |
| Two hoppers within the safety distance | − |
| Two hoppers using the relay at once | − |

**Argon rewrites the reward.** Argon reads the training curves and sample episodes. When learning stalls or the fleet finds a loophole (for example, hoppers learning to sit still to avoid fuel costs), Argon rewrites the reward code and training continues. This is the Eureka approach.

**Where training happens.** Inside the shared world model for speed, with regular checks in the real simulator so the policy doesn't exploit the world model's mistakes.

## 8. Multi-agent system

Two layers, with one rule: the hoppers act, Argon decides.

**The fleet.** Four to five hoppers trained together with MAPPO. They learn to split up the craters, keep clear of each other and take turns on the single relay link. Cooperation comes out of training, not hand-written rules.

**The orchestrator.** Gemini 4 Argon is the single orchestrator above the fleet. Nothing else plans, reads documents or judges results.

## 9. Argon, the orchestrator

| Job | Input | Output | Thinking level |
|---|---|---|---|
| Mission documents | Safety standards, mission constraints | Safety rules, each with an id, a plain-English statement and a check function | high |
| Reward design | Training curves, sample episodes | Reward code, and rewrites when learning stalls | medium |
| Trajectory planning | Terrain, crater list, fleet state, relay windows, the policy's suggestions | A fleet plan: which hopper visits which craters, and every hop | medium |
| Verification loop | The verifier's verdict and failure statistics | A revised fleet plan | medium |
| Log review | Hopper logs, world-model error map | Plain explanation of what went wrong and what to change | medium |

All calls use the Interactions API (`client.interactions.create`, `google-genai` 2.30.1 or later). Long calls run with `background=True` and are polled. Multi-turn state uses `previous_interaction_id`.

**Why it needs Argon.** Long documents have to become exact, executable rules. Reward code has to be written and rewritten from evidence. A fleet plan has to satisfy many interacting constraints at once. Failure statistics have to be read and turned into the right fix, not a random retry. A smaller model makes the rules vague and the revisions random.

## 10. Verifier

Plain code, not a model.

1. Take the fleet plan from Argon.
2. Replay it about 1000 times in the physics simulator, each time with fresh thrust error, landing scatter, sensor noise and dust.
3. Check every run against every safety rule: fuel reserve, landing slope, battery reserve, minimum separation between hoppers, relay conflicts.
4. Verdict: pass if at least 99% of runs break no rule. Otherwise report the pass rate and which rule failed, on which hop, for which hopper, how often.
5. On a fail, Argon reads the statistics and revises the plan. Repeat until it passes.

An optional stricter layer (ADR-0003, issue #4) would use the Z3 solver to prove the simplest rules, such as the fuel reserve, hold for every value inside the noise bounds, not only the sampled ones.

## 11. Human control

Nothing is uplinked until two things are true: the verifier has passed the plan, and a relay window is open. Then a person approves it. No code path skips either step.

## 12. Data

| Layer | Source | Real or simulated |
|---|---|---|
| Terrain, slope, roughness | SLDEM2015 (LOLA and Kaguya), LOLA global DEM | Real |
| Night temperature | Diviner global temperature maps | Real |
| Craters | Robbins lunar crater catalogue | Real |
| Relay windows | Orbit model | Simulated |
| Hop flight, fuel, battery, thermal | Physics models above | Simulated |
| Readings at each crater | Depends on the payload decision (issue #2) | Simulated |

Sources and processing: [`data/DATASETS.md`](data/DATASETS.md).

## 13. Models

| Model | Role |
|---|---|
| `gemini-4-argon` | The orchestrator: every planning, reading and judging job |
| Shared world model | Small neural network trained in this repo |
| Shared MAPPO policy | Small neural network trained in this repo |

Only Gemini models run inside the app. Claude Code is used to write the code, never at runtime.

## 14. Open decisions

- What the hoppers collect at each crater, [issue #2](https://github.com/Nabzx/selene/issues/2).
- Which far-side region, [issue #3](https://github.com/Nabzx/selene/issues/3).
- Verifier depth, [issue #4](https://github.com/Nabzx/selene/issues/4).

## 15. Risks

| Risk | Mitigation |
|---|---|
| MAPPO doesn't converge in a day | Small grid, small networks, start with fewer hoppers and scale up |
| Policy exploits the world model's mistakes | Regular evaluation in the real simulator |
| Argon calls are slow | Background mode, start long calls early, cached results labelled as cached |
| Venue wifi | All data on disk before kickoff |
