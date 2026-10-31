# Roadmap

Built in one hackathon day by two people: one on the 3D world, one on training and Argon. Phases are time-boxed: when the box ends, cut scope, don't extend it.

| Phase | Box | Python and Argon | 3D world | Done when |
|---|---|---|---|---|
| 0 | Before kickoff | First Argon call, data downloaded | Scene skeleton reading `episodes.example.json` | Both sides run against the example file |
| 1 | 2 h | Grid and simulator on real terrain | Far-side terrain at night, hopper model, plume | A route runs in the sim; the view replays the example |
| 2 | 2 h | Argon: safety spec, reward, route | Crater markers, relay orbiter, HUD | Argon writes a spec and a route from documents |
| 3 | 2 h | RL training and world model, logging `episodes.json` | Training replay with episode counter and reward curve | Real episodes replay in the view |
| 4 | 1.5 h | Verifier, Argon revision, world-model review | Trajectory cloud, world-model ghost, approve button | Red cloud turns green after a revision |
| 5 | 1 h | Demo hardening | Polish, Halloween styling pass | 3-minute script rehearsed, backup recording made |

## Cut order

1. Flash-Lite guardrail
2. Argon's world-model review (keep the ghost, drop the explanation)
3. Dreamer-style planning (policy trains in the simulator directly, world model only predicts)
4. Reward rewrite loop (keep one Argon-written reward)
