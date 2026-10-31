# Context

The vocabulary of this project. Use these terms in code, issues, commits and docs.

## Terms

**Far side**
The hemisphere of the Moon that never faces Earth. It is not permanently dark. What makes it hard is that Earth is never in line of sight, so all contact goes through a relay orbiter.
_Avoid_: dark side (except as a deliberate Halloween joke in UI copy).

**Lunar night**
The roughly 14.75 Earth days without sunlight at a given spot. No solar charging, surface below -170 °C. The simulation's setting.

**Hopper**
The drone. A small rocket-propelled lander that moves by short ballistic hops. No air on the Moon, so no rotors.
_Avoid_: quadcopter, rover.

**Hop**
One flight from a launch cell to a landing cell. Costs fuel set by distance and lunar gravity (1.62 m/s²).

**Route**
An ordered list of craters to visit, with the hops between them. The unit Argon proposes and the verifier checks.

**Simulator**
The hand-built ground truth: terrain from LOLA, hop physics, battery, temperature, relay windows. Deterministic given a seed.
_Avoid_: world model (that is the learned one).

**World model**
The small neural network the hopper learns from simulator transitions. Given a state and an action, it predicts the next state. The policy plans inside it.

**World-model error**
The gap between what the world model predicted and what the simulator did. Logged per step, mapped per cell, read by Argon.

**Relay window**
A period when the relay orbiter is above the horizon and the hopper can send data and receive an uplink.

**Safety spec**
The machine-checkable rules a route must satisfy, written by Argon from mission documents. Each rule has an id, a plain-English statement and a check function.

**Rollout**
One simulated execution of a route with sampled noise (thrust error, landing scatter, dust, sensor noise). Used by the verifier.

**Episode**
One training run of the hopper, logged step by step to `data/episodes.json` for the 3D view. Not the same as a rollout.

**Trajectory cloud**
The set of rollouts for one route drawn together in the 3D view. Green passes every rule, red breaks at least one.

**Verdict**
The verifier's output for a route: pass or fail, the pass rate, and failure statistics per rule and per hop.

**Uplink**
Sending an approved route to the hopper during a relay window. Needs a passing verdict and a person's approval.

**Payload**
What the hopper collects at each crater. Not yet decided, see the open issue.
