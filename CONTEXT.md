# Context

The vocabulary of this project. Use these terms in code, issues, commits and docs.

## Terms

**Hopper**
The drone. A small rocket-propelled lander that moves by short ballistic hops. There is no air on the Moon, so it has no rotors.
_Avoid_: quadcopter, rover.

**Hop**
One flight from a launch cell to a landing cell. Costs fuel set by distance and the 1.62 m/s² lunar gravity.

**Hop plan**
An ordered list of hops, from a sunlit start to one or more scan sites and back. The unit Argon proposes and the verifier checks.

**PSR**
Permanently shadowed region. Terrain that never receives direct sunlight, where water ice can survive. From the LOLA PSR map.

**Safety spec**
The list of machine-checkable rules a hop plan must satisfy, written by Argon from mission documents. Each rule has an id, a plain-English statement and a check function.

**Rollout**
One simulated execution of a hop plan with sampled noise (thrust error, landing scatter, dust, sensor noise).

**Trajectory cloud**
The set of rollouts for one hop plan drawn together in the 3D view. Green lines pass every rule, red lines break at least one.

**Verdict**
The verifier's output for a hop plan: pass or fail, the pass rate, and the failure statistics per rule and per hop.

**Revision**
Argon's new hop plan after a failed verdict, given the failure statistics.

**Uplink**
Sending an approved hop plan to the hopper. Needs a passing verdict and a person's approval.

**Ice field**
The simulated, hidden ice abundance per cell, sampled from PSR, Diviner and LAMP data plus noise. Not a measurement.
_Avoid_: ice map, real ice data.

**Grid**
The simulation's 2D cell grid, cropped from the LOLA DEM around Shackleton crater and downsampled (around 100 m per cell). The 3D view uses the full-resolution crop.
