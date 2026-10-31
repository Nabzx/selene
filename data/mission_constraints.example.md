# Selene Far-Side Night Exploration: Mission Safety Standards (Example)

> **Source:** `simulated` (example mission constraints document for the Daedalus region demo).
> Terrain height, slope and temperature are derived from LOLA SLDEM2015 and Diviner datasets (`docs/data/DATASETS.md`). Hopper subsystem limits, relay windows and separation thresholds below are simulated mission constraints matching `SimConfig`.

## 1. Scope & Physical Constants

These safety standards govern all autonomous rocket-hopper trajectories planned for the Selene fleet (`h0..h3`, up to `h4`) operating in the Daedalus `20 x 20 km` region on the lunar far side during lunar night (surface temperatures `-225.0 °C` to `-155.0 °C`).

- **Lunar gravity (`g_moon`):** `1.62 m/s^2`
- **Specific impulse (`isp_s`):** `220.0 s` (`v_e = 2157.46 m/s`), dry mass `35.0 kg`, initial fuel `15.0 kg` (`fuel = 1.0` equals `15.0 kg`), gravity-loss margin `0.08`
- **Maximum single-hop range (`max_hop_range_m`):** `3000.0 m` at `45.0 deg` launch elevation
- **Relay orbiter cycle:** `120.0 s` period with a `40.0 s` open window (`[k*120, k*120 + 40] s`)

Every candidate fleet plan must be verified in the ground-truth physics simulator (never the learned world model) across `1,000` Monte Carlo rollouts (`seed = 42`) with sampled thrust error, landing scatter, slope noise and dust. A fleet plan passes verification only if at least **99%** (`pass_rate >= 0.99`) of rollouts satisfy all rules below simultaneously, and a human operator explicitly approves the uplink during an open relay window.

## 2. Subsystem & Trajectory Safety Rules

### Rule R1_FUEL_RESERVE: Propellant Reserve
- **Scope:** `route`
- **Requirement:** Each hopper must retain a minimum propellant fraction of **0.10** (`10%`, equivalent to `1.5 kg` out of `15.0 kg` initial fuel) after final landing to guarantee a contingency landing burn. Any rollout or step where final fuel drops below `0.10` (`fuel_kg < 1.5`) violates this rule.

### Rule R2_BATTERY_RESERVE: Lunar Night Battery Reserve
- **Scope:** `step`
- **Requirement:** Because there is no solar charging during lunar night and avionics heaters draw continuous power, each hopper must maintain a battery fraction of at least **0.20** (`20.0%` state of charge) at every stage of the mission. Any step with `battery < 0.20` (`battery_pct < 20.0`) violates this rule.

### Rule R3_LANDING_SLOPE: Safe Landing Slope & No Crash
- **Scope:** `hop`
- **Requirement:** Crater rims are steep and rough. At every landing, the local surface slope at the landing cell must not exceed **15.0 degrees** (`landing_slope_deg <= 15.0`), and no hopper may crash (`crashed == False`, `event not in ("crash", "crashed", "tip_over")`).

### Rule R4_FLEET_SEPARATION: Fleet Separation Distance
- **Scope:** `fleet`
- **Requirement:** To prevent mid-flight or landing collisions and plume impingement, no two hoppers in the fleet may come within **100.0 metres** (`min_separation_m >= 100.0`) of one another at any stage.

### Rule R5_SINGLE_CHANNEL_RELAY: Single-Channel Relay Window Discipline
- **Scope:** `fleet`
- **Requirement:** All contact with Earth goes through a single-channel relay orbiter. At most one hopper may execute a `"send"` / `"transmit"` action during any stage, and only while the relay window is open (`relay_conflict == False`).
