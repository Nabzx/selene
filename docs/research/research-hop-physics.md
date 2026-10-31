# Ballistic hop and rocket-equation modelling

## Closed-form hop evaluation vs 3D replay interpolation

During environment stepping (`env.py`) and Monte Carlo verification (`verify.py`), each hop is computed **once in closed form** (one decision = one hop), avoiding step-by-step flight integration inside the training loop:

1. Given start coordinates $(x_0, y_0, z_0)$ and target coordinates $(x_1, y_1, z_1)$ (plus sampled landing scatter $(\delta x, \delta y)$), look up landing elevation $z_1$, landing slope $\alpha_{\text{land}}$, and surface night temperature $T_{\text{surf}}$ from the $200 \times 200$ `TerrainGrid`.
2. Compute horizontal range $R = \sqrt{(x_1 - x_0)^2 + (y_1 - y_0)^2}$ and elevation change $\Delta z = z_1 - z_0$.
3. Under lunar gravity $g = 1.62\text{ m/s}^2$ at launch angle $\theta = \pi/4$:
   - On flat ground ($\Delta z = 0$), launch speed is $v = \sqrt{R g / \sin(2\theta)} = \sqrt{R g}$.
   - With elevation difference $\Delta z$, adjust launch angle $\theta = \max(\pi/4, \arctan(\Delta z / R) + \pi/8)$ and compute:
     $$v_{\text{launch}} = \sqrt{\frac{g R^2}{2 \cos^2\theta \,(R \tan\theta - \Delta z)}}, \quad t_{\text{flight}} = \frac{R}{v_{\text{launch}} \cos\theta}$$
4. Total $\Delta v$ covers ascent and landing braking burn with gravity-loss factor $\kappa = 1.15$ and sampled thrust error $\epsilon_{\text{thrust}}$:
   $$\Delta v = \kappa \left(v_{\text{launch}} + \sqrt{(v_{\text{launch}}\cos\theta)^2 + (v_{\text{launch}}\sin\theta - g t_{\text{flight}})^2}\right) (1 + |\epsilon_{\text{thrust}}|)$$
5. Fuel mass burned follows the Tsiolkovsky rocket equation ($m_{\text{dry}} = 35\text{ kg}$, $m_{\text{fuel},0} = 25\text{ kg}$, $v_e = 2942\text{ m/s}$):
   $$\Delta m = m \left(1 - \exp\left(-\frac{\Delta v}{v_e}\right)\right)$$
6. Battery drains by base load, heater load proportional to $\max(0, T_{\text{internal}} - T_{\text{surf}})$ over $\Delta t = t_{\text{flight}}$, and a fixed hop cost.
7. Landing crashes if `landing_slope_deg > max_safe_slope_deg` ($15^\circ$) or if fuel is exhausted ($\Delta m > \text{fuel} \cdot m_{\text{fuel},0}$).

When exporting sampled episodes to `data/episodes.json` for Friend A's 3D view, a pure helper `interpolate_hop_arc(p0, p1, t0, t_flight, step_dt=0.5)` expands each hop into $0.5\text{ s}$ keyframes with `phase` in `{"ascent", "coast", "descent", "ground"}`.

## Sources

- Barker et al. (2016), *A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera (SLDEM2015)*, Icarus 273, 346-355. [https://doi.org/10.1016/j.icarus.2015.07.039](https://doi.org/10.1016/j.icarus.2015.07.039)
- Robbins (2019), *A New Global Database of Lunar Impact Craters >1-2 km*, JGR Planets 124(4), 871-892. [https://doi.org/10.1029/2018JE005592](https://doi.org/10.1029/2018JE005592)

## Recommendation

Evaluate each hop in closed form in `sim.py` so `env.py` and `verify.py` execute a full hop in a single vectorised numpy call, and expand hops into $0.5\text{ s}$ flight-phase keyframes only when writing `data/episodes.json`.
