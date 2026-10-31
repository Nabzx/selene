# Ballistic hop and rocket-equation modelling

## Right level of physical detail

For a $20 \times 20\text{ km}$ lunar crop (`SLDEM2015`, $200 \times 200$ cells at $100\text{ m/cell}$), individual hops cover $0.5\text{ km}$ to $4\text{ km}$. Over a $4\text{ km}$ distance, lunar curvature drops the horizon by only $d^2 / (2 R_{\text{Moon}}) \approx 4.6\text{ m}$, so a flat-Moon uniform gravity field ($g = 1.62\text{ m/s}^2$) with exact elevation differences $\Delta z = z_{\text{target}} - z_{\text{start}}$ from the terrain grid is physically accurate and fast to compute in closed form.

## Ballistic trajectory equations

Given launch position $\mathbf{p}_0 = (x_0, y_0, z_0)$, target cell $\mathbf{p}_1 = (x_1, y_1, z_1)$, horizontal range $R = \sqrt{(x_1 - x_0)^2 + (y_1 - y_0)^2}$, elevation change $\Delta z = z_1 - z_0$, and nominal launch angle $\theta = 45^\circ$ ($\pi / 4\text{ rad}$, which minimises flat-ground $\Delta v$):

1. **Flat-ground baseline launch speed:**
   $$v_{\text{flat}} = \sqrt{\frac{R \cdot g}{\sin(2\theta)}}$$
2. **Elevation-corrected launch speed:** To reach height $\Delta z$ at horizontal distance $R$ when launching at angle $\theta$ (provided $R \tan\theta > \Delta z$; for steep uphill hops we clamp $\theta = \max(\pi/4, \arctan(\Delta z / R) + \pi/8)$):
   $$v_{\text{launch}} = \sqrt{\frac{g R^2}{2 \cos^2\theta \,(R \tan\theta - \Delta z)}}$$
   When $\Delta z = 0$, this reduces exactly to $v_{\text{launch}} = \sqrt{R g / \sin(2\theta)}$.
3. **Time of flight and intermediate arc:**
   $$t_{\text{flight}} = \frac{R}{v_{\text{launch}} \cos\theta}$$
   At time $\tau \in [0, t_{\text{flight}}]$ during a hop (sampled every $0.5\text{ s}$ for `episodes.json` step logging):
   $$x(\tau) = x_0 + \frac{\tau}{t_{\text{flight}}}(x_1 - x_0), \quad y(\tau) = y_0 + \frac{\tau}{t_{\text{flight}}}(y_1 - y_0)$$
   $$z(\tau) = z_0 + (v_{\text{launch}} \sin\theta)\,\tau - \frac{1}{2} g \tau^2$$
   The flight phase is `"ascent"` for $\tau < 0.2\,t_{\text{flight}}$, `"coast"` for $0.2\,t_{\text{flight}} \le \tau \le 0.8\,t_{\text{flight}}$, and `"descent"` for $\tau > 0.8\,t_{\text{flight}}$.
4. **Impact speed and total $\Delta v$:**
   At touchdown ($\tau = t_{\text{flight}}$), vertical velocity is $v_{z,\text{land}} = v_{\text{launch}}\sin\theta - g t_{\text{flight}}$ and horizontal velocity is $v_{x,\text{land}} = v_{\text{launch}}\cos\theta$, giving landing burn magnitude:
   $$v_{\text{land}} = \sqrt{(v_{\text{launch}}\cos\theta)^2 + (v_{\text{launch}}\sin\theta - g t_{\text{flight}})^2}$$
   With gravity/steering margin factor $\kappa_{\text{grav}} = 1.15$ and sampled relative thrust error $\epsilon_{\text{thrust}} \sim \mathcal{N}(0, \sigma_{\text{thrust}}^2)$:
   $$\Delta v = \kappa_{\text{grav}} (v_{\text{launch}} + v_{\text{land}}) (1 + |\epsilon_{\text{thrust}}|)$$

## Fuel consumption (Tsiolkovsky rocket equation)

Let the hopper have dry mass $m_{\text{dry}} = 35\text{ kg}$, initial propellant mass $m_{\text{fuel},0} = 25\text{ kg}$ (total wet mass $m_0 = 60\text{ kg}$), and bipropellant effective exhaust velocity $v_e = I_{\text{sp}} g_0 = 300\text{ s} \times 9.80665\text{ m/s}^2 = 2942\text{ m/s}$.

For a hop starting at current total mass $m = m_{\text{dry}} + \text{fuel} \cdot m_{\text{fuel},0}$:
$$\Delta m = m \left(1 - \exp\left(-\frac{\Delta v}{v_e}\right)\right)$$
Normalised fuel fraction $[0, 1]$ updates as:
$$\text{fuel}' = \max\left(0.0,\; \text{fuel} - \frac{\Delta m}{m_{\text{fuel},0}}\right)$$
A $2\text{ km}$ hop ($R = 2000\text{ m}$, $\Delta z = 0$) has $v_{\text{launch}} = \sqrt{2000 \times 1.62} = 56.9\text{ m/s}$, $\Delta v \approx 131\text{ m/s}$, consuming $\Delta m \approx 2.61\text{ kg}$ ($\approx 10.4\%$ of total fuel), allowing roughly 8 to 10 medium hops per hopper with a $10\%$ safety reserve.

## Battery, thermal, and crash mechanics

- **Night temperature and heater drain:** Surface temperature $T_{\text{surf}} \in [-195, -165]\ ^\circ\text{C}$ comes from the grid (Diviner night map or synthetic crater cold traps). Over step duration $\Delta t$:
  $$\Delta \text{battery} = \left(P_{\text{base}} + P_{\text{heat}} \cdot \frac{\max(0, T_{\text{target}} - T_{\text{surf}})}{150.0} (1 + c_{\text{dust}})\right) \Delta t + c_{\text{hop}} \cdot \mathbb{I}_{\text{hop}} + c_{\text{act}} \cdot \mathbb{I}_{\text{scan/send}}$$
- **Landing crash condition:** A hop crashes if `landing_slope[iy, ix] > max_safe_slope` (default $15^\circ \approx 0.2618\text{ rad}$) or if fuel runs out mid-flight ($\Delta m > \text{fuel} \cdot m_{\text{fuel},0}$).

## Sources

- Barker et al. (2016), *A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera (SLDEM2015)*, Icarus 273, 346-355. [https://doi.org/10.1016/j.icarus.2015.07.039](https://doi.org/10.1016/j.icarus.2015.07.039)
- Robbins (2019), *A New Global Database of Lunar Impact Craters >1-2 km*, JGR Planets 124(4), 871-892. [https://doi.org/10.1029/2018JE005592](https://doi.org/10.1029/2018JE005592)

## Recommendation

Use the elevation-corrected ballistic arc under $g = 1.62\text{ m/s}^2$ with $\theta = \pi/4$, $v_e = 2942\text{ m/s}$, $m_{\text{dry}} = 35\text{ kg}$, and $m_{\text{fuel},0} = 25\text{ kg}$ in both `sim.py` and `verify.py`. Start with a deterministic seeded synthetic far-side crater terrain generator matching the $200 \times 200$ (`100 m/cell`) grid interface so training and verification run immediately, and load real cropped SLDEM2015 + Diviner `data/grid.npz` behind the same `TerrainGrid` class.
