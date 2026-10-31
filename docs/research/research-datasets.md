# Far-side datasets and region selection

## Candidate regions compared

We need a $20 \times 20\text{ km}$ far-side crop ($200 \times 200$ cells at $100\text{ m/cell}$) within the $\pm 60^\circ$ latitude band covered by SLDEM2015, containing 6 to 10 smaller craters ($0.3\text{ km}$ to $2.0\text{ km}$ radius) within hop range ($1\text{ km}$ to $3.5\text{ km}$ apart) and steep rim features where the world model can make visible prediction errors.

| Region | Centre coordinates | Diameter | Terrain character | Fit for $20 \times 20\text{ km}$ hopper crop |
|---|---|---|---|---|
| **Daedalus crater** (floor & inner rim terrace) | $5.9^\circ\text{S},\ 179.4^\circ\text{E}$ | $93\text{ km}$ | Near the exact antipode of Earth (most radio-quiet spot on the Moon; NASA NIAC Lunar Crater Radio Telescope candidate). Flat central floor ringed by terraced walls ($15^\circ$ to $28^\circ$ slopes), central peak, and secondary craters of $300\text{ m}$ to $1500\text{ m}$ radius. | **Best.** Combines navigable floor hops with hazardous rim terraces and strong science story. |
| **Tsiolkovskiy crater** | $20.38^\circ\text{S},\ 128.97^\circ\text{E}$ | $184\text{ km}$ | Smooth dark mare basalt floor around a massive central peak. | Floor is very flat over $20\text{ km}$ patches; fewer medium craters to hop between unless cropped right on the peak edge. |
| **Apollo basin** (southern mare / Chang'e 6 area) | $41.6^\circ\text{S},\ 154.0^\circ\text{W}$ | $537\text{ km}$ | Double-ring basin inside South Pole-Aitken basin. | Good crater density, but less iconic for the radio-quiet far-side autonomy story than Daedalus. |

## Primary datasets and PDS files

1. **Elevation and slope: SLDEM2015 (Barker et al., 2016)**
   - **Source:** NASA PDS Geosciences Node (`lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/jp2/` or USGS Astrogeology SLDEM2015 tiles).
   - **Resolution:** 512 pixels per degree ($\approx 59\text{ m/pixel}$ at Daedalus's $5.9^\circ\text{S}$ latitude), vertical accuracy $3\text{ to }4\text{ m}$.
   - **Crop box for Daedalus SW inner floor/rim transition:** Latitude $[-6.25^\circ, -5.59^\circ]$, Longitude $[178.95^\circ\text{E}, 179.61^\circ\text{E}]$ ($\approx 20 \times 20\text{ km}$). Resampled to $200 \times 200$ at $100\text{ m/cell}$ (`height_m` and gradient-derived `slope_deg`).
2. **Night surface temperature: LRO Diviner Global Bolometric Temperature (Williams et al., 2017)**
   - **Source:** PDS Diviner Gridded Data Records (`diviner.ucla.edu`, `dgdr_tb_avg_cyl`).
   - **Thermal profile at Daedalus:** Lunar night surface temperatures sit around $92\text{ K}$ to $98\text{ K}$ ($-181\ ^\circ\text{C}$ to $-175\ ^\circ\text{C}$) on the floor, dropping to $83\text{ K}$ ($-190\ ^\circ\text{C}$) inside shadowed secondary crater bowls and rising slightly on rockier rim slopes with higher thermal inertia.
3. **Crater catalogue: Robbins Lunar Crater Database (Robbins, 2019)**
   - **Source:** PDS Annex / USGS (`lunar_crater_database_robbins_2018.csv`).
   - **Selection:** Filter craters inside the $20 \times 20\text{ km}$ Daedalus crop box with diameter $0.5\text{ km}$ to $3.0\text{ km}$, selecting the top $C = 8$ craters as target sites `c1..c8`.

## Offline fallback and derived artifact (`data/grid.npz`)

Downloading full multi-gigabyte PDS JP2/IMG tiles during a one-day hackathon is slow. We therefore use a two-tier setup behind `TerrainGrid`:
1. `TerrainGrid.synthetic(seed=7)` generates a deterministic Daedalus-profile $200 \times 200$ elevation, slope, roughness, and Diviner night-temperature grid with 8 realistic crater bowls and a terraced rim wall in $< 10\text{ ms}$, saved directly to `data/grid.npz` and `data/terrain.json`.
2. If raw SLDEM2015 / Diviner tiles are placed in `materials/data/`, `scripts/build_grid.py` crops the exact Daedalus box into `data/grid.npz` with no code change in `sim.py`, `env.py`, or `verify.py`.

## Sources

- Barker et al. (2016), *A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera*, Icarus 273, 346-355. [https://doi.org/10.1016/j.icarus.2015.07.039](https://doi.org/10.1016/j.icarus.2015.07.039)
- NASA PDS LOLA GDR SLDEM2015 Archive. [https://pds-geosciences.wustl.edu/missions/lro/lola.htm](https://pds-geosciences.wustl.edu/missions/lro/lola.htm)
- Williams et al. (2017), *The global surface temperatures of the Moon as measured by the Diviner Lunar Radiometer Experiment*, Icarus 283, 300-325. [https://doi.org/10.1016/j.icarus.2016.08.012](https://doi.org/10.1016/j.icarus.2016.08.012)
- Robbins (2019), *A New Global Database of Lunar Impact Craters >1-2 km*, JGR Planets 124(4), 871-892. [https://doi.org/10.1029/2018JE005592](https://doi.org/10.1029/2018JE005592)

## Recommendation

Select the **Daedalus crater** inner-floor-to-rim transition ($5.9^\circ\text{S}, 179.4^\circ\text{E}$, $20 \times 20\text{ km}$ crop, $200 \times 200$ cells at $100\text{ m/cell}$) as the mission region, with 8 secondary craters ($300\text{ m}$ to $1200\text{ m}$ radius) and night temperatures in $[-190, -172]\ ^\circ\text{C}$. Export `data/grid.npz` and `data/terrain.json` from `TerrainGrid` so both the simulator and the 3D view share identical surface heights.
