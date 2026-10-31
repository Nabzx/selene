# Terrain preprocessing pipeline (`data/terrain/daedalus_200x200.npz`)

## Overview

The ground-truth physics simulator (`selene_sim/sim.py`) and the 3D lunar viewer (`apps/web`) operate on a shared $20 \times 20\text{ km}$ local East-North-Up (ENU) grid centred on the south-western inner floor and terraced rim transition of Daedalus crater ($5.9^\circ\text{S}, 179.4^\circ\text{E}$). At $100\text{ m/cell}$ horizontal spacing, this yields a $200 \times 200$ array schema stored in `data/terrain/daedalus_200x200.npz`.

To avoid multi-gigabyte PDS downloads blocking development or test runs, `scripts/build_terrain_crop.py` supports two modes behind the identical `.npz` output contract:
1. **Ingest mode (`--pds-dir`):** Crops and resamples raw NASA LRO LOLA + Kaguya SLDEM2015 elevation tiles, LRO Diviner night bolometric temperature maps, and Robbins lunar crater catalogue entries.
2. **Deterministic synthetic Daedalus mode (`--synthetic`, default when raw PDS rasters are absent):** Generates a physically calibrated Daedalus floor-to-rim elevation, Horn slope, Diviner night temperature, and 12-to-15 crater catalogue in under $20\text{ ms}$.

## Coordinate projection from cylindrical degrees to local metres

Both SLDEM2015 (`512 pixels/deg`) and LRO Diviner GDR (`128 pixels/deg`) are distributed in simple cylindrical (equirectangular) projection referenced to the IAU mean lunar sphere of radius $R_{\text{moon}} = 1\,737\,400\text{ m}$.

Given crop centre $(\phi_0, \lambda_0) = (-5.9^\circ, 179.4^\circ)$:
- Meridional scale:
  $$s_y = R_{\text{moon}} \cdot \frac{\pi}{180^\circ} \approx 30\,323.35\text{ m/deg}$$
- Zonal scale at latitude $\phi_0 = -5.9^\circ$:
  $$s_x = R_{\text{moon}} \cos(\phi_0) \cdot \frac{\pi}{180^\circ} \approx 30\,162.89\text{ m/deg}$$

For a $20\,000\text{ m} \times 20\,000\text{ m}$ local Cartesian grid where cell $(i, j)$ with $i, j \in \{0, \dots, 199\}$ has cell centre coordinates:
$$x_j = \left(j + 0.5\right) \times 100.0\text{ m}, \quad y_i = \left(i + 0.5\right) \times 100.0\text{ m}$$
the corresponding lunar latitude and longitude are:
$$\phi(y_i) = \phi_0 + \frac{y_i - 10\,000.0}{s_y}, \quad \lambda(x_j) = \lambda_0 + \frac{x_j - 10\,000.0}{s_x}$$
This spans approximately $\Delta \phi = 0.660^\circ$ in latitude and $\Delta \lambda = 0.663^\circ$ in longitude.

## Bilinear resampling to $100\text{ m/cell}$

SLDEM2015 stores elevation values $h_{\text{raw}}$ as signed 16-bit integers (`int16`) with a scale factor of $0.5\text{ m}$ relative to the $1\,737.4\text{ km}$ reference sphere (or float32 metres in USGS GeoTIFF converts). At $512\text{ pixels/deg}$, native spacing is $\approx 59\text{ m/pixel}$.

To avoid aliasing staircase artefacts in surface slope calculations, we resample onto the $200 \times 200$ target grid at $100\text{ m/cell}$ using 2D bilinear interpolation:
$$h(x_j, y_i) = (1 - \alpha)(1 - \beta)\,h_{u,v} + \alpha(1 - \beta)\,h_{u+1,v} + (1 - \alpha)\beta\,h_{u,v+1} + \alpha\beta\,h_{u+1,v+1}$$
where $(u, v)$ are the integer pixel indices in the cylindrical raster and $(\alpha, \beta) \in [0, 1)^2$ are the sub-pixel fractional offsets. Elevations are shifted so the lowest point on the Daedalus basin floor sits at $0.0\text{ m}$ local height (`height_m` $\in [0.0, 1450.0]\text{ m}$).

## Finite-difference Horn slope calculation

Landing stability for a lunar hopper depends on the local terrain slope angle $\gamma(x, y)$ at touchdown (`max_landing_slope_deg = 15.0`). Rather than a noisy 2-point forward difference, we compute the surface gradient on the $200 \times 200$ grid using Horn's $3 \times 3$ weighted finite-difference stencil (Horn, 1981), which is standard in planetary GIS tools (GDAL `gdaldem slope`):

$$\frac{\partial z}{\partial x} = \frac{(z_{i-1,j+1} + 2z_{i,j+1} + z_{i+1,j+1}) - (z_{i-1,j-1} + 2z_{i,j-1} + z_{i+1,j-1})}{8\,\Delta s}$$

$$\frac{\partial z}{\partial y} = \frac{(z_{i+1,j-1} + 2z_{i+1,j} + z_{i+1,j+1}) - (z_{i-1,j-1} + 2z_{i-1,j} + z_{i-1,j+1})}{8\,\Delta s}$$

with cell spacing $\Delta s = 100.0\text{ m}$ and edge cells padded by reflection. The slope angle in degrees is then:
$$\text{slope\_deg}_{i,j} = \frac{180^\circ}{\pi} \arctan\!\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right)$$

## Crater rim/floor classification and Robbins catalogue filtering

1. **Robbins catalogue filtering:** Entries from `lunar_crater_database_robbins_2018.csv` within the crop bounding box having diameter $D \in [0.6, 3.0]\text{ km}$ (radius $r_c \in [300, 1500]\text{ m}$) are projected into local $(x_c, y_c)$ metres.
2. **Bowl & rim profile parameterisation:** Each crater $c \in \{0, \dots, C-1\}$ ($C = 12\text{ to }15$ in full mode, $C = 6$ in `--quick` mode) is characterised by centre $(x_c, y_c, z_c)$, radius $r_c$, depth $d_c \approx 0.2\,r_c$, and rim crest height $h_{\text{rim}} \approx 0.04\,r_c$:
   - **Floor region ($r \le 0.65\,r_c$):** Low slope ($2^\circ$ to $8^\circ$), colder night temperatures ($-188\ ^\circ\text{C}$ to $-182\ ^\circ\text{C}$) due to topographic radiative trapping, and high science value.
   - **Inner wall ($0.65\,r_c < r \le 1.0\,r_c$):** Steep slopes ($14^\circ$ to $26^\circ$), high crash probability if a hopper scatters onto the wall during touchdown.
   - **Raised rim flank ($1.0\,r_c < r \le 1.35\,r_c$):** Moderate slopes ($8^\circ$ to $16^\circ$) with elevated rock abundance and higher world-model transition uncertainty.

## NPZ schema (`data/terrain/daedalus_200x200.npz`)

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `height_m` | `(200, 200)` | `float32` | Surface elevation in metres above basin datum |
| `slope_deg` | `(200, 200)` | `float32` | Horn $3 \times 3$ finite-difference slope angle in degrees |
| `temp_night_c` | `(200, 200)` | `float32` | Diviner bolometric night surface temperature in $^\circ\text{C}$ ($[-190, -168]$) |
| `crater_id_map` | `(200, 200)` | `int32` | Index $c \in \{0..C-1\}$ if cell lies inside crater $c$, else `-1` |
| `crater_zone_map` | `(200, 200)` | `uint8` | `0 = inter-crater plains`, `1 = crater floor`, `2 = inner wall`, `3 = raised rim`, `4 = terrace wall` |
| `crater_pos` | `(C, 3)` | `float32` | Crater centre coordinates `(x_m, y_m, z_m)` |
| `crater_radii` | `(C,)` | `float32` | Crater rim radii in metres (`300` to `1500`) |
| `cell_size_m` | `()` | `float32` | Grid resolution (`100.0`) |

## Sources

- Barker et al. (2016), *A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera*, Icarus 273, 346-355.
- Horn (1981), *Hill shading and the reflectance map*, Proceedings of the IEEE 69(1), 14-47.
- Williams et al. (2017), *The global surface temperatures of the Moon as measured by the Diviner Lunar Radiometer Experiment*, Icarus 283, 300-325.
- Robbins (2019), *A New Global Database of Lunar Impact Craters >1-2 km*, JGR Planets 124(4), 871-892.

## Recommendation

Implement `apps/sim/scripts/build_terrain_crop.py` to generate `data/terrain/daedalus_200x200.npz` (and optional `data/terrain.json` mesh preview) using the Horn $3 \times 3$ slope stencil and local ENU metre projection, so `TerrainGrid.from_npz()` in `selene_sim/sim.py` loads deterministically in $< 5\text{ ms}$.
