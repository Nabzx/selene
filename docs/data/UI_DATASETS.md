# 3D UI datasets and Three.js terrain guide

Practical guide for building the 3D far-side lunar night view in `apps/web` (plain HTML, CSS and JavaScript, Three.js via CDN import map, no build step). Raw downloads go in `materials/data/` (gitignored). Small processed crops live in `data/` and are the only files the browser loads.

---

## 1. Terrain height

### SLDEM2015 (primary DEM)

- **What it is:** Near-global digital elevation model merging about 4.5 billion LRO Lunar Orbiter Laser Altimeter (LOLA) laser altimetry points with 43,200 stereo-derived DEM tiles from the JAXA SELENE (Kaguya) Terrain Camera (TC). Typical vertical accuracy is 3 to 4 m. Values in the USGS GeoTIFF and PDS float tiles are elevations in metres relative to a reference sphere of radius 1,737,400 m (PDS `.IMG` integer tiles store values in 0.5 m units or kilometres depending on product; always check `SCALING_FACTOR` in the `.LBL` or use the float/GeoTIFF releases).
- **Resolution:** 512 pixels per degree, about 59 m per pixel at the equator (184,320 × 61,440 pixels globally).
- **Coverage:** 60°S to 60°N, 0° to 360°E (simple cylindrical / equirectangular projection). Covers all three candidate regions.
- **Licence and citation:** Public domain (NASA / JAXA / USGS PDS). Cite Barker, M. K., Mazarico, E., Neumann, G. A., Zuber, M. T., Haruyama, J., & Smith, D. E. (2016), *A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera*, Icarus, 273, 346-355, `doi:10.1016/j.icarus.2015.07.039`.
- **Download locations, filenames and sizes:**
  - **USGS Astrogeology Astropedia ("Moon LRO LOLA - SELENE Kaguya TC DEM Merge 60N60S 59m"):**
    - Catalogue page: <https://astrogeology.usgs.gov/search/map/moon_lro_lola_selene_kaguya_tc_dem_merge_60n60s_59m>
    - Full GeoTIFF: `Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif` at <https://planetarymaps.usgs.gov/mosaic/LolaKaguya_Topo/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif> (22.65 GB, 22,650,225,603 bytes). Because it is a Cloud-Optimised / HTTP-range-addressable GeoTIFF on S3, `gdalwarp` and `rasterio` can crop a 20 × 20 km window over HTTP via `/vsicurl/` without downloading the full 22.65 GB file.
  - **PDS Geosciences LOLA Node (MIT, 30° × 45° regional tiles):**
    - Float32 PDS tiles (`SLDEM2015_512_<lat1>_<lat2>_<lon1>_<lon2>_FLOAT.IMG` + `.LBL`): <http://imbrium.mit.edu/DATA/SLDEM2015/TILES/FLOAT_IMG/> (1.42 GB / 1,415,577,600 bytes per tile, 32 tiles covering 60°S to 60°N).
    - JPEG2000 tiles (`SLDEM2015_512_<lat1>_<lat2>_<lon1>_<lon2>.JP2` + `_JP2.LBL`): <http://imbrium.mit.edu/DATA/SLDEM2015/TILES/JP2/> (156 MB to 185 MB per tile).
    - Relevant tiles for our candidate regions:
      - Daedalus (5.9°S, 179.4°E): `SLDEM2015_512_30S_00S_135_180_FLOAT.IMG` (and `180_225` if cropping across 180°E).
      - Tsiolkovskiy (20.4°S, 129.1°E): `SLDEM2015_512_30S_00S_090_135_FLOAT.IMG`.
      - Apollo basin (36.1°S, 208.3°E / 151.7°W): `SLDEM2015_512_60S_30S_180_225_FLOAT.IMG`.

### LOLA global DEM 118 m (fallback and polar latitudes)

- **What it is:** Gridded global elevation model (`LDEM_256`) interpolated from LRO LOLA laser altimetry profiles. Vertical accuracy is about 1 m; horizontal accuracy is better than 20 m.
- **Resolution:** 256 pixels per degree, about 118 m per pixel at the equator (92,160 × 46,080 pixels).
- **Coverage:** Global, 90°S to 90°N, 0° to 360°E. Used as a fallback and for any target poleward of 60° latitude.
- **Licence and citation:** Public domain (NASA PDS / USGS). Cite Smith, D. E. et al. (2010), *Initial observations from the Lunar Orbiter Laser Altimeter (LOLA)*, Geophysical Research Letters, 37, L18204, `doi:10.1029/2010GL043751`.
- **Download locations, filenames and sizes:**
  - **USGS Astrogeology Astropedia ("Moon LRO LOLA DEM 118m v1"):**
    - Catalogue page: <https://astrogeology.usgs.gov/search/details/Moon/LRO/LOLA/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014/cub>
    - Full GeoTIFF: `Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif` at <https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif> (8.49 GB, 8,494,203,833 bytes). Note: 16-bit signed integer with `SCALING_FACTOR = 0.5` m (`height_m = DN * 0.5`).
  - **PDS Geosciences LOLA Node (MIT, quarter-globe tiles):**
    - Directory: <http://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/>
    - Files: `LDEM_256_00N_90N_000_180.IMG`, `LDEM_256_00N_90N_180_360.IMG`, `LDEM_256_90S_00S_000_180.IMG`, `LDEM_256_90S_00S_180_360.IMG` (each with a matching `.LBL`), 2.12 GB (2,123,366,400 bytes) per tile.

### LROC NAC DTMs (close-up metre-scale detail)

- **What it is:** Stereo-derived Digital Terrain Models generated from Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera (NAC) geometric stereo pairs using USGS ISIS and SOCET SET, registered to LOLA profiles. Paired with 0.5 to 4 m/px orthophotos (`_M*_*.IMG` / `.TIF`) and slope maps.
- **Resolution:** 2 to 5 m per pixel horizontally, about 1 m vertical precision.
- **Coverage:** Targeted strips only (typically 5 to 10 km wide by 20 to 40 km long), covering less than 2% of the Moon. Availability in our candidate regions:
  - **Tsiolkovskiy:** `NAC_DTM_TSILKVSKIY2` covers the northeastern rim and floor edge near 19.1°S, 130.3°E at 3.0 m/px (3,504 × 16,239 pixels, about 10.5 × 48.7 km).
  - **Apollo basin:** Several NAC DTMs cover the southern mare plains and surrounding craters, notably `NAC_DTM_CHANGE6` (Chang'e 6 landing site south of Chaffee S at 42.1°S, 206.0°E, 4.0 m/px, 4,185 × 11,724 pixels), `NAC_DTM_BARRINGERZ1` and `NAC_DTM_BARRINGERZ2` (Barringer Z on the northern rim), and `NAC_DTM_OPENHIMERF` / `NAC_DTM_OPNHEIMRE*` (adjacent Oppenheimer floor).
  - **Daedalus:** No pre-built `NAC_DTM` currently covers the interior floor of Daedalus (the closest is `NAC_DTM_FRSHCRATER16` east of Stein M). Use SLDEM2015 for Daedalus.
- **Licence and citation:** Public domain (NASA / GSFC / Arizona State University). Cite Henriksen, M. R. et al. (2017), *Extracting accurate and precise topography from LROC narrow angle camera stereo observations*, Icarus, 283, 122-137, `doi:10.1016/j.icarus.2016.05.012`.
- **Download locations, filenames and sizes:**
  - **LROC RDR Product Select:** <https://data.lroc.im-ldi.com/lroc/rdr_product_select> (filter prefix `NAC_DTM`).
  - **Tsiolkovskiy (`NAC_DTM_TSILKVSKIY2`):** <https://data.lroc.im-ldi.com/lroc/view_rdr_product/NAC_DTM_TSILKVSKIY2_E191S1303>
    - GeoTIFF (`NAC_DTM_TSILKVSKIY2_E191S1303.TIF`): 54.33 MB.
    - PDS Float32 (`NAC_DTM_TSILKVSKIY2_E191S1303.IMG`): 217.08 MB.
    - Matching 3 m/px orthophoto (`NAC_DTM_TSILKVSKIY2_M1122736739_3M.TIF`): ~25 MB.
  - **Apollo basin (`NAC_DTM_CHANGE6`):** <https://data.lroc.im-ldi.com/lroc/view_rdr_product/NAC_DTM_CHANGE6_E421S2060>
    - GeoTIFF (`NAC_DTM_CHANGE6_E421S2060.TIF`): 46.88 MB.
    - PDS Float32 (`NAC_DTM_CHANGE6_E421S2060.IMG`): 187.18 MB.

---

## 2. Surface imagery and texture

### LROC WAC global mosaic (albedo and morphology)

- **What it is:** Global 643 nm greyscale mosaics compiled from over 15,000 LROC Wide Angle Camera (WAC) images. Two versions exist:
  1. **WAC Global Morphology Mosaic (`WAC_GLOBAL`):** Solar incidence angles between 55° and 75°, showing crisp crater shadows and relief. Good for map inspection, but **avoid baking daytime sun shadows into a dynamic night scene** where a moving orbiter light illuminates the surface from changing angles.
  2. **WAC Empirically Normalized Reflectance Mosaic (`WAC_EMP`, 643 nm):** Photometrically normalised to a standard 30° incidence, 0° emission, 30° phase angle (true surface albedo without baked shadows). Ideal as `MeshStandardMaterial.map` in Three.js so all shadows come from our real-time night lighting, or blended 70/30 with the morphology mosaic for micro-relief detail.
- **Resolution:** 100 m per pixel (`WAC_GLOBAL`, 303.23 ppd) or 304 pixels per degree (`WAC_EMP` 643 nm, ~100 m per pixel at the equator).
- **Coverage:** Global, 90°S to 90°N, 0° to 360°E.
- **Licence and citation:** Public domain (NASA / GSFC / Arizona State University). Cite Speyerer, E. J. et al. (2011), *Lunar Reconnaissance Orbiter Camera global morphological map of the Moon*, LPSC XLII, Abstract #2387; and Boyd, A. K. et al. (2012), *Global empiric normalization of LROC WAC reflectance data*, LPSC XLIII, Abstract #2795 (or Sato et al. 2014, `doi:10.1002/2013JE004580`).
- **Download locations, filenames and sizes:**
  - **USGS Astrogeology Astropedia ("Moon LRO LROC WAC Global Morphology Mosaic 100m"):**
    - Catalogue page: <https://astrogeology.usgs.gov/search/map/moon_lro_lroc_wac_global_morphology_mosaic_100m>
    - Full GeoTIFF: `Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif` at <https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif> (5.96 GB, 5,959,263,751 bytes; range-addressable via `/vsicurl/`).
  - **LROC RDR Archive (regional 60° × 90° tiles, much smaller):**
    - Morphology (`WAC_GLOBAL`): <https://data.lroc.im-ldi.com/lroc/view_rdr/WAC_GLOBAL>. Far-side southern tiles covering our regions are `WAC_GLOBAL_E300S1350_100M.TIF` (0° to 60°S, 90°E to 180°E; covers Tsiolkovskiy and Daedalus west of 180°E, 473.67 MB GeoTIFF or 1.85 GB `.IMG`) and `WAC_GLOBAL_E300S2250_100M.TIF` (0° to 60°S, 180°E to 270°E; covers Apollo basin and Daedalus east of 180°E, ~474 MB GeoTIFF).
    - Albedo (`WAC_EMP`, 643 nm): <https://data.lroc.im-ldi.com/lroc/view_rdr/WAC_EMP>. Matching 304 ppd tiles `WAC_EMP_643NM_E300S1350_304P.TIF` (476.07 MB GeoTIFF, 1.86 GB `.IMG`) and `WAC_EMP_643NM_E300S2250_304P.TIF` (~476 MB GeoTIFF).

### LROC QuickMap (interactive region selection and crop export)

- **What it is:** Browser GIS portal hosted by ASU and Applied Coherent Technology for inspecting and exporting co-registered lunar layers (SLDEM2015 height and slope, WAC albedo, NAC mosaics, NAC DTM footprints, Diviner temperatures and Robbins craters) without downloading multi-gigabyte archives first.
- **Resolution and coverage:** Global, up to 0.5 m per pixel where NAC imagery exists.
- **Licence and citation:** Public domain data; cite NASA / GSFC / ASU / ACT Corp.
- **Link:** <https://quickmap.lroc.asu.edu/>
- **How to export a 20 × 20 km crop:**
  1. Switch projection to **Equirectangular** and pan to the target coordinates.
  2. Enable the layer to inspect or export (for example *Terrain > SLDEM2015 Elevation*, *LROC WAC > Global Mosaic* or *Normalized Reflectance (643nm)*, or *LROC NAC > NAC ROI / DTM*).
  3. Click the **Draw / Search** tool (polygon/rectangle icon in the left toolbar), select **Box**, and drag a 20 × 20 km rectangle (at the equator, 20 km is `0.659°` of latitude/longitude, given $1^\circ = 30.323\text{ km}$ on a $1,737.4\text{ km}$ sphere).
  4. In the selection panel, open **Export / Download**, choose **GeoTIFF (full resolution)**, and download. A 20 × 20 km crop is typically **1 to 15 MB**.

---

## 3. Craters: Robbins lunar crater catalogue

- **What it is:** Global database of 1,296,795 lunar impact craters manually identified and measured on LROC WAC mosaics, Kaguya TC imagery and SLDEM2015/LOLA topography. Includes crater ID (`CRATER_ID`), planetocentric latitude (`LAT_CIRC_IMG`), positive-east longitude (`LON_CIRC_IMG`), rim diameter in km (`DIAM_CIRC_IMG`), ellipse parameters and rim point counts. International Astronomical Union (IAU) official names for named craters can be joined from the USGS Gazetteer of Planetary Nomenclature (`MOON_nomenclature_center_pts.zip`, <https://planetarynames.wr.usgs.gov/Page/MOON/target>, ~1 MB); unnamed craters get mission/Halloween display names in `episodes.json` (such as `"Hollow 1"`, `"Wraith Pit"`).
- **Resolution / completeness:** Statistically complete for all craters with diameter $\ge 1\text{ to }2\text{ km}$ globally.
- **Coverage:** Global, 90°S to 90°N, 0° to 360°E.
- **Licence and citation:** Public domain (USGS / NASA PDS). Cite Robbins, S. J. (2019), *A new global database of lunar impact craters >1-2 km: 1. Crater locations and sizes, comparisons with published databases, and global analysis*, Journal of Geophysical Research: Planets, 124(4), 871-892, `doi:10.1029/2018JE005592`.
- **Download locations, filenames and sizes:**
  - **USGS Astrogeology Astropedia ("Moon Crater Database v1 Robbins"):**
    - Catalogue page: <https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins>
    - Direct zip download: <https://astrogeology.usgs.gov/ckan/dataset/f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/lunar_crater_database_robbins_2018> (91.77 MB / 96,227,201 bytes compressed; extracts to `lunar_crater_database_robbins_2018.csv`, 238.48 MB, plus shapefiles and README).

---

## 4. Night temperature: Diviner maps

- **What it is:** Gridded thermal infrared measurements from the LRO Diviner Lunar Radiometer Experiment (DLRE). Two PDS product families are useful for lunar night:
  1. **Level 3 Gridded Data Records (`DGDR_STN_AVG_CYL` and `DGDR_RA_AVG_CYL`):** Nighttime regolith surface temperature (`STN`, local time 19:30 to 05:30, in Kelvin, `T_K = DN * 0.02`) and rock abundance (`RA`), slope-corrected with a 128 ppd DEM.
  2. **Global Cumulative Products (`GCP`):** Bolometric brightness temperatures binned at $0.5^\circ \times 0.5^\circ$ across 96 local-time bins (0.25-hour steps across the full lunar day/night cycle), showing the surface cooling below 100 K (-173 °C) through the 14.75-Earth-day night.
- **Resolution:**
  - `DGDR_*_128`: 128 pixels per degree (~237 m per pixel at the equator, 46,080 × 20,480 pixels).
  - `DGDR_*_032`: 32 pixels per degree (~948 m per pixel at the equator, 11,520 × 5,120 pixels).
  - `GCP`: 2 pixels per degree (~15 km per cell).
- **Coverage:** 80°S to 80°N (`DGDR_STN` / `DGDR_RA`), 90°S to 90°N (`GCP`), 0° to 360°E.
- **Licence and citation:** Public domain (NASA PDS / UCLA). Cite Bandfield, J. L., Hayne, P. O., Williams, J.-P., Greenhagen, B. T., & Paige, D. A. (2018), *Lunar surface roughness derived from LRO Diviner Radiometer observations*, Icarus, 313, 111-123 (`doi:10.1016/j.icarus.2018.05.018`), and Williams, J.-P., Paige, D. A., Greenhagen, B. T., & Sefton-Nash, E. (2017), *The global surface temperatures of the Moon as measured by the Diviner Lunar Radiometer Experiment*, Icarus, 283, 300-325 (`doi:10.1016/j.icarus.2016.08.012`).
- **Download locations, filenames and sizes:**
  - **PDS Geosciences Node (Washington University in St. Louis):**
    - Level 3 Cylindrical IMG directory: <https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gdr_l3/cylindrical/img/>
      - 32 ppd nighttime regolith temperature: `dgdr_stn_avg_cyl_032_img.img` + `.lbl` (84.38 MB / 88,473,600 bytes).
      - 128 ppd nighttime regolith temperature: `dgdr_stn_avg_cyl_128_img.img` + `.lbl` (1.76 GB / 1,887,436,800 bytes; JPEG2000 version `dgdr_stn_avg_cyl_128_jp2.jp2` in `../jp2/` is 589.77 MB).
    - Global Cumulative Products (10° latitude bands): <https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gcp/>
      - Files: `global_cumul_avg_cyl_10s00s_002.tab` + `.lbl` (Daedalus), `global_cumul_avg_cyl_30s20s_002.tab` (Tsiolkovskiy), `global_cumul_avg_cyl_40s30s_002.tab` (Apollo basin), each 148.97 MB (156,211,313 bytes).

---

## 5. Candidate regions

All three sit on the lunar far side (never in line of sight to Earth, requiring a relay orbiter) and are covered by SLDEM2015, WAC and Diviner.

| Region | Centre (lat, lon) | Size | Why it looks good at lunar night |
|---|---|---|---|
| **Daedalus crater** | 5.9°S, 179.4°E | 93 km diameter (~3 km deep) | Almost at the anti-Earth point (180°E), the most radio-shielded crater near Earth and the site of NASA's Lunar Crater Radio Telescope (LCRT) concept. Terraced walls, a rugged central peak complex, and bowls of secondary craters give strong vertical relief and deep shadow pockets under a passing orbiter light. |
| **Tsiolkovskiy crater** | 20.4°S, 129.1°E | 180 km diameter | Unmistakable high-contrast geology: a smooth, dark basaltic mare floor (`WAC_EMP` albedo ~0.06) pierced by a jagged, bright W-shaped central peak (~3.2 km above the floor, albedo ~0.16) and steep terraced walls with `NAC_DTM_TSILKVSKIY2` coverage. Under low-angle night lighting, the bright peak and rim catch the light sharply while the dark mare floor drops into near-black. |
| **Apollo basin (southern mare / Chaffee S)** | 36.1°S, 208.3°E (Chang'e 6 site at 41.6°S, 206.0°E) | 492 km multi-ring basin | Transition between smooth mare lava flows, wrinkle ridges and densely cratered highlands on the northern rim of the South Pole-Aitken basin. A 20 × 20 km crop here contains dozens of 0.3 to 2 km craters from Robbins at hop-friendly spacing (1 to 3 km apart) plus 4 m/px `NAC_DTM_CHANGE6` coverage. |

---

## 6. From data to Three.js

### Coordinate frame alignment

Every asset exported for the 3D view must share the local east-north-up (ENU) frame defined in [`docs/data/EPISODE_SCHEMA.md`](EPISODE_SCHEMA.md):
- Origin `(0, 0, 0)` at the centre of the 20 × 20 km crop (`x` in `[-10000, 10000]` m east, `y` in `[-10000, 10000]` m north).
- `z` is elevation in metres relative to the crop's mean or reference datum (same vertical datum used when `apps/sim` logs `pos: [x, y, z]` and `craters[].z` in `data/episodes.json`).

### Step 1: Crop 20 × 20 km in a local metre projection

Use a local orthographic or azimuthal equidistant projection on the lunar sphere ($R = 1,737,400\text{ m}$) centred on `(lat0, lon0)` so pixel coordinates are directly in metres east (`x`) and north (`y`):

```bash
# Example centred on Daedalus floor/rim patch (lat0 = -5.9, lon0 = 179.4)
PROJ="+proj=ortho +lat_0=-5.9 +lon_0=179.4 +R=1737400 +units=m +no_defs"

# Crop 20 x 20 km (-10000 to +10000 m) at 512 x 512 pixels (39.0625 m/px)
gdalwarp -t_srs "$PROJ" \
  -te -10000 -10000 10000 10000 \
  -ts 512 512 -r cubicspline \
  materials/data/SLDEM2015_512_30S_00S_135_180_FLOAT.LBL \
  materials/data/crop_dem_512.tif

# Crop matching 2048 x 2048 WAC albedo texture (9.7656 m/px upsampled or native 100 m/px at 256 x 256 / 1024 x 1024)
gdalwarp -t_srs "$PROJ" \
  -te -10000 -10000 10000 10000 \
  -ts 1024 1024 -r bilinear \
  materials/data/WAC_EMP_643NM_E300S1350_304P.TIF \
  data/wac_albedo.png
```

### Step 2: Export decimated heightmap and `data/terrain.json` sidecar

Write a small Python script (using `rasterio` and `numpy`) during data prep to emit:
1. **`data/terrain.bin`** (1,048,576 bytes / 1.0 MB): raw little-endian `Float32Array` of $512 \times 512$ elevations in metres, row-major from north-west `(-10000, +10000)` to south-east `(+10000, -10000)`, matching Three.js `PlaneGeometry` vertex order. *(Alternative: a 16-bit greyscale PNG `data/terrain.png`, ~180 KB, where $z = z_{\min} + (\text{uint16} / 65535) \cdot (z_{\max} - z_{\min})$, decoded on the GPU or via a 2D canvas).*
2. **`data/terrain.json`** (~1 KB): metadata sidecar in the episode schema's frame:

```json
{
  "schema_version": 1,
  "region": "Daedalus",
  "centre_lat_deg": -5.9,
  "centre_lon_deg": 179.4,
  "frame": "ENU",
  "origin_m": [0.0, 0.0, 0.0],
  "width_m": 20000.0,
  "height_m": 20000.0,
  "cols": 512,
  "rows": 512,
  "cell_size_m": 39.0625,
  "min_z_m": -1420.5,
  "max_z_m": 612.8,
  "heightmap_file": "terrain.bin",
  "heightmap_dtype": "float32_le",
  "albedo_texture": "wac_albedo.png",
  "thermal_texture": "diviner_night.png"
}
```

### Step 3: Build the mesh in Three.js

Keep the scene itself in the ENU frame by setting `camera.up.set(0, 0, 1)` (or map ENU `(x, y, z)` to Three.js `(x, z, -y)` in a single helper). With `camera.up.set(0, 0, 1)`, `PlaneGeometry(width_m, height_m, cols - 1, rows - 1)` lies directly in the `x-y` plane and vertex `i` has its `z` coordinate set from `terrain.bin`:

```js
const meta = await fetch('/data/terrain.json').then(r => r.json());
const buf = await fetch(`/data/${meta.heightmap_file}`).then(r => r.arrayBuffer());
const heights = new Float32Array(buf);

const geom = new THREE.PlaneGeometry(
  meta.width_m,
  meta.height_m,
  meta.cols - 1,
  meta.rows - 1
);
const pos = geom.attributes.position;
for (let i = 0; i < heights.length; i++) {
  pos.setZ(i, heights[i]);
}
pos.needsUpdate = true;
geom.computeVertexNormals();

const albedoTex = new THREE.TextureLoader().load(`/data/${meta.albedo_texture}`);
albedoTex.colorSpace = THREE.SRGBColorSpace;
albedoTex.anisotropy = 4;

const mat = new THREE.MeshStandardMaterial({
  map: albedoTex,
  roughness: 0.95,
  metalness: 0.0
});
const terrainMesh = new THREE.Mesh(geom, mat);
scene.add(terrainMesh);
```

### Step 4: Scientifically credible lunar night lighting and Halloween styling

On the far side at lunar night there is **no direct sunlight and no Earthshine** (Earth never rises on the far side). Keep the lighting physically grounded while remaining readable on a laptop screen:

1. **Starlight and zodiacal/galactic glow:** A faint `HemisphereLight(0x1e1b38, 0x05050a, 0.18)` provides the low ambient baseline so unlit terrain silhouettes are discernible without washing out shadows.
2. **Relay orbiter as a moving light:** During a relay window (`relay_windows[]` in `episodes.json`), animate a high-altitude `PointLight` or narrow `SpotLight` across the sky arc above the terrain (`z ≈ 8000` m in scene coordinates, `castShadow = true`, `shadow.mapSize` of `1024 × 1024`). Its moving grazing rays sweep across crater rims and cast long dynamic shadows across the floor.
3. **Hopper plumes and strobes:** Give each hopper a short-range `PointLight(0xff7a1a, 2.5, 600)` that brightens during `"ascent"`, `"coast"` and `"descent"` phases to illuminate the regolith directly beneath the rocket plume.
4. **Darker crater floors:** Multiply vertex colours (or an ambient-occlusion channel baked from SLDEM2015 sky-view factor) by normalised elevation `(z - min_z) / (max_z - min_z)` clamped to `[0.45, 1.0]` so deep crater bowls stay noticeably darker than exposed rims.
5. **Halloween palette (styling only):** Per [`AGENTS.md`](../../AGENTS.md), the Halloween theme never changes a physical number. Apply it purely in presentation:
   - Deep violet sky/starlight fill (`#1a1033`) and subtle rim-light shader tint.
   - Pumpkin orange (`#ff6b1a`) for active hopper thrusters, selected crater rings and relay link beams.
   - Optional toggleable Diviner thermal overlay (`data/diviner_night.png`) mapped to a violet-to-ember colour ramp (coldest crater traps in dark violet `< 90 K`, warmer rocky rims in ember orange `> 115 K`), with a legend showing true Kelvin / Celsius values.

### Step 5: Performance budget

Target **60 fps at 1080p on an integrated laptop GPU** (Apple M-series or Intel Iris Xe):

| Resource | Budget | Notes |
|---|---|---|
| Terrain mesh | $512 \times 512$ vertices (**522,242 triangles**) | Single draw call, static `BufferGeometry` (~6 MB VRAM for positions, normals and UVs). Drop to $256 \times 256$ (**130,050 triangles**) on low-power devices. |
| Hoppers, crater rings, trajectory cloud | **< 50,000 triangles** total | Use `InstancedMesh` or merged `LineSegments` for the ~1,000 verifier rollouts in the trajectory cloud. |
| Surface textures | $1024 \times 1024$ PNG (or $2048 \times 2048$ max) | 1 to 4 MB VRAM per texture; at most 2 textures active on terrain (`wac_albedo.png` + optional `diviner_night.png`). |
| Shadow maps | One `1024 × 1024` shadow map | Only on the relay orbiter light; hopper plume point lights do not cast shadows. |
| Pixel ratio | `Math.min(window.devicePixelRatio, 2)` | Caps retina fill-rate cost. |

### Step 6: Files loaded by the 3D view (`data/` contract)

The front end in `apps/web` never imports or runs Python training code. It loads only static files from `data/` (or identical JSON payloads from the FastAPI backend):

| File | Size | Required | Contents |
|---|---|---|---|
| `data/terrain.json` | ~1 KB | Yes | Crop origin, ENU bounds, grid dimensions (`512 × 512`), `cell_size_m`, `min_z_m`, `max_z_m` and asset filenames. |
| `data/terrain.bin` *(or `data/terrain.png`)* | 1.0 MB *(or ~180 KB)* | Yes | $512 \times 512$ little-endian `Float32` elevations in metres *(or 16-bit PNG scaled between `min_z_m` and `max_z_m`)*. |
| `data/wac_albedo.png` | ~800 KB | Yes | $1024 \times 1024$ 8-bit greyscale WAC albedo texture co-registered to the ENU crop. |
| `data/diviner_night.png` | ~50 KB | Optional | $256 \times 256$ 8-bit normalised Diviner night temperature texture for the thermal overlay toggle. |
| `data/episodes.json` | 100 KB to 2 MB | Yes | Fleet metadata, `craters[]`, `relay_windows[]`, reward history and step-by-step hopper trajectories per [`docs/data/EPISODE_SCHEMA.md`](EPISODE_SCHEMA.md). |
