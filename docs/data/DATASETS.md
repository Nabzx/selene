# Datasets

What the simulator and the 3D view use. Download before the event. Raw files go in `materials/data/` (gitignored). Only small derived files are committed, under `data/`.

| Layer | Dataset | Format | Used for |
|---|---|---|---|
| Terrain (higher resolution) | SLDEM2015: LOLA and Kaguya merged DEM, about 60 m per pixel, 60°N to 60°S (Barker et al. 2016). PDS Geosciences LOLA node | GeoTIFF / IMG | Height grid, slope, landing hazard, 3D mesh |
| Terrain (global) | LOLA global gridded DEM (LDEM), PDS Geosciences LOLA node | IMG | Fallback, and any region above 60° latitude |
| Night temperature | Diviner global surface temperature maps, [diviner.ucla.edu](https://www.diviner.ucla.edu/), PDS | Gridded maps | Night thermal limits, battery heating cost |
| Crater catalogue | Robbins lunar crater database (2018), PDS | CSV | Crater centres and radii for `craters[]` |
| Imagery (optional) | LROC WAC global mosaic, [lroc.asu.edu](https://www.lroc.asu.edu/) | GeoTIFF | Terrain texture for the 3D view |

Exact filenames change between releases. Check them before scripting downloads.

## Candidate regions

| Region | Why |
|---|---|
| Daedalus crater | Near the far-side centre, the most radio-quiet place near Earth, a proposed radio telescope site |
| Tsiolkovskiy crater | Dark mare floor against bright highlands, very striking at low light |
| Apollo basin | Large, varied terrain, many smaller craters to hop between |

## Documents for Argon

PDFs go in `materials/docs/`.

- NASA-STD-8719 series (safety)
- Far-side mission studies (relay operations, lunar night survival)
- Chang'e 4 and Queqiao relay operations papers

## Processing

1. Crop about 20 × 20 km around the chosen region from SLDEM2015.
2. Resample to about 100 m per cell for the simulation grid (about 200 × 200 cells).
3. Keep a decimated full-resolution crop for the 3D mesh, written to `data/terrain.json` in the episode schema's frame.
4. Derive slope per cell. Pull craters in the crop from the Robbins catalogue.
5. Sample night temperature for the crop from Diviner.

## Attribution

Cite Barker et al. (2016) for SLDEM2015, Robbins (2018) for the crater database, and the Diviner team for temperatures.
