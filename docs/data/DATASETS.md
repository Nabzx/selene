# Datasets

Everything the simulator and the 3D view use. Download before the event. Raw files go in `materials/data/` (gitignored). Only small derived crops are committed, under `data/`.

| Layer | Dataset | Format | Used for |
|---|---|---|---|
| Terrain | [LOLA south pole DEM, 80 to 90°S, 20 m/px](https://pgda.gsfc.nasa.gov/products/90) (NASA PGDA, Barker et al.) | Cloud-optimised GeoTIFF, polar stereographic, metres, MOON_ME frame | Height grid, slope, landing hazard, 3D mesh |
| Permanent shadow | Same page, `LPSR_80S_20MPP_ADJ.TIF` (148 MB) | GeoTIFF | PSR mask: no solar charging, extreme cold, ice candidates |
| Sunlight | [Lunar polar illumination](https://pgda.gsfc.nasa.gov/products/69) (Mazarico et al. 2011) | GeoTIFF (COG) | Charging and safe landing windows |
| Roughness | [LOLA polar roughness](https://pgda.gsfc.nasa.gov/products/98) | GeoTIFF | Landing risk |
| Temperature | [Diviner Polar Resource Product](https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/Lunar%20Reconnaissance%20Orbiter%20(LRO)/DIVINER/PRP.htm), [diviner.ucla.edu](https://www.diviner.ucla.edu/) | Gridded maps (PDS) | Cold traps, thermal limits |
| Ice candidates | [Ice-bearing PSR database](https://zenodo.org/record/4646092) (Lemelin et al. 2021, CC-BY 4.0) | Shapefile, Excel | Prior for the ice field |
| Surface frost | [LRO-LAMP south pole maps](https://openpolar.no/Record/ftharvardunivdvn:doi:10.7910%2FDVN%2FW3KEWG) (Harvard Dataverse) | Maps, spectra | Extra prior for the ice field |
| Hydrogen | LEND map, via [NASA SVS](https://svs.gsfc.nasa.gov/cgi-bin/details.cgi?aid=3690) | PDS | Coarse search prior |

Exact filenames on the PGDA pages change between releases. Check them before scripting the download.

## Documents for Argon

These are what Argon reads to write the safety spec. PDFs go in `materials/docs/`.

- NASA-STD-8719 series (safety)
- Artemis III candidate landing region announcements
- VIPER mission overview papers (traverse and PSR operations constraints)

## Processing

1. Crop a box around Shackleton crater, about 20 × 20 km, from the DEM and PSR map (`rasterio` window read, or `gdalwarp -te`).
2. Resample to about 100 m per cell for the simulation grid (roughly 200 × 200 cells).
3. Keep the 20 m crop, decimated, for the 3D mesh.
4. Derive slope per cell from the DEM.
5. Write `data/grid.npz` (height, slope, psr, illumination, temperature) and `data/mesh.bin` for the view.

## Attribution

Cite Barker et al. for LOLA products, Mazarico et al. (2011) for illumination, and Lemelin et al. (2021) for the PSR ice database.
