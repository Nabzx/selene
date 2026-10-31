# 0004: Real terrain, simulated ice and flight

## Status

Accepted

## Context

Terrain, shadow, illumination and temperature at the lunar south pole are mapped and public. Ice abundance per cell is not: the instruments give indications, not quantities. No public dataset describes a hopper.

## Decision

Use the real datasets in `docs/data/DATASETS.md` for terrain, PSR, illumination and temperature. Sample a hidden ice field from the PSR ice database and LAMP data plus noise. Model hopper flight ballistically at 1.62 m/s² with a simple fuel model.

## Consequences

The UI labels the ice field and flight model as simulated. The terrain claims can be checked against NASA products.
