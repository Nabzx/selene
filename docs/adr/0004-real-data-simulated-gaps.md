# 0004: Real terrain, simulated hopper and payload

## Status

Accepted. Revised by ADR-0005 for the move to the far side.

## Context

Far-side terrain and night temperatures are mapped and public. No public dataset describes a hopper, a relay orbit for our scenario, or the payload readings.

## Decision

Use the real datasets in `docs/data/DATASETS.md` for terrain and temperature. Model hopper flight ballistically at 1.62 m/s² with simple fuel, battery and thermal models. Simulate relay windows and payload readings.

## Consequences

The UI labels the hopper, relay and payload models as simulated. Terrain claims can be checked against NASA products.
