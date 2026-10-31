# 0005: Far side at lunar night, Halloween styling

## Status

Accepted

## Context

The first plan was the south pole's permanently shadowed craters. The team wants a darker, Halloween-themed setting. The far side is not permanently dark, but it never sees Earth, and lunar night there lasts about two weeks.

## Decision

Set the simulation on real far-side terrain during lunar night. The core constraint is autonomy: no line of sight to Earth, contact only in relay windows. Styling is eerie but credible: night palette in orange and violet, drifting dust, glowing scan beams, Halloween crater names. Copy may joke about "the dark side" but the docs and the science stay accurate.

## Consequences

Drops the PSR and illumination datasets, adds relay windows and night thermal limits. The region is still open (Daedalus, Tsiolkovskiy or Apollo basin). Styling never changes a number.
