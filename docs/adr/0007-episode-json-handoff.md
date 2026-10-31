# 0007: Episode JSON as the handoff to the 3D view

## Status

Accepted

## Context

A teammate builds the 3D world while training runs in Python. The two must not block each other or drift apart.

## Decision

Training logs sampled episodes to `data/episodes.json` in the schema in `docs/data/EPISODE_SCHEMA.md`. The 3D view replays that file and reads nothing else for replay. An example file lets the view be built before training exists. A test validates the example against the schema.

## Consequences

The view can be built and demoed from a scripted file on day one, labelled `"source": "scripted"`. Swapping in trained episodes needs no view changes. Live streaming of training is out of scope.
