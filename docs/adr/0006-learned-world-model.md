# 0006: A learned world model inside a hand-built simulator

## Status

Accepted

## Context

"World model" can mean the simulated environment or a model the agent learns of it. The pitch, shared with Kessel, is a machine that learns an environment it has never seen and plans inside that model.

## Decision

The simulator is hand-built ground truth. The hopper learns a small world model (state and action in, next state out) from simulator transitions, and the policy plans inside it, Dreamer-style. The verifier always uses the simulator, never the world model. World-model error is logged per step and mapped per cell, and Argon reviews it.

## Consequences

Two models to keep apart in code and in the UI. The interesting failure, the world model being confidently wrong on unvisited terrain, becomes a feature of the demo. Training is real; the 3D view replays logged episodes (ADR-0007).
