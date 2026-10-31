# 0008: A hopper fleet under a single Argon orchestrator

## Status

Accepted

## Context

The project needs a multi-agent system. Two designs were possible: many LLM agents with separate roles, or many physical agents coordinated by one LLM.

## Decision

The multi-agent system is a fleet of four to five hoppers trained together with MAPPO: one shared policy, a centralised critic during training, each hopper acting on its own observations. All hoppers feed one shared world model. Gemini 4 Argon is the single orchestrator above the fleet. It reads mission documents, writes the safety rules and reward, plans every trajectory, drives the verification loop and reviews the drones' logs. No other model plans or judges.

## Consequences

The verifier must check fleet-level rules (separation between hoppers, relay conflicts) as well as per-hopper ones. Training cost grows with fleet size, so training starts with fewer hoppers and scales up. One orchestrator keeps decisions traceable to a single source.
