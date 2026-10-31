# 0003: Verifier

## Status

Proposed

## Context

A hop plan must be shown safe before uplink. The check has to be fast enough to run live (under 10 s), produce statistics Argon can act on, and draw well as a trajectory cloud.

## Options

**A. Monte Carlo with rule checks.** Replay the plan about 1000 times with sampled thrust error, landing scatter, dust and sensor noise. Check every rollout against every rule in the safety spec. Pass when at least 99% of rollouts break no rule. Output pass rate and failures per rule and per hop.

**B. Option A plus Z3 for the strict rules.** For rules with simple arithmetic (fuel reserve, maximum slope at a landing cell), encode the plan and the noise bounds in Z3 and prove the rule holds for every value in the bounds, not only the sampled ones.

## Decision

Pending the maintainer's call. Option A is the default if no decision is made before Phase 3.

## Consequences

Option A is a statistical claim and is quick to build. Option B is a stronger claim for a subset of rules and risks several hours.
