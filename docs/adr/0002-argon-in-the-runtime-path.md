# 0002: Argon in the runtime path

## Status

Accepted

## Context

The judges ask whether a project does work that needed Gemini 4 Argon. Code is written with Claude Code, which the judges never see. A token Argon call at the end of the pipeline would fail that test.

## Decision

Argon does five jobs while the app runs: writing the safety spec from documents, writing and rewriting the reward function, planning the crater route, revising a route from failure statistics, and reviewing the world model's errors. Smaller Gemini models handle summaries and guardrails. The verifier is deterministic code. No non-Gemini model runs inside the app.

## Consequences

The demo depends on Argon latency, so long calls run with `background=True` and the demo keeps labelled cached fallbacks. Any change that moves one of Argon's four jobs to a smaller model needs a new ADR.
