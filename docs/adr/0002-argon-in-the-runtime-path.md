# 0002: Argon in the runtime path

## Status

Accepted

## Context

The judges ask whether a project does work that needed Gemini 4 Argon. Code is written with Claude Code, which the judges never see. A token Argon call at the end of the pipeline would fail that test.

## Decision

Argon does four jobs while the app runs: writing the safety spec from documents, writing the reward function, proposing hop plans, and revising a plan from failure statistics. Smaller Gemini models handle summaries and guardrails. The verifier is deterministic code. No non-Gemini model runs inside the app.

## Consequences

The demo depends on Argon latency, so long calls run with `background=True` and the demo keeps labelled cached fallbacks. Any change that moves one of Argon's four jobs to a smaller model needs a new ADR.
