# Domain docs

How an engineering agent should consume this repo's domain documentation.

## Before working in an area, read these

- [`CONTEXT.md`](../../CONTEXT.md) at the repo root: the glossary.
- [`docs/adr/`](../adr): the ADRs that touch the area you are about to change.
- [`docs/OVERVIEW.md`](../OVERVIEW.md): what the project is and how the demo runs.

## Use the glossary's vocabulary

When your output names a domain concept, in an issue title, an ADR, a commit or a variable name, use the term as defined in `CONTEXT.md`. Do not drift to a synonym the glossary rules out (the hopper is never a quadcopter, the ice field is never real ice data).

If the concept you need is not in the glossary, either you are inventing language the project does not use, or there is a real gap. Note it and propose an addition.

## Flag ADR conflicts

If your change would contradict an Accepted ADR, say so openly:

> Contradicts ADR-0002 (Argon in the runtime path), but worth reopening because ...

An Accepted ADR is changed only by a new ADR that supersedes it.
