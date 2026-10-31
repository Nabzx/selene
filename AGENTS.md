# AGENTS.md

Structure and conventions for anyone, human or agent, working in this repository.

## Skills

### Gemini API

Any code that calls a Gemini model must follow the `gemini-api-dev` skill in
[`.agents/skills/`](.agents/skills) (mirrored in `.claude/skills/`). Your
training data predates Gemini 4 Argon and the Interactions API. Do not write
`generate_content`, `thinking_budget`, `temperature`, `top_p` or `top_k`.

### Domain docs

Single-context layout: [`CONTEXT.md`](CONTEXT.md) and [`docs/adr/`](docs/adr)
at the repo root. See [`docs/agents/domain.md`](docs/agents/domain.md).

### Issue tracker

Issues live in this repo's GitHub Issues. See
[`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md).

## Invariants

These do not change without an ADR that supersedes this list.

### One maintainer

Selene has a single maintainer, Nabil Shah. Every commit, issue and pull
request is authored by him. No commit carries a co-author trailer, including
no AI tool attribution of any kind.

### Argon does the hard part at runtime

Gemini 4 Argon is in the runtime path, not a cameo. It writes the safety spec
and the reward, plans the route, revises it after a failed check, and reviews
the world model's errors. If a change would let a smaller model do one of
those jobs without the demo changing, that change needs an ADR. Only Gemini
models run inside the app. See ADR-0002.

### The simulator and the world model stay separate

The simulator is ground truth. The world model is learned from it and is
allowed to be wrong. Never read world-model output as truth, and never let the
verifier use the world model. See ADR-0006.

### Nothing uplinks without a pass and a person

A route reaches the flight view only after the verifier passes it and a
person presses approve. No code path skips either step, including in demo
mode.

### Scripted is labelled scripted

Anything precomputed or hardcoded for the demo (a cached Argon response, a
pre-trained policy, the concept GIF) is labelled as such on screen and in
code. Live output is never faked.

### Real data, honest gaps

Terrain and temperature come from the datasets in
[`docs/data/DATASETS.md`](docs/data/DATASETS.md). Hopper physics, relay
windows and payload readings are simulated, and the UI says so. The Halloween
theme is styling only: it never changes a number.

### The front end is data-only and has no build step

`apps/web` is plain HTML, CSS and JavaScript with Three.js from a CDN import
map. It draws only what `data/episodes.json` and the backend send, in the
schema in [`docs/data/EPISODE_SCHEMA.md`](docs/data/EPISODE_SCHEMA.md). If the
view needs a field the schema lacks, change the schema first. No bundler, no
`node_modules`.

## Phases

Work is organised into phases in [`ROADMAP.md`](ROADMAP.md). Each phase opens
with one `wayfinder:map` issue holding **Fog**, **Decisions so far** and
**Frontier**. Child tickets carry `Part of: #n` and one of
`wayfinder:research`, `wayfinder:grilling`, `wayfinder:prototype` or
`wayfinder:task`.

## Research and decisions

Research output lands in `docs/research/issue-<n>-<slug>.md` and ends with a
recommendation. An adopted recommendation becomes a numbered ADR in
`docs/adr/`, starting from `0000-template.md`. The index in
`docs/adr/README.md` stays in sync with the files present.

## Pull requests and commits

- Conventional Commits: `type(scope): summary`, lowercase, imperative.
- PR body: *What this does*, *Why*, and the template checklist.
- Reference issues with `refs #n`, not `closes #n`.
- Small commits, one logical change each, no co-author trailers.

## House style

- British English throughout.
- No em dashes.
- Short and direct. No filler, no hedging, no hype.

## Documents

Reference materials (PDFs, briefs, raw downloads) go in `/materials/`, which
is gitignored and never committed.
