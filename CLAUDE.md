# CLAUDE.md

Conventions for Claude Code when working in this repository.

## Quick orientation

Read `AGENTS.md` first: invariants, phases, the wayfinder protocol, research and ADR conventions. Read `CONTEXT.md` for the vocabulary. This file covers the Claude Code angle.

Claude writes the code. Claude never runs inside the app. Every model call at runtime is a Gemini model (ADR-0002).

## Gemini code

- Load the `gemini-api-dev` skill (`.claude/skills/gemini-api-dev/`) before touching anything that calls a model. Its rules override your training data.
- Interactions API only: `client.interactions.create(...)`, `google-genai` 2.30.1 or later.
- `thinking_level` is `"low"`, `"medium"` or `"high"`. No `thinking_budget`, `temperature`, `top_p` or `top_k`.
- Output is capped at 65,536 tokens unless `background=True` and a higher `max_output_tokens`. Poll with `client.interactions.get(id=...)`.
- `"high"` thinking can run 15 to 20 minutes. Use background mode or long timeouts, never a blocking call in a request handler.
- Use `previous_interaction_id` for multi-turn state instead of resending history.

## House style

- British English throughout (organise, colour, behaviour, licence as a noun).
- No em dashes. Use a comma, a colon, or a full stop instead.
- Short, direct sentences. No padding, no hedging.
- No AI filler: "comprehensive", "robust", "seamless", "leverage" only if they do real work.
- No abstractions, error handling or options for scenarios that can't happen.

## Commits

- Conventional Commits: `type(scope): summary`, for example `feat(verifier): add slope check (#4)`.
- One logical change per commit.
- `refs #n` in the body, not `closes #n`.
- No co-author trailers.

## Skills to reach for

- `gemini-api-dev` for any model call.
- `frontend-design` for the 3D view and HUD, to avoid default-looking UI.
- `simplify` and `code-review` before calling a change finished.
- `commit-commands` for commit, push and PR.
- `claude-md-management` to keep this file and `AGENTS.md` current.

## Documents

Reference materials go in `/materials/`, which is gitignored. Never commit raw documents or dataset downloads.
