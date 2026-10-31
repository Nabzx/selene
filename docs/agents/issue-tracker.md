# Issue tracker: GitHub

Issues live as GitHub issues. Use the `gh` CLI. `gh` infers the repo from `git remote -v`.

## Conventions

- **Create**: `gh issue create --title "..." --body "..."`, heredoc for multi-line bodies. Titles describe the job plainly, not as a Conventional Commits line.
- **Read**: `gh issue view <n> --comments`.
- **List**: `gh issue list --state open --label phase-1 --json number,title,labels`.
- **Close**: `gh issue close <n> --comment "..."`.

## Labels

**Phase**: `phase-0` to `phase-6`, matching `ROADMAP.md`.

**Wayfinder**: `wayfinder:map`, `wayfinder:research`, `wayfinder:grilling`, `wayfinder:prototype`, `wayfinder:task`, `ready-for-agent`.

**Priority** (bugs): `p1` the demo is broken or an invariant is violated, `p2` ordinary defect, `p3` minor.

**Area**: `sim`, `verifier`, `argon`, `rl`, `view`, `data`, `demo`, `docs`, `adr`.

## Wayfinder

The map is one issue per phase labelled `wayfinder:map`, holding Fog, Decisions so far and Frontier. Child tickets are sub-issues of the map. The frontier is the set of open child tickets with no open blocker. Work them in map order.
