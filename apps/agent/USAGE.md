# Selene orchestrator usage

Selene runs as a Google managed agent. Your laptop does not need to stay
running. Call the Interactions API with the agent ID; no separate web server
is required.

| Setting | Value |
| --- | --- |
| Agent ID | `selene-orchestrator` |
| Model | `gemini-4-argon` |
| Base agent | `antigravity-preview-09-2026` |
| Google Cloud project | `imp-hallow26lon-5582` |
| API endpoint | `https://generativelanguage.googleapis.com/v1beta/interactions` |

The agent proposes safety specs, rewards and fleet plans, revises plans from
verifier statistics, and reviews logs and world-model errors. Supply any
markdown, text or JSON inputs in your call. Sandbox network access is disabled.
Its calculations are estimates, never verifier verdicts. It cannot approve
or uplink a route.

## Load your API key

Use a Gemini API key from the same Google Cloud project as the agent. An
unrelated project's key cannot access this registered agent. Keep keys out
of Git, messages and shared request examples. Requests use the project's
quota and billing.

From the repository root:

```bash
cd apps/agent
```

Put your key in the ignored `apps/agent/.env` file:

```dotenv
GEMINI_API_KEY=your_key_here
```

Load it into your shell before using curl:

```bash
set -a
source .env
set +a

# Check without printing the key.
[[ -n "$GEMINI_API_KEY" ]] && echo "API key loaded" || echo "API key missing"
```

## Tools available to the agent

For reward design, trajectory planning, verification and log review, the
orchestrator passes function tools from [`selene_agent/tools.py`](selene_agent/tools.py)
and automatically handles `requires_action` tool-call turns:

| Tool | Job | What it does |
| --- | --- | --- |
| `get_mission_state` | Trajectory planning | Returns craters, relay windows and fleet hoppers from `episodes.json` or custom state |
| `estimate_hop` | Trajectory planning | Calculates lunar ballistic hop distance, flight time, delta-v, fuel and battery estimates |
| `get_training_summary` | Reward design | Reads `episode_rewards`, `reward_versions` and episode outcomes |
| `test_reward_code` | Reward design | Compiles and runs candidate `compute_reward` code against logged steps |
| `verify_fleet_plan` | Verification | Runs noisy rollouts (or a custom verifier callback) against the safety spec and returns pass rate and failures |
| `get_hopper_logs` | Drone logs | Filters logged episodes and world-model prediction error (`wm_error`) hotspots |

## Five structured jobs from the command line

Every job accepts optional markdown/JSON files or raw text strings, lets Argon
call the tools above to fetch mission data and verify its work, and prints
validated JSON. Pass `-o` to save to disk, `--cache` to store a labelled
fallback, and `--use-cache` to load the cached file offline (`"source": "cached"`).

```bash
# 1. Mission documents -> SafetySpec JSON (with executable check_code per rule)
uv run --env-file .env python -m selene_agent.orchestrator safety-spec \
  ../../docs/SUMMARY.md -o ../../data/safety_spec.json --cache ../../data/cached/safety_spec.json

# 2a. Initial reward design (uses get_mission_state, get_training_summary, test_reward_code)
uv run --env-file .env python -m selene_agent.orchestrator reward \
  --episode 1 -o ../../data/reward.json

# 2b. Reward rewrite from training curves and episode logs
uv run --env-file .env python -m selene_agent.orchestrator rewrite-reward \
  ../../data/reward.json --episode 200 -o ../../data/reward_v2.json

# 3. Trajectory planning (uses get_mission_state, estimate_hop, verify_fleet_plan)
uv run --env-file .env python -m selene_agent.orchestrator plan \
  -o ../../data/fleet_plan.json

# 4. Verification revision (uses verify_fleet_plan and estimate_hop)
uv run --env-file .env python -m selene_agent.orchestrator revise \
  ../../data/fleet_plan.json -o ../../data/fleet_plan_revised.json

# 5. Drone logs and world-model error review (uses get_hopper_logs)
uv run --env-file .env python -m selene_agent.orchestrator review-logs \
  -o ../../data/log_review.json
```

## Use from Python (in `apps/sim` or scripts)

Teammates can pass a custom `verifier_fn`, `episodes_data` or register custom
tools on `AgentToolset`:

```python
from selene_agent import AgentToolset, Orchestrator, verify_and_revise

toolset = AgentToolset(
    episodes_data="../../data/episodes.example.json",
    verifier_fn=my_sim_verifier,  # optional: defaults to built-in Monte Carlo rollout checker
)
orch = Orchestrator(toolset=toolset)

# 1. Safety spec from any markdown string or Path (automatically linked into toolset)
spec = orch.write_safety_spec("Keep at least 15% fuel reserve and land on slopes under 12 deg.")

# 2. Reward design and rewrite (Argon calls get_training_summary and test_reward_code)
reward = orch.write_reward()
revised_reward = orch.rewrite_reward(reward, episode=200)

# 3 & 4. Fleet planning and verification (Argon calls get_mission_state, estimate_hop, verify_fleet_plan)
plan = orch.plan_fleet()
final_plan, verdict = verify_and_revise(orch.client, plan, toolset=toolset)

# 5. Hopper log and world-model error review (Argon calls get_hopper_logs)
review = orch.review_logs()
```

## Ask free-form questions from the command line

Needs uv and Python 3.12 or later. From `apps/agent`:

```bash
uv sync --locked
uv run --env-file .env python -m selene_agent.orchestrator ask \
  "What information do you need to write a safety spec?"
```

The command starts a background interaction, polls every five seconds and
prints the answer when complete. Token usage is printed to standard error.
Each `ask` starts a fresh conversation and sandbox.

## Ask through HTTP

```bash
curl --fail-with-body --silent --show-error \
  https://generativelanguage.googleapis.com/v1beta/interactions \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "selene-orchestrator",
    "input": "What information do you need to write a safety spec?",
    "environment": {"type": "remote"},
    "background": true
  }'
```

The initial response includes `id`, `status` and `environment_id`. A status
of `queued` or `in_progress` means the answer is not ready. Save both IDs.

Fetch the interaction, replacing the placeholder with its returned `id`:

```bash
INTERACTION_ID='paste_returned_interaction_id_here'

curl --fail-with-body --silent --show-error \
  "https://generativelanguage.googleapis.com/v1beta/interactions/$INTERACTION_ID" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  | python -m json.tool
```

Repeat the GET request every five seconds while the status is `queued` or
`in_progress`. When it is `completed`, read the text content in the
`model_output` entries under `steps`. Stop polling on `failed`, `cancelled`,
`incomplete` or `requires_action` and inspect the response.

## Inspect the registered agent

```bash
curl --fail-with-body --silent --show-error \
  https://generativelanguage.googleapis.com/v1beta/agents/selene-orchestrator \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  | python -m json.tool
```

This returns the saved model, instructions and environment configuration.
For usage and costs, open the [AI Studio dashboard](https://aistudio.google.com/usage)
with the hackathon account and select the project.

## Replace the agent definition

Only run this when intentionally deploying a changed prompt, toolset or data bundle:

```bash
uv run --env-file .env python -m selene_agent.orchestrator create
```

By default, `create` deletes and recreates `selene-orchestrator` on Google Cloud and mounts the cloud workspace bundle into `base_environment.sources`:
- `/workspace/selene_tools.py` (standalone CLI and Python tool module)
- `/workspace/mission_constraints.md` (`--mission-doc` or `data/mission_constraints.example.md`)
- `/workspace/safety_spec.json` (`--safety-spec` or `data/safety_spec.json`)
- `/workspace/episodes.json` (`--episodes` or `data/episodes.example.json`)

Because these files live inside the remote sandbox on Google Cloud, anyone calling `https://generativelanguage.googleapis.com/v1beta/interactions` directly over HTTP or from AI Studio Playground can run all five jobs without running local Python tools. The prompt lives in [`selene_agent/prompts/orchestrator.md`](selene_agent/prompts/orchestrator.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| 403, unregistered caller | The shell variable may be empty. Load `.env` and check it without printing the key. |
| Agent not found or access denied | Confirm the agent ID and that the key belongs to the agent's project. |
| Only IDs and `in_progress`, no answer | Background execution has started. Poll the interaction GET endpoint. |
| CLI cannot find `selene_agent` | Run from `apps/agent` using `uv run`. |
| Agent asks for missing data | Check `missing_data` in the JSON output and supply the missing documents or state. |

## Google documentation

- [Building managed agents](https://ai.google.dev/gemini-api/docs/custom-agents)
- [Managed agents quickstart](https://ai.google.dev/gemini-api/docs/managed-agents-quickstart)
- [AI Studio agent playground](https://ai.google.dev/gemini-api/docs/aistudio-agents)
