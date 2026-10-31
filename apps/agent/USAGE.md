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
verifier statistics, and reviews logs and world-model errors. It has no
connection yet to the simulator, verifier, terrain data or mission documents.
Supply the relevant data in your message. Sandbox network access is disabled.
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

## Ask from the command line

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

## Continue a conversation in Python

Pass both IDs to preserve the conversation and sandbox. Run this after
loading your key, with the dependencies installed:

```python
import time

from google import genai

client = genai.Client()
interaction = client.interactions.create(
    agent="selene-orchestrator",
    input="Explain which mission constraints I should supply first.",
    previous_interaction_id="paste_previous_interaction_id_here",
    environment="paste_previous_environment_id_here",
    background=True,
)

while interaction.status in ("queued", "in_progress"):
    time.sleep(5)
    interaction = client.interactions.get(id=interaction.id)

if interaction.status != "completed":
    raise RuntimeError(f"Interaction ended as {interaction.status}")

print(interaction.output_text)
```

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

Only run this when intentionally deploying a changed prompt or configuration:

```bash
uv run --env-file .env python -m selene_agent.orchestrator create
```

The current command deletes `selene-orchestrator` before recreating it. It is
not needed to use the existing deployment, and a failed recreation leaves
the agent unavailable. The prompt lives in
[`selene_agent/prompts/orchestrator.md`](selene_agent/prompts/orchestrator.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| 403, unregistered caller | The shell variable may be empty. Load `.env` and check it without printing the key. |
| Agent not found or access denied | Confirm the agent ID and that the key belongs to the agent's project. |
| Only IDs and `in_progress`, no answer | Background execution has started. Poll the interaction GET endpoint. |
| CLI cannot find `selene_agent` | Run from `apps/agent` using `uv run`. |
| Agent asks for missing data | Mission documents and simulator tools are not connected. Supply the inputs explicitly. |

## Google documentation

- [Building managed agents](https://ai.google.dev/gemini-api/docs/custom-agents)
- [Managed agents quickstart](https://ai.google.dev/gemini-api/docs/managed-agents-quickstart)
- [AI Studio agent playground](https://ai.google.dev/gemini-api/docs/aistudio-agents)
