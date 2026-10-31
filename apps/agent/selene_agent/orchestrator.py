import argparse
import sys
import time
from pathlib import Path

from google import genai

AGENT_ID = "selene-orchestrator"
BASE_AGENT = "antigravity-preview-09-2026"
MODEL = "gemini-4-argon"
PROMPT = Path(__file__).parent / "prompts" / "orchestrator.md"
POLL_SECONDS = 5


def create_agent(client):
    # The API has no update, so a prompt change means replacing the agent.
    client.agents.delete(id=AGENT_ID)
    return client.agents.create(
        id=AGENT_ID,
        base_agent=BASE_AGENT,
        agent_config={"type": "antigravity", "model": MODEL},
        system_instruction=PROMPT.read_text(),
        base_environment={"type": "remote", "network": "disabled"},
    )


def ask(client, text):
    interaction = client.interactions.create(
        agent=AGENT_ID,
        input=text,
        environment="remote",
        background=True,
    )
    while interaction.status in ("queued", "in_progress"):
        time.sleep(POLL_SECONDS)
        interaction = client.interactions.get(id=interaction.id)
    if interaction.status != "completed":
        raise RuntimeError(f"interaction {interaction.id} ended as {interaction.status}")
    return interaction


def main():
    parser = argparse.ArgumentParser(description="Selene's Argon orchestrator")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create", help="create or replace the agent")
    commands.add_parser("ask", help="send the agent a message").add_argument("text")
    args = parser.parse_args()

    client = genai.Client()
    if args.command == "create":
        agent = create_agent(client)
        print(f"{agent.id} on {agent.agent_config.model}")
    else:
        interaction = ask(client, args.text)
        print(interaction.output_text)
        print(f"tokens: {interaction.usage.total_tokens}", file=sys.stderr)


if __name__ == "__main__":
    main()
