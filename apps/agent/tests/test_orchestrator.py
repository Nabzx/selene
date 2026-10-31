from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from selene_agent import orchestrator


def interaction(status):
    return SimpleNamespace(id="i-1", status=status)


def test_create_agent_replaces_agent_on_argon_with_network_disabled():
    client = MagicMock()

    orchestrator.create_agent(client)

    client.agents.delete.assert_called_once_with(id="selene-orchestrator")
    client.agents.create.assert_called_once_with(
        id="selene-orchestrator",
        base_agent="antigravity-preview-09-2026",
        agent_config={"type": "antigravity", "model": "gemini-4-argon"},
        system_instruction=orchestrator.PROMPT.read_text(),
        base_environment={"type": "remote", "network": "disabled"},
    )


def test_ask_polls_in_background_until_completed(monkeypatch):
    monkeypatch.setattr(orchestrator.time, "sleep", lambda _: None)
    client = MagicMock()
    client.interactions.create.return_value = interaction("queued")
    client.interactions.get.side_effect = [interaction("in_progress"), interaction("completed")]

    result = orchestrator.ask(client, "hello")

    assert result.status == "completed"
    assert client.interactions.get.call_count == 2
    client.interactions.create.assert_called_once_with(
        agent="selene-orchestrator",
        input="hello",
        environment="remote",
        background=True,
    )


@pytest.mark.parametrize("status", ["failed", "cancelled", "requires_action", "incomplete"])
def test_ask_raises_unless_completed(status):
    client = MagicMock()
    client.interactions.create.return_value = interaction(status)

    with pytest.raises(RuntimeError, match=status):
        orchestrator.ask(client, "hello")
