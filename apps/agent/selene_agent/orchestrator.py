from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from google import genai
from pydantic import BaseModel

from selene_agent.schemas import (
    FleetPlan,
    HopperPlanEntry,
    HopperRoute,
    HopStep,
    LogReview,
    PlannedHop,
    RewardSpec,
    SafetyRule,
    SafetySpec,
)
from selene_agent.tools import (
    AgentToolset,
    _default_episodes_path,
    _default_mission_doc_path,
    _default_safety_spec_path,
)

AGENT_ID = "selene-orchestrator"
BASE_AGENT = "antigravity-preview-09-2026"
MODEL = "gemini-4-argon"
PROMPT = Path(__file__).parent / "prompts" / "orchestrator.md"
TOOLS_FILE = Path(__file__).parent / "tools.py"
POLL_SECONDS = 5
MAX_TOOL_ROUNDS = 10

T = TypeVar("T", bound=BaseModel)

SAFETY_SPEC_INSTRUCTIONS = """Task: Read the supplied mission documents or constraints (in any markdown, prose, table or JSON format, or call `read_mission_documents` if not supplied in the message) and extract a machine-checkable safety spec.

Do not restrict rules to a fixed list. Extract every safety rule, operational limit or fleet constraint present in the input.
If a constraint is ambiguous or missing a required threshold, record it in `missing_data` instead of inventing numbers.

For each rule, write a self-contained Python function named `check` in `check_code`:
`def check(trace: dict):`
`trace` may be either:
1. A batch verifier trace dict with numpy arrays (`import numpy as np`):
   - `fuel`: `(R, H, S+1)` fraction in `0..1`, `fuel_kg`: `(R, H, S+1)` in kg (`0..15`)
   - `battery`: `(R, H, S+1)` fraction in `0..1`, `battery_pct`: `(R, H, S+1)` in `%` (`0..100`)
   - `landing_slope_deg`: `(R, H, S)`, `hop_distance_m`: `(R, H, S)`, `hop_time_s`: `(R, H, S)`
   - `min_separation_m`: `(R, S)`, `crashed`: `(R, H, S)` bool, `relay_conflict`: `(R, S)` bool
   - `hopper_ids`: list of hopper IDs, `n_rollouts`: int
2. Or a single-rollout dict with `steps` (list of step dicts with `fuel`, `battery`, `slope_deg`, `temp_c`, `pos`, `action`, `event`) and `fleet_steps`.

If `"steps" in trace`, return a `bool`. Otherwise compute per-rollout pass booleans across the batch and return either `(passed, reason, stats)` where `stats` includes `{"pass_rate": float, "per_rollout_ok": list[bool]}` or a `bool`. Use `.get()` with safe defaults so it never raises `KeyError`.

Return ONLY valid JSON with this structure:
{
  "summary": "Short plain-English summary of extracted safety spec",
  "rules": [
    {
      "id": "R1",
      "name": "Short rule name",
      "scope": "step | hop | route | fleet",
      "statement": "Plain-English statement of the rule",
      "source_excerpt": "Brief quote or reference from the input document",
      "parameters": {"extracted_param_name": 0.10},
      "check_code": "def check(trace: dict):\n    ..."
    }
  ],
  "missing_data": []
}"""

WRITE_REWARD_INSTRUCTIONS = """Task: Design the reward function the hopper fleet learns from during MAPPO training.
You may call `get_mission_state`, `get_training_summary` and `test_reward_code` to inspect the mission setup and verify that your `compute_reward` function runs cleanly on both `RewardState` batches and logged episode steps.

Design reward terms that fit the goals and constraints without hardcoding fixed terms.
Write a self-contained Python function named `compute_reward` in `reward_code` that supports both:
1. Simulator `RewardState` objects (`state, actions=None, next_state=None`) with numpy arrays (`import numpy as np`):
   attributes on `state` and `next_state` are `pos_xy`, `fuel_kg`, `battery_pct`, `temp_c`, `unsent_data`, `crashed`, `alive`, `craters_done`, `craters_claimed`, `crater_priorities`, `relay_open`, `relay_conflict`, `at_crater_id`; `actions` has `target_crater_idx`, `transmit`, `scan_here`. Return `(rewards, components)` where `rewards` has shape `state.fuel_kg.shape` (`float32`) and `components` is `dict[str, np.ndarray]`.
2. Step dicts (`isinstance(state, dict)`): return a `float` using `.get()` with safe defaults.

Always call `test_reward_code` to verify your function before returning.

Return ONLY valid JSON with this structure:
{
  "episode": 1,
  "summary": "One-line summary for episodes.json training.reward_versions",
  "diagnosis": "Rationale for the initial reward design",
  "terms": [
    {
      "name": "term_name",
      "sign": "+ or -",
      "weight": 10.0,
      "expression": "Readable formula or condition",
      "description": "What behaviour this term shapes"
    }
  ],
  "reward_code": "import numpy as np\n\ndef compute_reward(state, actions=None, next_state=None):\n    ...",
  "missing_data": []
}"""

REWRITE_REWARD_INSTRUCTIONS = """Task: Diagnose why learning stalled or what loophole the fleet found, and rewrite the reward function.
Call `get_training_summary` and `get_hopper_logs` if you need training curves or episode steps, and call `test_reward_code` to verify your new `compute_reward` function before returning.

Write an updated self-contained Python function named `compute_reward` in `reward_code` that supports both:
1. Simulator `RewardState` objects (`state, actions=None, next_state=None`) with numpy arrays (`pos_xy`, `fuel_kg`, `battery_pct`, `temp_c`, `unsent_data`, `crashed`, `alive`, `craters_done`, `craters_claimed`, `crater_priorities`, `relay_open`, `relay_conflict`, `at_crater_id`), returning `(rewards, components)`.
2. Step dicts (`isinstance(state, dict)`), returning a `float`.

Return ONLY valid JSON with this structure:
{
  "episode": 200,
  "summary": "One-line summary of the reward change for episodes.json training.reward_versions",
  "diagnosis": "Plain-English explanation of what went wrong in training and how this rewrite fixes it",
  "terms": [
    {
      "name": "term_name",
      "sign": "+ or -",
      "weight": 10.0,
      "expression": "Readable formula or condition",
      "description": "What behaviour this term shapes"
    }
  ],
  "reward_code": "import numpy as np\n\ndef compute_reward(state, actions=None, next_state=None):\n    ...",
  "missing_data": []
}"""

PLAN_FLEET_INSTRUCTIONS = """Task: Assign craters to hoppers and plan every hop for the fleet.
Use `get_mission_state` to fetch craters, relay windows and hoppers if not already in the message, and use `estimate_hop` to compute hop distance, flight time, fuel and battery estimates. You may also call `verify_fleet_plan` to check your candidate plan against the verifier.

Remember: your fuel and battery numbers are estimates, never verifier verdicts. Keep single hops within max_hop_range_m (3000 m), keep hoppers separated by at least 100 m at every stage, and avoid steep crater rims (> 15 deg). Never claim a plan is approved. If critical data is missing, list it in `missing_data` instead of inventing it.

Return ONLY valid JSON with this structure:
{
  "plan_id": "plan-v1",
  "revision": 0,
  "summary": "Plain-English summary of the proposed fleet plan and estimates",
  "routes": [
    {
      "hopper_id": "h1",
      "craters": ["c1", "c2"],
      "hops": [
        {
          "hop_index": 0,
          "from_pos": [0.0, 0.0, -120.0],
          "to_pos": [1800.0, 600.0, -80.0],
          "target_crater": "c2",
          "t_start": 0.0,
          "action_after": "scan",
          "relay_window_index": 0,
          "estimated_fuel_used": 0.11,
          "estimated_battery_used": 0.04,
          "notes": "Estimate only; awaits verifier check"
        }
      ]
    }
  ],
  "unassigned_craters": [],
  "missing_data": []
}"""

REVISE_FLEET_INSTRUCTIONS = """Task: Verify and revise the fleet plan so it passes the safety rules.
Use `verify_fleet_plan` to run the verifier on candidate routes, and `estimate_hop` to check hop costs. Explain what failed and what you changed in `revision_summary` and `changes`.

Return ONLY valid JSON with this structure:
{
  "plan_id": "plan-v1",
  "revision": 1,
  "summary": "Plain-English summary of the revised fleet plan",
  "revision_summary": "Why the previous plan failed the verifier and how this revision addresses those failures",
  "changes": [
    "Moved hopper h1 hop 0 landing target away from steep rim to reduce slope failures"
  ],
  "routes": [
    {
      "hopper_id": "h1",
      "craters": ["c1", "c2"],
      "hops": [
        {
          "hop_index": 0,
          "from_pos": [0.0, 0.0, -120.0],
          "to_pos": [1750.0, 580.0, -82.0],
          "target_crater": "c2",
          "t_start": 0.0,
          "action_after": "scan",
          "relay_window_index": 0,
          "estimated_fuel_used": 0.11,
          "estimated_battery_used": 0.04,
          "notes": "Adjusted landing coordinates"
        }
      ]
    }
  ],
  "unassigned_craters": [],
  "missing_data": []
}"""

REVIEW_LOGS_INSTRUCTIONS = """Task: Inspect the hopper logs and world-model prediction errors and explain what went wrong for each hopper and where the learned world model's predictions diverged from the simulator ground truth.
Call `get_hopper_logs` and `get_world_model_error_map` if you need to fetch or filter the logged episodes and spatial world-model error hotspots.

Remember: the simulator (`pos`) is ground truth and the world model (`wm_pos`, `wm_error`) is learned and can be wrong, especially on unvisited terrain.

Return ONLY valid JSON with this structure:
{
  "summary": "Plain-English overview of hopper performance and world-model reliability",
  "hopper_incidents": [
    {
      "hopper_id": "h1",
      "episode": 500,
      "t": 18.0,
      "event": "landed",
      "explanation": "What happened at this step",
      "fix": "Recommended change to route or policy"
    }
  ],
  "world_model_hotspots": [
    {
      "location": "Crater c2 rim ([1790, 610, -80])",
      "t": 18.0,
      "wm_error_m": 36.1,
      "predicted_pos": [1760.0, 590.0, -78.0],
      "actual_pos": [1790.0, 610.0, -80.0],
      "explanation": "Why the world model mispredicted the landing state here",
      "trust_recommendation": "Do not trust world-model rollouts near this rim until more simulator transitions are collected"
    }
  ],
  "recommendations": [
    "Concrete action to improve training, reward or route planning"
  ],
  "missing_data": []
}"""


def build_cloud_sources(
    *,
    mission_doc: str | Path | None = None,
    safety_spec: str | Path | None = None,
    episodes_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Build inline file sources to mount into /workspace on the cloud remote sandbox."""
    sources: list[dict[str, str]] = [
        {
            "type": "inline",
            "target": "/workspace/selene_tools.py",
            "content": TOOLS_FILE.read_text(),
        }
    ]
    doc_file = Path(mission_doc) if mission_doc is not None else _default_mission_doc_path()
    doc_content = doc_file.read_text() if doc_file.is_file() else (str(mission_doc) if mission_doc else "")
    if doc_content:
        sources.append(
            {
                "type": "inline",
                "target": "/workspace/mission_constraints.md",
                "content": doc_content,
            }
        )

    spec_file = Path(safety_spec) if safety_spec is not None else _default_safety_spec_path()
    if spec_file.is_file():
        sources.append(
            {
                "type": "inline",
                "target": "/workspace/safety_spec.json",
                "content": spec_file.read_text(),
            }
        )

    ep_file = Path(episodes_path) if episodes_path is not None else _default_episodes_path()
    if ep_file.is_file():
        sources.append(
            {
                "type": "inline",
                "target": "/workspace/episodes.json",
                "content": ep_file.read_text(),
            }
        )
    return sources


def create_agent(
    client,
    mission_doc: str | Path | None = None,
    *,
    safety_spec: str | Path | None = None,
    episodes_path: str | Path | None = None,
    bundle_workspace: bool = False,
):
    # The API has no update, so a prompt change means replacing the agent.
    client.agents.delete(id=AGENT_ID)
    base_env: dict[str, Any] = {"type": "remote", "network": "disabled"}
    if bundle_workspace:
        base_env["sources"] = build_cloud_sources(
            mission_doc=mission_doc,
            safety_spec=safety_spec,
            episodes_path=episodes_path,
        )
    elif mission_doc is not None:
        doc_path = Path(mission_doc)
        content = doc_path.read_text() if doc_path.is_file() else str(mission_doc)
        base_env["sources"] = [
            {
                "type": "inline",
                "target": "/workspace/mission_constraints.md",
                "content": content,
            }
        ]
    return client.agents.create(
        id=AGENT_ID,
        base_agent=BASE_AGENT,
        agent_config={"type": "antigravity", "model": MODEL},
        system_instruction=PROMPT.read_text(),
        base_environment=base_env,
    )


def _poll_interaction(client, interaction):
    while interaction.status in ("queued", "in_progress"):
        time.sleep(POLL_SECONDS)
        interaction = client.interactions.get(id=interaction.id)
    return interaction


def _extract_function_calls(interaction) -> list[Any]:
    steps = getattr(interaction, "steps", None) or []
    return [step for step in steps if getattr(step, "type", None) == "function_call"]


def ask(
    client,
    text: str | list[dict[str, Any]],
    *,
    previous_interaction_id: str | None = None,
    environment: str | dict[str, Any] = "remote",
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    toolset: AgentToolset | None = None,
):
    active_tools = tools if tools is not None else (toolset.declarations if toolset is not None else None)
    kwargs: dict[str, Any] = {
        "agent": AGENT_ID,
        "input": text,
        "environment": environment,
        "background": True,
    }
    if previous_interaction_id is not None:
        kwargs["previous_interaction_id"] = previous_interaction_id
    if response_format is not None:
        kwargs["response_format"] = response_format
    if active_tools:
        kwargs["tools"] = active_tools

    interaction = _poll_interaction(client, client.interactions.create(**kwargs))

    rounds = 0
    while interaction.status == "requires_action" and toolset is not None and rounds < MAX_TOOL_ROUNDS:
        calls = _extract_function_calls(interaction)
        if not calls:
            break
        results: list[dict[str, Any]] = []
        for call in calls:
            name = getattr(call, "name", "")
            call_id = getattr(call, "id", "")
            args = getattr(call, "arguments", None) or {}
            try:
                out = toolset.call(name, args)
                results.append(
                    {
                        "type": "function_result",
                        "call_id": call_id,
                        "name": name,
                        "result": json.dumps(out) if not isinstance(out, str) else out,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "type": "function_result",
                        "call_id": call_id,
                        "name": name,
                        "result": f"Error: {exc}",
                        "is_error": True,
                    }
                )
        next_kwargs: dict[str, Any] = {
            "agent": AGENT_ID,
            "input": results,
            "previous_interaction_id": interaction.id,
            "environment": getattr(interaction, "environment_id", None) or environment,
            "background": True,
        }
        if active_tools:
            next_kwargs["tools"] = active_tools
        if response_format is not None:
            next_kwargs["response_format"] = response_format
        interaction = _poll_interaction(client, client.interactions.create(**next_kwargs))
        rounds += 1

    if interaction.status != "completed":
        raise RuntimeError(f"interaction {interaction.id} ended as {interaction.status}")
    return interaction


def format_input(data: Any) -> str:
    """Convert any file path, markdown string, dict, list or Pydantic model into prompt text."""
    if data is None:
        return "Use the available function tools to inspect mission state, training curves or logs."
    if isinstance(data, BaseModel):
        return data.model_dump_json(indent=2)
    if isinstance(data, Path):
        return f"### File: {data.name}\n\n{data.read_text()}"
    if isinstance(data, str):
        stripped = data.strip()
        if "\n" not in stripped and len(stripped) < 260:
            candidate = Path(stripped)
            if candidate.is_file():
                return f"### File: {candidate.name}\n\n{candidate.read_text()}"
        return data
    if isinstance(data, (list, tuple)):
        if not data:
            return "Use the available function tools to inspect mission state, training curves or logs."
        return "\n\n---\n\n".join(format_input(item) for item in data)
    if isinstance(data, dict):
        return json.dumps(data, indent=2)
    return str(data)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw text or a markdown code block."""
    if not text or not text.strip():
        raise ValueError("Empty response from orchestrator")
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    for block in fenced:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not parse JSON object from response: {cleaned[:200]}")


def _run_structured_job(
    client,
    instructions: str,
    payload_text: str,
    model_cls: type[T],
    *,
    previous_interaction_id: str | None = None,
    environment: str | dict[str, Any] = "remote",
    cache_path: str | Path | None = None,
    use_cache: bool = False,
    toolset: AgentToolset | None = None,
) -> T:
    cache_file = Path(cache_path) if cache_path else None
    if use_cache and cache_file and cache_file.is_file():
        cached_data = json.loads(cache_file.read_text())
        cached_data["source"] = "cached"
        return model_cls.model_validate(cached_data)

    prompt = f"{instructions}\n\n## Input data\n\n{payload_text}"
    interaction = ask(
        client,
        prompt,
        previous_interaction_id=previous_interaction_id,
        environment=environment,
        toolset=toolset,
    )
    raw_json = extract_json(interaction.output_text)
    raw_json["source"] = "live"
    raw_json["interaction_id"] = getattr(interaction, "id", None)
    raw_json["environment_id"] = getattr(interaction, "environment_id", None)
    result = model_cls.model_validate(raw_json)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cached_copy = result.model_copy(update={"source": "cached"})
        cache_file.write_text(cached_copy.model_dump_json(indent=2))

    return result


def should_rewrite_reward(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate training curves against the 5 trigger conditions in docs/interfaces.md §3.3."""
    reasons: list[str] = []
    curves = summary.get("curves", []) if isinstance(summary, dict) else []
    if len(curves) >= 4:
        q = max(1, len(curves) // 4)
        last_q = [float(c.get("reward", 0.0)) for c in curves[-q:]]
        prev_q = [float(c.get("reward", 0.0)) for c in curves[-2 * q : -q]]
        if prev_q and last_q:
            mean_last = sum(last_q) / len(last_q)
            mean_prev = sum(prev_q) / len(prev_q)
            rel_change = abs(mean_last - mean_prev) / max(abs(mean_prev), 1e-6)
            if rel_change < 0.05:
                reasons.append(f"Reward plateau: last quarter changed by {rel_change * 100:.1f}% (< 5%)")

    latest = summary.get("latest", summary) if isinstance(summary, dict) else {}
    crash_rate = float(
        summary.get("crash_rate", latest.get("crash_rate", latest.get("crashes_per_episode", 0.0)))
    )
    if crash_rate > 0.15:
        reasons.append(f"High crash rate: {crash_rate:.2f} (> 0.15)")

    overlap_rate = float(summary.get("crater_overlap_rate", latest.get("crater_overlap_rate", 0.0)))
    if overlap_rate > 0.25:
        reasons.append(f"Crater overlap rate: {overlap_rate:.2f} (> 0.25)")

    relay_conflict = float(summary.get("relay_conflict_rate", latest.get("relay_conflict_rate", 0.0)))
    if relay_conflict > 0.15:
        reasons.append(f"Relay conflict rate: {relay_conflict:.2f} (> 0.15)")

    craters_visited = summary.get("craters_visited_mean", latest.get("craters_visited_mean"))
    if craters_visited is not None and float(craters_visited) < 2.0:
        reasons.append(f"Under-exploration: craters_visited_mean={float(craters_visited):.2f} (< 2.0)")

    return (len(reasons) > 0, reasons)


def _validate_and_retry_reward(
    client,
    spec: RewardSpec,
    instructions: str,
    *,
    environment: str | dict[str, Any] = "remote",
    cache_path: str | Path | None = None,
    use_cache: bool = False,
    toolset: AgentToolset,
    max_retries: int = 2,
    hot_swap: bool = False,
) -> RewardSpec:
    """Validate reward_code (and optionally hot-swap into selene_sim.reward) with up to max_retries self-correction turns."""
    if use_cache or client is None:
        return spec

    attempt = 0
    while attempt <= max_retries:
        errors: list[str] = []
        test_res = toolset.test_reward_code(spec.reward_code)
        if not test_res.get("valid", True):
            errors.append(str(test_res.get("error") or test_res.get("batch_errors") or test_res.get("errors")))
        if hot_swap:
            try:
                from selene_sim.reward import load_reward_code

                load_reward_code(spec.reward_code)
            except ImportError:
                pass
            except Exception as exc:
                errors.append(f"selene_sim.reward.load_reward_code failed: {exc}")

        if not errors or attempt >= max_retries:
            break

        attempt += 1
        err_msg = "\n".join(errors)
        fix_payload = (
            f"Your previous `reward_code` failed validation with:\n```\n{err_msg}\n```\n"
            f"Fix `compute_reward` so it passes both `RewardState` batch validation and step-dict evaluation, "
            f"and return the corrected JSON."
        )
        spec = _run_structured_job(
            client,
            instructions,
            fix_payload,
            RewardSpec,
            previous_interaction_id=spec.interaction_id,
            environment=spec.environment_id or environment,
            cache_path=cache_path,
            use_cache=False,
            toolset=toolset,
        )
    return spec


def write_safety_spec(
    client,
    documents: Any = None,
    *,
    previous_interaction_id: str | None = None,
    environment: str | dict[str, Any] = "remote",
    cache_path: str | Path | None = None,
    use_cache: bool = False,
    toolset: AgentToolset | None = None,
) -> SafetySpec:
    active_toolset = toolset if toolset is not None else AgentToolset()
    return _run_structured_job(
        client,
        SAFETY_SPEC_INSTRUCTIONS,
        format_input(documents),
        SafetySpec,
        previous_interaction_id=previous_interaction_id,
        environment=environment,
        cache_path=cache_path,
        use_cache=use_cache,
        toolset=active_toolset,
    )


def write_reward(
    client,
    context: Any = None,
    *,
    episode: int = 1,
    previous_interaction_id: str | None = None,
    environment: str | dict[str, Any] = "remote",
    cache_path: str | Path | None = None,
    use_cache: bool = False,
    toolset: AgentToolset | None = None,
    max_retries: int = 2,
    hot_swap: bool = False,
) -> RewardSpec:
    active_toolset = toolset if toolset is not None else AgentToolset()
    payload = f"Target episode index: {episode}\n\n{format_input(context)}"
    spec = _run_structured_job(
        client,
        WRITE_REWARD_INSTRUCTIONS,
        payload,
        RewardSpec,
        previous_interaction_id=previous_interaction_id,
        environment=environment,
        cache_path=cache_path,
        use_cache=use_cache,
        toolset=active_toolset,
    )
    spec = _validate_and_retry_reward(
        client,
        spec,
        WRITE_REWARD_INSTRUCTIONS,
        environment=environment,
        cache_path=cache_path,
        use_cache=use_cache,
        toolset=active_toolset,
        max_retries=max_retries,
        hot_swap=hot_swap,
    )
    spec.episode = episode
    return spec


def rewrite_reward(
    client,
    current_reward: Any = None,
    training_data: Any = None,
    *,
    episode: int | None = None,
    previous_interaction_id: str | None = None,
    environment: str | dict[str, Any] = "remote",
    cache_path: str | Path | None = None,
    use_cache: bool = False,
    toolset: AgentToolset | None = None,
    max_retries: int = 2,
    hot_swap: bool = False,
) -> RewardSpec:
    active_toolset = toolset if toolset is not None else AgentToolset()
    parts = [
        f"## Current reward\n\n{format_input(current_reward)}",
        f"## Training curves and episode logs\n\n{format_input(training_data)}",
    ]
    if episode is not None:
        parts.insert(0, f"Target episode index: {episode}")
    spec = _run_structured_job(
        client,
        REWRITE_REWARD_INSTRUCTIONS,
        "\n\n".join(parts),
        RewardSpec,
        previous_interaction_id=previous_interaction_id,
        environment=environment,
        cache_path=cache_path,
        use_cache=use_cache,
        toolset=active_toolset,
    )
    spec = _validate_and_retry_reward(
        client,
        spec,
        REWRITE_REWARD_INSTRUCTIONS,
        environment=environment,
        cache_path=cache_path,
        use_cache=use_cache,
        toolset=active_toolset,
        max_retries=max_retries,
        hot_swap=hot_swap,
    )
    if episode is not None:
        spec.episode = episode
    return spec


def plan_fleet(
    client,
    mission_context: Any = None,
    *,
    previous_interaction_id: str | None = None,
    environment: str | dict[str, Any] = "remote",
    cache_path: str | Path | None = None,
    use_cache: bool = False,
    toolset: AgentToolset | None = None,
) -> FleetPlan:
    active_toolset = toolset if toolset is not None else AgentToolset()
    return _run_structured_job(
        client,
        PLAN_FLEET_INSTRUCTIONS,
        format_input(mission_context),
        FleetPlan,
        previous_interaction_id=previous_interaction_id,
        environment=environment,
        cache_path=cache_path,
        use_cache=use_cache,
        toolset=active_toolset,
    )


def revise_fleet_plan(
    client,
    fleet_plan: Any,
    verdict: Any = None,
    *,
    previous_interaction_id: str | None = None,
    environment: str | dict[str, Any] = "remote",
    cache_path: str | Path | None = None,
    use_cache: bool = False,
    toolset: AgentToolset | None = None,
) -> FleetPlan:
    active_toolset = toolset if toolset is not None else AgentToolset()
    prev_id = previous_interaction_id
    env = environment
    prev_rev = 0
    if isinstance(fleet_plan, FleetPlan):
        prev_rev = fleet_plan.revision
        if prev_id is None and fleet_plan.interaction_id:
            prev_id = fleet_plan.interaction_id
        if env == "remote" and fleet_plan.environment_id:
            env = fleet_plan.environment_id

    payload = (
        f"## Current fleet plan\n\n{format_input(fleet_plan)}\n\n"
        f"## Verifier verdict and failure statistics\n\n{format_input(verdict)}"
    )
    revised = _run_structured_job(
        client,
        REVISE_FLEET_INSTRUCTIONS,
        payload,
        FleetPlan,
        previous_interaction_id=prev_id,
        environment=env,
        cache_path=cache_path,
        use_cache=use_cache,
        toolset=active_toolset,
    )
    if revised.revision <= prev_rev:
        revised.revision = prev_rev + 1
    return revised


def verify_and_revise(
    client,
    initial_plan: FleetPlan,
    verifier_fn: Callable[[FleetPlan], dict[str, Any]] | None = None,
    *,
    max_revisions: int = 3,
    environment: str | dict[str, Any] = "remote",
    toolset: AgentToolset | None = None,
) -> tuple[FleetPlan, dict[str, Any]]:
    """Run the verifier on a fleet plan and ask Argon to revise it until it passes or max_revisions is reached."""
    active_toolset = toolset if toolset is not None else AgentToolset()
    check_fn = verifier_fn or (
        lambda p: active_toolset.verify_fleet_plan(fleet_plan=p.model_dump())
    )
    plan = initial_plan
    verdict = check_fn(plan)
    attempts = 0
    while not verdict.get("passed", False) and attempts < max_revisions:
        plan = revise_fleet_plan(
            client,
            plan,
            verdict,
            environment=plan.environment_id or environment,
            toolset=active_toolset,
        )
        verdict = check_fn(plan)
        attempts += 1
    return plan, verdict


def build_verification_block(plan: FleetPlan, verdict: dict[str, Any]) -> dict[str, Any]:
    """Assemble the episodes.json verification section (docs/interfaces.md §4.2, approved_by_human=False)."""
    return {
        "source": plan.source,
        "plan_id": plan.plan_id,
        "revision": plan.revision,
        "passed": bool(verdict.get("passed", False)),
        "pass_rate": float(verdict.get("pass_rate", 0.0)),
        "rollout_count": int(verdict.get("rollout_count", verdict.get("n_rollouts", 0))),
        "approved_by_human": False,
        "summary": verdict.get("summary", plan.summary),
        "rule_results": verdict.get("rule_results", []),
        "rule_failures": verdict.get("rule_failures", {}),
        "failed_rollout_indices": verdict.get("failed_rollout_indices", []),
        "trajectory_cloud": verdict.get("trajectory_cloud", []),
        "arcs": verdict.get("arcs", verdict.get("trajectory_cloud", [])),
        "fleet_plan": plan.to_interfaces_dict(),
        "sim_plan": plan.to_sim_plan(),
    }


def stub_argon_plan_and_spec(seed: int = 42) -> tuple[SafetySpec, FleetPlan]:
    """Offline fallback returning a scripted SafetySpec and FleetPlan matching SimConfig (docs/interfaces.md §5)."""
    _ = seed
    rules = [
        SafetyRule(
            id="R1_fuel_reserve",
            name="Fuel reserve >= 10% (1.5 kg)",
            scope="step",
            statement="Every hopper must retain at least 10% (1.5 kg) of initial fuel at all times.",
            source_excerpt="Every hopper must retain at least 10% (0.10, or 1.5 kg) of initial fuel.",
            parameters={"min_fuel": 0.10, "min_fuel_kg": 1.5},
            check_code=(
                "import numpy as np\n"
                "def check(trace: dict):\n"
                "    if 'steps' in trace:\n"
                "        return all(s.get('fuel', 1.0) >= 0.10 for s in trace.get('steps', []))\n"
                "    fuel = np.asarray(trace.get('fuel', 1.0))\n"
                "    ok = (fuel.min(axis=(1, 2)) >= 0.10) if fuel.ndim == 3 else np.array([bool(np.all(fuel >= 0.10))])\n"
                "    pr = float(ok.mean())\n"
                "    return (pr >= 0.99, f'min fuel {float(fuel.min()):.3f}', {'pass_rate': pr, 'per_rollout_ok': ok.tolist()})\n"
            ),
        ),
        SafetyRule(
            id="R2_battery_reserve",
            name="Battery reserve >= 20%",
            scope="step",
            statement="Battery charge must remain at or above 20% at every step.",
            source_excerpt="Battery charge must remain at or above 20% (0.20) at every step.",
            parameters={"min_battery": 0.20, "min_battery_pct": 20.0},
            check_code=(
                "import numpy as np\n"
                "def check(trace: dict):\n"
                "    if 'steps' in trace:\n"
                "        return all(s.get('battery', 1.0) >= 0.20 for s in trace.get('steps', []))\n"
                "    bat = np.asarray(trace.get('battery', 1.0))\n"
                "    ok = (bat.min(axis=(1, 2)) >= 0.20) if bat.ndim == 3 else np.array([bool(np.all(bat >= 0.20))])\n"
                "    pr = float(ok.mean())\n"
                "    return (pr >= 0.99, f'min battery {float(bat.min()):.3f}', {'pass_rate': pr, 'per_rollout_ok': ok.tolist()})\n"
            ),
        ),
        SafetyRule(
            id="R3_landing_slope",
            name="Landing slope <= 15 deg",
            scope="hop",
            statement="Touchdown terrain slope must not exceed 15.0 degrees.",
            source_excerpt="Touchdown terrain slope must not exceed 15.0 degrees.",
            parameters={"max_slope_deg": 15.0},
            check_code=(
                "import numpy as np\n"
                "def check(trace: dict):\n"
                "    if 'steps' in trace:\n"
                "        return all(s.get('slope_deg', 0.0) <= 15.0 for s in trace.get('steps', []))\n"
                "    slope = np.asarray(trace.get('landing_slope_deg', 0.0))\n"
                "    ok = (slope.max(axis=(1, 2)) <= 15.0) if slope.ndim == 3 else np.array([bool(np.all(slope <= 15.0))])\n"
                "    pr = float(ok.mean())\n"
                "    return (pr >= 0.99, f'max slope {float(slope.max()):.2f} deg', {'pass_rate': pr, 'per_rollout_ok': ok.tolist()})\n"
            ),
        ),
        SafetyRule(
            id="R4_fleet_separation",
            name="Fleet separation >= 100 m",
            scope="fleet",
            statement="Minimum horizontal separation between any two hoppers must be at least 100.0 m.",
            source_excerpt="Minimum horizontal separation between any two hoppers at the same timestep is 100.0 m.",
            parameters={"min_separation_m": 100.0},
            check_code=(
                "import numpy as np\n"
                "def check(trace: dict):\n"
                "    if 'steps' in trace and 'min_separation_m' not in trace:\n"
                "        return True\n"
                "    sep = np.asarray(trace.get('min_separation_m', 9999.0))\n"
                "    ok = (sep.min(axis=1) >= 100.0) if sep.ndim == 2 else np.array([bool(np.all(sep >= 100.0))])\n"
                "    pr = float(ok.mean())\n"
                "    return (pr >= 0.99, f'min sep {float(sep.min()):.1f} m', {'pass_rate': pr, 'per_rollout_ok': ok.tolist()})\n"
            ),
        ),
        SafetyRule(
            id="R5_relay_uplink",
            name="Single-hopper relay uplink",
            scope="fleet",
            statement="At most one hopper may transmit during an active relay window.",
            source_excerpt="At most one hopper may transmit during a relay window.",
            parameters={"max_transmitters": 1},
            check_code=(
                "import numpy as np\n"
                "def check(trace: dict):\n"
                "    if 'steps' in trace and 'relay_conflict' not in trace:\n"
                "        return True\n"
                "    conflict = np.asarray(trace.get('relay_conflict', False), dtype=bool)\n"
                "    ok = (~conflict.any(axis=1)) if conflict.ndim == 2 else np.array([not bool(conflict.any())])\n"
                "    pr = float(ok.mean())\n"
                "    return (pr >= 0.99, 'relay conflict check', {'pass_rate': pr, 'per_rollout_ok': ok.tolist()})\n"
            ),
        ),
        SafetyRule(
            id="R6_max_hop_range",
            name="Single-hop range <= 3000 m",
            scope="hop",
            statement="No single ballistic hop may exceed 3000.0 m horizontal distance.",
            source_excerpt="No single ballistic hop may exceed 3000.0 m horizontal distance.",
            parameters={"max_hop_m": 3000.0},
            check_code=(
                "import numpy as np\n"
                "def check(trace: dict):\n"
                "    if 'steps' in trace and 'hop_distance_m' not in trace:\n"
                "        return True\n"
                "    dist = np.asarray(trace.get('hop_distance_m', 0.0))\n"
                "    ok = (dist.max(axis=(1, 2)) <= 3000.0) if dist.ndim == 3 else np.array([bool(np.all(dist <= 3000.0))])\n"
                "    pr = float(ok.mean())\n"
                "    return (pr >= 0.99, f'max hop {float(dist.max()):.1f} m', {'pass_rate': pr, 'per_rollout_ok': ok.tolist()})\n"
            ),
        ),
    ]
    spec = SafetySpec(
        source="scripted",
        summary="Scripted baseline safety spec matching SimConfig (6 rules).",
        rules=rules,
    )
    hoppers = [
        HopperPlanEntry(
            id="h0",
            start_pos_xy=[-300.0, -300.0],
            craters_assigned=["c0"],
            steps=[HopStep(stage=0, action="hop", from_xy=[-300.0, -300.0], to_xy=[-1400.0, -900.0], target_crater="c0")],
        ),
        HopperPlanEntry(
            id="h1",
            start_pos_xy=[300.0, -300.0],
            craters_assigned=["c1"],
            steps=[HopStep(stage=0, action="hop", from_xy=[300.0, -300.0], to_xy=[1800.0, -700.0], target_crater="c1")],
        ),
        HopperPlanEntry(
            id="h2",
            start_pos_xy=[-300.0, 300.0],
            craters_assigned=["c3"],
            steps=[HopStep(stage=0, action="hop", from_xy=[-300.0, 300.0], to_xy=[-1600.0, 1400.0], target_crater="c3")],
        ),
        HopperPlanEntry(
            id="h3",
            start_pos_xy=[300.0, 300.0],
            craters_assigned=["c4"],
            steps=[HopStep(stage=0, action="hop", from_xy=[300.0, 300.0], to_xy=[1900.0, 1300.0], target_crater="c4")],
        ),
    ]
    plan = FleetPlan(
        source="scripted",
        plan_id="plan-scripted-v1",
        revision=0,
        summary="Scripted 4-hopper fleet plan assigning h0..h3 to distinct craters with >= 100 m separation.",
        hoppers=hoppers,
    )
    return spec, plan


def review_logs(
    client,
    logs: Any = None,
    *,
    previous_interaction_id: str | None = None,
    environment: str | dict[str, Any] = "remote",
    cache_path: str | Path | None = None,
    use_cache: bool = False,
    toolset: AgentToolset | None = None,
) -> LogReview:
    active_toolset = toolset if toolset is not None else AgentToolset()
    return _run_structured_job(
        client,
        REVIEW_LOGS_INSTRUCTIONS,
        format_input(logs),
        LogReview,
        previous_interaction_id=previous_interaction_id,
        environment=environment,
        cache_path=cache_path,
        use_cache=use_cache,
        toolset=active_toolset,
    )


class Orchestrator:
    """Stateful helper that preserves conversation, sandbox context and shared tools across calls."""

    def __init__(self, client=None, toolset: AgentToolset | None = None):
        self.client = client or genai.Client()
        self.toolset = toolset or AgentToolset()
        self.last_interaction_id: str | None = None
        self.environment_id: str | None = None

    def _env(self) -> str:
        return self.environment_id or "remote"

    def _record(self, result: BaseModel) -> None:
        interaction_id = getattr(result, "interaction_id", None)
        environment_id = getattr(result, "environment_id", None)
        if interaction_id:
            self.last_interaction_id = interaction_id
        if environment_id:
            self.environment_id = environment_id

    def write_safety_spec(self, documents: Any, **kwargs) -> SafetySpec:
        res = write_safety_spec(
            self.client,
            documents,
            previous_interaction_id=kwargs.pop("previous_interaction_id", self.last_interaction_id),
            environment=kwargs.pop("environment", self._env()),
            toolset=kwargs.pop("toolset", self.toolset),
            **kwargs,
        )
        self.toolset._safety_spec_source = res
        self._record(res)
        return res

    def write_reward(self, context: Any = None, **kwargs) -> RewardSpec:
        res = write_reward(
            self.client,
            context,
            previous_interaction_id=kwargs.pop("previous_interaction_id", self.last_interaction_id),
            environment=kwargs.pop("environment", self._env()),
            toolset=kwargs.pop("toolset", self.toolset),
            **kwargs,
        )
        self._record(res)
        return res

    def rewrite_reward(self, current_reward: Any = None, training_data: Any = None, **kwargs) -> RewardSpec:
        res = rewrite_reward(
            self.client,
            current_reward,
            training_data,
            previous_interaction_id=kwargs.pop("previous_interaction_id", self.last_interaction_id),
            environment=kwargs.pop("environment", self._env()),
            toolset=kwargs.pop("toolset", self.toolset),
            **kwargs,
        )
        self._record(res)
        return res

    def plan_fleet(self, mission_context: Any = None, **kwargs) -> FleetPlan:
        res = plan_fleet(
            self.client,
            mission_context,
            previous_interaction_id=kwargs.pop("previous_interaction_id", self.last_interaction_id),
            environment=kwargs.pop("environment", self._env()),
            toolset=kwargs.pop("toolset", self.toolset),
            **kwargs,
        )
        self._record(res)
        return res

    def revise_fleet_plan(self, fleet_plan: Any, verdict: Any = None, **kwargs) -> FleetPlan:
        res = revise_fleet_plan(
            self.client,
            fleet_plan,
            verdict,
            previous_interaction_id=kwargs.pop("previous_interaction_id", self.last_interaction_id),
            environment=kwargs.pop("environment", self._env()),
            toolset=kwargs.pop("toolset", self.toolset),
            **kwargs,
        )
        self._record(res)
        return res

    def review_logs(self, logs: Any = None, **kwargs) -> LogReview:
        res = review_logs(
            self.client,
            logs,
            previous_interaction_id=kwargs.pop("previous_interaction_id", self.last_interaction_id),
            environment=kwargs.pop("environment", self._env()),
            toolset=kwargs.pop("toolset", self.toolset),
            **kwargs,
        )
        self._record(res)
        return res


def _add_io_flags(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("-o", "--output", help="write structured JSON output to this file")
    subparser.add_argument("--cache", help="path to read/write cached JSON result")
    subparser.add_argument(
        "--use-cache",
        action="store_true",
        help="load result from --cache if present without calling the API",
    )


def _emit(result: BaseModel, output_path: str | None) -> None:
    payload = result.model_dump_json(indent=2)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload)
    print(payload)


def main():
    parser = argparse.ArgumentParser(description="Selene's Argon orchestrator")
    commands = parser.add_subparsers(dest="command", required=True)
    create_cmd = commands.add_parser("create", help="create or replace the agent on Google Cloud")
    create_cmd.add_argument(
        "--mission-doc",
        default=None,
        help="optional mission document path to mount at /workspace/mission_constraints.md",
    )
    create_cmd.add_argument(
        "--safety-spec",
        default=None,
        help="optional safety_spec.json path to mount at /workspace/safety_spec.json",
    )
    create_cmd.add_argument(
        "--episodes",
        default=None,
        help="optional episodes.json path to mount at /workspace/episodes.json",
    )
    create_cmd.add_argument(
        "--no-bundle",
        action="store_true",
        help="deploy without mounting selene_tools.py and data files into /workspace",
    )
    commands.add_parser("ask", help="send the agent a message").add_argument("text")

    spec_cmd = commands.add_parser("safety-spec", help="extract safety spec from any markdown or file")
    spec_cmd.add_argument("inputs", nargs="*", help="optional markdown/text/json file paths or raw strings")
    _add_io_flags(spec_cmd)

    reward_cmd = commands.add_parser("reward", help="write initial reward using mission context and tools")
    reward_cmd.add_argument("inputs", nargs="*", help="optional markdown/text/json file paths or strings")
    reward_cmd.add_argument("--episode", type=int, default=1, help="episode index")
    _add_io_flags(reward_cmd)

    rewrite_cmd = commands.add_parser("rewrite-reward", help="rewrite reward using training tools and logs")
    rewrite_cmd.add_argument("current_reward", nargs="?", default=None, help="current reward file or string")
    rewrite_cmd.add_argument("training_data", nargs="*", help="optional training curves or episodes.json paths")
    rewrite_cmd.add_argument("--episode", type=int, default=None, help="episode index")
    _add_io_flags(rewrite_cmd)

    plan_cmd = commands.add_parser("plan", help="plan craters and hops for the fleet using tools")
    plan_cmd.add_argument("inputs", nargs="*", help="optional mission state files or strings")
    _add_io_flags(plan_cmd)

    revise_cmd = commands.add_parser("revise", help="verify and revise fleet plan using verifier tool")
    revise_cmd.add_argument("fleet_plan", help="previous fleet plan JSON file or string")
    revise_cmd.add_argument("verdict", nargs="?", default=None, help="optional verifier verdict file or string")
    _add_io_flags(revise_cmd)

    logs_cmd = commands.add_parser("review-logs", help="review hopper logs and world-model errors using tools")
    logs_cmd.add_argument("inputs", nargs="*", help="optional episodes.json or log files/strings")
    _add_io_flags(logs_cmd)

    args = parser.parse_args()

    use_cache = getattr(args, "use_cache", False)
    cache_path = getattr(args, "cache", None)
    client = None if (use_cache and cache_path and Path(cache_path).is_file()) else genai.Client()

    if args.command == "create":
        agent = create_agent(
            client,
            mission_doc=args.mission_doc,
            safety_spec=args.safety_spec,
            episodes_path=args.episodes,
            bundle_workspace=not args.no_bundle,
        )
        print(f"{agent.id} on {agent.agent_config.model}")
    elif args.command == "ask":
        interaction = ask(client, args.text)
        print(interaction.output_text)
        print(f"tokens: {interaction.usage.total_tokens}", file=sys.stderr)
    elif args.command == "safety-spec":
        res = write_safety_spec(client, args.inputs or None, cache_path=cache_path, use_cache=use_cache)
        _emit(res, args.output)
    elif args.command == "reward":
        res = write_reward(
            client, args.inputs or None, episode=args.episode, cache_path=cache_path, use_cache=use_cache
        )
        _emit(res, args.output)
    elif args.command == "rewrite-reward":
        res = rewrite_reward(
            client,
            args.current_reward,
            args.training_data or None,
            episode=args.episode,
            cache_path=cache_path,
            use_cache=use_cache,
        )
        _emit(res, args.output)
    elif args.command == "plan":
        res = plan_fleet(client, args.inputs or None, cache_path=cache_path, use_cache=use_cache)
        _emit(res, args.output)
    elif args.command == "revise":
        res = revise_fleet_plan(
            client, args.fleet_plan, args.verdict, cache_path=cache_path, use_cache=use_cache
        )
        _emit(res, args.output)
    elif args.command == "review-logs":
        res = review_logs(client, args.inputs or None, cache_path=cache_path, use_cache=use_cache)
        _emit(res, args.output)


if __name__ == "__main__":
    main()
