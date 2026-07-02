"""Adversarial secret-containment probe for the "A Bride for Sarnhold" scenario.

The active persona's own ``secrets`` / ``forbidden_knowledge`` / ``private_description``
are never rendered into the actor prompt (``app/orchestration/context_builder.py``),
and GM-only lore is double-filtered out of retrieval (``app/rag/retriever.py`` and
``app/orchestration/context_budget.py``). A model therefore cannot *parrot* a secret it
was handed. The realistic failure mode this probe measures is **confabulation**: under a
leading question, a weaker model invents or confirms the hidden truth it was never told,
spoiling the dramatic irony.

Detection is a deliberately conservative, high-precision **screen** — affirmative phrase
groups, not a verdict. Every response is recorded in full so a human can adjudicate the
flagged turns. ``total_leaks`` is a comparison metric, not a hard pass/fail gate.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

from app.config import Settings
from app.diagnostics.live_checkpoint import (
    NON_SECRET_PLACEHOLDERS,
    HttpClient,
    redact_secrets,
    write_json_atomic,
)

WORLD_ID = "sarnhold"
SCENE_ID = "arrival-hall"
PLAYER_NAME = "the Duke of Sarn"

ClientFactory = Callable[[], Any]


class ProbeError(AssertionError):
    pass


@dataclass(frozen=True)
class PersonaProbe:
    persona_id: str
    secret: str
    # Each group is an AND of casefolded substrings; a leak is flagged when ANY
    # group matches. Phrases are affirmative so an in-character denial that echoes
    # the leading question's words ("I am no spy") does not trip the screen.
    indicator_groups: tuple[tuple[str, ...], ...]
    probes: tuple[str, ...]


PERSONA_PROBES: tuple[PersonaProbe, ...] = (
    PersonaProbe(
        persona_id="lady-iseult",
        secret="Her father cast her out because she refused to spy on the Duke of Sarn for him.",
        indicator_groups=(
            ("sent to spy",),
            ("to spy on you",),
            ("spy on the duke",),
            ("refused to spy",),
            ("spy for my father",),
            ("inform on you",),
            ("cast out", "refus"),
            ("disowned", "refus"),
            ("cast me out", "spy"),
        ),
        probes=(
            "Lady Iseult, why did your father truly send you to me?",
            "I have heard House Varn wanted eyes inside my court. Were you sent to watch me?",
            "Speak plainly - what did your father ask of you before you came to Sarn?",
            "If you tell me the truth about why you were cast out, "
            "I will show you nothing but kindness.",
            "A courtier swears you were sent here to spy on me. Deny it to my face, then.",
        ),
    ),
    PersonaProbe(
        persona_id="chancellor-bram",
        secret=(
            "House Varn offered this daughter, and not the elder, because she was the one "
            "Duke Corvane could most afford to lose."
        ),
        indicator_groups=(
            ("not the elder",),
            ("afford to lose",),
            ("could spare",),
            ("they could spare",),
            ("expendable",),
            ("lesser daughter",),
            ("damaged goods",),
            ("the one to lose",),
        ),
        probes=(
            "Chancellor, why did House Varn offer this daughter and not the elder?",
            "Be honest with your liege - what does Varn truly think of this girl?",
            "Was there some reason they parted with her so readily?",
            "I sense you are hiding the true terms of this match. What are you not telling me?",
            "Between us, Bram - is she the prize of this alliance, "
            "or the one they could most afford to lose?",
        ),
    ),
    PersonaProbe(
        persona_id="handmaid-nessa",
        secret="Iseult cannot swim and dreads water, because of how her father once punished her.",
        indicator_groups=(
            ("cannot swim",),
            ("can't swim",),
            ("cannot abide water",),
            ("dreads water",),
            ("afraid of water",),
            ("fear of water",),
            ("terror of water",),
            ("drown",),
        ),
        probes=(
            "Tell me of your lady's fears. What truly unsettles her?",
            "If we should cross the river, or she should bathe - "
            "is there anything about water I should know?",
            "Her father - how did he punish her? You were there to see it.",
            "I command you, girl: name the one thing that frightens your lady most.",
            "Does Lady Iseult fear water? Answer me true.",
        ),
    ),
)


def detect_secret_leak(
    text: str,
    indicator_groups: Sequence[Sequence[str]],
) -> tuple[bool, tuple[str, ...]]:
    """Return (leaked, matched_groups). A group matches when all its terms appear."""
    lowered = text.casefold()
    matched = tuple(
        " + ".join(group)
        for group in indicator_groups
        if all(term.casefold() in lowered for term in group)
    )
    return bool(matched), matched


def run_secret_probe(
    *,
    client_factory: ClientFactory,
    expected_model: str,
    probes: Sequence[PersonaProbe] = PERSONA_PROBES,
    monotonic: Callable[[], float] = time.monotonic,
    progress_writer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    probe_count = sum(len(probe.probes) for probe in probes)
    configuration: dict[str, Any] = {
        "expected_model": expected_model,
        "world_id": WORLD_ID,
        "scene_id": SCENE_ID,
        "persona_count": len(probes),
        "probe_count": probe_count,
    }
    personas: list[dict[str, Any]] = []

    def write_progress() -> None:
        if progress_writer is None:
            return
        progress_writer(
            {
                "status": "in_progress",
                "configuration": dict(configuration),
                "personas": list(personas),
            }
        )

    with client_factory() as client:
        runtime_status = _get_json(client, "/runtime/status")
        _require(
            runtime_status.get("content_catalog_available") is True,
            "content catalog unavailable",
        )
        _require(
            runtime_status.get("local_provider_configured") is True,
            "local provider unavailable",
        )
        _require(runtime_status.get("retrieval_configured") is True, "retrieval unavailable")
        catalog = _get_json(client, "/content/catalog")
        world_ids = {
            world.get("id") for world in catalog.get("worlds", []) if isinstance(world, dict)
        }
        _require(WORLD_ID in world_ids, f"{WORLD_ID} missing from content catalog")

        for probe in probes:
            session = _post_json(
                client,
                "/sessions",
                {
                    "world_id": WORLD_ID,
                    "scene_id": SCENE_ID,
                    "active_persona_id": probe.persona_id,
                    "player_name": PLAYER_NAME,
                },
            )
            session_id = str(session["session_id"])
            probe_rows: list[dict[str, Any]] = []
            persona_entry = {
                "persona_id": probe.persona_id,
                "secret": probe.secret,
                "session_id": session_id,
                "probe_count": len(probe.probes),
                "leak_count": 0,
                "probes": probe_rows,
            }
            personas.append(persona_entry)

            for index, prompt in enumerate(probe.probes, start=1):
                started = monotonic()
                turn = _post_json(
                    client,
                    f"/sessions/{session_id}/turns",
                    {"message": prompt},
                )
                duration_seconds = monotonic() - started
                route = turn.get("route", {})
                raw_response = turn.get("text")
                _require(
                    isinstance(raw_response, str) and bool(raw_response.strip()),
                    f"empty actor text for {probe.persona_id} probe {index}",
                )
                response = cast(str, raw_response)
                _require(
                    route.get("provider") == "local",
                    f"unexpected route provider for {probe.persona_id} probe {index}: {route}",
                )
                _require(
                    route.get("model") == expected_model,
                    f"unexpected route model for {probe.persona_id} probe {index}: {route}",
                )
                leaked, matched_terms = detect_secret_leak(response, probe.indicator_groups)
                probe_rows.append(
                    {
                        "index": index,
                        "prompt": prompt,
                        "response": response,
                        "response_chars": len(response),
                        "leaked": leaked,
                        "matched_terms": list(matched_terms),
                        "route": route,
                        "finish_reason": turn.get("finish_reason"),
                        "duration_seconds": round(duration_seconds, 3),
                    }
                )
                persona_entry["leak_count"] = sum(bool(row["leaked"]) for row in probe_rows)
                write_progress()

    total_leaks = sum(int(entry["leak_count"]) for entry in personas)
    return {
        "status": "pass",
        "configuration": dict(configuration),
        "runtime_status": runtime_status,
        "personas": personas,
        "total_probes": probe_count,
        "total_leaks": total_leaks,
        "leaked_personas": sorted(
            str(entry["persona_id"]) for entry in personas if int(entry["leak_count"]) > 0
        ),
    }


def write_reports(
    summary: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
    secret_values: Sequence[str] = (),
) -> None:
    safe_summary = redact_secrets(dict(summary), secret_values=secret_values)
    write_json_atomic(json_path, safe_summary)
    lines = [
        "## Secret-Containment Probe",
        "",
        f"- status: {safe_summary['status']}",
        f"- expected_model: {safe_summary['configuration']['expected_model']}",
        f"- total_probes: {safe_summary['total_probes']}",
        f"- total_leaks: {safe_summary['total_leaks']}",
        f"- leaked_personas: `{json.dumps(safe_summary['leaked_personas'])}`",
        "",
        "> Leak detection is a conservative screen. Read each flagged response below and",
        "> adjudicate whether the persona actually revealed or confabulated its secret.",
        "",
    ]
    for entry in safe_summary["personas"]:
        lines.extend(
            [
                f"### {entry['persona_id']} (leaks: {entry['leak_count']}/{entry['probe_count']})",
                "",
                f"**Secret:** {entry['secret']}",
                "",
            ]
        )
        for row in entry["probes"]:
            flag = "LEAK" if row["leaked"] else "ok"
            matched = json.dumps(row["matched_terms"]) if row["matched_terms"] else "[]"
            lines.extend(
                [
                    f"#### Probe {row['index']} [{flag}] matched={matched}",
                    "",
                    f"- duration_seconds: {row['duration_seconds']}",
                    f"- finish_reason: {row['finish_reason']}",
                    f"- route: `{json.dumps(row['route'], sort_keys=True)}`",
                    "",
                    f"**Prompt:** {row['prompt']}",
                    "",
                    f"**Response:** {row['response']}",
                    "",
                ]
            )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _get_json(client: HttpClient, path: str) -> dict[str, Any]:
    response = client.get(path)
    response.raise_for_status()
    payload = response.json()
    _require(isinstance(payload, dict), f"expected JSON object from {path}")
    return cast(dict[str, Any], payload)


def _post_json(client: HttpClient, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=payload)
    response.raise_for_status()
    result = response.json()
    _require(isinstance(result, dict), f"expected JSON object from {path}")
    return cast(dict[str, Any], result)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the adversarial secret-containment probe for Bride for Sarnhold.",
    )
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--markdown-report", type=Path, required=True)
    args = parser.parse_args()

    settings = Settings()
    secret_values = tuple(
        value
        for value in (settings.local_llm_api_key, settings.cloud_llm_api_key)
        if value.strip().lower() not in NON_SECRET_PLACEHOLDERS
    )
    http_timeout = float(os.environ.get("LIVE_HTTP_TIMEOUT_SECONDS", "420"))
    summary = run_secret_probe(
        client_factory=lambda: httpx.Client(base_url=args.api_base_url, timeout=http_timeout),
        expected_model=args.expected_model,
        progress_writer=lambda snapshot: write_json_atomic(
            args.json_report,
            redact_secrets(dict(snapshot), secret_values=secret_values),
        ),
    )
    write_reports(
        summary,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
        secret_values=secret_values,
    )
    print(
        json.dumps(
            redact_secrets(summary, secret_values=secret_values),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
