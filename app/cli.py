from __future__ import annotations

import json
from typing import Annotated

import typer

from app.config import Settings, get_settings
from app.llm.router import ModelTask, choose_route

app = typer.Typer(help="RoleRAG Phase 0/1 CLI")


def _redact_settings(settings: Settings) -> dict[str, object]:
    values = settings.model_dump()
    values["local_llm_api_key"] = "***"
    values["cloud_llm_api_key"] = "***"
    return values


@app.command()
def config() -> None:
    settings = get_settings()
    typer.echo(json.dumps(_redact_settings(settings), indent=2, sort_keys=True))


@app.command()
def route(
    task: Annotated[ModelTask, typer.Option(help="Model task to route")],
    failed_local_attempts: Annotated[int, typer.Option(min=0)] = 0,
    scene_complexity: Annotated[int, typer.Option(min=1)] = 1,
    retrieval_confidence: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
) -> None:
    settings = get_settings()
    chosen_route = choose_route(
        task=task,
        cloud_mode=settings.cloud_mode,
        local_model=settings.local_llm_model,
        cloud_model=settings.cloud_llm_model,
        local_max_tokens=settings.local_llm_max_tokens,
        cloud_max_tokens=settings.cloud_llm_max_tokens,
        local_temperature=settings.local_llm_temperature,
        cloud_temperature=settings.cloud_llm_temperature,
        failed_local_attempts=failed_local_attempts,
        retrieval_confidence=retrieval_confidence,
        scene_complexity=scene_complexity,
    )
    typer.echo(json.dumps(chosen_route.model_dump(), indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
