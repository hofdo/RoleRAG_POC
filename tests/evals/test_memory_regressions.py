from __future__ import annotations

import pytest

from app.agents.memory_curator import MemoryCurator, MemoryCuratorOutputError
from app.evals.fixtures import build_eval_fixture


@pytest.mark.asyncio
async def test_memory_curator_accepts_important_turn_and_returns_memory() -> None:
    fixture = build_eval_fixture()
    provider = fixture.build_memory_provider(kind="important")

    result = await MemoryCurator().curate(
        provider=provider,
        route=fixture.memory_route,
        session=fixture.session,
        scene=fixture.scene,
        persona=fixture.primary_persona,
        user_message=fixture.important_turn_user_message,
        assistant_message=fixture.important_turn_assistant_message,
    )

    assert result.write_memory is True
    assert len(result.memories) == 1
    assert result.memories[0].visibility.value == "player"


@pytest.mark.asyncio
async def test_memory_curator_rejects_trivial_turn_by_returning_no_memory() -> None:
    fixture = build_eval_fixture()
    provider = fixture.build_memory_provider(kind="trivial")

    result = await MemoryCurator().curate(
        provider=provider,
        route=fixture.memory_route,
        session=fixture.session,
        scene=fixture.scene,
        persona=fixture.primary_persona,
        user_message="Good evening.",
        assistant_message="Good evening.",
    )

    assert result.write_memory is False
    assert result.memories == []


@pytest.mark.asyncio
async def test_memory_curator_rejects_invalid_memory_visibility() -> None:
    fixture = build_eval_fixture()
    provider = fixture.build_memory_provider(kind="invalid_visibility")

    with pytest.raises(MemoryCuratorOutputError):
        await MemoryCurator().curate(
            provider=provider,
            route=fixture.memory_route,
            session=fixture.session,
            scene=fixture.scene,
            persona=fixture.primary_persona,
            user_message=fixture.important_turn_user_message,
            assistant_message=fixture.important_turn_assistant_message,
        )


@pytest.mark.asyncio
async def test_memory_curator_rejects_write_without_candidates() -> None:
    fixture = build_eval_fixture()
    provider = fixture.build_memory_provider(kind="write_without_candidates")

    with pytest.raises(MemoryCuratorOutputError):
        await MemoryCurator().curate(
            provider=provider,
            route=fixture.memory_route,
            session=fixture.session,
            scene=fixture.scene,
            persona=fixture.primary_persona,
            user_message=fixture.important_turn_user_message,
            assistant_message=fixture.important_turn_assistant_message,
        )
