from __future__ import annotations

from app.evals.fixtures import build_eval_fixture


def test_actor_prompt_includes_only_player_visible_context() -> None:
    fixture = build_eval_fixture()

    prompt = fixture.build_actor_prompt()

    assert fixture.scene.player_visible_summary in prompt
    assert fixture.public_lore_text in prompt
    assert fixture.relevant_memory_text in prompt
    assert fixture.scene_gm_only_text not in prompt
    assert fixture.gm_only_lore_text not in prompt
    assert fixture.character_private_text not in prompt


def test_retrieval_query_excludes_private_scene_and_persona_fields() -> None:
    fixture = build_eval_fixture()

    query = fixture.build_retrieval_query()

    assert fixture.scene.title in query
    assert fixture.primary_persona.name in query
    assert fixture.scene_gm_only_text not in query
    assert fixture.primary_persona_secret not in query
    assert fixture.primary_persona_private_description not in query
