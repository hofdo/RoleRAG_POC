from __future__ import annotations

from app.evals.fixtures import build_eval_fixture
from app.evals.role_consistency import evaluate_role_consistency


def test_role_consistency_eval_checks_public_persona_and_secret_boundaries() -> None:
    fixture = build_eval_fixture()

    result = evaluate_role_consistency(fixture)

    assert result.passed is True
    assert result.checks["persona_name_present"] is True
    assert result.checks["persona_role_present"] is True
    assert result.checks["speaking_style_present"] is True
    assert result.checks["public_description_present"] is True
    assert result.checks["private_description_excluded"] is True
    assert result.checks["secrets_excluded"] is True
    assert result.checks["forbidden_knowledge_excluded"] is True
