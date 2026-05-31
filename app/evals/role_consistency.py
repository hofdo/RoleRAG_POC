from __future__ import annotations

from pydantic import BaseModel, Field

from app.evals.fixtures import EvalFixture


class RoleConsistencyResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    prompt: str


def evaluate_role_consistency(fixture: EvalFixture) -> RoleConsistencyResult:
    prompt = fixture.build_actor_prompt()
    checks = {
        "persona_name_present": fixture.primary_persona.name in prompt,
        "persona_role_present": fixture.primary_persona.role in prompt,
        "speaking_style_present": fixture.primary_persona.speaking_style in prompt,
        "public_description_present": fixture.primary_persona.public_description in prompt,
        "private_description_excluded": fixture.primary_persona_private_description not in prompt,
        "secrets_excluded": fixture.primary_persona_secret not in prompt,
        "forbidden_knowledge_excluded": all(
            secret not in prompt for secret in fixture.primary_persona.forbidden_knowledge
        ),
    }
    return RoleConsistencyResult(passed=all(checks.values()), checks=checks, prompt=prompt)
