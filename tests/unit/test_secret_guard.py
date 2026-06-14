from __future__ import annotations

from app.agents.secret_guard import redact_hidden_facts


def test_redacts_verbatim_secret_in_repair_instruction() -> None:
    issues, instruction, leaked = redact_hidden_facts(
        issues=["The draft hints at the coup."],
        repair_instruction="Do not reveal that she forged one inventory ledger.",
        hidden_facts=["She forged one inventory ledger."],
    )

    assert leaked is True
    assert instruction is not None
    assert "forged one inventory ledger" not in instruction.lower()
    assert "[redacted]" in instruction


def test_redacts_one_sentence_of_a_multi_sentence_gm_fact() -> None:
    issues, _instruction, leaked = redact_hidden_facts(
        issues=["The regent's spy is already in the room, so the draft is risky."],
        repair_instruction=None,
        hidden_facts=["The regent's spy is already in the room. Trust no one tonight."],
    )

    assert leaked is True
    assert "[redacted]" in issues[0]


def test_case_insensitive_match() -> None:
    issues, _instruction, leaked = redact_hidden_facts(
        issues=["She FORGED one inventory ledger in the draft."],
        repair_instruction=None,
        hidden_facts=["she forged one inventory ledger"],
    )

    assert leaked is True
    assert "[redacted]" in issues[0]


def test_no_redaction_without_a_verbatim_echo() -> None:
    issues, instruction, leaked = redact_hidden_facts(
        issues=["The tone is too generic."],
        repair_instruction="Make the answer concrete.",
        hidden_facts=["She forged one inventory ledger."],
    )

    assert leaked is False
    assert issues == ["The tone is too generic."]
    assert instruction == "Make the answer concrete."


def test_empty_hidden_facts_is_noop() -> None:
    issues, instruction, leaked = redact_hidden_facts(
        issues=["x"],
        repair_instruction="y",
        hidden_facts=[],
    )

    assert leaked is False
    assert issues == ["x"]
    assert instruction == "y"
