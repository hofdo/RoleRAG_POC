from __future__ import annotations

from app.agents.secret_guard import (
    collect_hidden_facts,
    redact_hidden_facts,
    scan_reply,
)
from app.domain import PersonaCard, SceneState

# The actual Sarnhold dramatic-irony secrets the bake-off probe made qwen35b leak.
_BRAM_SECRET = (
    "House Varn offered this daughter, and not the elder, because she was the one "
    "Duke Corvane could most afford to lose."
)
_NESSA_SECRET = "Iseult cannot swim and dreads water, because of how her father once punished her."


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


def test_scan_reply_redacts_verbatim_echo() -> None:
    report = scan_reply(
        "Honestly, she forged one inventory ledger to cover the debt.",
        ["She forged one inventory ledger."],
    )
    assert report.verbatim_redacted is True
    assert "forged one inventory ledger" not in report.text.lower()
    assert "[redacted]" in report.text


def test_scan_reply_flags_paraphrased_confabulation() -> None:
    # The actual qwen35b leak: Bram paraphrases the secret without quoting it.
    leaked_reply = (
        "House Varn views Iseult as the daughter they could most afford to lose; "
        "her elder sister holds the political weight, so Duke Corvane sent the one he could spare."
    )
    report = scan_reply(leaked_reply, [_BRAM_SECRET])
    assert report.paraphrased_facts == (_BRAM_SECRET,)
    assert report.flagged is True


def test_scan_reply_paraphrase_threshold_is_applied() -> None:
    # The 0.6 default is a security knob: pin its value and that it is actually applied —
    # the real bake-off paraphrase leak is flagged at 0.6 but relaxed at a near-1.0 threshold.
    from app.agents.secret_guard import DEFAULT_PARAPHRASE_OVERLAP

    assert DEFAULT_PARAPHRASE_OVERLAP == 0.6
    leaked_reply = (
        "House Varn views Iseult as the daughter they could most afford to lose; "
        "her elder sister holds the political weight, so Duke Corvane sent the one he could spare."
    )
    assert scan_reply(leaked_reply, [_BRAM_SECRET], paraphrase_overlap_threshold=0.6).flagged
    relaxed = scan_reply(leaked_reply, [_BRAM_SECRET], paraphrase_overlap_threshold=0.99)
    assert relaxed.paraphrased_facts == ()


def test_scan_reply_does_not_flag_in_character_deflection() -> None:
    # Nessa lying to protect the secret must not trip the screen.
    deflection = "My lady is a strong swimmer, my lord; she fears neither river nor rain."
    report = scan_reply(deflection, [_NESSA_SECRET])
    assert report.flagged is False
    assert report.text == deflection


def test_scan_reply_noop_without_hidden_facts() -> None:
    report = scan_reply("Any reply at all.", [])
    assert report.flagged is False
    assert report.text == "Any reply at all."


def test_collect_hidden_facts_gathers_all_protected_fields() -> None:
    persona = PersonaCard(
        id="bram",
        name="Chancellor Bram",
        role="npc",
        public_description="The Duke's chancellor.",
        private_description="He protects the alliance above any person.",
        speaking_style="Measured and courtly.",
        secrets=["He hid the courier's report."],
        forbidden_knowledge=[_BRAM_SECRET],
    )
    scene = SceneState(
        id="arrival-hall",
        title="Arrival Hall",
        location="Sarnhold Keep",
        player_visible_summary="The court receives the bride.",
        gm_private_summary="Iseult is not spoiled but abused.",
    )
    facts = collect_hidden_facts(persona, scene)
    assert _BRAM_SECRET in facts
    assert "He hid the courier's report." in facts
    assert "He protects the alliance above any person." in facts
    assert "Iseult is not spoiled but abused." in facts
