from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PROFILE_LIBRARY = ROOT / "scripts/lib/local-model-profile.sh"


def _profile(profile: str) -> tuple[str, list[str]]:
    script = f"""
source {PROFILE_LIBRARY!s}
LOCAL_MODEL_PROFILE={profile}
resolve_local_model_profile
printf '%s\\n' "$PROFILE_HF_MODEL"
printf '%s\\0' "${{PROFILE_LLAMA_ARGS[@]}}"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
    )
    model, _, raw_args = result.stdout.partition(b"\n")
    args = [item.decode() for item in raw_args.split(b"\0") if item]
    return model.decode(), args


@pytest.mark.parametrize(
    ("profile", "expected_model"),
    [
        (
            "small",
            "DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking-GGUF:Q8_0",
        ),
        (
            "26b",
            "HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M",
        ),
    ],
)
def test_named_profiles_use_identical_inference_arguments(
    profile: str,
    expected_model: str,
) -> None:
    model, args = _profile(profile)

    assert model == expected_model
    assert args == [
        "--jinja",
        "--reasoning",
        "off",
        "-ngl",
        "all",
        "-c",
        "8192",
        "-fa",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q4_0",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--seed",
        "424242",
    ]
    kwargs_index = args.index("--chat-template-kwargs") + 1
    assert json.loads(args[kwargs_index]) == {"enable_thinking": False}


def test_unknown_profile_is_rejected() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {PROFILE_LIBRARY!s}; LOCAL_MODEL_PROFILE=large; resolve_local_model_profile",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "small or 26b" in result.stderr
