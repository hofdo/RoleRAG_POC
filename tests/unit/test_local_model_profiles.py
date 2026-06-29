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
    shared_args = [
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
    assert args[: len(shared_args)] == shared_args
    kwargs_index = args.index("--chat-template-kwargs") + 1
    assert json.loads(args[kwargs_index]) == {"enable_thinking": False}


def test_small_profile_uses_patched_no_think_chat_template() -> None:
    _, args = _profile("small")

    template_index = args.index("--chat-template-file") + 1
    template_path = Path(args[template_index]).resolve()
    assert template_path == (ROOT / "scripts/templates/small-model.jinja").resolve()
    first_line = template_path.read_text(encoding="utf-8").splitlines()[0]
    assert "enable_thinking is not defined" in first_line


def test_26b_profile_keeps_stock_chat_template() -> None:
    _, args = _profile("26b")

    assert "--chat-template-file" not in args


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
    assert "small, 26b, or 26b-mtp" in result.stderr


def _profile_model_path(profile: str, *, mtp_dir: str = "/models/mtp") -> str:
    script = f"""
source {PROFILE_LIBRARY!s}
LOCAL_MODEL_PROFILE={profile}
MODEL_26B_MTP_DIR={mtp_dir}
resolve_local_model_profile
printf '%s' "${{PROFILE_MODEL_PATH}}"
"""
    result = subprocess.run(["bash", "-c", script], check=True, capture_output=True)
    return result.stdout.decode()


def test_26b_mtp_profile_serves_local_files_with_speculative_draft() -> None:
    model, args = _profile("26b-mtp")

    assert model == ""  # served from a local -m path, not pulled via -hf
    assert args[args.index("-c") + 1] == "16384"  # bumped context for the MTP build
    assert args[args.index("--spec-type") + 1] == "draft-mtp"
    assert args[args.index("--spec-draft-n-max") + 1] == "3"
    assert args[args.index("-md") + 1].endswith("mtp-gemma-4-26B-A4B-it.gguf")
    # Sampling stays app-controlled (the router sets temperature per request).
    assert "--temp" not in args


def test_26b_mtp_profile_model_path_honors_dir_override() -> None:
    path = _profile_model_path("26b-mtp", mtp_dir="/models/mtp")

    assert path == "/models/mtp/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
