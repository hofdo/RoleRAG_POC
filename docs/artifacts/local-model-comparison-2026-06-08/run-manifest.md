# Local Model Comparison Run Manifest

- captured_at_utc: `2026-06-08T11:41:58Z`
- git_head: `22f23ffdb30e9da0bc13921a9042b157a45c9493`
- tracked_diff_sha256: `be8ab452972d5772419d240b4af97de834319432fd4183cd896f1482d1dd82b8`
- untracked_files_aggregate_sha256: `768bcbe02485be0e71358368b3e721f299aebc1c15588d18f6f6699d01070556`
- host: `MacBook Pro Mac17,2`
- chip: `Apple M5, 10 cores (4 performance, 6 efficiency)`
- memory: `32 GB`
- operating_system: `macOS 26.3 (25D125), Darwin 25.3.0 arm64`
- llama_cpp: `llama-server 8680 (15f786e65), AppleClang 21.0.0.21000099`
- docker: `Docker Desktop 4.54.0, engine 29.1.2, linux/arm64`
- python: `3.14.4`
- node: `20.19.6`
- npm: `11.6.4`
- free_disk_before_run: `514 GiB`
- ports_before_run: `8080`, `18080`, and `6334` free

## Models

| Profile | Hugging Face identifier | Revision | Quantization | GGUF SHA-256 | Cached size |
| --- | --- | --- | --- | --- | --- |
| small | `DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking-GGUF:Q8_0` | `cc4f8a5d19924a28456785fe6fb065a6910675a9` | `Q8_0` | `79307aae843e7910222645e3b8f5bf1c8e8caefef66535c2dad5575701955e35` | 11 GiB |
| 26b | `HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M` | `96c11c22b1128c3c8c655b21557b409f307c557f` | `Q4_K_M` | `f8b1da6dc139e6928159e536bc85602adbc1412018871732a878dedcad7ccafd` | 16 GiB |

Both repositories and exact GGUF blobs were present in the Hugging Face cache before execution.

## Immutable Settings

- seed: `424242`
- llama.cpp context: `8192`
- GPU layers: `all`
- flash attention: `on`
- K cache: `q8_0`
- V cache: `q4_0`
- Jinja chat template: enabled
- reasoning: disabled
- chat-template `enable_thinking`: `false`
- actor maximum output: `500` tokens
- actor temperature: `0.75`
- structured-task maximum output: `350` tokens
- structured-task temperature: `0.0`
- retrieved chunks: `5`
- maximum characters per retrieved chunk: `800`
- recent dialogue turns: `8`
- cloud mode: `off`
- structured warnings: report-only
- Qdrant image: `qdrant/qdrant:v1.18.1`
- application port: `18080`
- llama.cpp port: `8080`
- Qdrant port: `6334`

The small and 26B profiles differ only in the model identifier, quantization, and model alias.

## llama.cpp Command Arrays

Small:

```text
/opt/homebrew/bin/llama-server
-hf DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe-HERETIC-UNCENSORED-Thinking-GGUF:Q8_0
--host 127.0.0.1
--port 8080
--alias rolerag-small
--jinja
--reasoning off
-ngl all
-c 8192
-fa on
--cache-type-k q8_0
--cache-type-v q4_0
--chat-template-kwargs {"enable_thinking":false}
--seed 424242
```

26B:

```text
/opt/homebrew/bin/llama-server
-hf HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M
--host 127.0.0.1
--port 8080
--alias rolerag-26b
--jinja
--reasoning off
-ngl all
-c 8192
-fa on
--cache-type-k q8_0
--cache-type-v q4_0
--chat-template-kwargs {"enable_thinking":false}
--seed 424242
```

## Study Commands

20-turn qualification:

```bash
MODEL_COMPARE_ARTIFACT_DIR=/tmp/rolerag-model-comparison-20 \
MODEL_COMPARE_TURN_COUNT=20 \
PYTHON=.venv/bin/python \
bash scripts/test-local-model-matrix.sh
```

50-turn small profile:

```bash
LOCAL_MODEL_PROFILE=small \
LOCAL_LLM_MODEL=rolerag-small \
LIVE_ARTIFACT_DIR=/tmp/rolerag-model-comparison-50/small \
LIVE_TURN_COUNT=50 \
LIVE_FAIL_ON_STRUCTURED_WARNINGS=0 \
PYTHON=.venv/bin/python \
bash scripts/live-smoke.sh
```

50-turn 26B profile:

```bash
LOCAL_MODEL_PROFILE=26b \
LOCAL_LLM_MODEL=rolerag-26b \
LIVE_ARTIFACT_DIR=/tmp/rolerag-model-comparison-50/26b \
LIVE_TURN_COUNT=50 \
LIVE_FAIL_ON_STRUCTURED_WARNINGS=0 \
PYTHON=.venv/bin/python \
bash scripts/live-smoke.sh
```

Comparison aggregation:

```bash
.venv/bin/python -m app.diagnostics.model_comparison \
  --output-dir /tmp/rolerag-model-comparison-50 \
  --deterministic-exit 0 \
  --small-exit SMALL_EXIT \
  --model-26b-exit MODEL_26B_EXIT
```

Full transcripts, databases, Qdrant state, Playwright output, and llama.cpp logs remain under
`/tmp`. Only this manifest and sanitized `comparison.json` summaries are retained in the repository.
