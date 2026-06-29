# MTP speculative-decoding speed test — 2026-06-29

Point-in-time finding (not current-state docs). Compares the `26b-mtp` local-model profile
(Gemma4-26B-A4B Balanced QAT + an MTP draft model, speculative decoding on) against the **same
local weights with the draft off** — so the only variable is MTP speculative decoding.

## Setup

- Both runs: `scripts/live-smoke.sh`, `LIVE_TURN_COUNT=100`, `LIVE_SKIP_BROWSER=1`,
  `LIVE_FAIL_ON_STRUCTURED_WARNINGS=0`, `CLOUD_MODE=off`, same seed, same 100-turn Rose Gallery scenario.
- A = `LOCAL_MODEL_PROFILE=26b-mtp` (draft on, `-md … --spec-type draft-mtp --spec-draft-n-max 3`, 16384 ctx).
- B = `LOCAL_MODEL_PROFILE=26b` + `LLAMA_CPP_MODEL_PATH=<same local main .gguf>` + `LLAMA_CPP_CTX_SIZE=16384`
  (draft off). Identical weights/config otherwise.
- Apple Silicon, llama.cpp Metal, full GPU offload. One run each, sequential.

## Result

| Metric | A — MTP on | B — baseline | A advantage |
|---|---|---|---|
| Per-turn mean | 51.9s | 59.4s | **1.14× faster** |
| Per-turn median | 49.1s | 54.2s | 1.10× faster |
| Decode throughput | 26.6 tok/s (med 31.0) | 21.7 (med 22.8) | **1.23–1.36× faster** |
| Prefill throughput | 165 tok/s | 157 tok/s | ~tied |
| Draft acceptance | 0.71 | — | working |
| Total wall (100 turns) | 86.5 min | 99.0 min | ~12.5 min (~13%) saved |

Advantage grows with conversation length:

| Turns | A-mtp | B-base | speedup |
|---|---|---|---|
| 1–25 | 53.9s | 52.7s | 0.98× (tied) |
| 26–50 | 55.4s | 64.2s | 1.16× |
| 51–75 | 50.7s | 58.5s | 1.15× |
| 76–100 | 47.7s | 62.3s | 1.31× |

## Interpretation

- **MTP is a lossless ~10–14% net speedup** at realistic-to-long session length, growing the longer
  you play. Speculative decoding emits identical tokens, just faster — no quality cost by design.
- It **grows with length** because prefill plateaus (the recent-dialogue window of 8 + retrieval
  top-k bound the prompt size), so decode becomes a larger share of each turn — and decode is exactly
  what MTP accelerates.
- **Smaller runs undersold it**: 8 turns looked like a tie, 30 turns ~1.1× — both just undersampled.
  The **decode tok/s (1.23–1.36×) is the most reliable number** — steady across 8/30/100-turn runs.

## Caveats

- Single run each, sequential. B ran second, so some of the late-turn 1.31× may be Mac thermal
  throttling rather than pure MTP. Trust the per-token decode tok/s over the net wall-clock for the
  true MTP effect.
- No recall metrics at 100 turns: the live checkpoint's recent-conversation assertion aborts past
  ~8 turns (timings are still captured). Recall was identical (zero misses) at 8 and 30 turns; MTP
  is lossless, so recall is not expected to change.

## Recommendation

Enable `26b-mtp` for real play sessions — free, lossless, compounding speedup. Keep it opt-in
(`LOCAL_MODEL_PROFILE=26b-mtp`), not the default, since it depends on local `.gguf` files.

## Reproduce

```bash
LOCAL_MODEL_PROFILE=26b-mtp LIVE_TURN_COUNT=100 LIVE_SKIP_BROWSER=1 \
  LIVE_FAIL_ON_STRUCTURED_WARNINGS=0 LIVE_ARTIFACT_DIR=/tmp/A bash scripts/live-smoke.sh
LOCAL_MODEL_PROFILE=26b LLAMA_CPP_MODEL_PATH=<local main .gguf> LLAMA_CPP_CTX_SIZE=16384 \
  LIVE_TURN_COUNT=100 LIVE_SKIP_BROWSER=1 LIVE_FAIL_ON_STRUCTURED_WARNINGS=0 \
  LIVE_ARTIFACT_DIR=/tmp/B bash scripts/live-smoke.sh
```
Per-turn `duration_seconds` are in `<artifact>/raw/conversation-checkpoint.json`; decode/prefill
tok/s + draft acceptance are in `<artifact>/raw/llama-server.log`.
