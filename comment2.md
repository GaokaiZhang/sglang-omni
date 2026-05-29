# MOSS-TTS #607 — root-cause fixes for bad WER/CER + RTF

These are the **confirmed** causes of the high WER/CER (incl. the reference-copy
tail) and the slow decode. Apply both on top of the current `sup/607` (which
already has the CUDA-graph work — these are independent of it), update the unit
tests, then re-run. My push couldn't land (branch diverged after `34959d7`), so
the diffs are spelled out below to apply directly.

## Cause #1 (WER/CER + reference-copying): we greedy-decode a sampling model
The served checkpoint's own `generate()` (`models--OpenMOSS-Team--MOSS-TTS-v1.5/
.../modeling_moss_tts.py`) defaults to **sampling**: `text_temperature=1.5,
text_top_k=50, audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25`, and the
upstream reference (EN WER 1.84 / ZH CER 1.37) was produced with these. Our
integration forces **greedy** (`temp=0, top_p=1.0, top_k=-1`). Greedy on a
reference-conditioned RVQ codec LM collapses into **copying the reference audio**
(the ASR≈ref_text symptom) and wrecks WER/CER. "(no sampling)" in Seed-TTS-eval
means a single deterministic pass / no best-of-N — **not** temperature=0;
determinism comes from the fixed server `random_seed` (123) + `sampling_backend=
"pytorch"`.

**Fix — `sglang_omni/models/moss_tts/request_builders.py`:**

In `build_generation_kwargs`, replace the greedy block with the upstream defaults:
```python
        "text_temperature": 1.5,
        "audio_temperature": 1.7,
        "text_top_p": 1.0,
        "audio_top_p": 0.8,
        "text_top_k": 50,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
```
Also set the matching defaults on the `MossTTSSGLangRequestData` dataclass (it
currently defaults `text_temperature=0.0 / audio_temperature=0.0 / top_p=1.0 /
top_k=-1`) to `1.5 / 1.7 / (1.0,0.8) / (50,25)` so no code path falls back to
greedy. Keep the existing explicit-override logic (callers can still pass values).

## Cause #2 (RTF): per-step, per-codebook Python sampling with GPU syncs
`model_runner._sample_next_row` loops over ~32 codebooks doing one `.item()`
sync per codebook (`sampling_audio_mask[vq_idx].item()`) and rebuilding
`_previous_audio_tokens` every codebook even when `repetition_penalty == 1.0`.
That's ~32 serialized GPU round-trips per decode step.

**Fix — `sglang_omni/models/moss_tts/model_runner.py`** (in `_sample_next_row`,
replace the audio-channel loop):
```python
        sampling_audio_mask = self._sampling_audio_mask(data, n_vq=n_vq, device=device)
        active = sampling_audio_mask.tolist()          # one host sync, not n_vq
        rep_penalty = float(data.audio_repetition_penalty)
        for vq_idx in range(n_vq):
            if not active[vq_idx]:
                continue
            logits = channel_logits[vq_idx + 1][row_idx].clone()
            logits[int(cfg.audio_pad_code)] = float("-inf")
            audio_tokens[vq_idx] = self._sample_logits(
                logits,
                temperature=float(data.audio_temperature),
                top_p=float(data.audio_top_p),
                top_k=int(data.audio_top_k),
                repetition_penalty=rep_penalty,
                prev_tokens=(
                    self._previous_audio_tokens(data, vq_idx)
                    if rep_penalty != 1.0 else None
                ),
            )
```
Note: `rtf_mean` is end-to-end latency ÷ audio-duration at **concurrency 16**, so
it bakes in queueing — also report a **c=1** rtf for a clean decode number. The
larger follow-up lever is fully vectorizing the audio sampling across all heads
(mirror upstream `inference_utils.sample_token`, which reshapes to `[-1, vocab]`
and does one top-k/top-p/multinomial).

## Unit tests
Update `tests/unit_test/moss_tts/test_pipeline.py`: the defaults test must now
assert `text_temperature == 1.5`, `audio_temperature == 1.7`, `audio_top_p ==
0.8`, `audio_top_k == 25`. Run `pytest -q tests/unit_test/moss_tts
tests/unit_test/serve/test_openai_api.py` (expect green).

## Re-run plan
1. Restart the server (capture stdout to a log).
2. **EN subset sanity:** `--max-samples 64`, once **with** `--token-count auto`
   and once **without** (the auto path injects a `Tokens:` duration field — not a
   neutral request). Confirm WER drops toward ~2% and the ref-copy tail is gone.
3. If WER is good → full EN + ZH: `--generate-only` then `--transcribe-only
   --lang {en,zh} --device cuda:N` then `--similarity-only --device cuda:N`.
4. If reference-copy **persists** after the sampling fix → audit output
   extraction: `_resolve_audio_payload_bounds` takes the **first** `audio_start`
   in the assembled rows, which can grab the reference codes; prefer matching
   upstream `processor.decode()` semantics for the generated span.
5. Update the PR comment with EN+ZH **WER/CER + speaker-SIM + rtf_mean (+ c=1
   rtf)**, vs upstream ref EN 1.84/70.86, ZH 1.37/76.98.
