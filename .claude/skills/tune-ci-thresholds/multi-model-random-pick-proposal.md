# Proposal: calibration for multi-model random-pick TTS CI (issue #724)

Status: **proposal, not final**. This documents where the calibration skill
(`SKILL.md`, `tune.py`, `models/tts/config.yaml`) must change to support the
multi-model TTS CI (Qwen3-TTS / Higgs / FishAudio S2 Pro / MOSS TTS) with a
random model pick per CI run. Nothing in this document is wired up yet; the
first PR only adds MOSS in scaffold mode (thresholds blank, numbers printed).

## Current state

- `models/tts/config.yaml` is Higgs-specific: one `hf_model_id`, one set of
  stage keys (`tts_nonstream_*`, `tts_stream_*`), and `metric_sources` that
  assume the Higgs artifact layout.
- `test_tts_ci.py` holds a single set of threshold constants
  (`_VC_NON_STREAM_P95`, `_VC_STREAM_P95`, `VC_WER_MAX_CORPUS`,
  `VC_STREAM_WER_MAX_CORPUS`, `VC_SIMILARITY_MEAN_MIN`,
  `VC_UTMOS_MEAN_REFERENCE`). The apply step rewrites those literals in place.
- Since #724 scaffolding, `test_tts_ci.py` selects the served model via
  `TTS_CI_MODEL` (presets in `TTS_CI_MODEL_PRESETS`); only `higgs` gates on
  thresholds, every other model prints measured numbers.

## Required changes

### 1. Per-model calibration configs

One calibration config per CI TTS model, sharing the SeedTTS dataset and the
Qwen3-ASR stage:

```
models/tts/          (rename intent: tts-higgs; keep "tts" as alias)
models/tts-moss/
models/tts-qwen3/
models/tts-fishaudio-s2pro/
```

Each `config.yaml` needs:

- `hf_model_ids_by_test.test_tts_ci.py` pointing at that model's checkpoint.
- An `extra_env` (new key) of `TTS_CI_MODEL=<model>` and
  `TTS_MODEL_PATH=<checkpoint>` exported by `tune.py` for every pytest
  invocation of `test_tts_ci.py`, mirroring the CI workflow stage env.
- The same `metric_sources` shape (the artifact layout
  `vc_nonstream_c16/ vc_stream_c16/` is model-independent).

`tune.py` itself only needs the `extra_env` plumbing; metric extraction is
unchanged because all models write the same `speed_results.json` /
`wer_results.json` schema.

### 2. Per-model threshold constants in the test file

Today the apply step rewrites single-valued constants. With four models the
constants must become per-model. Proposal: move the calibrated values into
`TTS_CI_MODEL_PRESETS[<model>]` (e.g. keys `non_stream_p95`, `stream_p95`,
`wer_max_corpus`, `stream_wer_max_corpus`, `similarity_mean_min`,
`utmos_mean_reference`), keeping `None` = scaffold/not yet calibrated.
`tune.py`'s AST matcher (`match_metric()` / `_NESTED`) and the apply editor
must learn this nested per-model shape — this is the main tune.py code change.
Until that lands, apply works only for Higgs (whose constants stay where they
are; this PR deliberately did not move them).

### 3. Calibration procedure under random pick

The random pick changes *when* a model's thresholds are exercised, not how
they are measured. Implications the skill must encode:

- **Calibration is always explicit, never sampled.** `tune.py run --model
  tts-<m>` runs that model's full stage set N times regardless of any CI
  pick. The random pick exists only in the CI workflow.
- **A model enters the gated pool only after strict worst-of-N (N=5) on the
  H20 CI host.** Until then it stays in scaffold mode (thresholds `None`,
  print-only) — exactly MOSS's state after the first #724 PR.
- **Engine updates require recalibrating all enabled models**, not just the
  model that happened to be drawn on the regressing commit. A perf regression
  in a model that is sampled on ~1/4 of commits is detected with expected lag
  of ~4 commits (P(undetected after k commits) = 0.75^k); the skill's
  "Performance optimization checks" section must treat the *commit range since
  the model's last calibration*, not since the last CI run, as the comparison
  window.
- **Provenance per model.** Each model's thresholds carry their own
  calibration commit + date (today there is one implicit provenance for
  Higgs). Reports under `docs/calibration/` should be per model:
  `<timestamp>-tts-<model>-report.md`.
- **The CI must record the drawn model** (job name / log line / artifact
  prefix already include it via `TTS_CI_MODEL`) so a threshold failure can be
  attributed to the right model when reading CI history.
- **Reproducing a CI failure**: derive the pick deterministically (e.g. seed =
  commit SHA) and allow forcing via `TTS_CI_MODEL` so a rerun or a calibration
  run can pin the same model.

### 4. Out of scope for the first PR (done here only as scaffold)

- Random-pick wiring in the workflow (stages 5–7 currently run MOSS on every
  commit, additively to Higgs stages 2–4; the follow-up replaces both with one
  drawn model per run).
- Any MOSS / Qwen3-TTS / S2-Pro thresholds.
- Per-model speaker-similarity / UTMOS gates (skipped in scaffold mode).
