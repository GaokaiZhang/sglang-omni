# ZONOS2

[ZONOS2](https://huggingface.co/Zyphra) is a mixture-of-experts (MoE)
text-to-speech model from Zyphra. A MoE autoregressive decoder predicts
**9 DAC audio codebooks** scheduled in a **delay pattern**; the codes are then
decoded back to **44.1 kHz** speech by a DAC vocoder. It clones a voice from a
short reference clip and accepts an optional target-language hint. In
SGLang-Omni it runs as a `preprocessing → speaker_encode → tts_engine →
vocoder` pipeline and is served through the OpenAI-compatible
`/v1/audio/speech` endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then
download the model:

```bash
hf download Zyphra/zonos2
```

The processor ships with the checkpoint, so no extra TTS package is needed. Decoding base64
(data-URI) reference audio additionally requires `soundfile` (`uv pip install soundfile`).

## Server Configuration

The pipeline is `preprocessing → speaker_encode → tts_engine → vocoder`.

ZONOS2 ships a `params.json` whose `model_type` (`zonos2`) auto-selects the
`Zonos2ForCausalLM` architecture, so `serve` needs only `--model-path` — no
`--config` (mirrors Higgs).

```bash
sgl-omni serve \
  --model-path Zyphra/zonos2 \
  --port 8000
```

## Synthesizing Speech

### Voice Cloning

ZONOS2 clones a voice from a reference clip. The `references` field accepts `audio_path`
(a local path, HTTP URL, or base64 data URI) and `text` (the transcript of that clip). Supplying
the transcript materially improves cloning quality.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for `references[0].audio_path` and
`references[0].text`.

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Reference Audio Sources

`audio_path` / `ref_audio` may be a local filesystem path readable by the server, an HTTP(S)
URL, or a base64 **data URI** (`data:audio/wav;base64,<...>`, decoded with `soundfile`):

```json
{"ref_audio": "data:audio/wav;base64,UklGR.....", "ref_text": "Transcript of the clip."}
```

### Streaming

Set `"stream": true` to receive Server-Sent Events (SSE). Audio events carry base64-encoded WAV
bytes in `audio.data`; the final metadata event has `audio: null`, followed by `data: [DONE]`.

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stream": true
  }'
```

### Language

An optional `language` hint biases the target language; omit it to let the model infer from the
text.

```json
{
  "input": "今天天气不错，就该出去晒晒太阳。",
  "ref_audio": "...", "ref_text": "...",
  "language": "Chinese"
}
```

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Text to synthesize |
| `references` | `null` | Reference clip for cloning; each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `stream` | `false` | Enable SSE streaming |
| `language` | `null` | Optional target-language hint; omit to let the model infer |
| `max_new_tokens` | (model default) | Maximum generated frames; an explicit value must be `> 0` |
| `temperature` | (model default) | Sampling temperature |
| `top_p` | (model default) | Top-p sampling |
| `top_k` | (model default) | Top-k sampling |
| `min_p` | (model default) | Min-p sampling |
| `repetition_penalty` | (model default) | Audio repetition penalty |
| `seed` | `null` | Non-negative integer for reproducible sampling |

## Benchmarking

ZONOS2 clones from each prompt (`--ref-format references`). Run the seed-tts-eval voice-clone
benchmark against a running server:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Zyphra/zonos2 --port 8000 \
  --ref-format references \
  --output-dir results/zonos2_en --lang en --max-concurrency 16
```

Use `--lang zh` for the Chinese split. See `benchmarks/README.md` for the full workflow.

## Benchmark Results

<!-- TODO(calibrate on CI GPU): fill from a real ZONOS2 seed-tts-eval run.
Mirror the columns below from the calibration JSON
(speed_results.json['summary'] + wer_results.json['summary']). Do not copy
another model's numbers. -->

Seed-TTS-Eval, concurrency 16, `--ref-format references`. WER is scored with the Qwen3-ASR
router (same scorer as `tests/test_model/test_zonos2_tts_ci.py`).

| Lang | WER (corpus) | Latency mean (s) | RTF mean | Throughput (qps) |
|---|---|---|---|---|
| EN | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

## Known Limitations

- **Voice cloning depends on the reference.** Provide the transcript (`text` / `ref_text`) for
  the best speaker similarity when cloning.
- **Language is a hint.** `language` biases the target language but is not a hard constraint.
- **Rare runaway generation.** A small fraction of utterances can loop and generate up to
  `max_new_tokens`; lowering `max_new_tokens` bounds the output.
