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

The processor ships with the checkpoint, so no extra TTS package is needed. Voice cloning
transcodes reference audio (file, URL, or base64 data-URI) with **ffmpeg**, so `ffmpeg` must
be on the server's `PATH` (e.g. `apt-get install ffmpeg`).

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
URL, or a base64 **data URI** (`data:audio/wav;base64,<...>`, transcoded via `ffmpeg`):

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

Seed-TTS-Eval EN (1088/1088 successful), 1× H100, concurrency 16, `--ref-format references`,
`fp8 + frame_graph + async_decode + compile_sampler`. WER scored with whisper-large-v3 +
`EnglishTextNormalizer` + jiwer (the CI gate in `tests/test_model/test_zonos2_tts_ci.py` also
checks the Qwen3-ASR router; both agree ~1.3–1.5%). Means over 2 runs — seed-tts generation is
unseeded, so per-run corpus WER varies ~±0.15pt.

| Config | WER (corpus) | RTF mean | Throughput (qps) |
|---|---|---|---|
| non-streaming | ~1.3% | 0.68 | 6.3 |
| streaming, `stream_emit_chunk_frames=1` | ~1.4% | 0.92 | 4.6 |
| **streaming, adaptive `24→32` (default)** | ~1.5% | **0.75** | **5.7** |
| streaming, 2-GPU (`multi_gpu`) + emit=32 | ~1.5% | 0.69 | 6.5 |

### Streaming throughput (`stream_emit_chunk_frames`)

In streaming mode the AR engine pushes sampled frames to the vocoder over an in-process queue.
By default the engine **coalesces `stream_emit_chunk_frames=32` frames into one message** instead
of one `put()` per frame; the per-frame puts run on the resolve host loop and serialize against
the next decode launch, so batching them gives **−17% streaming RTF and +21% throughput,
WER-neutral**, vs the per-frame path (inter-chunk latency also improves, 0.37 s → 0.28 s). To keep
first-audio latency low, the default also emits a **smaller first chunk**
(`stream_emit_first_chunk_frames=24`), so time-to-first-chunk stays at the per-frame level
(~0.30 s) instead of waiting for a full 32-frame batch (which would add ~0.09 s). Set
`stream_emit_first_chunk_frames=0` to disable the adaptive first chunk, or
`stream_emit_chunk_frames=1` for the lowest-latency per-frame streaming. The 2-GPU `multi_gpu`
pipeline (codec + speaker encoder on `cuda:1`) stacks on top for the best streaming RTF (~0.69).

> ZH (`--lang zh`, 2020 samples) tracks the same RTF/throughput pattern; CER stays flat across
> the streaming configs (the coalescing is audio-neutral after the `on_stream_done` tail fix).

## Known Limitations

- **Voice cloning depends on the reference.** Provide the transcript (`text` / `ref_text`) for
  the best speaker similarity when cloning.
- **Language is a hint.** `language` biases the target language but is not a hard constraint.
- **Rare runaway generation.** A small fraction of utterances can loop and generate up to
  `max_new_tokens`; lowering `max_new_tokens` bounds the output.
