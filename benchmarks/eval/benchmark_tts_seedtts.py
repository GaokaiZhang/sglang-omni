# SPDX-License-Identifier: Apache-2.0
"""SeedTTS benchmark for S2-Pro TTS: speed measurement and WER evaluation.

Combines benchmark_tts_speed.py and voice_clone_tts_wer.py into a two-phase
pipeline: generate audio while the server is running, then transcribe without
the server to avoid GPU OOM.

Usage:
    # Full pipeline (generate + transcribe)
    python benchmarks/eval/benchmark_tts_seedtts.py \
        --meta seedtts_testset/en/meta.lst \
        --model fishaudio/s2-pro --port 8000

    # Full pipeline, streaming, high concurrency
    python benchmarks/eval/benchmark_tts_seedtts.py \
        --meta seedtts_testset/en/meta.lst \
        --model fishaudio/s2-pro --port 8000 \
        --concurrency 8 --stream

    # Phase 1: generate audio only (server must be running)
    python benchmarks/eval/benchmark_tts_seedtts.py \
        --generate-only \
        --meta seedtts_testset/en/meta.lst \
        --model fishaudio/s2-pro --port 8000 --concurrency 8

    # Phase 2: transcribe + WER only (server not needed)
    python benchmarks/eval/benchmark_tts_seedtts.py \
        --transcribe-only \
        --meta seedtts_testset/en/meta.lst \
        --model fishaudio/s2-pro \
        --output-dir results/s2pro_en \
        --lang en --device cuda:0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts import (
    SampleOutput,
    _transcribe_and_compute_wer,
    build_speed_results,
    calculate_wer_metrics,
    load_asr_model,
    make_tts_send_fn,
    print_speed_summary,
    print_wer_summary,
    save_generated_audio_metadata,
    save_speed_results,
    save_wer_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TtsSeedttsBenchmarkConfig:
    model: str
    meta: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    no_ref_audio: bool = False
    output_dir: str = "results/tts_seedtts"
    max_samples: int | None = None
    max_new_tokens: int | None = 2048
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    warmup: int = 1
    concurrency: int = 1
    request_rate: float = float("inf")
    stream: bool = False
    disable_tqdm: bool = False
    # Transcribe phase
    lang: str = "en"
    device: str = "cuda:0"


def _build_base_url(config: TtsSeedttsBenchmarkConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


def _build_generation_kwargs(config: TtsSeedttsBenchmarkConfig) -> dict:
    generation_kwargs: dict = {}
    if config.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = config.max_new_tokens
    if config.temperature is not None:
        generation_kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        generation_kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        generation_kwargs["top_k"] = config.top_k
    if config.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = config.repetition_penalty
    return generation_kwargs


def _build_results_config(
    config: TtsSeedttsBenchmarkConfig,
    *,
    base_url: str,
) -> dict:
    return {
        "model": config.model,
        "base_url": base_url,
        "meta": config.meta,
        "no_ref_audio": config.no_ref_audio,
        "stream": config.stream,
        "max_samples": config.max_samples,
        "max_new_tokens": config.max_new_tokens,
        "warmup": config.warmup,
        "concurrency": config.concurrency,
        "request_rate": config.request_rate,
    }


async def run_tts_seedtts_benchmark(
    config: TtsSeedttsBenchmarkConfig,
) -> dict:
    """Generate audio and measure speed. Always saves audio for WER use.

    Returns a dict with keys: summary, per_request, config.
    """
    if not os.path.isfile(config.meta):
        raise FileNotFoundError(f"Meta file not found: {config.meta}")

    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/audio/speech"

    samples = load_seedtts_samples(config.meta, config.max_samples)
    logger.info("Prepared %d requests", len(samples))

    save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
    os.makedirs(save_audio_dir, exist_ok=True)

    generation_kwargs = _build_generation_kwargs(config)
    send_fn = make_tts_send_fn(
        config.model,
        api_url,
        stream=config.stream,
        no_ref_audio=config.no_ref_audio,
        save_audio_dir=save_audio_dir,
        **generation_kwargs,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    outputs = await runner.run(samples, send_fn)

    metrics = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    results_config = _build_results_config(config, base_url=base_url)
    benchmark_results = build_speed_results(outputs, metrics, results_config)
    save_speed_results(outputs, metrics, results_config, config.output_dir)
    save_generated_audio_metadata(outputs, samples, config.output_dir)
    return benchmark_results


def run_tts_seedtts_transcribe(config: TtsSeedttsBenchmarkConfig) -> dict:
    """Transcribe saved audio and compute WER. Server need not be running.

    Returns a dict with keys: summary, per_sample.
    """
    generation_mode = "streaming" if config.stream else "non-streaming"
    if "cuda" in config.device:
        torch.cuda.set_device(config.device)

    generated_path = os.path.join(config.output_dir, "generated.json")
    with open(generated_path) as f:
        generated: list[dict] = json.load(f)
    logger.info("Loaded %d entries from %s", len(generated), generated_path)

    asr = load_asr_model(config.lang, config.device, generation_mode)

    outputs: list[SampleOutput] = []
    for entry in tqdm(generated, desc="WER transcribe", unit="sample"):
        output = SampleOutput(
            sample_id=entry["sample_id"],
            target_text=entry["target_text"],
        )
        if not entry.get("is_success", False):
            output.error = f"Generation failed: {entry.get('error', 'unknown')}"
            outputs.append(output)
            continue

        output.latency_s = entry.get("latency_s", 0.0)
        output.audio_duration_s = entry.get("audio_duration_s", 0.0)
        output = _transcribe_and_compute_wer(
            output, entry["wav_path"], asr, config.lang, config.device
        )
        if not output.is_success:
            logger.warning(
                "Transcription failed: %s -- %s", entry["sample_id"], output.error
            )
        outputs.append(output)

    metrics = calculate_wer_metrics(outputs, config.lang)
    print_wer_summary(metrics, config.model, generation_mode)

    wer_config = {
        "model": config.model,
        "meta": config.meta,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "max_samples": config.max_samples,
        "stream": config.stream,
        "concurrency": config.concurrency,
    }
    save_wer_results(outputs, metrics, wer_config, config.output_dir)
    return {"summary": metrics, "per_sample": outputs}


def _config_from_args(args: argparse.Namespace) -> TtsSeedttsBenchmarkConfig:
    return TtsSeedttsBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        meta=args.meta,
        no_ref_audio=args.no_ref_audio,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        warmup=args.warmup,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        stream=args.stream,
        disable_tqdm=args.disable_tqdm,
        lang=args.lang,
        device=args.device,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_tts_seedtts_benchmark(config)
    print_speed_summary(
        results["summary"], config.model, concurrency=config.concurrency
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SeedTTS benchmark for S2-Pro TTS: speed and WER evaluation."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="fishaudio/s2-pro",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="seedtts_testset/en/meta.lst",
        help="Path to a meta.lst file (seed-tts-eval format).",
    )
    parser.add_argument(
        "--no-ref-audio",
        action="store_true",
        help="Skip ref audio/text from testset (TTS without voice cloning).",
    )
    parser.add_argument("--output-dir", type=str, default="results/tts_seedtts")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming SSE for TTS generation.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Language for ASR model (transcribe phase).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for ASR model (transcribe phase).",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Only synthesize audio and measure speed; skip WER transcription.",
    )
    mode.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only run ASR transcription and WER on existing output-dir.",
    )
    args = parser.parse_args()

    if args.transcribe_only:
        config = _config_from_args(args)
        run_tts_seedtts_transcribe(config)
    elif args.generate_only:
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        wait_for_service(base_url)
        asyncio.run(benchmark(args))
    else:
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        wait_for_service(base_url)
        asyncio.run(benchmark(args))
        config = _config_from_args(args)
        run_tts_seedtts_transcribe(config)


if __name__ == "__main__":
    main()
