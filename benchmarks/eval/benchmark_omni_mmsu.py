# SPDX-License-Identifier: Apache-2.0
"""MMSU benchmark."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.mmsu import (
    build_mmsu_results,
    compute_mmsu_metrics,
    make_mmsu_send_fn,
    print_mmsu_summary,
    save_mmsu_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


async def run(args: argparse.Namespace) -> dict:
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"
    modalities = ["text", "audio"] if args.modalities == "text+audio" else ["text"]

    samples = load_mmsu_samples(
        max_samples=args.max_samples,
        task_names=args.task_names.split(",") if args.task_names else None,
        categories=args.categories.split(",") if args.categories else None,
        seed=args.seed,
    )

    save_audio_dir = None
    if args.save_audio and args.output_dir:
        save_audio_dir = os.path.join(args.output_dir, "audio")
        os.makedirs(save_audio_dir, exist_ok=True)

    send_fn_kwargs = dict(
        modalities=modalities,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        save_audio_dir=save_audio_dir,
    )
    if args.prompt:
        send_fn_kwargs["prompt"] = args.prompt
    send_fn = make_mmsu_send_fn(args.model, api_url, **send_fn_kwargs)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=args.max_concurrency,
            request_rate=args.request_rate,
            warmup=args.warmup,
            disable_tqdm=args.disable_tqdm,
        )
    )
    request_results = await runner.run(samples, send_fn)

    results = build_mmsu_results(request_results, samples, modalities)
    metrics = compute_mmsu_metrics(results)
    speed = compute_speed_metrics(request_results, wall_clock_s=runner.wall_clock_s)
    audio_mode = "audio" in modalities

    print_mmsu_summary(metrics, args.model, speed_metrics=speed if audio_mode else None)

    if args.output_dir:
        save_mmsu_results(
            results,
            metrics,
            {
                "model": args.model,
                "base_url": base_url,
                "modalities": modalities,
                "max_samples": args.max_samples,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            args.output_dir,
            speed_metrics=speed if audio_mode else None,
        )

    return {"accuracy": metrics, "speed": speed}


def main() -> None:
    p = argparse.ArgumentParser(description="MMSU benchmark.")
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", type=str, default="qwen3-omni")
    p.add_argument("--modalities", choices=["text", "text+audio"], default="text")
    p.add_argument("--output-dir", type=str, default="results/mmsu")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--task-names", type=str, default=None)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--request-rate", type=float, default=float("inf"))
    p.add_argument("--save-audio", action="store_true")
    p.add_argument("--disable-tqdm", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args()
    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
