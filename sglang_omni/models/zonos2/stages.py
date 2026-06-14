# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the ZONOS2 pipeline.

    preprocessing -> speaker_encode -> tts_engine -> vocoder

Each stage is a SimpleScheduler compute-fn over a Zonos2State dict carried in
``StagePayload.data``; the terminal vocoder merges an audio payload.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

import numpy as np
import torch

from sglang_omni.models.zonos2.payload_types import Zonos2State
from sglang_omni.models.zonos2.request_builders import (
    build_zonos2_state,
    ref_audio_to_encoder_input,
)
from sglang_omni.models.zonos2.text_frontend import (
    TTSSamplingParams,
    build_prompt_rows,
    make_speaker_slot,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

# Default quality conditioning: only trailing-silence (feature 5); rest None.
_QUALITY_FEATURES = [
    "lufs",
    "estimated_snr",
    "max_pause",
    "estimated_bandlimit_hz",
    "leading_silence_s",
    "trailing_silence_s",
]
_DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}

_SAMPLING_FIELD_NAMES = {f.name for f in dataclasses.fields(TTSSamplingParams)}


def _default_quality_list() -> list[int | None]:
    return [_DEFAULT_QUALITY_BUCKETS.get(f) for f in _QUALITY_FEATURES]


def _store(payload: StagePayload, state: Zonos2State) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id, request=payload.request, data=state.to_dict()
    )


# ---- preprocessing (text frontend) ----


def create_preprocessing_executor(
    model_path: str, *, max_concurrency: int = 16, **_: Any
) -> SimpleScheduler:
    def _preprocess(payload: StagePayload) -> StagePayload:
        state = build_zonos2_state(payload)
        rows = build_prompt_rows(
            state.text,
            language=state.language,
            quality_buckets=_default_quality_list(),
            normalize=True,
        )
        state.input_ids = rows.to(torch.long)
        return _store(payload, state)

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


# ---- speaker encode (Qwen3 voice embedding) ----


def create_speaker_encode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    speaker_cache_max_items: int = 256,
    max_concurrency: int = 4,
    **_: Any,
) -> SimpleScheduler:
    from sglang_omni.models.zonos2.speaker_encoder import SpeakerEncoder

    encoder = SpeakerEncoder(device=device, cache_max_items=speaker_cache_max_items)

    def _speaker(payload: StagePayload) -> StagePayload:
        state = Zonos2State.from_dict(payload.data)
        if state.ref_audio is not None:
            ref = ref_audio_to_encoder_input(state.ref_audio)
            state.speaker_emb = encoder.encode(ref)
            state.speaker_fingerprint = encoder.fingerprint()
        return _store(payload, state)

    return SimpleScheduler(_speaker, max_concurrency=max_concurrency)


# ---- AR decode engine ----


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    gpu_id: int | None = 0,
    dtype: str = "bfloat16",
    max_concurrency: int = 16,
    **_: Any,
) -> SimpleScheduler:
    from sglang_omni.models.zonos2.engine import Zonos2Engine

    device = f"cuda:{int(gpu_id)}" if gpu_id is not None else "cuda:0"
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    engine = Zonos2Engine.from_pretrained(model_path, device=device, dtype=torch_dtype)
    cfg = engine.cfg
    n, audio_pad = cfg.n_codebooks, cfg.audio_pad_id
    # Conditioning ids occupy the tail of text_vocab:
    #   ... speaking_rate | quality | bg(clean, noisy) | accurate_mode
    base = (
        cfg.text_vocab - cfg.speaking_rate_num_buckets - cfg.quality_num_buckets - 2 - 1
    )
    cond = cfg.speaking_rate_num_buckets + cfg.quality_num_buckets
    bg_noisy_id = base + cond + 1  # clean_speaker_background=False -> noisy
    accurate_id = base + cond + 2

    def _marker_row(tok: int) -> list[int]:
        return [audio_pad] * n + [tok]

    def _generate(payload: StagePayload) -> StagePayload:
        state = Zonos2State.from_dict(payload.data)
        rows = state.input_ids
        if not isinstance(rows, torch.Tensor):
            rows = torch.tensor(rows, dtype=torch.long)
        rows = rows.to(torch.long)

        speaker_emb = state.speaker_emb
        if speaker_emb is not None:
            if not isinstance(speaker_emb, torch.Tensor):
                speaker_emb = torch.tensor(speaker_emb, dtype=torch.float32)
            # Train-time prefix order: speaker_slot, bg, accurate, then text.
            slot = make_speaker_slot().to(torch.long)
            bg = torch.tensor([_marker_row(bg_noisy_id)], dtype=torch.long)
            acc = torch.tensor([_marker_row(accurate_id)], dtype=torch.long)
            rows = torch.cat([slot, bg, acc, rows], dim=0)

        gen = {
            k: v
            for k, v in state.generation_kwargs.items()
            if k in _SAMPLING_FIELD_NAMES
        }
        params = TTSSamplingParams(**gen)
        res = engine.generate_one(
            rows, params, speaker_emb=speaker_emb, speaker_position=0
        )
        frames = res["audio_tokens"]
        state.audio_codes = torch.tensor(frames, dtype=torch.long) if frames else None
        state.eos_frame = res["eos_frame"]
        state.prompt_tokens = int(rows.shape[0])
        state.completion_tokens = len(frames)
        return _store(payload, state)

    return SimpleScheduler(_generate, max_concurrency=max_concurrency)


# ---- vocoder (DAC 44.1 kHz, terminal) ----


def create_vocoder_executor(
    model_path: str, *, device: str = "cuda:0", **_: Any
) -> Any:
    from sglang_omni.models.zonos2.streaming_vocoder import (
        Zonos2StreamingVocoderScheduler,
        decode_batch,
        decode_to_pcm,
    )

    def _result_payload(
        payload: StagePayload, state: Zonos2State, pcm: Any
    ) -> StagePayload:
        pcm_np = (
            pcm.detach().cpu().numpy()
            if isinstance(pcm, torch.Tensor)
            else np.asarray(pcm, dtype=np.float32)
        ).reshape(-1)
        # Terminal payload is msgpack'd back to the server: emit only
        # serializable values, never the upstream state tensors.
        data: dict[str, Any] = dict(
            audio_waveform_payload(pcm_np, source_hint="ZONOS2")
        )
        data["sample_rate"] = int(state.sample_rate)
        data["modality"] = "audio"
        if state.prompt_tokens or state.completion_tokens:
            data["usage"] = {
                "prompt_tokens": int(state.prompt_tokens),
                "completion_tokens": int(state.completion_tokens),
                "total_tokens": int(state.prompt_tokens + state.completion_tokens),
            }
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=data
        )

    def _coerce_codes(state: Zonos2State) -> torch.Tensor:
        codes = state.audio_codes
        if isinstance(codes, torch.Tensor):
            return codes
        if codes is None:
            return torch.empty((0, 9), dtype=torch.long)
        return torch.as_tensor(codes, dtype=torch.long)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = Zonos2State.from_dict(payload.data)
        codes = state.audio_codes
        if codes is None or (isinstance(codes, torch.Tensor) and codes.numel() == 0):
            raise ValueError("ZONOS2 generated no audio codes")
        if not isinstance(codes, torch.Tensor):
            codes = torch.tensor(codes, dtype=torch.long)
        pcm = decode_to_pcm(codes, state.eos_frame, device=device)
        return _result_payload(payload, state, pcm)

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        states = [Zonos2State.from_dict(p.data) for p in payloads]
        pcms = decode_batch(
            [_coerce_codes(s) for s in states],
            [s.eos_frame for s in states],
            device=device,
        )
        return [_result_payload(p, s, pcm) for p, s, pcm in zip(payloads, states, pcms)]

    def _request_cost(payload: StagePayload) -> int:
        codes = Zonos2State.from_dict(payload.data).audio_codes
        if codes is None:
            return 0
        try:
            return int(codes.shape[0])
        except (AttributeError, IndexError):
            return int(len(codes))

    # note (Yue Yin): batched DAC decode coalesces non-stream requests, but joint
    # right-padding lets ConvTranspose bleed across items and changes the gate
    # output vs the single decode. Keep it opt-in (default off) until GPU
    # allclose-vs-single parity is confirmed; default path stays single-decode.
    batch_enabled = os.environ.get("ZONOS2_DAC_BATCH", "0") == "1"
    return Zonos2StreamingVocoderScheduler(
        device=device,
        compute_fn=_vocode,
        batch_compute_fn=_vocode_batch if batch_enabled else None,
        max_batch_size=16 if batch_enabled else 1,
        max_batch_wait_ms=10 if batch_enabled else 0,
        request_cost_fn=_request_cost if batch_enabled else None,
        max_batch_cost=32768 if batch_enabled else None,
    )
