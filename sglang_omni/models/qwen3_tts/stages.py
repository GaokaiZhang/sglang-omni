# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Qwen3-TTS Base pipeline."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.request_builders import build_qwen3_tts_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)

_QWEN_TTS_INSTALL_HINT = (
    "Qwen3-TTS support requires the official `qwen-tts` package. "
    "Install `qwen-tts==0.1.1` and its Transformers 4.57.3 requirement "
    "in the serving environment before launching Qwen3-TTS."
)


def load_state(payload: StagePayload) -> Qwen3TTSState:
    return Qwen3TTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: Qwen3TTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_qwen3_tts_model(
    model_path: str,
    *,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(_QWEN_TTS_INSTALL_HINT) from exc

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    kwargs: dict[str, Any] = {
        "device_map": device,
        "dtype": torch_dtype,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info(f"Loading Qwen3-TTS model from {checkpoint_dir} on {device}")
    return Qwen3TTSModel.from_pretrained(checkpoint_dir, **kwargs)


def _load_qwen3_tts_tokenizer(
    model_path: str,
    *,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ImportError as exc:
        raise RuntimeError(_QWEN_TTS_INSTALL_HINT) from exc

    checkpoint_dir = _resolve_checkpoint(model_path)
    tokenizer_path = os.path.join(checkpoint_dir, "speech_tokenizer")
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    kwargs: dict[str, Any] = {
        "device_map": device,
        "dtype": torch_dtype,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    logger.info(f"Loading Qwen3-TTS speech tokenizer from {tokenizer_path} on {device}")
    return Qwen3TTSTokenizer.from_pretrained(tokenizer_path, **kwargs)


def _audio_to_list(audio: Any) -> list[float]:
    if isinstance(audio, torch.Tensor):
        return audio.detach().float().cpu().flatten().tolist()
    try:
        import numpy as np

        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        return array.tolist()
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported Qwen3-TTS audio output type: {type(audio)}"
        ) from exc


def _build_usage(state: Qwen3TTSState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "total_tokens": state.prompt_tokens + state.completion_tokens,
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


def _run_voice_clone_generation(model: Any, state: Qwen3TTSState) -> torch.Tensor:
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=state.ref_audio,
        ref_text=state.ref_text,
        x_vector_only_mode=state.x_vector_only_mode,
    )
    if len(prompt_items) != 1:
        raise ValueError("Qwen3-TTS expected exactly one voice-clone prompt")

    prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)
    input_ids = model._tokenize_texts([model._build_assistant_text(state.text)])

    ref_text = prompt_items[0].ref_text
    if ref_text:
        ref_ids = [model._tokenize_texts([model._build_ref_text(ref_text)])[0]]
    else:
        ref_ids = [None]

    gen_kwargs = model._merge_generate_kwargs(**state.generation_kwargs)
    talker_codes, _ = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=prompt,
        languages=[state.language],
        non_streaming_mode=state.non_streaming_mode,
        **gen_kwargs,
    )
    if not talker_codes:
        raise RuntimeError("Qwen3-TTS did not return any codec tokens")

    codes = talker_codes[0]
    ref_codes = prompt.get("ref_code")
    if ref_codes and ref_codes[0] is not None:
        ref_code = ref_codes[0].to(codes.device)
        state.ref_code_len = int(ref_code.shape[0])
        codes = torch.cat([ref_code, codes], dim=0)
    else:
        state.ref_code_len = 0
    state.completion_tokens = max(int(codes.shape[0]) - state.ref_code_len, 0)
    state.prompt_tokens = state.ref_code_len
    return codes


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    del model_path

    def _preprocess(payload: StagePayload) -> StagePayload:
        state = build_qwen3_tts_state(payload)
        return store_state(payload, state)

    return SimpleScheduler(_preprocess)


def create_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> SimpleScheduler:
    model = _load_qwen3_tts_model(
        model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    def _generate(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        if state.seed is not None:
            torch.manual_seed(int(state.seed))

        start = time.perf_counter()
        codes = _run_voice_clone_generation(model, state)
        state.engine_time_s = time.perf_counter() - start
        state.audio_codes = codes
        return store_state(payload, state)

    return SimpleScheduler(_generate)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> SimpleScheduler:
    tokenizer = _load_qwen3_tts_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    def _vocode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError("Qwen3-TTS vocoder requires audio_codes from tts_engine")

        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        wavs, sample_rate = tokenizer.decode([{"audio_codes": codes}])
        if not wavs:
            raise RuntimeError("Qwen3-TTS speech tokenizer did not return audio")

        wav = wavs[0]
        if state.ref_code_len:
            total_len = int(codes.shape[0])
            cut = int(state.ref_code_len / max(total_len, 1) * wav.shape[0])
            wav = wav[cut:]
        state.audio_samples = _audio_to_list(wav)
        state.sample_rate = int(sample_rate)

        payload = store_state(payload, state)
        audio = state.audio_samples or []
        payload.data["audio_data"] = audio
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    return SimpleScheduler(_vocode)
