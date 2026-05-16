# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for Qwen3-TTS Base."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.proto import StagePayload

QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS = 2048

_GENERATION_FIELDS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "subtalker_dosample",
    "subtalker_temperature",
    "subtalker_top_p",
    "subtalker_top_k",
    "max_new_tokens",
)

_CLIENT_SAMPLING_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
}


def build_qwen3_tts_state(payload: StagePayload) -> Qwen3TTSState:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = normalize_qwen3_tts_inputs(inputs)
    ref_audio, ref_text = resolve_voice_clone_reference(references, tts_params)
    language = normalize_language(tts_params.get("language") or params.get("language"))
    x_vector_only_mode = resolve_x_vector_only_mode(
        params=params,
        tts_params=tts_params,
        ref_text=ref_text,
    )

    return Qwen3TTSState(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
        non_streaming_mode=bool(params.get("non_streaming_mode", False)),
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
        seed=tts_params["seed"] if "seed" in tts_params else params.get("seed"),
    )


def normalize_qwen3_tts_inputs(inputs: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(inputs, str):
        return inputs, []
    if isinstance(inputs, dict):
        text = inputs.get("text", inputs.get("input", ""))
        references = inputs.get("references") or []
        if not isinstance(references, list):
            raise ValueError("Qwen3-TTS references must be a list")
        normalized_references = [
            dict(reference) for reference in references if isinstance(reference, dict)
        ]
        return str(text), normalized_references
    return str(inputs) if inputs is not None else "", []


def resolve_voice_clone_reference(
    references: list[dict[str, Any]],
    tts_params: dict[str, Any],
) -> tuple[Any, str | None]:
    reference = references[0] if references else {}
    ref_audio = (
        reference.get("audio_path")
        or reference.get("ref_audio")
        or reference.get("audio")
        or tts_params.get("ref_audio")
    )
    ref_text = reference.get("text") or tts_params.get("ref_text")
    if ref_audio is None:
        raise ValueError(
            "Qwen3-TTS Base requires reference audio via ref_audio or references[0].audio_path"
        )
    return ref_audio, str(ref_text) if ref_text is not None else None


def normalize_language(language: Any) -> str:
    if language is None or language == "":
        return "auto"
    return str(language)


def resolve_x_vector_only_mode(
    *,
    params: dict[str, Any],
    tts_params: dict[str, Any],
    ref_text: str | None,
) -> bool:
    for source in (params, tts_params):
        if "x_vector_only_mode" in source:
            return bool(source["x_vector_only_mode"])
    return not bool(ref_text)


def build_generation_kwargs(
    params: dict[str, Any],
    *,
    tts_params: dict[str, Any],
) -> dict[str, Any]:
    explicit_fields = tts_params.get("explicit_generation_params")
    if isinstance(explicit_fields, list):
        selected_fields = {str(field) for field in explicit_fields}
    else:
        selected_fields = set()
        for field in _GENERATION_FIELDS:
            value = params.get(field)
            if value is None:
                continue
            if field in _CLIENT_SAMPLING_DEFAULTS:
                if value == _CLIENT_SAMPLING_DEFAULTS[field]:
                    continue
            selected_fields.add(field)

    max_new_tokens = params.get("max_new_tokens")
    if max_new_tokens is None:
        max_new_tokens = QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS
    generation_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    for field in _GENERATION_FIELDS:
        if field == "max_new_tokens":
            continue
        if field in selected_fields and params.get(field) is not None:
            generation_kwargs[field] = params[field]
    return generation_kwargs
