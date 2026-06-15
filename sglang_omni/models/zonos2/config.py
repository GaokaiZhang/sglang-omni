# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 pipeline configuration.

Four-stage pipeline: preprocessing -> speaker_encode -> tts_engine -> vocoder.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.zonos2"


def _stages(*, codec_device: str, speaker_device: str) -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={
                "ref_audio_cache": True,
                "ref_audio_cache_max_items": 256,
                "ref_audio_cache_max_bytes": 64 * 1024 * 1024,
                "tts_norm": True,
                "tts_norm_cache_dir": None,
            },
            gpu=0,
            next="speaker_encode",
        ),
        StageConfig(
            name="speaker_encode",
            process="pipeline",
            factory=f"{_PKG}.stages.create_speaker_encode_executor",
            factory_args={
                "device": speaker_device,
                "speaker_cache": True,
                "speaker_cache_max_items": 256,
                "spk_compile": False,
            },
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.sglang_stages.create_sglang_omni_tts_engine_executor",
            factory_args={
                "gpu_id": 0,
                "dtype": "bfloat16",
                "fp8": False,
                "frame_graph": False,
                "compile_sampler": False,
                "async_decode": False,
            },
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={
                "device": codec_device,
                "dac_batch": False,
                "vocoder_warmup": False,
            },
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class Zonos2PipelineConfig(PipelineConfig):
    """Single-GPU colocated default."""

    architecture: ClassVar[str] = "Zonos2ForCausalLM"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "Zonos2",
        "Zonos2Model",
        "ZONOS2",
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", speaker_device="cuda:0")
    )


class Zonos2MultiGPUPipelineConfig(Zonos2PipelineConfig):
    """Offload codec + speaker encoder to cuda:1, leaving the AR engine alone on cuda:0."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:1", speaker_device="cuda:1")
    )


EntryClass = Zonos2PipelineConfig

Variants = {
    "default": Zonos2PipelineConfig,
    "multi_gpu": Zonos2MultiGPUPipelineConfig,
}
