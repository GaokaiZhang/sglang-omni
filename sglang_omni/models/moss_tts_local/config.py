# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Local (v1.5)."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.moss_tts_local"


class MossTTSLocalPipelineConfig(PipelineConfig):
    """MOSS-TTS Local pipeline: preprocessing -> AR engine -> vocoder.

    Resolves OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5, whose checkpoint
    declares ``architectures: ["MossTTSLocalModel"]``, so serving by model id
    works without an explicit config file. The AR engine runs on GPU 0; the
    48 kHz stereo v2 codec (reference encode + vocoder) auto-places on the
    second visible GPU when one exists — its ~1B-param encoder otherwise
    competes with the AR decode loop — and falls back to GPU 0 on
    single-GPU hosts.
    """

    architecture: ClassVar[str] = "MossTTSLocalModel"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSLocal",
        "MossTTSLocalForConditionalGeneration",
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = MossTTSLocalPipelineConfig
