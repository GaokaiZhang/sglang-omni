# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.tasks.tts import (
    MOSS_TTS_TOKEN_COUNT_AUTO,
    _build_tts_payload,
    estimate_moss_tts_duration_tokens,
)
from sglang_omni.models.moss_tts.codec import split_moss_audio_segments
from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import (
    _INF_DELAY,
    build_moss_tts_state,
    build_row_cache_key_ids,
    build_sglang_moss_tts_request,
    clear_moss_tts_preprocessing_context,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput


def install_fake_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReq:
        def __init__(
            self,
            *,
            rid,
            origin_input_text,
            origin_input_ids,
            sampling_params,
            eos_token_ids=None,
            vocab_size=None,
            **kwargs,
        ) -> None:
            del kwargs
            self.rid = rid
            self.origin_input_text = origin_input_text
            self.origin_input_ids = origin_input_ids
            self.sampling_params = sampling_params
            self.eos_token_ids = eos_token_ids
            self.vocab_size = vocab_size
            self.output_ids = []
            self.prefix_indices = []
            self.extend_input_len = len(origin_input_ids)

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def normalize(self, tokenizer) -> None:
            del tokenizer

        def verify(self, vocab_size) -> None:
            self.vocab_size = vocab_size

    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.schedule_batch": types.ModuleType(
            "sglang.srt.managers.schedule_batch"
        ),
        "sglang.srt.sampling": types.ModuleType("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_params": types.ModuleType(
            "sglang.srt.sampling.sampling_params"
        ),
    }
    for name in ("sglang", "sglang.srt", "sglang.srt.managers", "sglang.srt.sampling"):
        modules[name].__path__ = []
    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].managers = modules["sglang.srt.managers"]
    modules["sglang.srt"].sampling = modules["sglang.srt.sampling"]
    modules["sglang.srt.managers"].schedule_batch = modules[
        "sglang.srt.managers.schedule_batch"
    ]
    modules["sglang.srt.sampling"].sampling_params = modules[
        "sglang.srt.sampling.sampling_params"
    ]
    modules["sglang.srt.managers.schedule_batch"].Req = FakeReq
    modules["sglang.srt.sampling.sampling_params"].SamplingParams = FakeSamplingParams
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def make_payload(
    *,
    inputs,
    params: dict | None = None,
    tts_params: dict | None = None,
    request_id: str = "req-moss",
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def test_moss_tts_config_and_registry_contracts() -> None:
    config = MossTTSPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("MossTTSDelayModel")
        is MossTTSPipelineConfig
    )


def test_moss_tts_state_round_trip_keeps_tensors_native() -> None:
    codes = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    state = MossTTSState(
        text="hello",
        ref_audio="ref.wav",
        ref_text="reference",
        language="en",
        instructions="warm",
        token_count=180,
        generation_kwargs={"max_new_tokens": 64},
        delayed_audio_codes=codes,
        assistant_start_length=2,
    )
    restored = MossTTSState.from_dict(state.to_dict())

    assert restored.text == "hello"
    assert restored.ref_audio == "ref.wav"
    assert restored.ref_text == "reference"
    assert restored.language == "en"
    assert restored.instructions == "warm"
    assert restored.token_count == 180
    assert torch.equal(restored.delayed_audio_codes, codes)
    assert restored.assistant_start_length == 2


def test_moss_tts_maps_references_token_count_and_deterministic_defaults() -> None:
    payload = make_payload(
        inputs={
            "text": "${token:120}hello [pause 0.5s] ni3 hao3 /hello/",
            "references": [{"audio_path": "voice.wav", "text": "reference"}],
        },
        params={"temperature": 0.8, "top_p": 0.8, "top_k": 30},
        tts_params={"language": "en"},
    )

    state = build_moss_tts_state(payload)

    assert state.text.startswith("${token:120}hello")
    assert state.ref_audio == "voice.wav"
    assert state.ref_text == "reference"
    assert state.language == "en"
    assert state.token_count == 120
    assert state.generation_kwargs["max_new_tokens"] == 4096
    assert state.generation_kwargs["text_temperature"] == 0.0
    assert state.generation_kwargs["audio_temperature"] == 0.0


def test_moss_tts_benchmark_auto_token_count_uses_openmoss_estimate() -> None:
    sample = SampleInput(
        sample_id="sample-1",
        ref_text="reference",
        ref_audio="ref.wav",
        target_text="hello world",
    )

    payload = _build_tts_payload(
        sample,
        "OpenMOSS-Team/MOSS-TTS-v1.5",
        token_count=MOSS_TTS_TOKEN_COUNT_AUTO,
    )

    assert payload["token_count"] == estimate_moss_tts_duration_tokens("hello world")
    assert payload["token_count"] == 32


def test_moss_tts_preserves_explicit_standard_sampling_values() -> None:
    payload = make_payload(
        inputs="hello",
        params={"temperature": 0.7, "top_p": 0.9, "top_k": 40},
        tts_params={
            "explicit_generation_params": ["temperature", "top_p", "top_k"],
            "token_count": 42,
        },
    )

    state = build_moss_tts_state(payload)

    assert state.token_count == 42
    assert state.generation_kwargs["text_temperature"] == 0.7
    assert state.generation_kwargs["audio_temperature"] == 0.7
    assert state.generation_kwargs["text_top_p"] == 0.9
    assert state.generation_kwargs["audio_top_k"] == 40


def test_moss_row_cache_keys_are_content_based() -> None:
    rows = torch.tensor([[1, 1024, 1024], [2, 1024, 1024]], dtype=torch.long)
    same = rows.clone()
    different = torch.tensor([[1, 1024, 1024], [2, 1024, 1023]], dtype=torch.long)

    assert build_row_cache_key_ids(rows) == build_row_cache_key_ids(same)
    assert build_row_cache_key_ids(rows) != build_row_cache_key_ids(different)


def test_moss_preprocess_and_sglang_request_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)

    class FakeProcessor:
        def __init__(self) -> None:
            self.message_kwargs = None

        def build_user_message(self, **kwargs):
            self.message_kwargs = kwargs
            return {"role": "user", **kwargs}

        def __call__(self, conversations, mode):
            assert mode == "generation"
            assert conversations[0][0]["text"] == "hello"
            return {
                "input_ids": torch.tensor(
                    [
                        [
                            [1, 1024, 1024],
                            [151644, 1024, 1024],
                            [198, 1024, 1024],
                        ]
                    ],
                    dtype=torch.long,
                )
            }

    processor = FakeProcessor()
    payload = make_payload(
        inputs="hello",
        params={"max_new_tokens": 12},
        tts_params={"token_count": 80, "language": "en"},
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size_list=[200000, 1025, 1025],
            im_end_token_id=151645,
            im_start_token_id=151644,
            audio_start_token_id=151652,
            audio_assistant_gen_slot_token_id=151656,
        )
    )

    try:
        set_moss_tts_preprocessing_context(processor=processor)
        prepared_payload = preprocess_moss_tts_payload(payload)
        data = build_sglang_moss_tts_request(prepared_payload, model=model)
    finally:
        clear_moss_tts_preprocessing_context()

    assert processor.message_kwargs["tokens"] == 80
    assert processor.message_kwargs["language"] == "en"
    assert data.req._input_embeds_are_projected is True
    assert data.input_embeds_are_projected is True
    assert data.max_new_tokens == 12
    assert data.prompt_rows.shape == (3, 3)
    assert data.state.assistant_start_length == 0
    assert data.req.sampling_params.stop_token_ids == [151645]


def test_moss_delay_runner_samples_audio_and_appends_feedback() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    cfg = SimpleNamespace(
        pad_token_id=0,
        audio_start_token_id=10,
        audio_end_token_id=11,
        audio_assistant_gen_slot_token_id=12,
        audio_assistant_delay_slot_token_id=13,
        audio_pad_code=4,
        im_end_token_id=14,
    )
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=cfg,
        hidden_size=3,
        device=torch.device("cpu"),
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32)[:, :3],
    )
    data = SimpleNamespace(
        audio_length=0,
        delayed_length=_INF_DELAY,
        is_audio=False,
        generation_steps=0,
        text_temperature=0.0,
        text_top_p=1.0,
        text_top_k=-1,
        audio_temperature=0.0,
        audio_top_p=1.0,
        audio_top_k=-1,
        audio_repetition_penalty=1.0,
        prompt_rows=None,
        output_rows=[],
        pending_feedback_queue=[],
    )
    text_logits = torch.full((1, 20), -100.0)
    text_logits[0, cfg.audio_start_token_id] = 10.0
    audio0_logits = torch.tensor([[-1.0, 0.0, 5.0, 1.0, -100.0]])
    audio1_logits = torch.tensor([[-1.0, 6.0, 0.0, 1.0, -100.0]])

    text_token, audio_tokens = runner._sample_next_row(
        [text_logits, audio0_logits, audio1_logits],
        row_idx=0,
        data=data,
        n_vq=2,
    )

    assert text_token == cfg.audio_start_token_id
    assert audio_tokens.tolist() == [cfg.audio_pad_code, cfg.audio_pad_code]
    assert data.is_audio is True
    assert data.audio_length == 1

    data.generation_steps = 1
    text_logits[0] = -100.0
    text_logits[0, cfg.audio_assistant_gen_slot_token_id] = 10.0
    text_token, audio_tokens = runner._sample_next_row(
        [text_logits, audio0_logits, audio1_logits],
        row_idx=0,
        data=data,
        n_vq=2,
    )

    assert text_token == cfg.audio_assistant_gen_slot_token_id
    assert audio_tokens.tolist() == [2, cfg.audio_pad_code]
    assert data.audio_length == 2


def test_moss_post_process_outputs_skips_im_end() -> None:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(im_end_token_id=14))
    runner._pending_rows = torch.tensor([[12, 2, 4], [14, 4, 4]], dtype=torch.long)
    runner._pending_embeds = torch.ones((2, 3))
    requests = [
        SimpleNamespace(
            request_id="active",
            data=SimpleNamespace(output_rows=[], pending_feedback_queue=[]),
        ),
        SimpleNamespace(
            request_id="eos",
            data=SimpleNamespace(output_rows=[], pending_feedback_queue=[]),
        ),
    ]

    runner.post_process_outputs(
        object(),
        SimpleNamespace(requests=requests),
        {
            "active": RequestOutput("active", data=12),
            "eos": RequestOutput("eos", data=14),
        },
    )

    assert [row.tolist() for row in requests[0].data.output_rows] == [[12, 2, 4]]
    assert len(requests[0].data.pending_feedback_queue) == 1
    assert requests[1].data.output_rows == []
    assert requests[1].data.pending_feedback_queue == []


def test_moss_delay_codec_splits_non_pad_segments() -> None:
    delayed = torch.tensor(
        [
            [1, 1024],
            [2, 3],
            [1024, 4],
            [1024, 1024],
        ],
        dtype=torch.long,
    )

    segments = split_moss_audio_segments(delayed, audio_pad_code=1024)

    assert [segment.tolist() for segment in segments] == [[[1, 3], [2, 4]]]
