# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.request_builders import (
    Qwen3TTSSGLangRequestData,
    apply_sglang_qwen3_tts_result,
    build_embedding_cache_key_ids,
    build_qwen3_tts_state,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(
    *,
    inputs,
    params: dict | None = None,
    tts_params: dict | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="req-qwen3-tts",
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def test_qwen3_tts_config_and_registry_contracts() -> None:
    config = Qwen3TTSPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.stages[1].factory.endswith("create_sglang_tts_engine_executor")
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3TTSForConditionalGeneration")
        is Qwen3TTSPipelineConfig
    )


def test_qwen3_tts_state_round_trip_preserves_request_fields() -> None:
    state = Qwen3TTSState(
        text="hello",
        language="en",
        ref_audio="voice.wav",
        ref_text="reference",
        generation_kwargs={"max_new_tokens": 128, "temperature": 0.7},
        seed=123,
        audio_codes=[[1, 2], [3, 4]],
        ref_code_len=1,
        audio_samples=[0.0, 0.1],
        sample_rate=24000,
    )
    restored = Qwen3TTSState.from_dict(state.to_dict())
    assert restored.text == "hello"
    assert restored.language == "en"
    assert restored.ref_audio == "voice.wav"
    assert restored.ref_text == "reference"
    assert restored.generation_kwargs["max_new_tokens"] == 128
    assert restored.audio_codes == [[1, 2], [3, 4]]
    assert restored.ref_code_len == 1
    assert restored.audio_samples == [0.0, 0.1]


def test_qwen3_tts_maps_references_and_keeps_upstream_sampling_defaults() -> None:
    payload = make_payload(
        inputs={
            "text": "target",
            "references": [{"audio_path": "voice.wav", "text": "reference"}],
        },
        params={
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
        },
    )

    state = build_qwen3_tts_state(payload)

    assert state.text == "target"
    assert state.language == "auto"
    assert state.ref_audio == "voice.wav"
    assert state.ref_text == "reference"
    assert state.x_vector_only_mode is False
    assert state.generation_kwargs == {"max_new_tokens": 2048}


def test_qwen3_tts_ignores_client_sampling_defaults() -> None:
    payload = make_payload(
        inputs="target",
        params={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
        },
        tts_params={"ref_audio": "voice.wav", "ref_text": "reference"},
    )

    state = build_qwen3_tts_state(payload)

    assert state.generation_kwargs == {"max_new_tokens": 2048}


def test_qwen3_tts_embedding_cache_keys_are_stable_and_content_based() -> None:
    """Protects radix-cache keys for Qwen requests that prefill with embeddings."""
    embeds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    same = embeds.clone()
    different_same_length = torch.tensor([[1.0, 2.0], [3.0, 5.0]])

    assert build_embedding_cache_key_ids(embeds) == build_embedding_cache_key_ids(same)
    assert build_embedding_cache_key_ids(embeds) != build_embedding_cache_key_ids(
        different_same_length
    )


def test_qwen3_tts_maps_ref_audio_form_and_explicit_sampling() -> None:
    payload = make_payload(
        inputs="target",
        params={"temperature": 0.7, "top_k": 40, "max_new_tokens": 256},
        tts_params={
            "ref_audio": "voice.wav",
            "ref_text": "reference",
            "language": "en",
        },
    )

    state = build_qwen3_tts_state(payload)

    assert state.text == "target"
    assert state.language == "en"
    assert state.ref_audio == "voice.wav"
    assert state.generation_kwargs == {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_k": 40,
    }


def test_qwen3_tts_uses_x_vector_only_when_ref_text_is_missing() -> None:
    payload = make_payload(
        inputs={"text": "target", "references": [{"audio_path": "voice.wav"}]},
    )

    state = build_qwen3_tts_state(payload)

    assert state.ref_audio == "voice.wav"
    assert state.ref_text is None
    assert state.x_vector_only_mode is True


def test_qwen3_tts_rejects_missing_reference_audio() -> None:
    payload = make_payload(inputs="target")

    with pytest.raises(ValueError, match="requires reference audio"):
        build_qwen3_tts_state(payload)


def test_qwen3_tts_predictor_codec_embeddings_use_talker_hidden_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects 1.7B loading where talker and predictor hidden sizes differ."""
    from torch import nn

    from sglang_omni.models.qwen3_tts import sglang_model

    class FakeDecoderLayer(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeReplicatedLinear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            bias: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        def forward(self, x):
            return self.linear(x), None

    monkeypatch.setattr(sglang_model, "Qwen3TTSTalkerDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(sglang_model, "ReplicatedLinear", FakeReplicatedLinear)
    monkeypatch.setattr(
        sglang_model,
        "RMSNorm",
        lambda hidden_size, eps=1e-6: nn.LayerNorm(hidden_size, eps=eps),
    )

    predictor_config = SimpleNamespace(
        vocab_size=2048,
        hidden_size=1024,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
    )
    talker_config = SimpleNamespace(
        hidden_size=2048,
        num_code_groups=16,
        code_predictor_config=predictor_config,
    )

    predictor = sglang_model.Qwen3TTSCodePredictor(talker_config)

    assert predictor.model.codec_embedding[0].weight.shape == (2048, 2048)
    assert predictor.small_to_mtp_projection.weight.shape == (1024, 2048)


def test_qwen3_tts_vocoder_batches_decode_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects Qwen3-TTS vocoder throughput from regressing to serial decode."""
    from sglang_omni.models.qwen3_tts import stages

    decode_batch_sizes: list[int] = []

    class FakeTokenizer:
        def decode(self, encoded):
            decode_batch_sizes.append(len(encoded))
            return [
                torch.arange(6, dtype=torch.float32),
                torch.arange(8, dtype=torch.float32),
            ], 24000

    monkeypatch.setattr(
        stages,
        "_load_qwen3_tts_tokenizer",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        max_batch_size=2,
        max_batch_wait_ms=3,
    )
    first = make_payload(inputs="first")
    first.data = Qwen3TTSState(
        audio_codes=torch.tensor([[1, 2], [3, 4]]),
        ref_code_len=1,
    ).to_dict()
    second = make_payload(inputs="second")
    second.data = Qwen3TTSState(
        audio_codes=torch.tensor([[5, 6], [7, 8]]),
    ).to_dict()

    results = scheduler._batch_fn([first, second])

    assert scheduler._max_batch_size == 2
    assert scheduler._max_batch_wait_s == pytest.approx(0.003)
    assert decode_batch_sizes == [2]
    assert results[0].data["sample_rate"] == 24000
    assert results[0].data["audio_data"] == [3.0, 4.0, 5.0]
    assert "audio_codes" not in results[0].data
    assert results[1].data["audio_data"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_qwen3_tts_result_adapter_keeps_code_handoff_tensor_native() -> None:
    """Avoids list serialization between the AR stage and vocoder stage."""
    payload = make_payload(inputs="target")
    data = Qwen3TTSSGLangRequestData(
        req=SimpleNamespace(output_ids=[]),
        output_codes=[torch.tensor([1, 2]), torch.tensor([3, 4])],
        ref_code=torch.tensor([[9, 9]]),
        ref_code_len=1,
        stage_payload=payload,
    )

    result = apply_sglang_qwen3_tts_result(payload, data)

    assert isinstance(result.data["audio_codes"], torch.Tensor)
    assert result.data["audio_codes"].tolist() == [[9, 9], [1, 2], [3, 4]]
    assert result.data["completion_tokens"] == 2


def test_qwen3_tts_engine_reenables_cuda_graph_after_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects the SGLang decode path from silently falling back to eager mode."""
    from transformers import AutoProcessor

    from sglang_omni.models.qwen3_tts import model_runner as model_runner_mod
    from sglang_omni.models.qwen3_tts import stages
    from sglang_omni.scheduling import bootstrap as bootstrap_mod
    from sglang_omni.scheduling import omni_scheduler as scheduler_mod
    from sglang_omni.scheduling import sglang_backend

    build_kwargs: dict = {}
    infrastructure_saw_graph_disabled: list[bool] = []
    init_graph_saw_graph_enabled: list[bool] = []

    class FakeModel:
        def load_speech_tokenizer(self, tokenizer) -> None:
            self.speech_tokenizer = tokenizer

    class FakeSGLangRunner:
        def __init__(self, server_args) -> None:
            self.server_args = server_args
            self.model = FakeModel()

        def init_device_graphs(self) -> None:
            init_graph_saw_graph_enabled.append(
                not bool(self.server_args.disable_cuda_graph)
            )

    class FakeWorker:
        def __init__(self, server_args) -> None:
            self.model_runner = FakeSGLangRunner(server_args)

    class FakeQwen3TTSModel:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    qwen_tts_module = types.ModuleType("qwen_tts")
    qwen_tts_module.Qwen3TTSModel = FakeQwen3TTSModel
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts_module)

    monkeypatch.setattr(stages, "_register_qwen3_tts_hf_config", lambda: None)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages,
        "_load_qwen3_tts_tokenizer",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        stages,
        "make_qwen3_tts_scheduler_adapters",
        lambda **kwargs: (lambda payload: payload, lambda data: data),
    )

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        del model_path, context_length
        build_kwargs.update(kwargs)
        return SimpleNamespace(
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            disable_overlap_schedule=kwargs["disable_overlap_schedule"],
            page_size=1,
            chunked_prefill_size=0,
            max_prefill_tokens=kwargs["max_prefill_tokens"],
            max_running_requests=kwargs["max_running_requests"],
        )

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id, kwargs
        infrastructure_saw_graph_disabled.append(bool(server_args.disable_cuda_graph))
        return (
            FakeWorker(server_args),
            object(),
            object(),
            object(),
            object(),
            object(),
            SimpleNamespace(),
        )

    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        bootstrap_mod,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        model_runner_mod,
        "Qwen3TTSModelRunner",
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )
    monkeypatch.setattr(
        scheduler_mod,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    scheduler = stages.create_sglang_tts_engine_executor("model", device="cuda:0")

    assert build_kwargs["disable_cuda_graph"] is False
    assert build_kwargs["sampling_backend"] == "pytorch"
    assert infrastructure_saw_graph_disabled == [True]
    assert init_graph_saw_graph_enabled == [True]
    assert scheduler.server_args.disable_cuda_graph is False
