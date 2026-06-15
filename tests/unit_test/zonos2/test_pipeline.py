# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang_omni.models.zonos2 import sglang_stages
from sglang_omni.models.zonos2.config import Zonos2PipelineConfig


def test_zonos2_streaming_pipeline_routes_chunks_to_vocoder() -> None:
    config = Zonos2PipelineConfig(model_path="fake-model")
    stages_by_name = {stage.name: stage for stage in config.stages}

    assert stages_by_name["tts_engine"].stream_to == ["vocoder"]
    assert stages_by_name["vocoder"].can_accept_stream_before_payload is True


def test_zonos2_tts_engine_disables_chunked_prefill(monkeypatch) -> None:
    """The per-frame feedback/EOS state machine has no rollback, so the engine
    must disable chunked prefill regardless of the ServerArgs default."""
    captured: dict[str, object] = {}

    def fake_build_sglang_server_args(checkpoint_dir, context_length, **overrides):
        # Start from a non-zero default to prove the factory overrides it.
        server_args = SimpleNamespace(
            disable_cuda_graph=overrides["disable_cuda_graph"],
            chunked_prefill_size=8192,
        )
        captured["context_length"] = context_length
        captured["server_args"] = server_args
        return server_args

    def fake_create_sglang_infrastructure(
        server_args, gpu_id, model_arch_override=None
    ):
        captured["model_arch_override"] = model_arch_override
        captured["chunked_at_infra"] = server_args.chunked_prefill_size
        model_runner = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(dim=8)),
            init_device_graphs=lambda: captured.__setitem__("graphs_captured", True),
        )
        worker = SimpleNamespace(model_runner=model_runner)
        return (worker, object(), object(), object(), object(), object(), object())

    class FakeOutputProcessor:
        def __init__(self, **kwargs) -> None:
            pass

    class FakeModelRunner:
        def __init__(self, model_worker, output_proc, **kwargs) -> None:
            pass

        def set_stream_outbox(self, outbox) -> None:
            self._outbox = outbox

    class FakeScheduler:
        def __init__(self, **kwargs) -> None:
            self.outbox = object()

    monkeypatch.setattr(
        sglang_stages,
        "load_zonos2_pretrained_config",
        lambda model_path: SimpleNamespace(max_seqlen=6144),
    )
    monkeypatch.setattr(sglang_stages, "_register_zonos2_autoconfig", lambda: None)
    monkeypatch.setattr(
        sglang_stages, "_build_config_shim", lambda model_path, cfg: "/tmp/shim"
    )
    # Engine imports these lazily from their source modules, so patch there.
    monkeypatch.setattr(
        "sglang_omni.scheduling.sglang_backend.build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        "sglang_omni.scheduling.bootstrap.create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )
    monkeypatch.setattr(
        "sglang_omni.scheduling.sglang_backend.SGLangOutputProcessor",
        FakeOutputProcessor,
    )
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler.OmniScheduler", FakeScheduler
    )
    monkeypatch.setattr(
        "sglang_omni.models.zonos2.model_runner.Zonos2ModelRunner", FakeModelRunner
    )
    monkeypatch.setattr(
        "sglang_omni.models.zonos2.sglang_request_builders."
        "make_zonos2_scheduler_adapters",
        lambda model: (None, None),
    )

    sglang_stages.create_sglang_omni_tts_engine_executor("fake-zonos2")

    assert captured["context_length"] == 6144
    assert captured["model_arch_override"] == "Zonos2SGLangModel"
    # Disabled before infra/graph capture builds the scheduler.
    assert captured["chunked_at_infra"] == 0
    assert captured["server_args"].chunked_prefill_size == 0
