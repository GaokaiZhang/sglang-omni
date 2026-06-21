# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the ZONOS2 pipeline.

    preprocessing -> speaker_encode -> tts_engine -> vocoder

Each stage is a SimpleScheduler compute-fn over a Zonos2State dict carried in
``StagePayload.data``; the terminal vocoder merges an audio payload.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from sglang_omni.models.zonos2.hf_config import (
    Zonos2Config,
    load_zonos2_pretrained_config,
)
from sglang_omni.models.zonos2.payload_types import N_CODEBOOKS, Zonos2State
from sglang_omni.models.zonos2.request_builders import (
    build_zonos2_state,
    ref_audio_to_encoder_input,
)
from sglang_omni.models.zonos2.text_frontend import (
    build_prompt_rows,
    configure_tts_norm_cache_root,
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


def _default_quality_list() -> list[int | None]:
    return [_DEFAULT_QUALITY_BUCKETS.get(f) for f in _QUALITY_FEATURES]


def _store(payload: StagePayload, state: Zonos2State) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id, request=payload.request, data=state.to_dict()
    )


# ---- preprocessing (text frontend) ----


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 16,
    tts_norm: bool = True,
    tts_norm_cache_dir: str | None = None,
    **_: Any,
) -> SimpleScheduler:
    configure_tts_norm_cache_root(tts_norm_cache_dir)

    def _preprocess(payload: StagePayload) -> StagePayload:
        state = build_zonos2_state(payload)
        rows = build_prompt_rows(
            state.text,
            language=state.language,
            quality_buckets=_default_quality_list(),
            normalize=tts_norm,
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
    spk_compile: bool = False,
    **_: Any,
) -> SimpleScheduler:
    from sglang_omni.models.zonos2.speaker_encoder import SpeakerEncoder

    encoder = SpeakerEncoder(
        device=device,
        cache_max_items=speaker_cache_max_items,
        compile_forward=spk_compile,
    )

    def _speaker(payload: StagePayload) -> StagePayload:
        state = Zonos2State.from_dict(payload.data)
        if state.ref_audio is not None:
            ref = ref_audio_to_encoder_input(state.ref_audio)
            state.speaker_emb = encoder.encode(ref)
            state.speaker_fingerprint = encoder.fingerprint()
        return _store(payload, state)

    return SimpleScheduler(_speaker, max_concurrency=max_concurrency)


# ---- vocoder (DAC 44.1 kHz, terminal) ----


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dac_batch: bool = False,
    vocoder_warmup: bool = False,
    **_: Any,
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
    batch_enabled = dac_batch
    scheduler = Zonos2StreamingVocoderScheduler(
        device=device,
        compute_fn=_vocode,
        batch_compute_fn=_vocode_batch if batch_enabled else None,
        max_batch_size=16 if batch_enabled else 1,
        max_batch_wait_ms=10 if batch_enabled else 0,
        request_cost_fn=_request_cost if batch_enabled else None,
        max_batch_cost=32768 if batch_enabled else None,
    )
    # note (Yue Yin): opt-in warmup moves the one-time DAC load + conv autotune
    # off the first request's critical path to startup. N_CODEBOOKS+1 rows keep
    # 2 aligned frames after the de-shear so a real decode warms the kernels.
    # Never fatal -- the lazy load still happens on first decode if this fails.
    if vocoder_warmup:
        try:
            decode_to_pcm(
                torch.zeros((N_CODEBOOKS + 1, N_CODEBOOKS), dtype=torch.long),
                device=device,
            )
            logger.info("ZONOS2 DAC vocoder warmed up at startup")
        except Exception:  # noqa: BLE001 - warmup must never block server start
            logger.warning("ZONOS2 vocoder warmup failed", exc_info=True)
    return scheduler


# ---- AR engine stage (OmniScheduler-backed ZONOS2 backbone) ----

def _build_config_shim(model_path: str, cfg: Zonos2Config) -> str:
    shim = tempfile.mkdtemp(prefix="zonos2_sglang_")
    atexit.register(shutil.rmtree, shim, ignore_errors=True)
    with open(os.path.join(model_path, "params.json")) as f:
        params = json.load(f)
    params.update(
        architectures=["Zonos2SGLangModel"],
        model_type="zonos2",
        hidden_size=cfg.dim,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.audio_vocab,
        max_position_embeddings=cfg.max_seqlen,
        rms_norm_eps=cfg.norm_eps,
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
    )
    with open(os.path.join(shim, "config.json"), "w") as f:
        json.dump(params, f)
    src = os.path.join(model_path, "model.pth")
    dst = os.path.join(shim, "pytorch_model.bin")
    # Prefer a symlink; fall back to a hardlink, then a copy, where symlinks are
    # unsupported (some network / Windows filesystems).
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    return shim


def _register_zonos2_autoconfig() -> None:
    from transformers import AutoConfig

    try:
        AutoConfig.register("zonos2", Zonos2Config)
    except (ValueError, KeyError):
        pass  # already registered


def _install_tuned_moe_configs() -> None:
    # note (Yue Yin): the fused-MoE Triton kernel (46% of decode GPU time,
    # profiled) ships no config for this deployment shape (E=16,N=3072 on H100),
    # so it falls back to get_default_config -> "Performance might be sub-optimal".
    # Install the bundled tuned configs into sglang's config dir so the kernel
    # picks them up. Quality-neutral (kernel tiling only); never clobbers an
    # existing config; device/triton-version-keyed filenames auto-ignore on a
    # mismatch (falls back to default). Best-effort: never block startup.
    import shutil

    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            fused_moe_triton_config as _fc,
        )

        dst_root = os.path.join(
            os.path.dirname(os.path.realpath(_fc.__file__)), "configs"
        )
        src_root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "moe_configs"
        )
        for vdir in os.listdir(src_root):
            sdir = os.path.join(src_root, vdir)
            if not os.path.isdir(sdir):
                continue
            ddir = os.path.join(dst_root, vdir)
            os.makedirs(ddir, exist_ok=True)
            for fn in os.listdir(sdir):
                dst = os.path.join(ddir, fn)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(sdir, fn), dst)
    except Exception:
        pass


def _cuda_graph_buckets(max_bs: int) -> list[int]:
    """Power-of-two decode buckets up to max_bs (+ max_bs itself)."""
    bs = [b for b in (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256) if b <= max_bs]
    if not bs or bs[-1] != max_bs:
        bs.append(max_bs)
    return bs


def create_sglang_omni_tts_engine_executor(
    model_path: str,
    *,
    gpu_id: int | None = 0,
    dtype: str = "bfloat16",
    mem_fraction_static: float = 0.5,
    fp8: bool = False,
    frame_graph: bool = False,
    compile_sampler: bool = False,
    async_decode: bool = False,
    stream_emit_chunk_frames: int = 1,
    stream_emit_first_chunk_frames: int = 0,
    max_running_requests: int = 16,
    cuda_graph_max_bs: int = 16,
    server_args_overrides: dict | None = None,
    **_: Any,
) -> Any:
    from sglang_omni.models.zonos2.model_runner import Zonos2ModelRunner
    from sglang_omni.models.zonos2.request_builders import (
        make_zonos2_scheduler_adapters,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    cfg = load_zonos2_pretrained_config(model_path)
    _register_zonos2_autoconfig()
    _install_tuned_moe_configs()
    shim = _build_config_shim(model_path, cfg)
    gpu = int(gpu_id) if gpu_id is not None else 0

    # note (Yue Yin): concurrency knobs (generic generation-stage override from the
    # serve CLI arrives as server_args_overrides). Higher max_running_requests +
    # cuda_graph_max_bs raise single-card throughput at the cost of per-request RTF;
    # decode buckets scale with cuda_graph_max_bs so large batches stay graph-captured.
    sao = dict(server_args_overrides or {})
    mrr = int(sao.pop("max_running_requests", max_running_requests))
    cg_max = int(sao.pop("cuda_graph_max_bs", cuda_graph_max_bs))
    cg_bs = _cuda_graph_buckets(cg_max)

    # Opt-in dynamic FP8 on the MoE experts (bf16 -> fp8 at load, halving the
    # expert weights); bf16 nn.Linear projections are unaffected. Off by default;
    # needs sm80+ (native fp8 tensor-core speedup on Hopper/Ada).
    fp8_kwargs = {"quantization": "fp8"} if fp8 else {}

    server_args = build_sglang_server_args(
        shim,
        context_length=cfg.max_seqlen,
        dtype=dtype,
        disable_cuda_graph=False,
        cuda_graph_bs=cg_bs,
        cuda_graph_max_bs=cg_max,
        # async-decode lookahead overlaps the resolve D2H with the next forward;
        # the overlap scheduler must be enabled for it (opt-in via async_decode).
        disable_overlap_schedule=not async_decode,
        enable_torch_compile=True,
        max_running_requests=mrr,
        mem_fraction_static=mem_fraction_static,
        sampling_backend="pytorch",
        trust_remote_code=True,
        **fp8_kwargs,
        **sao,
    )
    # Note:(Chenchen Hong) per-frame feedback/EOS state has no rollback, so a
    # non-final chunked-prefill chunk would queue a spurious frame; disable
    # chunking after construction (mirrors the Qwen3-Omni talker).
    server_args.chunked_prefill_size = 0

    # Defer graph capture until weights are loaded and the runner is wired.
    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args, gpu, model_arch_override="Zonos2SGLangModel"
    )

    model = model_worker.model_runner.model
    if want_cuda_graph:
        server_args.disable_cuda_graph = False
        model_worker.model_runner.init_device_graphs()

    # Opt-in tail CUDA graph: capture the per-frame head+sample+embed+hash tail
    # (otherwise eager in the runner). Captured per decode bucket with the default
    # sampling params; the runner falls back to eager for other params.
    if frame_graph:
        from sglang_omni.models.zonos2.text_frontend import TTSSamplingParams

        model.capture_tail_graphs(cg_bs, TTSSamplingParams())

    output_proc = SGLangOutputProcessor(
        capture_hidden=False, capture_hidden_layers=None, model=model
    )
    request_builder, result_adapter = make_zonos2_scheduler_adapters(model=model)

    runner = Zonos2ModelRunner(
        model_worker,
        output_proc,
        compile_sampler=compile_sampler,
        frame_graph=frame_graph,
        async_decode=async_decode,
        stream_emit_chunk_frames=stream_emit_chunk_frames,
        stream_emit_first_chunk_frames=stream_emit_first_chunk_frames,
    )
    scheduler = OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        enable_async_decode=async_decode,
    )
    runner.set_stream_outbox(scheduler.outbox)
    return scheduler
