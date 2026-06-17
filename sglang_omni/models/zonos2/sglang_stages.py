# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler-backed ZONOS2 AR engine stage (radix cache + batched decode).

Builds the SGLang infrastructure for the custom ZONOS2 backbone and drives it
with Zonos2ModelRunner. The checkpoint ships params.json + a flat model.pth
(no config.json / safetensors), so we synthesize an HF config shim and symlink
model.pth as pytorch_model.bin for the loader; our model.load_weights does the
key mapping.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from typing import Any

from sglang_omni.models.zonos2.hf_config import (
    Zonos2Config,
    load_zonos2_pretrained_config,
)


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
    **_: Any,
) -> Any:
    from sglang_omni.models.zonos2.model_runner import Zonos2ModelRunner
    from sglang_omni.models.zonos2.sglang_request_builders import (
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

    # Opt-in dynamic FP8 on the MoE experts (bf16 -> fp8 at load, halving the
    # expert weights); bf16 nn.Linear projections are unaffected. Off by default;
    # needs sm80+ (native fp8 tensor-core speedup on Hopper/Ada).
    fp8_kwargs = {"quantization": "fp8"} if fp8 else {}

    server_args = build_sglang_server_args(
        shim,
        context_length=cfg.max_seqlen,
        dtype=dtype,
        disable_cuda_graph=False,
        cuda_graph_bs=[1, 2, 4, 8, 16],
        cuda_graph_max_bs=16,
        # async-decode lookahead overlaps the resolve D2H with the next forward;
        # the overlap scheduler must be enabled for it (opt-in via async_decode).
        disable_overlap_schedule=not async_decode,
        enable_torch_compile=True,
        max_running_requests=16,
        mem_fraction_static=mem_fraction_static,
        sampling_backend="pytorch",
        trust_remote_code=True,
        **fp8_kwargs,
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

        model.capture_tail_graphs([1, 2, 4, 8, 16], TTSSamplingParams())

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
