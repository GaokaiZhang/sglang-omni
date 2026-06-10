# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Local (v1.5) pipeline."""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
from typing import Any

import torch

from sglang_omni.models.moss_tts.stages import (
    _load_moss_processor_class,
    _moss_transformers_processor_compat,
    _resolve_checkpoint,
)
from sglang_omni.models.moss_tts_local.payload_types import (
    MossTTSLocalState,
    moss_tts_local_special_token_defaults,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    cleanup_prepared_moss_tts_local_request,
    make_moss_tts_local_scheduler_adapters,
    preprocess_moss_tts_local_payload,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_MOSS_TTS_LOCAL_INSTALL_HINT = (
    "MOSS-TTS Local support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2."
)

# NOTE: the preprocessing and vocoder stages each load their own processor
# (and thus their own ~4.3 GB bf16 codec instance). The codec's chunked decode
# flips module-global streaming state (`model.streaming()`), so a decode on a
# shared instance corrupts any concurrently running reference encode; with
# separate instances the encoder side only ever runs stateless forwards and the
# streaming decode stays confined to the single-threaded vocoder batch loop.


def load_state(payload: StagePayload) -> MossTTSLocalState:
    return MossTTSLocalState.from_dict(payload.data)


def store_state(payload: StagePayload, state: MossTTSLocalState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _normalize_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_local_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _resolve_codec_device(device: str | None, gpu_id: int | None) -> str:
    """Pick the codec GPU for the preprocessing/vocoder stages.

    The ~1B-param codec encoder costs ~0.25 GPU-seconds per reference, which
    at concurrency 16 starves the AR engine when both share one device.
    The default config passes an explicit ``device`` so the second-GPU codec
    placement is visible in the pipeline config. ``gpu_id`` remains a fallback
    for custom colocated configs and launcher-injected runtime defaults.
    """
    if device:
        return device
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return "cuda:0"


def _load_moss_tts_local_processor(model_path: str, *, device: str) -> Any:
    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info(
        "Loading MOSS-TTS Local processor from %s on %s", checkpoint_dir, device
    )
    try:
        with _moss_transformers_processor_compat():
            processor_cls = _load_moss_processor_class(checkpoint_dir)
            processor = processor_cls.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_LOCAL_INSTALL_HINT) from exc

    _normalize_processor_config(processor)
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is not None:
        if hasattr(audio_tokenizer, "eval"):
            audio_tokenizer.eval()
        if hasattr(audio_tokenizer, "to"):
            # Device move only: the v2 codec manages its own dtypes (bf16
            # encoder/decoder with an fp32 quantizer); a blanket dtype cast
            # would corrupt the quantizer codebooks.
            audio_tokenizer.to(device)
    return processor


def _build_usage(state: MossTTSLocalState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


class _ReferenceCodesCache:
    """Content-keyed LRU of encoded reference codes (Higgs-style, #562).

    Keys are a blake2b hash of the file contents, memoized per
    ``(path, size, mtime_ns)`` stat signature so unchanged files hash once.
    Values are the CPU ``[T, n_vq]`` code tensors. Repeated reference audio
    (the common voice-cloning pattern: one voice, many requests) then skips
    the codec encode entirely.
    """

    def __init__(self, max_items: int = 256, max_bytes: int = 256 * 1024 * 1024):
        import collections

        self._max_items = int(max_items)
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._stat_memo: dict[str, tuple[tuple[int, int], str]] = {}
        self._entries: "collections.OrderedDict[str, torch.Tensor]" = (
            collections.OrderedDict()
        )
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._max_items > 0 and self._max_bytes > 0

    def _content_key(self, path: str) -> str | None:
        import hashlib
        import os

        try:
            stat = os.stat(path)
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
        except OSError:
            return None
        with self._lock:
            memo = self._stat_memo.get(path)
            if memo is not None and memo[0] == signature:
                return memo[1]
        try:
            digest = hashlib.blake2b(open(path, "rb").read(), digest_size=16)
        except OSError:
            return None
        key = digest.hexdigest()
        with self._lock:
            self._stat_memo[path] = (signature, key)
        return key

    def get(self, key: str | None) -> torch.Tensor | None:
        if key is None:
            return None
        with self._lock:
            codes = self._entries.get(key)
            if codes is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return codes

    def put(self, key: str | None, codes: torch.Tensor) -> None:
        if key is None or not self.enabled:
            return
        size = codes.numel() * codes.element_size()
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous.numel() * previous.element_size()
            self._entries[key] = codes
            self._bytes += size
            while self._entries and (
                len(self._entries) > self._max_items or self._bytes > self._max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.numel() * evicted.element_size()


class _BatchedReferenceEncoder:
    """Coalesces concurrent reference-audio encodes into batched codec calls.

    Each request needs its reference run through the ~1B-param codec encoder
    (~0.25 GPU-seconds). The preprocessing workers call :meth:`encode`
    concurrently; a single daemon thread drains the queue and encodes up to
    ``max_batch_size`` files in one ``batch_encode`` forward, which costs
    barely more than a single encode. Encoded codes land in a content-keyed
    LRU so repeated reference audio skips the codec. Failures fall back to
    per-item encodes so one bad file only fails its own request.
    """

    # Mirrors the Higgs reference-audio cap: bounds both encoder runtime and
    # the batch-padding memory amplification.
    MAX_REFERENCE_SECONDS = 100.0
    # An encode batch takes well under a second; a result this late means the
    # worker died or wedged, so fail the request instead of hanging the slot.
    ENCODE_TIMEOUT_S = 120.0

    def __init__(
        self,
        processor: Any,
        *,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 4,
        cache: _ReferenceCodesCache | None = None,
    ) -> None:
        self._processor = processor
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._cache = cache
        self._queue: queue.Queue[tuple[str, concurrent.futures.Future]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, name="moss-local-ref-encode", daemon=True
        )
        self._thread.start()

    @classmethod
    def _check_reference_duration(cls, path: str) -> None:
        try:
            import torchaudio

            info = torchaudio.info(path)
            duration = info.num_frames / max(int(info.sample_rate), 1)
        except Exception:
            return  # unreadable files fail with a clearer error in the codec
        if duration > cls.MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"reference audio is {duration:.1f}s long; the limit is "
                f"{cls.MAX_REFERENCE_SECONDS:.0f}s"
            )

    def encode(self, path: str) -> torch.Tensor:
        """Encode one reference file; blocks until its batch completes."""
        path = str(path)
        cache_key = None
        if self._cache is not None and self._cache.enabled:
            cache_key = self._cache._content_key(path)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        self._check_reference_duration(path)
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((path, future))
        codes = future.result(timeout=self.ENCODE_TIMEOUT_S)
        if self._cache is not None:
            self._cache.put(cache_key, codes)
        return codes

    def _drain_batch(self) -> list[tuple[str, concurrent.futures.Future]]:
        batch = [self._queue.get()]
        while len(batch) < self._max_batch_size:
            try:
                if self._max_wait_s > 0:
                    batch.append(self._queue.get(timeout=self._max_wait_s))
                else:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _worker(self) -> None:
        while True:
            batch = self._drain_batch()
            unique_paths = list(dict.fromkeys(path for path, _ in batch))
            results: dict[str, Any] = {}
            try:
                encoded = self._processor.encode_audios_from_path(unique_paths)
                results = dict(zip(unique_paths, encoded))
            except Exception:
                logger.exception(
                    "MOSS-TTS Local batched reference encode failed; "
                    "retrying per item"
                )
                for path in unique_paths:
                    try:
                        results[path] = self._processor.encode_audios_from_path([path])[
                            0
                        ]
                    except Exception as exc:
                        results[path] = exc
            for path, future in batch:
                outcome = results.get(path)
                if isinstance(outcome, Exception):
                    # Fresh exception per future: a shared instance would be
                    # mutated concurrently by every waiter's traceback raise.
                    future.set_exception(
                        RuntimeError(f"reference encode failed for {path}: {outcome}")
                    )
                elif outcome is None:
                    future.set_exception(
                        RuntimeError(f"reference encode produced no codes: {path}")
                    )
                else:
                    future.set_result(outcome)


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 16,
    encode_batch_size: int = 8,
    encode_batch_wait_ms: int = 4,
    encode_cache_items: int = 256,
    encode_cache_mb: int = 256,
) -> SimpleScheduler:
    device = _resolve_codec_device(device, gpu_id)
    processor = _load_moss_tts_local_processor(model_path, device=device)
    reference_encoder = _BatchedReferenceEncoder(
        processor,
        max_batch_size=encode_batch_size,
        max_batch_wait_ms=encode_batch_wait_ms,
        cache=_ReferenceCodesCache(
            max_items=encode_cache_items,
            max_bytes=encode_cache_mb * 1024 * 1024,
        ),
    )
    set_moss_tts_local_preprocessing_context(
        processor=processor, reference_encoder=reference_encoder
    )
    # Reference encoding runs through the ~1B-param causal codec encoder, so
    # unlike MOSS Delay the audio tokenizer must live on the GPU; threads
    # release the GIL during the codec forward, keeping the AR engine fed.
    return SimpleScheduler(
        preprocess_moss_tts_local_payload,
        abort_callback=cleanup_prepared_moss_tts_local_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        "dtype": dtype,
        "cuda_graph_bs": [1, 2, 4, 8, 16],
        "cuda_graph_max_bs": 16,
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": False,
        "max_prefill_tokens": 8192,
        "max_running_requests": 16,
        # Leave headroom for the two ~4.3 GB bf16 codec instances plus their
        # activations: on multi-GPU hosts the codec lives on the second GPU
        # (0.6 of an 80 GB card still gives the 4B backbone a ~35 GB KV pool);
        # on a single GPU everything co-locates, so back off further.
        "mem_fraction_static": 0.6 if torch.cuda.device_count() > 1 else 0.5,
        "sampling_backend": "pytorch",
        "torch_compile_max_bs": 16,
        "trust_remote_code": True,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=8192,
        **overrides,
    )

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
        server_args,
        gpu_id,
        model_arch_override="MossTTSLocalSGLangModel",
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False

    model = model_worker.model_runner.model
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()
        # Also graph the per-frame local-transformer decode (1 + n_vq
        # micro-steps and 13 seeded sampling passes per frame): eager it is
        # kernel-launch-bound at ~22 ms/frame independent of batch size.
        model.init_frame_decode_graphs(
            list(overrides.get("cuda_graph_bs") or [1, 2, 4, 8, 16])
        )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_moss_tts_local_scheduler_adapters(
        model=model
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=MossTTSLocalModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=cleanup_prepared_moss_tts_local_request,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    device = _resolve_codec_device(device, gpu_id)
    processor = _load_moss_tts_local_processor(model_path, device=device)

    def _prepare_codes(
        payload: StagePayload,
    ) -> tuple[MossTTSLocalState, torch.Tensor | None]:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError("MOSS-TTS Local vocoder requires audio_codes")
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if codes.numel() == 0:
            # Immediate stop decision: emit no audio so only this request
            # fails downstream instead of poisoning the whole decode batch.
            return state, None
        return state, codes

    def _store_vocoder_result(
        payload: StagePayload,
        state: MossTTSLocalState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        # The v2 codec is natively stereo: keep the [channels, samples]
        # layout end to end so the client receives a 2-channel waveform.
        audio_payload = audio_waveform_payload(
            wav, source_hint="MOSS-TTS Local", keep_channels=True
        )
        state.audio_codes = None
        state.sample_rate = int(sample_rate)
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _sample_rate() -> int:
        return int(
            getattr(getattr(processor, "model_config", None), "sampling_rate", 0)
            or getattr(
                getattr(getattr(processor, "audio_tokenizer", None), "config", None),
                "sampling_rate",
                0,
            )
            or 48000
        )

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        prepared = [_prepare_codes(payload) for payload in payloads]
        codes_list = [codes for _, codes in prepared if codes is not None]
        decoded = iter(processor.decode_audio_codes(codes_list))
        sample_rate = _sample_rate()
        results = []
        for payload, (state, codes) in zip(payloads, prepared):
            if codes is None:
                # No audio fields: the client surfaces a per-request
                # "no audio output" error without failing batch peers.
                state.audio_codes = None
                results.append(store_state(payload, state))
                continue
            wav = torch.as_tensor(next(decoded)).detach().to("cpu")
            results.append(_store_vocoder_result(payload, state, wav, sample_rate))
        return results

    def _vocode(payload: StagePayload) -> StagePayload:
        return _vocode_batch([payload])[0]

    return SimpleScheduler(
        _vocode,
        batch_compute_fn=_vocode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
