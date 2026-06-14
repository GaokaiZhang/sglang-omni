# SPDX-License-Identifier: Apache-2.0
"""Streaming entry point for the ZONOS2 DAC vocoder.

``decode_to_pcm`` decodes the full delayed code sequence in one shot (the
non-streaming terminal path). ``Zonos2StreamingVocoderScheduler`` adds true
streaming: it consumes the per-frame delayed code rows emitted by the AR engine
and decodes them incrementally with raised-cosine overlap-add (OLA), withholding
the delay/flush tail until ``stream_done``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.tts_streaming import resolve_initial_codec_chunk_frames
from sglang_omni.models.zonos2.audio_codec import DAC_HOP_LENGTH, Zonos2DACVocoder
from sglang_omni.models.zonos2.payload_types import (
    N_CODEBOOKS,
    ZONOS2_SAMPLE_RATE,
    Zonos2State,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

# Process-wide cache: load the DAC checkpoint at most once per device.
_vocoder: Zonos2DACVocoder | None = None
_vocoder_device: str | None = None

# New streaming-chunk constants (not changes to existing values). At 86.1328125
# fps (44100/512) one frame = DAC_HOP_LENGTH=512 PCM samples.
_STREAM_STEADY_CHUNK_FRAMES = 25  # ~0.29 s steady chunk
_STREAM_INITIAL_CHUNK_FRAMES = 5  # ~0.06 s first chunk for low TTFB
_STREAM_OLA_OVERLAP_FRAMES = 2  # note (Yue Yin): cross-fade width; TODO calibrate
# shear_up needs the trailing N_CODEBOOKS-1 future rows to de-shear a frame.
_STREAM_WITHHOLD_TAIL = N_CODEBOOKS - 1


def _get_vocoder(device: str) -> Zonos2DACVocoder:
    global _vocoder, _vocoder_device
    if _vocoder is None or _vocoder_device != device:
        _vocoder = Zonos2DACVocoder(device=device)
        _vocoder_device = device
    return _vocoder


def decode_to_pcm(
    audio_codes: torch.Tensor,
    eos_frame: int | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Decode delayed ``[T, 9]`` AR codes to 1-D float32 PCM @ 44.1 kHz.

    Args:
        audio_codes: delayed per-frame codes ``[T, 9]`` (or batched ``[B, T, 9]``)
            straight from AR decode, before the delay is sheared out.
        eos_frame: number of aligned frames to keep before EOS, if known.
        device: torch device the DAC model runs on.

    Returns:
        1-D ``float32`` PCM tensor at 44.1 kHz on CPU.
    """
    return _get_vocoder(device).decode(audio_codes, eos_frame=eos_frame)


def decode_batch(
    audio_codes_list: list[torch.Tensor],
    eos_frames: list[int | None],
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Batched analogue of ``decode_to_pcm``: one DAC forward for many items.

    Reuses the process-wide DAC cache, so no second checkpoint load.
    """
    return _get_vocoder(device).decode_batch(audio_codes_list, eos_frames)


# ---- streaming (incremental raised-cosine OLA) ----


class _Zonos2OLADecoder:
    """Incremental de-shear + DAC decode of delayed rows with OLA cross-fade.

    Reuses ``Zonos2DACVocoder.decode`` per window: given delayed rows
    ``[lo, hi + WITHHOLD)`` it returns the aligned PCM for frames ``[lo, hi)``.
    Consecutive windows overlap by ``overlap`` frames; the overlap PCM is held
    back and raised-cosine cross-faded into the next window to mask the DAC
    ConvTranspose edge transients at chunk boundaries.
    """

    def __init__(self, device: str, overlap_frames: int, hop: int) -> None:
        self.device = device
        self.overlap = int(overlap_frames)
        self.hop = int(hop)
        self.rows: list[torch.Tensor] = []
        self.emitted = 0  # next aligned frame to emit
        self.decoded_to = 0  # right edge of decoded frames (== emitted + overlap)
        self.tail: torch.Tensor | None = None  # held PCM for the overlap region

    def add(self, rows: list[torch.Tensor]) -> None:
        self.rows.extend(rows)

    @staticmethod
    def _ramps(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.arange(n, dtype=torch.float32)
        up = 0.5 * (1.0 - torch.cos(math.pi * i / max(n - 1, 1)))
        return up, 1.0 - up

    def pull(
        self,
        vocoder: Zonos2DACVocoder,
        *,
        chunk_frames: int,
        flush: bool,
        eos_frame: int | None = None,
    ) -> list[torch.Tensor]:
        chunks: list[torch.Tensor] = []
        ovl = self.overlap
        hop = self.hop
        hold = ovl * hop
        max_frame = len(self.rows) - _STREAM_WITHHOLD_TAIL
        if eos_frame is not None:
            max_frame = min(max_frame, int(eos_frame))
        if max_frame <= 0:
            return chunks
        while True:
            hi = max_frame if flush else self.decoded_to + chunk_frames
            if not flush and hi > max_frame:
                break
            lo = self.emitted
            if hi <= lo:
                break
            block = torch.stack(self.rows[lo : hi + _STREAM_WITHHOLD_TAIL], dim=0)
            pcm = vocoder.decode(block)  # aligned PCM for frames [lo, hi)
            if pcm.numel() == 0:
                break
            if self.tail is not None and hold > 0 and pcm.numel() >= hold:
                up, down = self._ramps(hold)
                pcm = pcm.clone()
                pcm[:hold] = self.tail * down + pcm[:hold] * up
            if flush:
                self.tail = None
                self.emitted = hi
                self.decoded_to = hi
                if pcm.numel() > 0:
                    chunks.append(pcm.contiguous())
                break
            if pcm.numel() > hold:
                chunks.append(pcm[: pcm.numel() - hold].contiguous())
            self.tail = pcm[pcm.numel() - hold :].clone() if hold > 0 else None
            self.emitted = hi - ovl
            self.decoded_to = hi
        return chunks


@dataclass
class _Zonos2StreamState:
    decoder: _Zonos2OLADecoder | None = None
    n_codebooks: int = N_CODEBOOKS
    initial_chunk_frames: int = 0
    emitted_any: bool = False
    latched: bool = False


class Zonos2StreamingVocoderScheduler(StreamingSimpleScheduler):
    """Decode ZONOS2 delayed code rows incrementally with raised-cosine OLA.

    Non-streaming requests use the one-shot ``compute_fn`` / ``batch_compute_fn``
    (the terminal DAC decode) unchanged.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        compute_fn: Any = None,
        batch_compute_fn: Any = None,
        steady_chunk_frames: int = _STREAM_STEADY_CHUNK_FRAMES,
        initial_chunk_frames: int = _STREAM_INITIAL_CHUNK_FRAMES,
        overlap_frames: int = _STREAM_OLA_OVERLAP_FRAMES,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 0,
        request_cost_fn: Any = None,
        max_batch_cost: int | None = None,
    ) -> None:
        if steady_chunk_frames <= 0:
            raise ValueError(
                f"steady_chunk_frames must be positive, got {steady_chunk_frames}"
            )
        self._device = device
        self._sample_rate = ZONOS2_SAMPLE_RATE
        self._steady_chunk_frames = int(steady_chunk_frames)
        self._default_initial_chunk_frames = max(
            0, min(int(initial_chunk_frames), int(steady_chunk_frames))
        )
        self._overlap_frames = int(overlap_frames)
        self._stream_states: dict[str, _Zonos2StreamState] = {}
        super().__init__(
            compute_fn,
            batch_compute_fn=batch_compute_fn,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            request_cost_fn=request_cost_fn,
            max_batch_cost=max_batch_cost,
        )

    # ---- streaming hooks ----

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        params = getattr(payload.request, "params", None)
        return bool(isinstance(params, dict) and params.get("stream"))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        state = self._stream_states.setdefault(request_id, _Zonos2StreamState())
        params = getattr(payload.request, "params", None)
        self._latch(state, params if isinstance(params, dict) else None)

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self._stream_states.setdefault(request_id, _Zonos2StreamState())
        self._latch(state, item.metadata)
        row = item.data
        if not isinstance(row, torch.Tensor):
            raise TypeError(
                f"ZONOS2 stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(row).__name__}"
            )
        row = row.to(dtype=torch.long).reshape(-1)[: state.n_codebooks]
        if state.decoder is None:
            state.decoder = _Zonos2OLADecoder(
                self._device, self._overlap_frames, DAC_HOP_LENGTH
            )
        state.decoder.add([row])
        chunk_frames = (
            state.initial_chunk_frames
            if (not state.emitted_any and state.initial_chunk_frames > 0)
            else self._steady_chunk_frames
        )
        messages: list[OutgoingMessage] = []
        for pcm in state.decoder.pull(
            _get_vocoder(self._device), chunk_frames=chunk_frames, flush=False
        ):
            if pcm.numel() > 0:
                state.emitted_any = True
                messages.append(self._chunk_message(request_id, pcm))
        return messages

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload = self._stream_payloads[request_id]
        state = self._stream_states.setdefault(request_id, _Zonos2StreamState())
        zstate = Zonos2State.from_dict(payload.data)

        messages: list[OutgoingMessage] = []
        if state.decoder is not None and state.decoder.rows:
            for pcm in state.decoder.pull(
                _get_vocoder(self._device),
                chunk_frames=self._steady_chunk_frames,
                flush=True,
                eos_frame=zstate.eos_frame,
            ):
                if pcm.numel() > 0:
                    state.emitted_any = True
                    messages.append(self._chunk_message(request_id, pcm))
        if not state.emitted_any and zstate.audio_codes is not None:
            # Nothing streamed (slot-starved / sub-window utterance): fall back to
            # the one-shot decode so streaming output matches the non-stream path.
            codes = torch.as_tensor(zstate.audio_codes, dtype=torch.long)
            if codes.numel() > 0:
                pcm = decode_to_pcm(codes, zstate.eos_frame, device=self._device)
                if pcm.numel() > 0:
                    state.emitted_any = True
                    messages.append(self._chunk_message(request_id, pcm))

        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": int(zstate.sample_rate),
        }
        if zstate.prompt_tokens or zstate.completion_tokens:
            final_data["usage"] = {
                "prompt_tokens": int(zstate.prompt_tokens),
                "completion_tokens": int(zstate.completion_tokens),
                "total_tokens": int(zstate.prompt_tokens + zstate.completion_tokens),
            }
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self._stream_states.pop(request_id, None)

    # ---- internals ----

    def _latch(self, state: _Zonos2StreamState, params: dict | None) -> None:
        if state.latched or not isinstance(params, dict):
            return
        n_vq = params.get("n_codebooks")
        if n_vq is not None:
            state.n_codebooks = int(n_vq)
        state.initial_chunk_frames = (
            resolve_initial_codec_chunk_frames(
                params, steady_chunk_frames=self._steady_chunk_frames
            )
            or self._default_initial_chunk_frames
        )
        state.latched = True

    def _chunk_message(self, request_id: str, pcm: torch.Tensor) -> OutgoingMessage:
        data = audio_waveform_payload(
            pcm.detach().to("cpu", torch.float32),
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="ZONOS2 streaming",
        )
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=data,
            metadata={"modality": "audio"},
        )


__all__ = ["decode_to_pcm", "decode_batch", "Zonos2StreamingVocoderScheduler"]
