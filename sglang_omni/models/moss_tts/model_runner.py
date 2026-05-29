# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Delay model runner for OmniScheduler."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.request_builders import _INF_DELAY
from sglang_omni.scheduling.types import RequestOutput


class MossTTSModelRunner(ModelRunner):
    """Samples MOSS-TTS text/audio channels and maintains delay-pattern state."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._pending_rows: torch.Tensor | None = None
        self._pending_embeds: torch.Tensor | None = None

    def prepare_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch,
            requests,
        )
        return None

    def prepare_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del schedule_batch
        self._write_decode_input_embedding(forward_batch, requests)
        return None

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_moss_step(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_moss_step(result, forward_batch, schedule_batch, requests)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rows = data.prompt_rows
            if rows is None:
                raise RuntimeError("MOSS-TTS prefill requires prompt_rows")
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            current_rows = rows[prefix_len : prefix_len + req_len]
            embeds = self.model._prepare_multi_modal_inputs(
                current_rows.to(device=forward_batch.input_ids.device)
            )
            pieces.append(embeds)
        if not pieces:
            return torch.empty(
                (0, self.model.hidden_size),
                device=forward_batch.input_ids.device,
                dtype=self.model.dtype,
            )
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=self.model.dtype,
        )

    def _write_decode_input_embedding(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        embedding = self.model._decode_input_embedding
        weight = embedding.weight
        rows = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            if not queue:
                rows.append(torch.zeros(self.model.hidden_size, device=weight.device))
                continue
            if hasattr(queue, "popleft"):
                rows.append(queue.popleft())
            else:
                rows.append(queue.pop(0))
        stacked = torch.stack(rows, dim=0).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            weight[:batch_size].copy_(stacked)

        row_ids = torch.arange(
            batch_size,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    def _collect_moss_step(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        channel_logits = self._channel_logits_from_result(result, forward_batch)
        n_vq = len(channel_logits) - 1
        if n_vq <= 0:
            raise RuntimeError("MOSS-TTS requires at least one audio codebook head")

        device = channel_logits[0].device
        rows = []
        embeds = []
        text_tokens = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            text_token, audio_tokens = self._sample_next_row(
                channel_logits,
                row_idx=row_idx,
                data=data,
                n_vq=n_vq,
            )
            row = torch.empty(n_vq + 1, dtype=torch.long, device=device)
            row[0] = int(text_token)
            row[1:] = audio_tokens
            rows.append(row)
            text_tokens.append(int(text_token))
            embeds.append(
                self.model._prepare_multi_modal_inputs(
                    row.unsqueeze(0).to(device=self.model.device)
                )[0].detach()
            )

        next_token_ids = torch.tensor(text_tokens, dtype=torch.long, device=device)
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids
        self._pending_rows = torch.stack(rows, dim=0)
        self._pending_embeds = torch.stack(embeds, dim=0)

    def _channel_logits_from_result(
        self,
        result: Any,
        forward_batch: Any,
    ) -> list[torch.Tensor]:
        logits_output = result.logits_output
        customized = getattr(logits_output, "customized_info", None)
        if isinstance(customized, dict):
            values = customized.get("moss_tts_channel_logits")
            if isinstance(values, list) and values:
                return values
        hidden_states = getattr(logits_output, "hidden_states", None)
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim == 3:
                hidden_states = hidden_states[:, -1, :]
            return self.model.compute_channel_logits(hidden_states, forward_batch)
        raise RuntimeError("MOSS-TTS model output did not include channel logits")

    def _sample_next_row(
        self,
        channel_logits: list[torch.Tensor],
        *,
        row_idx: int,
        data: Any,
        n_vq: int,
    ) -> tuple[int, torch.Tensor]:
        cfg = self.model.config
        device = channel_logits[0].device
        audio_tokens = torch.full(
            (n_vq,),
            int(cfg.audio_pad_code),
            dtype=torch.long,
            device=device,
        )

        text_token = self._next_text_token(
            channel_logits[0][row_idx],
            data=data,
            n_vq=n_vq,
        )
        sampling_audio_mask = self._sampling_audio_mask(data, n_vq=n_vq, device=device)
        # Materialize the mask once (single host sync) instead of one .item()
        # per codebook, and only rebuild the per-head history when the
        # repetition penalty is actually active (it is 1.0 by default).
        active = sampling_audio_mask.tolist()
        rep_penalty = float(data.audio_repetition_penalty)
        for vq_idx in range(n_vq):
            if not active[vq_idx]:
                continue
            logits = channel_logits[vq_idx + 1][row_idx].clone()
            logits[int(cfg.audio_pad_code)] = float("-inf")
            audio_tokens[vq_idx] = self._sample_logits(
                logits,
                temperature=float(data.audio_temperature),
                top_p=float(data.audio_top_p),
                top_k=int(data.audio_top_k),
                repetition_penalty=rep_penalty,
                prev_tokens=(
                    self._previous_audio_tokens(data, vq_idx)
                    if rep_penalty != 1.0
                    else None
                ),
            )

        self._update_delay_state(data, int(text_token), n_vq=n_vq)
        return int(text_token), audio_tokens

    def _next_text_token(self, logits: torch.Tensor, *, data: Any, n_vq: int) -> int:
        cfg = self.model.config
        delayed_length = self._delayed_length_value(data.delayed_length)
        if delayed_length is not None and delayed_length < n_vq:
            return int(cfg.audio_assistant_delay_slot_token_id)
        if delayed_length == n_vq:
            data.is_audio = False
            return int(cfg.audio_end_token_id)

        masked = logits.clone()
        if bool(data.is_audio):
            disallow = torch.ones(
                masked.shape[-1],
                dtype=torch.bool,
                device=masked.device,
            )
            for token_id in (
                int(cfg.audio_assistant_gen_slot_token_id),
                int(cfg.audio_assistant_delay_slot_token_id),
            ):
                if 0 <= token_id < masked.shape[-1]:
                    disallow[token_id] = False
            masked[disallow] = float("-inf")
        else:
            for token_id in (
                int(cfg.pad_token_id),
                int(cfg.audio_assistant_gen_slot_token_id),
                int(cfg.audio_assistant_delay_slot_token_id),
                int(cfg.audio_end_token_id),
            ):
                if 0 <= token_id < masked.shape[-1]:
                    masked[token_id] = float("-inf")
        if int(data.generation_steps) == 0:
            token_id = int(cfg.audio_assistant_delay_slot_token_id)
            if 0 <= token_id < masked.shape[-1]:
                masked[token_id] = float("-inf")
        if int(data.generation_steps) <= n_vq:
            token_id = int(cfg.im_end_token_id)
            if 0 <= token_id < masked.shape[-1]:
                masked[token_id] = float("-inf")

        return int(
            self._sample_logits(
                masked,
                temperature=float(data.text_temperature),
                top_p=float(data.text_top_p),
                top_k=int(data.text_top_k),
            ).item()
        )

    @staticmethod
    def _delayed_length_value(delayed_length: int) -> int | None:
        delayed = int(delayed_length)
        return None if delayed == _INF_DELAY else delayed

    @staticmethod
    def _sampling_audio_mask(
        data: Any,
        *,
        n_vq: int,
        device: torch.device,
    ) -> torch.Tensor:
        delayed = MossTTSModelRunner._delayed_length_value(data.delayed_length)
        indices = torch.arange(n_vq, dtype=torch.long, device=device)
        pre_audio = int(data.audio_length) > indices
        if delayed is None:
            post_audio = torch.ones(n_vq, dtype=torch.bool, device=device)
        else:
            post_audio = indices > delayed - 1
        return pre_audio & post_audio

    def _previous_audio_tokens(self, data: Any, vq_idx: int) -> torch.Tensor | None:
        parts = []
        if data.prompt_rows is not None and data.prompt_rows.numel() > 0:
            parts.append(data.prompt_rows[:, vq_idx + 1])
        if data.output_rows:
            parts.append(torch.stack(data.output_rows, dim=0)[:, vq_idx + 1])
        if not parts:
            return None
        return torch.cat(
            [part.to(dtype=torch.long, device=self.model.device) for part in parts]
        )

    @staticmethod
    def _sample_logits(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float = 1.0,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = logits.to(dtype=torch.float32).clone()
        if prev_tokens is not None and repetition_penalty != 1.0:
            valid = prev_tokens[(prev_tokens >= 0) & (prev_tokens < scores.shape[-1])]
            if valid.numel() > 0:
                unique = torch.unique(valid)
                prior = scores[unique]
                penalty = torch.tensor(
                    float(repetition_penalty),
                    dtype=prior.dtype,
                    device=prior.device,
                )
                scores[unique] = torch.where(
                    prior > 0,
                    prior / penalty,
                    prior * penalty,
                )

        if not torch.isfinite(scores).any():
            return torch.zeros((), dtype=torch.long, device=logits.device)
        if temperature <= 0:
            return torch.argmax(scores, dim=-1).to(dtype=torch.long)

        scores = scores / float(temperature)
        if top_k is not None and int(top_k) > 0 and int(top_k) < scores.shape[-1]:
            kth = torch.topk(scores, int(top_k)).values[-1]
            scores[scores < kth] = float("-inf")
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, descending=True)
            probs = torch.softmax(sorted_scores, dim=-1)
            remove = torch.cumsum(probs, dim=-1) > float(top_p)
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            scores[sorted_idx[remove]] = float("-inf")
        probs = torch.softmax(scores, dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
            return torch.argmax(logits, dim=-1).to(dtype=torch.long)
        return torch.multinomial(probs, 1).squeeze(0).to(dtype=torch.long)

    def _update_delay_state(self, data: Any, text_token: int, *, n_vq: int) -> None:
        cfg = self.model.config
        if text_token in (
            int(cfg.audio_start_token_id),
            int(cfg.audio_assistant_gen_slot_token_id),
            int(cfg.audio_assistant_delay_slot_token_id),
        ):
            data.audio_length = int(data.audio_length) + 1
        if text_token == int(cfg.audio_end_token_id):
            data.audio_length = 0
        if text_token == int(cfg.audio_start_token_id):
            data.is_audio = True
        if text_token == int(cfg.im_end_token_id):
            data.is_audio = False

        delayed = self._delayed_length_value(data.delayed_length)
        if delayed is None and text_token == int(
            cfg.audio_assistant_delay_slot_token_id
        ):
            delayed = 0
        if delayed is not None:
            delayed += 1
            if delayed > n_vq:
                delayed = None
        data.delayed_length = _INF_DELAY if delayed is None else int(delayed)

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        rows = self._pending_rows
        embeds = self._pending_embeds
        self._pending_rows = None
        self._pending_embeds = None
        if rows is None or embeds is None:
            return

        eos_id = int(self.model.config.im_end_token_id)
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == eos_id:
                continue
            sched_req.data.output_rows.append(rows[row_idx].detach().cpu().clone())
            sched_req.data.pending_feedback_queue.append(
                embeds[row_idx].detach().clone()
            )
