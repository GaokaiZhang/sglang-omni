# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 model runner for OmniScheduler.

Per step the radix-cached backbone yields one hidden state per request; the
runner runs the multi-codebook head, samples all 9 DAC codes for the whole
batch at once, advances the any-of-9 EOS state machine, and stages the next
frame's summed embedding into a fixed row-indexed buffer so the backbone decode
stays CUDA-graph-replayable (decode input_ids are row indices). No frame loop.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.zonos2.radix_hash import EOS_SENTINEL, poly_row_hash
from sglang_omni.models.zonos2.sampler import sample_tts


class Zonos2ModelRunner(ModelRunner):
    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    # ---- input embeddings ----

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_embeds(forward_batch, requests)
        return None

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead=False
    ):
        del schedule_batch, is_lookahead
        bs = len(requests)
        if bs == 0:
            return
        dim = self.model.config.dim
        buf = self.model._decode_input_embedding.weight
        rows = []
        for sr in requests:
            q = sr.data.pending_feedback_queue
            rows.append(
                q.popleft()
                if q
                else torch.zeros(dim, device=buf.device, dtype=buf.dtype)
            )
        with torch.no_grad():
            buf[:bs].copy_(torch.stack(rows, dim=0).to(buf.device, buf.dtype))
        # Decode reads the staged buffer by row index -> stable input for graph replay.
        forward_batch.input_ids = torch.arange(bs, device=buf.device, dtype=torch.long)
        forward_batch.input_embeds = None

    def _build_prefill_embeds(self, forward_batch, requests) -> torch.Tensor:
        model = self.model
        pieces = []
        for sr in requests:
            data = sr.data
            req = data.req
            prefix_len = len(req.prefix_indices)
            req_len = int(req.extend_input_len)
            rows = data.prompt_rows[prefix_len : prefix_len + req_len].to(model.device)
            emb = model.embed_frames(rows)
            if data.speaker_emb is not None:
                pos = int(data.speaker_position) - prefix_len
                if 0 <= pos < req_len:
                    s = model.speaker_lda_projection(
                        data.speaker_emb.to(
                            model.device, model.speaker_lda_projection.weight.dtype
                        )
                    )
                    s = model.speaker_projection(
                        s.to(model.speaker_projection.weight.dtype)
                    )
                    emb[pos] = s.to(emb.dtype)
            pieces.append(emb)
        return torch.cat(pieces, dim=0).to(device=model.device, dtype=model.dtype)

    # ---- frame collection (head + batched sample + EOS + feedback) ----

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_frame(
            result, forward_batch, schedule_batch, requests, is_prefill=True
        )

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        self._collect_frame(
            result, forward_batch, schedule_batch, requests, is_prefill=False
        )

    def _last_token_hidden(self, hidden, forward_batch, is_prefill) -> torch.Tensor:
        if not is_prefill:
            return hidden
        lens = forward_batch.extend_seq_lens
        idx = torch.cumsum(lens.to(hidden.device, torch.long), dim=0) - 1
        return hidden[idx]

    def _collect_frame(
        self, result, forward_batch, schedule_batch, requests, *, is_prefill
    ):
        model = self.model
        n = model.n_codebooks
        eoa = model.config.eoa_id
        text_pad = model.config.text_vocab
        cb_size = model.config.codebook_size

        hidden = self._last_token_hidden(
            result.logits_output.hidden_states, forward_batch, is_prefill
        )
        logits = model.compute_logits(hidden).float()  # [B, 9, 1026]
        b = len(requests)

        # Batched per-codebook sampling. The benchmark uses uniform sampling
        # params; a per-request seed falls back to that request's generator.
        params = requests[0].data.params
        rep_ids = self._rep_window(requests, n, cb_size, logits.device)
        self._break_frame_loops(logits, requests)
        gen = next(
            (sr.data.generator for sr in requests if sr.data.generator is not None),
            None,
        )
        codes = sample_tts(
            logits, params, rep_token_ids=rep_ids, generator=gen
        )  # [B, 9]

        # Feedback embeddings for the next step: [B, 10] rows (codes + text pad).
        text_col = torch.full((b, 1), text_pad, device=codes.device, dtype=torch.long)
        rows = torch.cat([codes, text_col], dim=1)
        feedback = model.embed_frames(rows)  # [B, dim]
        keys = poly_row_hash(rows)  # [B] (< RADIX_HASH_SPACE)

        codes_cpu = codes.to("cpu")
        keys_cpu = keys.tolist()  # one D2H, not B per-row int(keys[i]) syncs
        next_ids = [0] * b
        for i, sr in enumerate(requests):
            data = sr.data
            frame = codes_cpu[i].tolist()
            data.output_codes.append(codes_cpu[i].clone())
            data.rep_hist.append(frame)
            data.pending_feedback_queue.append(feedback[i].detach())

            step = data.generation_step
            if data.eos_frame is None:
                hits = [frame[j] == eoa for j in range(n)]
                if any(hits):
                    data.eos_frame = max(
                        0, step - max(j for j, h in enumerate(hits) if h)
                    )
                    data.eos_countdown = n + 1
            finished = False
            if data.eos_frame is not None:
                if data.eos_countdown > 0:
                    data.eos_countdown -= 1
                finished = data.eos_countdown <= 0
            data.generation_step += 1
            next_ids[i] = EOS_SENTINEL if finished else keys_cpu[i]

        next_ids = torch.tensor(next_ids, dtype=torch.long, device=logits.device)
        result.next_token_ids = next_ids
        schedule_batch.output_ids = next_ids

    def _rep_window(self, requests, n, cb_size, device):
        params = requests[0].data.params
        if params.repetition_penalty == 1.0:
            return None
        w = params.repetition_window
        hist = []
        for sr in requests:
            h = sr.data.rep_hist[-w:]
            if len(h) < w:  # left-pad with -1 (ignored)
                h = [[-1] * n] * (w - len(h)) + h
            hist.append(h)
        if not any(sr.data.rep_hist for sr in requests):
            return None
        # [B, W, n] -> [B, n, W]; mask eoa/pad (>= codebook_size); only first rc codebooks
        t = torch.tensor(hist, device=device, dtype=torch.long).transpose(1, 2)
        rc = min(params.repetition_codebooks, n)
        rep = torch.full_like(t, -1)
        rep[:, :rc] = torch.where(t[:, :rc] < cb_size, t[:, :rc], rep[:, :rc])
        return rep

    def _break_frame_loops(self, logits, requests, run: int = 8):
        # Loop-collapse guard: if a request's full 9-codebook frame has repeated
        # identically for `run` consecutive steps (a degenerate loop the windowed
        # rep-penalty misses on long runs), mask its primary-codebook token so the
        # next step must diverge. `run` byte-identical frames (~0.09s) is never
        # real speech, so this is inert on healthy generation.
        for i, sr in enumerate(requests):
            h = sr.data.rep_hist
            if len(h) < run:
                continue
            last = h[-1]
            if last[0] < 0:
                continue
            if all(h[-j] == last for j in range(1, run + 1)):
                logits[i, 0, last[0]] = float("-inf")
