"""
Beam search from ICAL.
Copied from: https://github.com/qingzhenduyu/ICAL
Adapted: uses vocab PAD/SOS/EOS indices passed as params instead of global vocab.
"""
from typing import List, Tuple

import torch
from torch import FloatTensor, LongTensor


class BeamSearchScorer:
    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        alpha: float,
        do_early_stopping: bool,
        device: torch.device,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ) -> None:
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.alpha = alpha
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self._beam_hyps = [
            BeamHypotheses(beam_size, alpha, do_early_stopping)
            for _ in range(batch_size)
        ]

        self._done = torch.tensor(
            [False for _ in range(batch_size)], dtype=torch.bool, device=self.device
        )

    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: LongTensor,
        next_scores: FloatTensor,
        next_tokens: LongTensor,
        next_indices: LongTensor,
    ) -> Tuple[FloatTensor, LongTensor, LongTensor]:
        next_beam_scores = torch.zeros(
            (self.batch_size, self.beam_size),
            dtype=next_scores.dtype,
            device=self.device,
        )
        next_beam_tokens = torch.zeros(
            (self.batch_size, self.beam_size),
            dtype=next_tokens.dtype,
            device=self.device,
        )
        next_beam_indices = torch.zeros(
            (self.batch_size, self.beam_size),
            dtype=next_indices.dtype,
            device=self.device,
        )

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert len(beam_hyp) >= self.beam_size
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = self.pad_idx
                next_beam_indices[batch_idx, :] = batch_idx * self.beam_size
                continue

            beam_idx = 0
            for beam_token_rank, (next_score, next_token, next_index) in enumerate(
                zip(
                    next_scores[batch_idx],
                    next_tokens[batch_idx],
                    next_indices[batch_idx],
                )
            ):
                batch_beam_idx = batch_idx * self.beam_size + next_index
                l2r_done = (
                    input_ids[batch_beam_idx][0].item() == self.sos_idx
                    and next_token.item() == self.eos_idx
                )
                r2l_done = (
                    input_ids[batch_beam_idx][0].item() == self.eos_idx
                    and next_token.item() == self.sos_idx
                )
                if l2r_done or r2l_done:
                    if beam_token_rank >= self.beam_size:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(), next_score.item())
                else:
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                if beam_idx == self.beam_size:
                    break

            assert beam_idx == self.beam_size

            self._done[batch_idx] = beam_hyp.is_done(
                best_sum_logprobs=next_beam_scores[batch_idx].max().item(),
                cur_len=input_ids.shape[-1],
            )

        return (
            next_beam_scores.view(-1),
            next_beam_tokens.view(-1),
            next_beam_indices.view(-1),
        )

    def finalize(
        self,
        input_ids: LongTensor,
        final_scores: FloatTensor,
    ) -> Tuple[List[LongTensor], FloatTensor]:
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            for beam_id in range(self.beam_size):
                batch_beam_idx = batch_idx * self.beam_size + beam_id
                final_score = final_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        all_hyps: List[LongTensor] = []
        scores: FloatTensor = torch.zeros(
            self.batch_size * self.beam_size, dtype=torch.float, device=self.device
        )

        for beam_hyp in self._beam_hyps:
            for score, seq in beam_hyp.beams:
                scores[len(all_hyps)] = score
                all_hyps.append(seq[1:])

        return all_hyps, scores


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams: List[Tuple[float, LongTensor]] = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp: LongTensor, sum_logprobs: float):
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)]
                )
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
