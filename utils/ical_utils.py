"""
ICAL utility functions: loss, target building, hypothesis.
Adapted from: https://github.com/qingzhenduyu/ICAL
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


def smooth_weight_adjustment(targets, class_of_interest=3, base_weight=1.0, max_weight=10.0):
    """
    Adjust weights based on frequency smoothing, using a logarithmic function.
    class_of_interest=3 corresponds to <space> (ICAL) or UNK (our vocab).
    """
    class_count = torch.sum(targets == class_of_interest)
    total_count = torch.numel(targets)
    frequency = class_count.float() / total_count

    weight = base_weight + torch.log1p(1 / ((1 - frequency) + 1e-6))
    weight = torch.clamp(weight, max=max_weight)
    return weight


def ce_loss(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    ignore_idx: int = 0,
    reduction: str = "mean",
    need_weight: bool = False,
    class_of_interest: int = 3,
    vocab_size: int = 0,
) -> torch.Tensor:
    """Compute cross-entropy loss (ICAL-style).

    Args:
        output_hat: [batch, len, vocab_size]
        output: [batch, len]
        ignore_idx: PAD index
        reduction: 'mean' or 'none'
        need_weight: if True, apply dynamic class weighting for implicit loss
        class_of_interest: the blank/space token index
        vocab_size: vocabulary size (needed if need_weight=True)
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    weights = None
    if need_weight:
        if vocab_size == 0:
            vocab_size = output_hat.size(-1)
        weights = torch.ones(vocab_size, device=output_hat.device)
        smooth_weight = smooth_weight_adjustment(flat, class_of_interest)
        weights[:] = smooth_weight
        weights[class_of_interest] = 1.0

    loss = F.cross_entropy(flat_hat, flat, weight=weights,
                           ignore_index=ignore_idx, reduction=reduction)
    return loss


def to_tgt_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    direction: str,
    device: torch.device,
    pad_idx: int = 0,
    sos_idx: int = 1,
    eos_idx: int = 2,
    space_idx: int = 3,
    structural_indices: Optional[set] = None,
    pad_to_len: Optional[int] = None,
    is_explicit: bool = False,
    is_implicit: bool = False,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices.

    Parameters
    ----------
    tokens : list of token index lists
    direction : 'l2r' or 'r2l'
    device : torch device
    pad_idx, sos_idx, eos_idx, space_idx : special token indices
    structural_indices : set of structural token indices ({, }, ^, _)
    is_explicit : if True, replace structural tokens with space
    is_implicit : if True, replace non-structural tokens with space

    Returns
    -------
    Tuple[LongTensor, LongTensor]: tgt [b, l], out [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if is_implicit and structural_indices is not None:
        filtered_tokens = []
        for token in tokens:
            mask = torch.ones(len(token), dtype=torch.bool)
            for si in structural_indices:
                mask &= (token != si)
            token = token.clone()
            token[mask] = space_idx
            filtered_tokens.append(token)
        tokens = filtered_tokens

    if is_explicit and structural_indices is not None:
        filtered_tokens = []
        for token in tokens:
            mask = torch.zeros(len(token), dtype=torch.bool)
            for si in structural_indices:
                mask |= (token == si)
            token = token.clone()
            token[mask] = space_idx
            filtered_tokens.append(token)
        tokens = filtered_tokens

    if direction == "l2r":
        tokens = tokens
        start_w = sos_idx
        stop_w = eos_idx
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = eos_idx
        stop_w = sos_idx

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=pad_idx,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=pad_idx,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1: (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def plicit_tgt_out(
    tokens: List[List[int]],
    device: torch.device,
    pad_idx: int = 0,
    sos_idx: int = 1,
    eos_idx: int = 2,
    space_idx: int = 3,
    structural_indices: Optional[set] = None,
    is_implicit: bool = False,
    is_explicit: bool = False,
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirectional tgt and out (ICAL-style).

    Returns
    -------
    Tuple[LongTensor, LongTensor]: tgt [2b, l], out [2b, l]
    """
    kwargs = dict(
        pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx,
        space_idx=space_idx, structural_indices=structural_indices,
        is_explicit=is_explicit, is_implicit=is_implicit,
    )
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device, **kwargs)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device, **kwargs)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out
