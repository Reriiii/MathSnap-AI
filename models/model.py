"""
Full HMER model: DenseNet Encoder + CoMER Decoder + Multi-scale Counting.

Architecture:
  DenseNet encoder  -> [B, S, d_model] feature sequence + (feat_h, feat_w)
                       + intermediate feature maps for multi-scale counting
  CoMER decoder(s)  -> [B, T, vocab_size] logits  (ARM coverage attention)
                       L2R + optional R2L for bidirectional training (BTTR)
  Counting module   -> [B, vocab_size] per-class presence probability
                       Multi-scale with channel attention (CAN)

References:
  CoMER (ECCV 2022): ARM coverage attention for Transformer decoders
  CAN   (ECCV 2022): multi-scale counting with channel attention
  BTTR  (ICCV 2021): bidirectional training for Transformer-based HMER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from models.encoder import DenseNetEncoder
from models.decoder import CoMERDecoder


# ---------------------------------------------------------------------------
# Multi-scale Counting Module (CAN, ECCV 2022)
# ---------------------------------------------------------------------------

class MultiScaleCountingModule(nn.Module):
    """
    Multi-scale counting with channel attention (CAN, ECCV 2022).

    Uses intermediate feature maps from multiple encoder stages, projects
    them to a common channel dimension, applies SE-style channel attention
    to weight the contribution of each scale, then predicts per-class
    presence probabilities.

    This is significantly more powerful than naive average-pool counting
    because it captures multi-scale spatial information (fine details for
    small subscripts, coarse structure for large operators).
    """

    def __init__(
        self,
        feature_channels: List[int],
        common_channels: int,
        vocab_size: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_channels: channel counts of each intermediate feature map
            common_channels:  projection dimension for all scales
            vocab_size:       output dimension
            dropout:          dropout rate
        """
        super().__init__()
        self.num_scales = len(feature_channels)

        # Per-scale 1x1 projection to common channel dim
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, common_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for c in feature_channels
        ])

        # SE-style channel attention across scales
        self.se_fc1 = nn.Linear(self.num_scales * common_channels, self.num_scales)
        self.se_fc2 = nn.Linear(self.num_scales, self.num_scales)

        # Final prediction
        self.drop = nn.Dropout(dropout)
        self.output_conv = nn.Conv2d(common_channels, vocab_size, kernel_size=1)

    def forward(self, intermediates: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            intermediates: list of [B, C_i, H_i, W_i] encoder feature maps

        Returns:
            counts: [B, vocab_size] probability each token is present
        """
        # Use the last N feature maps matching num_scales
        feats = intermediates[-self.num_scales:]
        B = feats[0].size(0)

        # Project each scale and upsample to largest spatial size
        target_h = max(f.size(2) for f in feats)
        target_w = max(f.size(3) for f in feats)

        projected = []
        for proj, f in zip(self.scale_projs, feats):
            p = proj(f)  # [B, common_channels, H_i, W_i]
            if p.size(2) != target_h or p.size(3) != target_w:
                p = F.interpolate(p, size=(target_h, target_w), mode='bilinear',
                                  align_corners=False)
            projected.append(p)

        # SE-style channel attention: weight each scale
        # Global avg pool each scale -> concatenate -> FC -> sigmoid
        scale_descriptors = []
        for p in projected:
            desc = F.adaptive_avg_pool2d(p, 1).view(B, -1)  # [B, common_channels]
            scale_descriptors.append(desc)

        se_input = torch.cat(scale_descriptors, dim=1)  # [B, num_scales * common_channels]
        scale_weights = torch.sigmoid(
            self.se_fc2(F.relu(self.se_fc1(se_input)))
        )  # [B, num_scales]

        # Weighted sum of projected feature maps
        fused = torch.zeros_like(projected[0])
        for i, p in enumerate(projected):
            w = scale_weights[:, i].view(B, 1, 1, 1)
            fused = fused + w * p  # [B, common_channels, H, W]

        # Predict per-class presence
        fused = self.drop(fused)
        logits = self.output_conv(fused)           # [B, vocab_size, H, W]
        return torch.sigmoid(logits.mean(dim=(2, 3)))  # [B, vocab_size]


# ---------------------------------------------------------------------------
# Full HMER model with bidirectional decoding
# ---------------------------------------------------------------------------

class HMERModel(nn.Module):
    """
    Handwritten Mathematical Expression Recognition model.

    DenseNet encodes the image into a spatial feature sequence.
    CoMER decoder(s) generate LaTeX token sequences with ARM coverage.
    Optional R2L decoder for bidirectional training (BTTR, ICCV 2021).
    MultiScaleCountingModule provides an auxiliary presence-prediction signal.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        # Encoder
        enc_growth_rate: int = 24,
        enc_block_config: tuple = (6, 12, 16, 8),
        enc_num_init_features: int = 64,
        enc_bn_size: int = 4,
        enc_drop_rate: float = 0.2,
        enc_compression: float = 0.8,
        enc_num_transitions: int = 2,
        enc_num_groups: int = 32,
        # Decoder
        dec_nhead: int = 8,
        dec_num_layers: int = 6,
        dec_dim_feedforward: int = 2048,
        dec_dropout: float = 0.3,
        max_seq_len: int = 200,
        pad_idx: int = 0,
        # ARM
        arm_kernel_size: int = 5,
        arm_d_coverage: int = 32,
        # Counting
        counting_dropout: float = 0.1,
        counting_common_channels: int = 128,
        # Bidirectional
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.encoder = DenseNetEncoder(
            in_channels=1,
            growth_rate=enc_growth_rate,
            block_config=enc_block_config,
            num_init_features=enc_num_init_features,
            bn_size=enc_bn_size,
            drop_rate=enc_drop_rate,
            compression=enc_compression,
            d_model=d_model,
            num_transitions=enc_num_transitions,
            num_groups=enc_num_groups,
        )

        dec_kwargs = dict(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=dec_nhead,
            num_layers=dec_num_layers,
            dim_feedforward=dec_dim_feedforward,
            dropout=dec_dropout,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
            arm_kernel_size=arm_kernel_size,
            arm_d_coverage=arm_d_coverage,
        )

        # L2R decoder (primary)
        self.decoder = CoMERDecoder(**dec_kwargs)
        # R2L decoder (bidirectional, separate parameters as in BTTR)
        self.decoder_r2l = CoMERDecoder(**dec_kwargs) if bidirectional else None

        # CTC auxiliary head (on encoder output)
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # Multi-scale counting module (CAN)
        # Use the last 3 dense block outputs (or all if fewer than 3)
        block_channels = self.encoder.block_out_channels
        num_counting_scales = min(3, len(block_channels))
        counting_channels = block_channels[-num_counting_scales:]
        self.counting_module = MultiScaleCountingModule(
            feature_channels=counting_channels,
            common_channels=counting_common_channels,
            vocab_size=vocab_size,
            dropout=counting_dropout,
        )

        self.pad_idx = pad_idx
        self.d_model = d_model
        self.vocab_size = vocab_size

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int, List[torch.Tensor]]:
        """Run encoder. Returns (memory, feat_h, feat_w, intermediates)."""
        return self.encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        targets_r2l: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Teacher-forcing forward pass.

        Args:
            images:      [B, 1, H, W]
            targets:     [B, T] L2R token sequence (SOS + content + EOS + padding)
            targets_r2l: [B, T] R2L token sequence (optional, for bidirectional)

        Returns:
            logits_l2r: [B, T-1, vocab_size]
            logits_r2l: [B, T-1, vocab_size] or None
            counts:     [B, vocab_size] counting module predictions
        """
        memory, feat_h, feat_w, intermediates = self.encoder(images)

        # L2R decoder
        tgt_input = targets[:, :-1]
        logits_l2r = self.decoder(tgt_input, memory, feat_h, feat_w)

        # R2L decoder (bidirectional)
        logits_r2l = None
        if self.bidirectional and self.decoder_r2l is not None and targets_r2l is not None:
            tgt_input_r2l = targets_r2l[:, :-1]
            logits_r2l = self.decoder_r2l(tgt_input_r2l, memory, feat_h, feat_w)

        # Multi-scale counting
        counts = self.counting_module(intermediates)

        return logits_l2r, logits_r2l, counts

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 200,
        beam_size: int = 1,
    ) -> torch.Tensor:
        """
        Generate LaTeX sequences from images.

        With bidirectional=True, runs both L2R and R2L decoders and
        picks the result with higher length-normalized log-probability.
        """
        self.eval()
        memory, feat_h, feat_w, _ = self.encoder(images)

        def _decode(decoder):
            if beam_size <= 1:
                return decoder.greedy_decode(
                    memory, sos_idx, eos_idx, feat_h, feat_w, max_len
                )
            results = []
            for i in range(images.size(0)):
                mem_i = memory[i:i + 1]
                pred = decoder.beam_search(
                    mem_i, sos_idx, eos_idx, feat_h, feat_w, beam_size, max_len
                )
                results.append(pred.squeeze(0))
            max_len_pred = max(r.size(0) for r in results)
            padded = torch.full(
                (len(results), max_len_pred), self.pad_idx,
                dtype=torch.long, device=images.device
            )
            for i, r in enumerate(results):
                padded[i, :r.size(0)] = r
            return padded

        preds_l2r = _decode(self.decoder)

        if not self.bidirectional or self.decoder_r2l is None:
            return preds_l2r

        # R2L decode and reverse content tokens
        preds_r2l_raw = _decode(self.decoder_r2l)
        preds_r2l = self._reverse_predictions(preds_r2l_raw, sos_idx, eos_idx)

        # Pick better result per sample using sequence log-prob
        return self._ensemble_predictions(
            preds_l2r, preds_r2l, memory, feat_h, feat_w, sos_idx, eos_idx
        )

    def _reverse_predictions(
        self, preds: torch.Tensor, sos_idx: int, eos_idx: int
    ) -> torch.Tensor:
        """Reverse content tokens in R2L predictions to get L2R order."""
        result = preds.clone()
        for i in range(preds.size(0)):
            seq = preds[i]
            content_mask = (seq != self.pad_idx) & (seq != sos_idx) & (seq != eos_idx)
            content = seq[content_mask]
            result[i, content_mask] = content.flip(0)
        return result

    def _ensemble_predictions(
        self,
        preds_l2r: torch.Tensor,
        preds_r2l: torch.Tensor,
        memory: torch.Tensor,
        feat_h: int,
        feat_w: int,
        sos_idx: int,
        eos_idx: int,
    ) -> torch.Tensor:
        """Pick the prediction with higher log-prob per sample."""
        B = preds_l2r.size(0)

        def _score(decoder, preds):
            """Compute length-normalized log-prob for each prediction."""
            scores = torch.zeros(B, device=preds.device)
            for i in range(B):
                seq = preds[i]
                # Find sequence length (up to and including EOS)
                eos_pos = (seq == eos_idx).nonzero(as_tuple=True)[0]
                seq_len = eos_pos[0].item() + 1 if len(eos_pos) > 0 else seq.size(0)
                if seq_len <= 1:
                    scores[i] = float('-inf')
                    continue

                tgt_in = seq[:seq_len - 1].unsqueeze(0)  # [1, T-1]
                mem_i = memory[i:i + 1]
                logits = decoder(tgt_in, mem_i, feat_h, feat_w)  # [1, T-1, V]
                log_probs = F.log_softmax(logits, dim=-1)

                tgt_out = seq[1:seq_len]  # [T-1]
                token_log_probs = log_probs[0, torch.arange(tgt_out.size(0)), tgt_out]
                scores[i] = token_log_probs.sum() / (seq_len ** 0.6)
            return scores

        scores_l2r = _score(self.decoder, preds_l2r)

        # For R2L scoring, need to re-reverse to get R2L order for the R2L decoder
        preds_r2l_reversed = self._reverse_predictions(preds_r2l, sos_idx, eos_idx)
        scores_r2l = _score(self.decoder_r2l, preds_r2l_reversed)

        # Pick better prediction per sample
        result = preds_l2r.clone()
        # Pad to same length if needed
        max_len = max(preds_l2r.size(1), preds_r2l.size(1))
        if preds_l2r.size(1) < max_len:
            result = F.pad(result, (0, max_len - preds_l2r.size(1)), value=self.pad_idx)
        if preds_r2l.size(1) < max_len:
            preds_r2l = F.pad(preds_r2l, (0, max_len - preds_r2l.size(1)), value=self.pad_idx)

        r2l_better = scores_r2l > scores_l2r
        if r2l_better.any():
            result[r2l_better] = preds_r2l[r2l_better]

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(vocab_size: int, config=None) -> HMERModel:
    """Build HMERModel from config."""
    from config import Config
    if config is None:
        config = Config()

    return HMERModel(
        vocab_size=vocab_size,
        d_model=config.decoder.d_model,
        enc_growth_rate=config.encoder.growth_rate,
        enc_block_config=config.encoder.block_config,
        enc_num_init_features=config.encoder.num_init_features,
        enc_bn_size=config.encoder.bn_size,
        enc_drop_rate=config.encoder.drop_rate,
        enc_compression=config.encoder.compression,
        enc_num_transitions=config.encoder.num_transitions,
        enc_num_groups=config.encoder.num_groups,
        dec_nhead=config.decoder.nhead,
        dec_num_layers=config.decoder.num_layers,
        dec_dim_feedforward=config.decoder.dim_feedforward,
        dec_dropout=config.decoder.dropout,
        max_seq_len=config.decoder.max_seq_len,
        pad_idx=0,
        arm_kernel_size=config.decoder.arm_kernel_size,
        arm_d_coverage=config.decoder.arm_d_coverage,
        counting_dropout=config.decoder.counting_dropout,
        counting_common_channels=config.decoder.counting_common_channels,
        bidirectional=config.decoder.bidirectional,
    )
