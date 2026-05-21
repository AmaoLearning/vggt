from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionContext, KVReductionHook
from .config import CompressionConfig
from .utils import gather_tokens


class ThresholdKSimilarityPruning(KVReductionHook):
    """D3: I/P-frame-style KV compression via K-space cosine similarity threshold.

    Frame 0 is kept as the I-frame (all tokens preserved).
    For each subsequent frame s, tokens whose K-vector cosine similarity with the
    reference frame at the same position exceeds d3_threshold are dropped from both
    K and V.  Q is never modified.

    Reference frame strategies (d3_reference):
        "adjacent": frame s compared to frame s-1  (default, standard P-frame chain)
        "first"   : every frame compared to frame 0 (fixed I-frame / GOP-style)

    Special tokens (camera token + registers) are always preserved.
    """

    def __init__(self, config: CompressionConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compress(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        ctx: CompressionContext,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Callable]]:
        if not ctx.is_global or ctx.S < self.config.min_frames_to_compress:
            return q, k, v, None

        keep_indices = self._build_keep_indices(k, ctx)
        gather_idx = keep_indices.view(1, 1, -1).expand(k.shape[0], k.shape[1], -1)
        k_new = gather_tokens(k, gather_idx)
        v_new = gather_tokens(v, gather_idx)
        return q, k_new, v_new, None

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _build_keep_indices(self, k: Tensor, ctx: CompressionContext) -> Tensor:
        """Build flat keep indices from the K-space cosine similarity threshold.

        Args:
            k:   [B, H, S*P, D]  — full key tensor for this layer
            ctx: CompressionContext

        Returns:
            keep_indices: [N_keep] long — flat token positions to retain in K/V
        """
        B, H, _SP, D = k.shape
        S, P = ctx.S, ctx.P
        special_tokens = ctx.special_tokens

        # Normalise and reshape: [B, H, S, P, D]
        k_norm = F.normalize(k.float(), dim=-1)
        k_frames = k_norm.reshape(B, H, S, P, D)

        # I-frame (frame 0) and all positions default to kept
        keep_mask = torch.ones(S, P, dtype=torch.bool, device=k.device)

        for s in range(1, S):
            ref_s = s - 1 if self.config.d3_reference == "adjacent" else 0
            # Per-position cosine similarity averaged over batch and heads: [P]
            sim = (k_frames[:, :, s] * k_frames[:, :, ref_s]).sum(dim=-1).mean(dim=(0, 1))
            # Mark patch tokens that exceed the threshold as redundant
            redundant = sim > self.config.d3_threshold
            redundant[:special_tokens] = False   # always keep special tokens
            keep_mask[s] = ~redundant

        return keep_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
