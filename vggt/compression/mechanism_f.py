import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionContext, KVReductionHook
from .config import CompressionConfig, get_f_keep_ratio
from .utils import gather_tokens


class LowSimilaritySaliencyPruning(KVReductionHook):
    """方案 F：保留与目标帧最不相似的显著 token。"""

    def __init__(self, config: CompressionConfig):
        self.config = config

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

    def _build_keep_indices(self, reference: Tensor, ctx: CompressionContext) -> Tensor:
        S, P, special_tokens = ctx.S, ctx.P, ctx.special_tokens
        patch_tokens = P - special_tokens
        grid_h = grid_w = math.isqrt(patch_tokens)
        if grid_h * grid_w != patch_tokens:
            raise ValueError(f"Patch token count {patch_tokens} is not a square number.")

        keep_ratio = get_f_keep_ratio(ctx.layer_idx, self.config)
        keep_patch_tokens = max(1, int(math.ceil(patch_tokens * keep_ratio)))

        frames = reference.reshape(reference.shape[0], reference.shape[1], S, P, reference.shape[-1])
        patch = frames[:, :, :, special_tokens:, :].mean(dim=(0, 1))  # [S, P_patch, D]
        candidates = valid_mask = None
        if self.config.f_match_mode == "window":
            candidates, valid_mask = self._build_window_candidates(grid_h, grid_w, self.config.f_window_radius, reference.device)

        keep_indices = []
        special_offsets = torch.arange(special_tokens, device=reference.device)

        for src_frame_idx in range(S):
            dst_frame_idx = self._select_dst_frame(src_frame_idx, S)
            src_patch = patch[src_frame_idx]
            dst_patch = patch[dst_frame_idx]
            if self.config.f_match_mode == "global":
                saliency = self._compute_global_saliency(src_patch, dst_patch)
            elif self.config.f_match_mode == "window":
                saliency = self._compute_window_saliency(src_patch, dst_patch, candidates, valid_mask)
            else:
                raise ValueError("f_match_mode must be either 'global' or 'window'.")

            patch_keep = torch.topk(saliency, k=keep_patch_tokens, largest=True, sorted=False).indices
            patch_keep = torch.sort(patch_keep).values + special_tokens
            frame_offset = src_frame_idx * P
            frame_special = frame_offset + special_offsets
            frame_patch = frame_offset + patch_keep
            keep_indices.append(torch.cat([frame_special, frame_patch], dim=0))

        return torch.cat(keep_indices, dim=0)

    def _select_dst_frame(self, frame_idx: int, num_frames: int) -> int:
        policy = self.config.f_dst_policy
        if policy == "next_frame":
            return frame_idx + 1 if frame_idx + 1 < num_frames else max(0, frame_idx - 1)
        if policy == "prev_frame":
            return frame_idx - 1 if frame_idx > 0 else min(num_frames - 1, frame_idx + 1)
        if policy == "anchor_frame":
            return 0
        raise ValueError("f_dst_policy must be one of 'next_frame', 'prev_frame', 'anchor_frame'.")

    def _compute_global_saliency(self, src_tokens: Tensor, dst_tokens: Tensor) -> Tensor:
        src_norm = F.normalize(src_tokens, dim=-1)
        dst_norm = F.normalize(dst_tokens, dim=-1)
        sim = src_norm @ dst_norm.transpose(0, 1)
        return 1.0 - sim.max(dim=-1).values

    def _compute_window_saliency(
        self,
        src_tokens: Tensor,
        dst_tokens: Tensor,
        candidates: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        src_norm = F.normalize(src_tokens, dim=-1)
        dst_norm = F.normalize(dst_tokens, dim=-1)
        safe_candidates = candidates.clamp(min=0)
        dst_candidates = dst_norm[safe_candidates]  # [P, K, D]
        sim = torch.einsum("pd,pkd->pk", src_norm, dst_candidates)
        sim = sim.masked_fill(~valid_mask, float("-inf"))
        return 1.0 - sim.max(dim=-1).values

    @staticmethod
    def _build_window_candidates(
        grid_h: int,
        grid_w: int,
        radius: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        max_candidates = (2 * radius + 1) ** 2
        candidates = torch.full((grid_h * grid_w, max_candidates), -1, dtype=torch.long, device=device)
        valid_mask = torch.zeros((grid_h * grid_w, max_candidates), dtype=torch.bool, device=device)

        for row in range(grid_h):
            for col in range(grid_w):
                patch_idx = row * grid_w + col
                write_idx = 0
                for drow in range(-radius, radius + 1):
                    for dcol in range(-radius, radius + 1):
                        neigh_row = row + drow
                        neigh_col = col + dcol
                        if 0 <= neigh_row < grid_h and 0 <= neigh_col < grid_w:
                            candidates[patch_idx, write_idx] = neigh_row * grid_w + neigh_col
                            valid_mask[patch_idx, write_idx] = True
                            write_idx += 1
        return candidates, valid_mask