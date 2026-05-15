import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionContext, KVReductionHook
from .config import CompressionConfig
from .utils import gather_tokens


class LocalRedundancyPairPruning(KVReductionHook):
    """方案 D2：相邻帧局部窗口冗余对删除。"""

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
        if self.config.d2_apply_to_q:
            raise NotImplementedError("D2 Q-path compression is not implemented yet; keep d2_apply_to_q=False.")

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

        frames = reference.reshape(reference.shape[0], reference.shape[1], S, P, reference.shape[-1])
        patch = frames[:, :, :, special_tokens:, :].mean(dim=(0, 1))  # [S, P_patch, D]
        delete_mask = torch.zeros(S, patch_tokens, dtype=torch.bool, device=reference.device)
        candidates, valid_mask = self._build_window_candidates(grid_h, grid_w, self.config.d2_window_radius, reference.device)

        for src_frame_idx in range(0, S - self.config.d2_pair_stride):
            dst_frame_idx = src_frame_idx + self.config.d2_pair_stride
            scores, matched_dst = self._match_local_windows(
                patch[src_frame_idx],
                patch[dst_frame_idx],
                candidates,
                valid_mask,
            )
            pair_count = max(1, int(math.ceil(scores.numel() * self.config.d2_drop_ratio)))
            selected_src, selected_dst = self._select_redundant_pairs(
                scores,
                matched_dst,
                pair_count,
                self.config.d2_similarity_policy,
            )
            if selected_src.numel() == 0:
                continue
            delete_mask[src_frame_idx, selected_src] = True
            delete_mask[dst_frame_idx, selected_dst] = True

        keep_mask = torch.ones(S, P, dtype=torch.bool, device=reference.device)
        keep_mask[:, special_tokens:] = ~delete_mask
        return keep_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

    def _match_local_windows(
        self,
        src_tokens: Tensor,
        dst_tokens: Tensor,
        candidates: Tensor,
        valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        src_norm = F.normalize(src_tokens, dim=-1)
        dst_norm = F.normalize(dst_tokens, dim=-1)
        safe_candidates = candidates.clamp(min=0)
        dst_candidates = dst_norm[safe_candidates]  # [P, K, D]
        sim = torch.einsum("pd,pkd->pk", src_norm, dst_candidates)
        sim = sim.masked_fill(~valid_mask, float("-inf"))
        scores, best_idx = sim.max(dim=-1)
        matched_dst = safe_candidates.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
        return scores, matched_dst

    def _select_redundant_pairs(
        self,
        scores: Tensor,
        matched_dst: Tensor,
        pair_count: int,
        policy: str,
    ) -> Tuple[Tensor, Tensor]:
        if pair_count <= 0:
            empty = torch.empty(0, dtype=torch.long, device=scores.device)
            return empty, empty

        if policy == "highest":
            order = torch.argsort(scores, descending=True)
        elif policy == "lowest":
            order = torch.argsort(scores, descending=False)
        else:
            raise ValueError("d2_similarity_policy must be either 'highest' or 'lowest'.")

        used_src = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
        used_dst = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
        selected_src = []
        selected_dst = []

        for src_idx in order.tolist():
            dst_idx = int(matched_dst[src_idx].item())
            if used_src[src_idx] or used_dst[dst_idx]:
                continue
            used_src[src_idx] = True
            used_dst[dst_idx] = True
            selected_src.append(src_idx)
            selected_dst.append(dst_idx)
            if len(selected_src) >= pair_count:
                break

        if not selected_src:
            empty = torch.empty(0, dtype=torch.long, device=scores.device)
            return empty, empty
        return (
            torch.tensor(selected_src, dtype=torch.long, device=scores.device),
            torch.tensor(selected_dst, dtype=torch.long, device=scores.device),
        )

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