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
        """向量化选对：消除 dst 冲突后按优先级取前 pair_count 对。

        由于 nearest-neighbor 匹配每个 src 恰好对应一个 dst，src 间不存在冲突，
        只需处理多个 src 映射到同一 dst 的情况：保留优先级最高（rank 最小）的 src。
        使用 scatter_reduce "amin" 实现全向量化，无 Python 循环。
        """
        if pair_count <= 0:
            empty = torch.empty(0, dtype=torch.long, device=scores.device)
            return empty, empty

        if policy not in ("highest", "lowest"):
            raise ValueError("d2_similarity_policy must be 'highest' or 'lowest'.")

        N = scores.shape[0]
        descending = (policy == "highest")
        order = torch.argsort(scores, descending=descending)   # [N] priority-sorted src indices

        # rank_of[src] = src 在 order 中的位置（越小优先级越高）
        rank_of = torch.empty(N, dtype=torch.long, device=scores.device)
        rank_of[order] = torch.arange(N, device=scores.device)

        # 对每个 dst，找到最高优先级（最小 rank）的 src
        best_rank_per_dst = torch.full((N,), N, dtype=torch.long, device=scores.device)
        best_rank_per_dst.scatter_reduce_(0, matched_dst, rank_of, reduce="amin", include_self=True)

        # canonical src：它是其 dst 的最高优先级 src（消除 dst 冲突后保留的唯一代表）
        is_canonical = rank_of == best_rank_per_dst[matched_dst]   # [N]

        # 按优先级顺序取前 pair_count 个 canonical 配对
        canonical_by_priority = order[is_canonical[order]]
        selected_src = canonical_by_priority[:pair_count]
        selected_dst = matched_dst[selected_src]
        return selected_src, selected_dst

    @staticmethod
    def _build_window_candidates(
        grid_h: int,
        grid_w: int,
        radius: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """预计算窗口候选索引表（向量化，无 Python 循环）。

        返回:
            candidates: [P, K] long，邻域 patch 的展平线性索引（越界处填 -1）
            valid_mask: [P, K] bool，True 表示对应候选合法
        其中 P = grid_h × grid_w，K = (2*radius+1)²。
        """
        # 偏移量网格 [K]
        dr = torch.arange(-radius, radius + 1, device=device)
        dc = torch.arange(-radius, radius + 1, device=device)
        drows, dcols = torch.meshgrid(dr, dc, indexing="ij")   # [2R+1, 2R+1]
        drows = drows.reshape(-1)                               # [K]
        dcols = dcols.reshape(-1)                               # [K]

        # 每个 patch 的行列坐标 [P]
        rows = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w).reshape(-1)
        cols = torch.arange(grid_w, device=device).unsqueeze(0).expand(grid_h, -1).reshape(-1)

        # 所有候选坐标 [P, K]
        nr = rows.unsqueeze(1) + drows.unsqueeze(0)
        nc = cols.unsqueeze(1) + dcols.unsqueeze(0)
        valid = (nr >= 0) & (nr < grid_h) & (nc >= 0) & (nc < grid_w)   # [P, K]

        clamped_idx = nr.clamp(0, grid_h - 1) * grid_w + nc.clamp(0, grid_w - 1)
        candidates = torch.where(valid, clamped_idx, torch.full_like(clamped_idx, -1))
        return candidates, valid