import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable, Tuple
from .base import KVReductionHook, CompressionContext
from .config import CompressionConfig, get_rQ, get_layer_rKV
from .mechanism_a import TemporalStridePruning


class QueryDCTMerging(KVReductionHook):
    """
    机制 E：Q 路径使用 DCT 代表元选择（替代机制 A 的循环偏移策略）

    KV 路径：与机制 A 完全相同（时序步长剪枝）
    Q  路径：在每组 G 帧内，对每个 patch 位置计算 Q 的时序均值（DC 分量），
             选取与 DC 余弦相似度最高的帧作为 destination，其余帧为 source 并合并入 destination。

    改进点：
      - DC 代表元在 L2 意义下最小化总合并误差（vs 循环偏移的无关内容选择）
      - 无需 O(G²) 相似度计算，O(G*P) 即可完成
    """

    def __init__(self, config: CompressionConfig):
        self.config = config
        # 复用机制 A 的 KV 路径实现
        self._stride_pruner = TemporalStridePruning(config)

    def compress(
        self,
        q: Tensor, k: Tensor, v: Tensor,
        ctx: CompressionContext,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Callable]]:
        if not ctx.is_global or ctx.S < self.config.min_frames_to_compress:
            return q, k, v, None

        S, P = ctx.S, ctx.P
        rKV = get_layer_rKV(S, ctx.layer_idx, self.config.kv_insensitive_multiplier)
        rQ = get_rQ(S)

        # KV 路径：与机制 A 相同（时序步长剪枝）
        k_new, v_new = self._stride_pruner._temporal_stride_prune(k, v, S, P, rKV)

        # Q 路径：DCT 代表元选择
        if self.config.enable_q_compression and rQ > 1:
            q_new, unmerge_fn = self._dct_representative_merge_q(
                q, S, P, rQ, ctx.special_tokens, self.config.q_group_size
            )
        else:
            q_new, unmerge_fn = q, None

        return q_new, k_new, v_new, unmerge_fn

    def _dct_representative_merge_q(
        self,
        q: Tensor,          # [B, H, S*P, D]
        S: int, P: int,
        rQ: int,
        special_tokens: int,
        G: int,             # 分组大小，默认 20
    ) -> Tuple[Tensor, Callable]:
        """
        在每组 G 帧内，对每个 patch 位置：
          1. 计算该位置 G 帧 Q 的均值（DC 分量，[B, H, P, D]）
          2. 找到与 DC 余弦相似度最高的帧作为 destination
          3. 将其余帧的 Q 合并入 destination（加权平均）
        """
        B, H, N, D = q.shape
        device = q.device
        q_3d = q.reshape(B, H, S, P, D)   # [B, H, S, P, D]

        merged_parts = []
        group_info = []
        dst_offset = 0

        group_starts = list(range(0, S, G))

        for g_start in group_starts:
            g_end = min(g_start + G, S)
            g_size = g_end - g_start

            q_group = q_3d[:, :, g_start:g_end, :, :]   # [B, H, g_size, P, D]

            # ── 1. 计算 DC 分量（时序均值）────────────────────────────────
            q_dc = q_group.mean(dim=2, keepdim=True)  # [B, H, 1, P, D]

            # ── 2. 计算每帧与 DC 的余弦相似度 ─────────────────────────────
            q_group_norm = F.normalize(q_group, dim=-1)   # [B, H, g_size, P, D]
            q_dc_norm    = F.normalize(q_dc,    dim=-1)   # [B, H, 1, P, D]
            sim = (q_group_norm * q_dc_norm).sum(dim=-1)  # [B, H, g_size, P]

            # 取 batch-mean 的 sim 后 argmax，得到统一的 destination 帧
            sim_mean = sim.mean(dim=(0, 1))       # [g_size, P]
            dst_frame_idx_unified = sim_mean.argmax(dim=0)  # [P]

            # ── 3. 构建 dst_mask [g_size, P] ────────────────────────────
            dst_mask = torch.zeros(g_size, P, dtype=torch.bool, device=device)
            # Special tokens：所有帧都是 destination
            dst_mask[:, :special_tokens] = True
            # Patch tokens：每个位置只有 destination 帧是 dst
            patch_positions = torch.arange(special_tokens, P, device=device)   # [P_patch]
            dst_frames_for_patches = dst_frame_idx_unified[special_tokens:]    # [P_patch]
            dst_mask[dst_frames_for_patches, patch_positions] = True

            # ── 4. 提取 dst/src tokens，执行合并 ──────────────────────────
            q_flat = q_group.reshape(B, H, g_size * P, D)
            dst_mask_flat = dst_mask.reshape(-1)

            abs_offset_val = g_start * P
            all_flat_idx = torch.arange(g_size * P, device=device)
            dst_flat_idx = all_flat_idx[dst_mask_flat]   # [n_dst]
            src_flat_idx = all_flat_idx[~dst_mask_flat]  # [n_src]
            n_dst = dst_flat_idx.shape[0]

            dst_tokens = q_flat[:, :, dst_flat_idx, :]   # [B, H, n_dst, D]
            src_tokens = q_flat[:, :, src_flat_idx, :]   # [B, H, n_src, D]

            # 位置匹配：source (frame_i, pos_p) → destination at (dst_frame[p], pos_p)
            src_patch_pos = src_flat_idx % P               # [n_src]
            dst_for_src_flat = (
                dst_frame_idx_unified[src_patch_pos] * P + src_patch_pos
            )  # [n_src]

            # special 位置：同帧的 special dst
            is_special_src = src_patch_pos < special_tokens
            if is_special_src.any():
                src_frame_for_special = src_flat_idx[is_special_src] // P
                dst_for_src_flat[is_special_src] = (
                    src_frame_for_special * P + src_patch_pos[is_special_src]
                )

            # 建立查找表 flat -> dst_position
            lookup = torch.full((g_size * P,), -1, dtype=torch.long, device=device)
            lookup.scatter_(0, dst_flat_idx, torch.arange(n_dst, device=device))
            src_to_dst_in_merged = lookup[dst_for_src_flat].clamp(min=0)  # [n_src]

            # 合并：dst ← average(dst, matched sources)
            dst_expanded = (
                src_to_dst_in_merged
                .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                .expand(B, H, -1, D)
            )
            count = torch.zeros(B, H, n_dst, 1, device=device, dtype=q.dtype)
            count.scatter_add_(2, dst_expanded[..., :1], torch.ones_like(src_tokens[..., :1]))
            dst_merged = dst_tokens.clone()
            dst_merged.scatter_add_(2, dst_expanded, src_tokens)
            dst_merged = dst_merged / (1.0 + count)

            merged_parts.append(dst_merged)

            group_info.append({
                "n_dst": n_dst,
                "dst_offset": dst_offset,
                "abs_src_flat": abs_offset_val + src_flat_idx,
                "abs_dst_flat": abs_offset_val + dst_flat_idx,
                "src_to_dst_in_merged": src_to_dst_in_merged,
            })
            dst_offset += n_dst

        q_merged = torch.cat(merged_parts, dim=2)   # [B, H, N_merged, D]

        def unmerge_fn(x_merged: Tensor) -> Tensor:
            """恢复 [B, H, S*P, D] 形状"""
            B2, H2, _, D2 = x_merged.shape
            out = torch.empty(B2, H2, S * P, D2, device=x_merged.device, dtype=x_merged.dtype)
            for info in group_info:
                n_dst = info["n_dst"]
                dst_off = info["dst_offset"]
                abs_dst = info["abs_dst_flat"]
                abs_src = info["abs_src_flat"]
                s2d = info["src_to_dst_in_merged"]
                merged_slice = x_merged[:, :, dst_off:dst_off + n_dst, :]
                idx_dst = abs_dst.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B2, H2, -1, D2)
                out.scatter_(2, idx_dst, merged_slice)
                src_output = merged_slice[:, :, s2d, :]
                idx_src = abs_src.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B2, H2, -1, D2)
                out.scatter_(2, idx_src, src_output)
            return out

        return q_merged, unmerge_fn
