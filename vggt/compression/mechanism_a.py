import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable, Tuple
from .base import KVReductionHook, CompressionContext
from .config import CompressionConfig, get_rQ, get_layer_rKV
from .utils import cosine_similarity_batched


class TemporalStridePruning(KVReductionHook):
    """
    机制 A：非对称时序步长压缩（Spark3R 基线实现）

    KV 路径：每 rKV 帧保留一帧的所有 tokens（轻量级 token pruning）
    Q  路径：组内 token merging，默认分组大小 G=20
    """

    def __init__(self, config: CompressionConfig):
        self.config = config

    # ──────────────────────────────────────────────────────────────────────
    # 主接口
    # ──────────────────────────────────────────────────────────────────────

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

        # KV 路径：时序步长剪枝
        k_new, v_new = self._temporal_stride_prune(k, v, S, P, rKV)

        # Q 路径：组内 token merging
        # 仅当 enable_q_compression=True 且 rQ > 1 时才启用
        # 设 enable_q_compression=False 可使机制 A 退化为纯 KV 压缩，与机制 B/C 公平对比
        if self.config.enable_q_compression and rQ > 1:
            q_new, unmerge_fn = self._intra_group_merge_q(q, S, P, rQ,
                                                           ctx.special_tokens,
                                                           self.config.q_group_size)
        else:
            q_new, unmerge_fn = q, None

        return q_new, k_new, v_new, unmerge_fn

    # ──────────────────────────────────────────────────────────────────────
    # KV 路径：时序步长剪枝
    # ──────────────────────────────────────────────────────────────────────

    def _temporal_stride_prune(
        self, k: Tensor, v: Tensor,
        S: int, P: int, rKV: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        保留每隔 rKV 帧的所有 P 个 tokens（包括 special + patch）。

        k, v: [B, H, S*P, D]
        返回: k_pruned, v_pruned: [B, H, n_anchor*P, D]
        """
        if rKV <= 1:
            return k, v

        B, H, N, D = k.shape
        k_3d = k.reshape(B, H, S, P, D)  # [B, H, S, P, D]
        v_3d = v.reshape(B, H, S, P, D)

        anchor_idx = torch.arange(0, S, rKV, device=k.device)  # [n_anchor]
        k_pruned = k_3d[:, :, anchor_idx, :, :].reshape(B, H, -1, D)
        v_pruned = v_3d[:, :, anchor_idx, :, :].reshape(B, H, -1, D)

        return k_pruned, v_pruned

    # ──────────────────────────────────────────────────────────────────────
    # Q 路径：组内 token merging（基于循环空间偏移）
    # ──────────────────────────────────────────────────────────────────────

    def _intra_group_merge_q(
        self,
        q: Tensor,           # [B, H, S*P, D]
        S: int, P: int,
        rQ: int,
        special_tokens: int,  # = 5（仅用于传递兼容性，不再控制行为）
        G: int,               # 分组大小，默认 20
    ) -> Tuple[Tensor, Callable]:
        """
        将 S 帧分成若干组，每组 G 帧（最后一组可以不足 G 帧）。
        在每组内部执行 token merging，reduction factor = rQ。

        循环空间偏移策略（Spark3R 风格，O(S·P) 复杂度）：
          - 对于所有位置 p（含 special tokens）和帧索引 i（在组内 0-indexed）：
              若 i % rQ == p % rQ → 该 token 是 destination
              否则              → 是 source，合并到同一位置的 destination 帧
          - Special tokens 与 patch tokens 一视同仁（Spark3R 论文未提及对 special tokens
            做任何豁免，此处严格遵循论文描述）

        返回:
          q_merged:   [B, H, N_merged, D]，N_merged = groups * G//rQ * P + special_extra
          unmerge_fn: callable(x [B,H,N_merged,D]) -> [B,H,S*P,D]
        """
        B, H, N, D = q.shape
        device = q.device

        q_3d = q.reshape(B, H, S, P, D)  # [B, H, S, P, D]

        # 按组处理
        merged_parts = []
        group_info = []  # list of dict with unmerge info
        dst_offset = 0   # 在合并后序列中的偏移量

        group_starts = list(range(0, S, G))

        for g_start in group_starts:
            g_end = min(g_start + G, S)
            g_size = g_end - g_start

            q_group = q_3d[:, :, g_start:g_end, :, :]  # [B, H, g_size, P, D]

            # ── 1. 构建 destination mask ─────────────────────────────────
            # dst_mask: [g_size, P]，True = destination
            dst_mask = self._build_dst_mask(g_size, P, rQ, special_tokens, device)

            # ── 2. 将 q_group 重排为 [B, H, g_size*P, D]
            q_flat = q_group.reshape(B, H, g_size * P, D)

            # dst_mask 对应到 flat 索引
            dst_mask_flat = dst_mask.reshape(-1)          # [g_size * P]

            # 在完整序列中的绝对 flat 索引
            abs_offset = g_start * P
            all_flat_idx = torch.arange(g_size * P, device=device)
            dst_flat_idx = all_flat_idx[dst_mask_flat]    # [n_dst]
            src_flat_idx = all_flat_idx[~dst_mask_flat]   # [n_src]

            # ── 3. 提取 destination 和 source tokens ────────────────────
            # dst_tokens: [B, H, n_dst, D]
            dst_tokens = q_flat[:, :, dst_flat_idx, :]
            # src_tokens: [B, H, n_src, D]
            src_tokens = q_flat[:, :, src_flat_idx, :]

            # ── 4. 位置匹配（全向量化版本）──────────────────────────────
            src_to_dst_in_merged = self._build_position_match_vectorized(
                src_flat_idx, dst_flat_idx, P, special_tokens, rQ, g_size, device
            )

            # ── 5. Merge：dst_merged = average(dst, matched sources) ────
            n_dst = dst_flat_idx.shape[0]
            dst_expanded = src_to_dst_in_merged.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            dst_expanded = dst_expanded.expand(B, H, -1, D)  # [B, H, n_src, D]

            # 先计数：每个 dst 被多少个 src 匹配
            count = torch.zeros(B, H, n_dst, 1, device=device, dtype=q.dtype)
            count.scatter_add_(2, dst_expanded[..., :1], torch.ones_like(src_tokens[..., :1]))

            # 累加 src 贡献到 dst
            dst_merged = dst_tokens.clone()
            dst_merged.scatter_add_(2, dst_expanded, src_tokens)
            # 除以 (1 + count)
            dst_merged = dst_merged / (1.0 + count)

            merged_parts.append(dst_merged)  # [B, H, n_dst, D]

            # 记录 unmerge 信息
            abs_src_idx = abs_offset + src_flat_idx    # [n_src]
            abs_dst_idx = abs_offset + dst_flat_idx    # [n_dst]
            group_info.append({
                "n_dst": n_dst,
                "dst_offset": dst_offset,
                "abs_src_flat": abs_src_idx,
                "abs_dst_flat": abs_dst_idx,
                "src_to_dst_in_merged": src_to_dst_in_merged,
            })
            dst_offset += n_dst

        q_merged = torch.cat(merged_parts, dim=2)  # [B, H, N_merged, D]

        # ── 构建 unmerge 函数 ─────────────────────────────────────────────
        def unmerge_fn(x_merged: Tensor) -> Tensor:
            """
            x_merged: [B, H, N_merged, D]
            Returns:  [B, H, S*P, D]
            """
            B2, H2, _, D2 = x_merged.shape
            out = torch.empty(B2, H2, S * P, D2, device=x_merged.device, dtype=x_merged.dtype)

            for info in group_info:
                n_dst = info["n_dst"]
                dst_off = info["dst_offset"]
                abs_dst = info["abs_dst_flat"]    # [n_dst]
                abs_src = info["abs_src_flat"]    # [n_src]
                src2dst = info["src_to_dst_in_merged"]  # [n_src]

                # merged 序列中该组的 dst 部分
                merged_slice = x_merged[:, :, dst_off:dst_off + n_dst, :]  # [B,H,n_dst,D]

                # 写回 dst 到原始序列
                idx_dst = abs_dst.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B2, H2, -1, D2)
                out.scatter_(2, idx_dst, merged_slice)

                # 写回 src：每个 src 复制其对应 dst 的输出
                src_output = merged_slice[:, :, src2dst, :]  # [B,H,n_src,D]
                idx_src = abs_src.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B2, H2, -1, D2)
                out.scatter_(2, idx_src, src_output)

            return out

        return q_merged, unmerge_fn

    # ──────────────────────────────────────────────────────────────────────
    # 辅助：构建 destination mask
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_dst_mask(
        g_size: int, P: int, rQ: int,
        special_tokens: int, device: torch.device,
    ) -> Tensor:
        """
        dst_mask: [g_size, P]，True = destination token

        规则（严格遵循 Spark3R 论文 Section IV-B，不对 special tokens 做豁免）：
          - 所有位置 p（含 camera/register special tokens）和帧 i（组内 0-indexed）：
              若 i % rQ == p % rQ → destination
              否则              → source
        Special tokens 与 patch tokens 采用完全相同的循环偏移规则。
        """
        dst_mask = torch.zeros(g_size, P, dtype=torch.bool, device=device)
        # 所有位置（含 special tokens）均按循环偏移规则决定 dst/src
        for i in range(g_size):
            offset = i % rQ
            all_positions = torch.arange(P, device=device)
            is_dst = (all_positions % rQ) == offset
            dst_mask[i, :] = is_dst
        return dst_mask

    @staticmethod
    def _build_position_match_vectorized(
        src_flat_idx: Tensor,  # [n_src]，在 g_size*P 中的 flat 位置
        dst_flat_idx: Tensor,  # [n_dst]
        P: int, special_tokens: int, rQ: int,
        g_size: int, device: torch.device,
    ) -> Tensor:
        """
        对每个 source token，找到同一 patch 位置的 destination token（在 dst_flat_idx 中的下标）。
        O(n_src) 全向量化版本。
        """
        dst_frame = dst_flat_idx // P
        dst_patch = dst_flat_idx % P
        src_frame = src_flat_idx // P
        src_patch = src_flat_idx % P
        n_src = src_flat_idx.shape[0]
        n_dst = dst_flat_idx.shape[0]

        # 所有 token（含 special tokens）均按同一公式匹配：
        # source (frame_i, pos_p) → destination 帧满足 frame % rQ == p % rQ
        # 最近 dst 帧 = (src_frame // rQ) * rQ + (p % rQ)，再 clip 到 [0, g_size)
        target_frame_raw = (src_frame // rQ) * rQ + (src_patch % rQ)
        target_frame = target_frame_raw.clamp(0, g_size - 1)
        # （无 special token 豁免，与 Spark3R 论文一致）

        # 计算目标 flat idx（在 g_size*P 中）
        target_flat = target_frame * P + src_patch  # [n_src]

        # 在 dst_flat_idx 中查找 target_flat 的位置
        lookup = torch.full((g_size * P,), -1, dtype=torch.long, device=device)
        dst_positions = torch.arange(n_dst, device=device)
        lookup.scatter_(0, dst_flat_idx, dst_positions)

        match = lookup[target_flat]   # [n_src]
        # 若 match == -1（未找到），回退到 0（安全兜底）
        match = match.clamp(min=0)
        return match
