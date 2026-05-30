"""
机制 G：DPP Attention —— 基于行列式点过程的 Global Attention Token Reduction

参考文献：
  CDPruner: Contextual Diversity Pruner
    arXiv:2506.10967 | https://github.com/Theia-4869/CDPruner

核心思路：
  在每个 Global Attention Block 执行前，对每帧的 patch token 用 DPP 子集选择：
    1. 帧内余弦相似度 Gram 矩阵（diversity）
    2. 帧间同位置 L×L 窗口相似度（relevance，跨帧变化越大越重要）
    3. L-ensemble Kernel = diag(q) * Sim * diag(q)
    4. Fast Greedy MAP（Cholesky 秩一更新）选出 M 个 patch token
  被选 patch token + 所有特殊 token（register + camera）参与 Global Attention，
  结果 scatter back 原位；未选 patch token 保持 frame attention 输出不变，
  由后续 frame attention 传递跨帧信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DPPConfig:
    """DPP Attention 的全部超参数。"""

    keep_ratio: float = 0.5
    """每帧保留 patch token 的比例，M = floor(keep_ratio × N)"""

    window_size: int = 3
    """Relevance 计算的帧内邻域窗口边长 L，实际窗口 L×L；建议奇数"""

    num_adj_frames: int = -1
    """参与 relevance 计算的相邻帧数 K_adj；-1 表示全部帧（除自身）"""

    relevance_agg: Literal["max", "mean"] = "max"
    """跨帧邻域相似度的聚合方式：max（激进）或 mean（平滑）"""

    on_all_layers: bool = True
    """True：每个 global block 前重新计算 DPP 子集；False：第一层计算后全层复用"""

    protect_special: bool = True
    """始终保留所有特殊 token（register + camera），不参与剪枝"""

    def __post_init__(self):
        assert 0.0 < self.keep_ratio <= 1.0, "keep_ratio 必须在 (0, 1]"
        assert self.window_size >= 1 and self.window_size % 2 == 1, (
            "window_size 必须为正奇数"
        )
        assert self.num_adj_frames == -1 or self.num_adj_frames >= 1


# ══════════════════════════════════════════════════════════════════════════════
# DPPTokenSelector
# ══════════════════════════════════════════════════════════════════════════════

class DPPTokenSelector:
    """
    为每帧 patch token 执行基于 DPP 的子集选择。

    典型调用方式（每个 global block 调用一次）：
        selector = DPPTokenSelector(cfg)
        select_idx = selector.select(patch_tokens, patch_h, patch_w)
        #   patch_tokens : [S, N, C]  —— 仅包含 patch token，不含特殊 token
        #   select_idx   : [S, M]     —— 每帧选出的 M 个 patch 索引（已排序）
    """

    def __init__(self, cfg: DPPConfig) -> None:
        self.cfg = cfg

    # ──────────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────────

    def select(
        self,
        patch_tokens: torch.Tensor,  # [S, N, C]
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:               # [S, M]  已排序的 patch 索引
        S, N, _ = patch_tokens.shape
        M = max(1, int(round(N * self.cfg.keep_ratio)))

        if M >= N:
            # keep_ratio == 1.0：直接返回全部索引
            return torch.arange(N, device=patch_tokens.device).unsqueeze(0).expand(S, -1)

        with torch.no_grad():
            sim    = self._compute_similarity(patch_tokens)                    # [S, N, N]
            rel    = self._compute_relevance(patch_tokens, patch_h, patch_w)   # [S, N]
            kernel = self._build_kernel(sim, rel)                               # [S, N, N]
            idx    = self._greedy_map(kernel, M)                                # [S, M]

        return idx

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1：帧内 Gram 矩阵（余弦相似度）
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_similarity(
        self,
        patch_tokens: torch.Tensor,  # [S, N, C]
    ) -> torch.Tensor:               # [S, N, N]
        """帧内 Gram 矩阵，元素为单位化 token 对之间的点积（余弦相似度）。"""
        v_hat = F.normalize(patch_tokens, dim=-1)              # [S, N, C]
        sim   = torch.bmm(v_hat, v_hat.transpose(1, 2))        # [S, N, N]
        return sim

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2：帧间相关性得分（Relevance）
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_relevance(
        self,
        patch_tokens: torch.Tensor,  # [S, N, C]；N = patch_h × patch_w
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:               # [S, N]  值域 [0, 2]，越大越值得保留
        """
        对每帧每个 patch token 计算跨帧独特性得分。
        若 token 与参考帧同位置邻域高度相似（可由其他帧代替），得分低；
        若跨帧变化显著（携带独特信息），得分高。

        实现：
            1. 将 patch token 重排为 feature map [S, C, H, W]
            2. F.unfold 一次性提取所有位置的 L×L 邻域
            3. 对参考帧集合计算点积（余弦相似度的近似）
            4. q_i = 1 - agg(cos)，agg 为 max 或 mean
        """
        S, N, C = patch_tokens.shape
        L    = self.cfg.window_size
        half = L // 2
        dev  = patch_tokens.device

        # 确定参考帧数
        k_adj = self.cfg.num_adj_frames if self.cfg.num_adj_frames > 0 else (S - 1)
        k_adj = min(k_adj, S - 1)

        # 单位化并重排为特征图 [S, C, H, W]
        v_hat    = F.normalize(patch_tokens, dim=-1)              # [S, N, C]
        feat_map = v_hat.view(S, patch_h, patch_w, C)
        feat_map = feat_map.permute(0, 3, 1, 2).contiguous()     # [S, C, H, W]

        # 展开 L×L 邻域：[S, C*L*L, N]
        # 边界用 0 填充；补零位置的点积为 0（偏保守估计，可接受）
        unfolded = F.unfold(feat_map, kernel_size=L, padding=half)  # [S, C*L*L, N]
        unfolded = unfolded.permute(0, 2, 1)                         # [S, N, C*L*L]
        unfolded = unfolded.view(S, N, L * L, C)                     # [S, N, L*L, C]

        relevance = torch.zeros(S, N, device=dev, dtype=patch_tokens.dtype)

        for s in range(S):
            # 按帧索引距离升序选取 k_adj 个参考帧
            others     = [s_ for s_ in range(S) if s_ != s]
            ref_frames = sorted(others, key=lambda s_: abs(s_ - s))[:k_adj]

            if not ref_frames:
                # 仅一帧时无跨帧参考，视全部 token 为独特
                relevance[s] = 1.0
                continue

            # ref_unfolded: [K, N, L*L, C]
            ref_unfolded = unfolded[ref_frames]         # [K, N, L*L, C]

            # 查询向量 v_hat[s]: [N, C] → [1, N, 1, C]
            query = v_hat[s].unsqueeze(0).unsqueeze(2)  # [1, N, 1, C]

            # 点积 → 近似余弦相似度（因为 v_hat 已归一化，但 unfolded 中补零位置模长为 0）
            # [K, N, L*L]
            cos = (query * ref_unfolded).sum(dim=-1)

            if self.cfg.relevance_agg == "max":
                # 取 K 帧 × L*L 邻域中最大相似度
                max_cos = cos.reshape(-1, N).max(dim=0).values   # [N]
                relevance[s] = 1.0 - max_cos.clamp(-1.0, 1.0)
            else:
                # 取均值（包含补零位置的 0，会轻微低估均值）
                mean_cos = cos.mean(dim=(0, 2))                  # [N]
                relevance[s] = 1.0 - mean_cos.clamp(-1.0, 1.0)

        return relevance  # [S, N]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3：L-ensemble Kernel
    # ──────────────────────────────────────────────────────────────────────────

    def _build_kernel(
        self,
        sim:       torch.Tensor,  # [S, N, N]
        relevance: torch.Tensor,  # [S, N]
    ) -> torch.Tensor:            # [S, N, N]
        """
        L[i,j] = q_i * sim[i,j] * q_j
        对角项 L[i,i] = q_i^2 * 1，编码 token i 的显著性；
        非对角项编码两 token 的冗余性。
        最大化 det(L_S) 促使被选 token 相互正交且各自相关性高。
        """
        q      = relevance.unsqueeze(2)              # [S, N, 1]
        kernel = q * sim * q.transpose(1, 2)         # [S, N, N]
        return kernel

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4：Fast Greedy MAP 推断（Cholesky 秩一更新，S 帧并行）
    # ──────────────────────────────────────────────────────────────────────────

    def _greedy_map(
        self,
        kernel: torch.Tensor,  # [S, N, N]
        M:      int,
    ) -> torch.Tensor:         # [S, M]  已排序的 patch 索引
        """
        贪心 MAP 推断：每步选使 log det 增量最大的 token，
        利用 Cholesky 秩一更新将每步的额外计算从 O(N^2) 降至 O(N)。
        总复杂度 O(M*N)，近似保证 (1-1/e) OPT（Chen et al. 2018）。

        参考：CDPruner llava/model/llava_arch.py 的批量 MAP 实现。
        """
        S, N, _ = kernel.shape
        dev     = kernel.device
        dtype   = kernel.dtype
        s_range = torch.arange(S, device=dev)

        # cis[t, s, n]：第 t 步第 s 帧的 Cholesky 列向量 e_t[n]
        cis  = torch.zeros(M, S, N, device=dev, dtype=dtype)
        # di2s[s, n]：det 增量的分母（剩余"平方长度"）；初始为对角元素
        di2s = kernel.diagonal(dim1=1, dim2=2).clone()     # [S, N]

        select_idx = torch.empty(M, S, dtype=torch.long, device=dev)

        for i in range(M):
            # 1. 选出每帧中 di2s 最大的 token（行列式增量最大化）
            j = di2s.argmax(dim=-1)        # [S]
            select_idx[i] = j

            # 2. 提取 kernel 的第 j[s] 行：kernel[s, j[s], :] → [S, N]
            row_j = kernel[s_range, j]     # [S, N]

            # 3. Cholesky 秩一更新
            if i > 0:
                # ci_j[t, s] = cis[t, s, j[s]]：前 i 步在 j[s] 处的 Cholesky 分量
                ci_j = cis[:i, s_range, j]                              # [i, S]
                # correction[s, n] = Σ_{t<i} cis[t,s,n] * ci_j[t,s]
                correction = torch.einsum("ts,tsn->sn", ci_j, cis[:i])  # [S, N]
                denom = torch.sqrt(di2s[s_range, j].clamp(min=1e-8)).unsqueeze(-1)
                eis   = (row_j - correction) / denom
            else:
                denom = torch.sqrt(di2s[s_range, j].clamp(min=1e-8)).unsqueeze(-1)
                eis   = row_j / denom

            # 4. 更新缓存
            cis[i]           = eis
            di2s            -= eis.pow(2)
            # 已选 token 的 di2s 设为 -inf，避免重复选择
            di2s[s_range, j] = float("-inf")

        # 对每帧选出的索引排序（保持位置稳定性，方便 scatter）
        select_idx = select_idx.t().contiguous()           # [S, M]
        select_idx = torch.sort(select_idx, dim=1).values  # [S, M]
        return select_idx
