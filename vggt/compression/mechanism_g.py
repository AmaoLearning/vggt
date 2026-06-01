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

    selection_mode: Literal["dpp", "random", "topk_stable", "dpp_diversity"] = "dpp"
    """Token 选择策略：
    - 'dpp'           : DPP Greedy MAP 推断（relevance × diversity）
    - 'random'        : 均匀随机采样（对照实验，验证 gather-attend-scatter 本身的影响）
    - 'topk_stable'   : 仅按 relevance 分数排序取 Top-M（无 diversity 约束），
                        用于隔离「稳定性选择」与「帧内多样性选择」两个效果
    - 'dpp_diversity' : 纯帧内多样性 DPP（令 q_i=1，即 L[i,j]=sim[i,j]），
                        不依赖任何跨帧信号，完全由帧内余弦相似度驱动；
                        与 'random' 对比可独立验证 DPP diversity 项的贡献"""

    proj_dim: int = 64
    """相似度 / 相关性计算所使用的嵌入维度（取 feature 前 proj_dim 维）；
    0 表示使用全部 C 维（精确但慢）；默认 64 以大幅降低计算开销"""

    max_map_steps: int = 128
    """Greedy MAP 推断的最大迭代步数，超出部分用均匀随机采样补全。

    Greedy MAP 每步约产生 1 ms Python→GPU kernel launch 延迟，
    总开销 ≈ 1 ms × M × num_blocks（线性于 M，与 FLOPs 无关）。
    限制为 128 步可将所有 keep_ratio 配置的 DPP 开销均匀控制在 ~1.5 s，
    同时由 DPP 保证前 128 个最重要 anchor token 的多样性，
    其余 slot 由随机采样填充。0 = 不限制（完整 Greedy MAP）"""

    def __post_init__(self):
        assert 0.0 < self.keep_ratio <= 1.0, "keep_ratio 必须在 (0, 1]"
        assert self.window_size >= 1 and self.window_size % 2 == 1, (
            "window_size 必须为正奇数"
        )
        assert self.num_adj_frames == -1 or self.num_adj_frames >= 1
        assert self.proj_dim >= 0, "proj_dim 必须为非负整数（0 = 使用全部 C 维）"
        assert self.max_map_steps >= 0, "max_map_steps 必须为非负整数（0 = 不限制）"


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

        # ── 随机采样路径（对照实验） ──────────────────────────────────────────
        if self.cfg.selection_mode == "random":
            # 向量化均匀随机采样（每帧独立，无 Python for-loop）
            noise = torch.rand(S, N, device=patch_tokens.device)
            idx   = noise.argsort(dim=1)[:, :M]          # [S, M]  未排序
            return torch.sort(idx, dim=1).values           # [S, M]  已排序

        # ── Top-K Stable 路径（纯 relevance 排序，无 diversity）─────────────
        # 用于隔离「稳定性选择」效果：排除 DPP diversity 项后，
        # 仅保留跨帧相似度最高的 M 个 token。
        if self.cfg.selection_mode == "topk_stable":
            with torch.no_grad():
                rel = self._compute_relevance(patch_tokens, patch_h, patch_w)  # [S, N]
            _, idx = rel.topk(M, dim=1)                   # [S, M]  未排序
            return torch.sort(idx, dim=1).values           # [S, M]  已排序

        # ── 纯帧内多样性 DPP 路径（q_i=1，无 relevance）─────────────────────
        # 令所有 relevance = 1，L-ensemble kernel 退化为帧内余弦相似度矩阵：
        #   L[i,j] = 1 * sim[i,j] * 1 = sim[i,j]
        # 选择完全由帧内 token 间的「互斥性」（越不相似越倾向同时被选）驱动，
        # 不依赖任何跨帧信号。与 random 对比可独立量化 DPP diversity 项的价值。
        if self.cfg.selection_mode == "dpp_diversity":
            with torch.no_grad():
                sim    = self._compute_similarity(patch_tokens)                # [S, N, N]
                rel    = torch.ones(S, N, device=patch_tokens.device,
                                    dtype=patch_tokens.dtype)                  # q_i = 1
                kernel = self._build_kernel(sim, rel)                          # [S, N, N]
                idx    = self._greedy_map(kernel, M)                           # [S, M]
            return idx

        # ── DPP 路径 ────────────────────────────────────────────────────────
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
        """帧内 Gram 矩阵（余弦相似度）。若 proj_dim>0，先截取前 proj_dim 维降低计算量。"""
        C     = patch_tokens.shape[-1]
        C_eff = self.cfg.proj_dim if 0 < self.cfg.proj_dim < C else C
        v_hat = F.normalize(patch_tokens[..., :C_eff], dim=-1)  # [S, N, C_eff]
        sim   = torch.bmm(v_hat, v_hat.transpose(1, 2))          # [S, N, N]
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
        对每帧每个 patch token 计算跨帧独特性得分（向量化实现）。

        核心优化：用 einsum("snc,tnc->stn") 一次性计算所有帧对的位置相似度 [S,S,N]，
        取代原先 Python for-loop + 大 ref_unfolded 分配（[K,N,L²,C] 逐帧分配导致
        巨量 GPU 内存往返）。通过 max_pool2d 在空间维度上实现 L×L 窗口聚合。

        proj_dim 只影响 C_eff 维度大小，不改变输出语义：
          C_eff = proj_dim（若 0 < proj_dim < C）或 C（全量）

        算法步骤：
            1. 取前 C_eff 维并单位化：v [S, N, C_eff]
            2. pairwise[s, t, n] = v[s,n] · v[t,n]，形状 [S, S, N]
            3. 若 L>1：对 pairwise 做 max_pool2d(kernel=L) 在空间维度扩展窗口
            4. 屏蔽自帧（s==t）和非相邻帧（超出 k_adj 范围）
            5. 对有效参考帧聚合（max 或 mean），得到每帧每 token 的跨帧最大相似度
            6. relevance = 1 - max_sim（越不相似 = 越独特 = 得分越高）
        """
        S, N, C = patch_tokens.shape
        L    = self.cfg.window_size
        half = L // 2
        dev  = patch_tokens.device
        dtype = patch_tokens.dtype

        if S <= 1:
            # 单帧无参考，所有 token 均视为独特
            return torch.ones(S, N, device=dev, dtype=dtype)

        # 确定参考帧数
        k_adj = self.cfg.num_adj_frames if self.cfg.num_adj_frames > 0 else (S - 1)
        k_adj = min(k_adj, S - 1)

        # ── Step 1：低维投影 + 单位化 ────────────────────────────────────────
        C_eff = self.cfg.proj_dim if 0 < self.cfg.proj_dim < C else C
        v = F.normalize(patch_tokens[..., :C_eff], dim=-1)  # [S, N, C_eff]

        # ── Step 2：向量化全帧对逐位置点积 ──────────────────────────────────
        # pairwise[s, t, n] = v[s,n] · v[t,n]，O(S²·N·C_eff)
        pairwise = torch.einsum("snc,tnc->stn", v, v)  # [S, S, N]

        # ── Step 3：空间窗口扩展（max-pooling 代替 F.unfold） ───────────────
        # 对 pairwise 中第 t 帧在位置 n 处的值，扩展到 n 周围 L×L 邻域的最大值，
        # 等效于：对参考帧 t 的 feature map 先做 max_pool 再与 query 帧比较。
        if L > 1:
            p      = pairwise.reshape(S * S, 1, patch_h, patch_w)
            p_win  = F.max_pool2d(p, kernel_size=L, stride=1, padding=half)
            pairwise = p_win.reshape(S, S, N)

        # ── Step 4：屏蔽无效帧 ───────────────────────────────────────────────
        # 4a. 自帧（s==t）置 -inf
        eye = torch.eye(S, device=dev, dtype=torch.bool)          # [S, S]
        pairwise = pairwise.masked_fill(eye.unsqueeze(-1), float("-inf"))

        # 4b. 仅保留距离最近的 k_adj 帧
        if k_adj < S - 1:
            s_idx = torch.arange(S, device=dev)
            dist  = (s_idx.unsqueeze(0) - s_idx.unsqueeze(1)).abs()  # [S, S]
            dist.fill_diagonal_(S + 1)                               # 自帧距离设大
            _, top_k = dist.topk(k_adj, dim=1, largest=False)        # [S, k_adj]
            keep = torch.zeros(S, S, device=dev, dtype=torch.bool)
            keep.scatter_(1, top_k, True)
            pairwise = pairwise.masked_fill(~keep.unsqueeze(-1), float("-inf"))

        # ── Step 5：跨帧聚合 ─────────────────────────────────────────────────
        # relevance 衡量 token 对 Global Attention 的「可用性」。
        # Global Attention 目标是跨帧空间对应（cross-frame correspondence），
        # 因此跨帧相似度高（稳定、可匹配）的 token 应有更高 relevance。
        # 公式：q_i = (1 + max_sim) / 2 ∈ [0, 1]
        #   max_sim → +1（高度相似，稳定特征） → q_i → 1（高优先保留）
        #   max_sim → -1（完全不同，动态/遮挡）→ q_i → 0（低优先）
        if self.cfg.relevance_agg == "max":
            max_sim   = pairwise.max(dim=1).values        # [S, N]（-inf 被忽略）
            relevance = (1.0 + max_sim.clamp(-1.0, 1.0)) / 2.0
        else:
            valid_mask  = pairwise != float("-inf")        # [S, S, N]
            count       = valid_mask.float().sum(dim=1).clamp(min=1.0)  # [S, N]
            pairwise_s  = pairwise.masked_fill(~valid_mask, 0.0)
            mean_sim    = pairwise_s.sum(dim=1) / count    # [S, N]
            relevance   = (1.0 + mean_sim.clamp(-1.0, 1.0)) / 2.0

        return relevance.to(dtype)  # [S, N]

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

        # 确定实际执行的 Greedy MAP 步数
        # max_map_steps=0 表示不限制；否则上限为 min(M, max_map_steps)
        max_steps    = self.cfg.max_map_steps
        actual_steps = M if max_steps <= 0 else min(M, max_steps)

        # Cholesky 更新状态强制使用 float32，避免 BFloat16（7位尾数）的累积舍入误差：
        # 实测在 BFloat16 下 ~130–150 步后 di2s 产生显著漂移（见 §10.7.4），
        # 导致 argmax 退化为随机甚至反向选择。float32 的 23 位尾数可稳定运行到 M=1369。
        # cis[t, s, n]：第 t 步第 s 帧的 Cholesky 列向量 e_t[n]
        cis  = torch.zeros(actual_steps, S, N, device=dev, dtype=torch.float32)
        # di2s[s, n]：det 增量的分母（剩余"平方长度"）；初始为对角元素
        di2s = kernel.diagonal(dim1=1, dim2=2).clone().float()  # float32

        select_idx    = torch.empty(M, S, dtype=torch.long, device=dev)
        # 标记已选 token，供随机补全时排除
        selected_mask = torch.zeros(S, N, dtype=torch.bool, device=dev)

        for i in range(actual_steps):
            # 1. 选出每帧中 di2s 最大的 token（行列式增量最大化）
            j = di2s.argmax(dim=-1)        # [S]
            select_idx[i] = j
            selected_mask[s_range, j] = True

            # 2. 提取 kernel 的第 j[s] 行并升至 float32
            row_j = kernel[s_range, j].float()  # [S, N]

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

        # ── 随机补全（actual_steps < M 时激活）──────────────────────────────
        # 从未被 Greedy MAP 选中的 token 中均匀随机采样，完全向量化。
        if actual_steps < M:
            fill_count = M - actual_steps
            noise = torch.rand(S, N, device=dev)
            noise.masked_fill_(selected_mask, -1.0)          # 已选位置赋极低分
            _, fill_idx = noise.topk(fill_count, dim=1)      # [S, fill_count]
            select_idx[actual_steps:] = fill_idx.t()         # [fill_count, S]

        # 对每帧选出的索引排序（保持位置稳定性，方便 scatter）
        select_idx = select_idx.t().contiguous()           # [S, M]
        select_idx = torch.sort(select_idx, dim=1).values  # [S, M]
        return select_idx
