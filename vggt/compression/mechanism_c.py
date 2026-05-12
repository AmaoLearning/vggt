import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable, Tuple
from .base import KVReductionHook, CompressionContext
from .config import CompressionConfig
from .utils import dct2d, idct2d, gather_tokens


class Spatial2DDCTCompression(KVReductionHook):
    """
    机制 C：空间 2D-DCT + 异常值保留

    在每帧的 37×37 patch 网格上执行 2D DCT，保留：
      - 低频区域（左上角 r×r 块对应的 patch 位置）
      - 异常值 patch（偏离低频重建最远的 top-k 个）
      - 全部 special tokens（camera + register）

    输出 KV 序列更短，Q 不变（SDPA 输出形状取决于 Q，无需 unmerge）。
    """

    def __init__(self, config: CompressionConfig):
        self.config = config

    def compress(
        self,
        q: Tensor, k: Tensor, v: Tensor,
        ctx: CompressionContext,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Callable]]:
        if not ctx.is_global or ctx.S < self.config.min_frames_to_compress:
            return q, k, v, None

        k_new = self._spatial_compress(k, ctx.S, ctx.P, ctx.special_tokens)
        v_new = self._spatial_compress(v, ctx.S, ctx.P, ctx.special_tokens)

        return q, k_new, v_new, None  # Q 不变，无需 unmerge

    def _spatial_compress(
        self,
        kv: Tensor,
        S: int, P: int, special_tokens: int = 5,
    ) -> Tensor:
        """
        kv: [B, H, S*P, D]
        Returns: [B, H, S*k_keep, D]，k_keep = special_tokens + n_low_freq + n_outlier（每帧固定）
        """
        B, H, N, D = kv.shape
        P_patch = P - special_tokens
        patch_h = patch_w = int(P_patch ** 0.5)   # 37×37 = 1369
        assert patch_h * patch_w == P_patch, f"Patch tokens {P_patch} not a perfect square"

        # 低频区域的线性边长
        r = max(1, int(patch_h * (self.config.spatial_low_freq_ratio ** 0.5)))
        n_low_freq = r * r

        # 异常值数量（相对于 patch token 总数）
        n_outlier = max(0, int(P_patch * self.config.spatial_outlier_ratio))

        kv_3d = kv.reshape(B, H, S, P, D)

        # ── 1. 分离 special tokens（始终保留）────────────────────────────
        kv_special = kv_3d[:, :, :, :special_tokens, :]    # [B, H, S, 5, D]
        kv_patch   = kv_3d[:, :, :, special_tokens:, :]    # [B, H, S, P_patch, D]

        # ── 2. 将 patch 重排为 2D 网格 ────────────────────────────────────
        # [B, H, S, P_patch, D] → [B, H, S, patch_h, patch_w, D]
        kv_2d = kv_patch.reshape(B, H, S, patch_h, patch_w, D)

        # ── 3. 对每个 patch 位置取跨 head 均值用于选择 patch 位置
        kv_mean = kv_2d.mean(dim=1)   # [B, S, patch_h, patch_w, D]

        # ── 4. 2D DCT（沿 patch_h, patch_w 维）──────────────────────────
        kv_dct = dct2d(kv_mean, dims=(-3, -2), norm="ortho")  # [B, S, patch_h, patch_w, D]

        # ── 5. 构建低频重建（仅保留左上角 r×r 系数）────────────────────
        kv_dct_low = kv_dct.clone()
        kv_dct_low[:, :, r:, :, :] = 0.0
        kv_dct_low[:, :, :, r:, :] = 0.0
        kv_low = idct2d(kv_dct_low, dims=(-3, -2), norm="ortho")  # [B, S, h, w, D]

        # ── 6. 计算残差，找 outlier patch 位置 ──────────────────────────
        residual = (kv_mean - kv_low).norm(dim=-1)  # [B, S, patch_h, patch_w]
        residual_flat = residual.reshape(B, S, -1)  # [B, S, P_patch]

        # 低频 patch 的 flat 位置（左上角 r×r 块）
        low_freq_mask = torch.zeros(patch_h, patch_w, dtype=torch.bool, device=kv.device)
        low_freq_mask[:r, :r] = True
        low_freq_flat_idx = low_freq_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        # [r*r]

        # 屏蔽低频位置的残差（不参与 outlier 选择）
        residual_for_outlier = residual_flat.clone()
        residual_for_outlier[:, :, low_freq_flat_idx] = -1.0

        # Top-k outlier（沿 P_patch 维）
        if n_outlier > 0:
            _topk_vals, outlier_flat_idx = residual_for_outlier.topk(
                n_outlier, dim=-1, largest=True, sorted=False
            )  # [B, S, n_outlier]
        else:
            outlier_flat_idx = torch.empty(
                B, S, 0, dtype=torch.long, device=kv.device
            )

        # ── 7. 合并 low_freq + outlier patch 索引 ────────────────────────
        # low_freq_flat_idx: [r*r]（对所有帧相同）
        # outlier_flat_idx:  [B, S, n_outlier]（随帧变化）

        # 将低频索引扩展到 [B, S, n_low_freq]
        low_freq_expanded = (
            low_freq_flat_idx
            .unsqueeze(0).unsqueeze(0)
            .expand(B, S, -1)
        )  # [B, S, n_low_freq]

        # 合并：[B, S, n_low_freq + n_outlier]
        if n_outlier > 0:
            patch_keep_idx = torch.cat([low_freq_expanded, outlier_flat_idx], dim=-1)
        else:
            patch_keep_idx = low_freq_expanded

        # 偏移：patch token 的绝对位置 = special_tokens + patch_keep_idx
        patch_keep_idx_abs = patch_keep_idx + special_tokens   # [B, S, n_keep_patch]
        n_keep_patch = patch_keep_idx_abs.shape[-1]

        # ── 8. Gather：从 kv_3d 中提取选中的 patch tokens ─────────────
        # kv_3d: [B, H, S, P, D]
        # Special tokens: [B, H, S, 5, D]（始终保留）
        kv_special_out = kv_special  # [B, H, S, 5, D]

        # Patch gather：patch_keep_idx_abs [B, S, n_keep_patch] → [B, H, S, n_keep_patch, D]
        gather_idx = (
            patch_keep_idx_abs
            .unsqueeze(1)                               # [B, 1, S, n_keep_patch]
            .expand(B, H, S, n_keep_patch)             # [B, H, S, n_keep_patch]
            .unsqueeze(-1)
            .expand(B, H, S, n_keep_patch, D)          # [B, H, S, n_keep_patch, D]
        )
        kv_patch_out = kv_3d.gather(3, gather_idx)    # [B, H, S, n_keep_patch, D]

        # ── 9. 拼接 special + patch，展平为 [B, H, S*k_keep, D] ─────────
        k_keep_total = special_tokens + n_keep_patch
        kv_out = torch.cat([kv_special_out, kv_patch_out], dim=3)  # [B,H,S,k_keep,D]
        return kv_out.reshape(B, H, S * k_keep_total, D)
