import torch
from torch import Tensor
from typing import Optional, Callable, Tuple
from .base import KVReductionHook, CompressionContext
from .config import CompressionConfig, get_temporal_keep_ratio
from .utils import dct1d, idct1d


class TemporalDCTKVCompression(KVReductionHook):
    """
    机制 B：时序 DCT KV 压缩

    对 k, v 张量沿帧轴（时序轴 S）做 1D DCT：
      1. reshape k/v: [B, H, S*P, D] → [B, H, S, P, D]
      2. DCT 沿 S 维：保留低频 keep_top = ceil(S * keep_ratio) 个系数，其余置零
      3. IDCT 恢复 → [B, H, S*P, D]（长度不变，但时序高频被滤除）

    Q 不压缩，SDPA 输出形状不变，无需 unmerge。
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

        keep_ratio = get_temporal_keep_ratio(ctx.layer_idx, self.config)
        k_new = self._temporal_dct_compress(k, ctx.S, ctx.P, keep_ratio)
        v_new = self._temporal_dct_compress(v, ctx.S, ctx.P, keep_ratio)

        return q, k_new, v_new, None   # 无 unmerge（长度不变）

    def _temporal_dct_compress(
        self, kv: Tensor, S: int, P: int, keep_ratio: float,
    ) -> Tensor:
        """
        kv: [B, H, S*P, D]
        在 S 维做 DCT，保留 keep_top 个低频系数，IDCT 恢复。
        输出形状与输入相同: [B, H, S*P, D]
        """
        B, H, N, D = kv.shape
        keep_top = max(1, int(S * keep_ratio))

        if keep_top >= S:
            return kv  # 无需压缩

        # reshape: [B, H, S, P, D] → 沿 S 轴做 DCT
        kv_3d = kv.reshape(B, H, S, P, D)   # [B, H, S, P, D]
        # 转置使 S 轴在最后，方便 dct1d(dim=-1)
        kv_t = kv_3d.permute(0, 1, 3, 4, 2).contiguous()  # [B, H, P, D, S]

        # DCT along last dim (S)
        kv_dct = dct1d(kv_t, dim=-1, norm="ortho")          # [B, H, P, D, S]

        # 置零高频系数（keep_top 之后的全部置零）
        kv_dct[..., keep_top:] = 0.0

        # IDCT 恢复
        kv_idct = idct1d(kv_dct, dim=-1, norm="ortho")      # [B, H, P, D, S]

        # 转置回原始维度顺序
        kv_3d_out = kv_idct.permute(0, 1, 4, 2, 3).contiguous()  # [B, H, S, P, D]
        return kv_3d_out.reshape(B, H, N, D)

    def _temporal_dct_compress_patch_only(
        self, kv: Tensor, S: int, P: int, keep_ratio: float, special_tokens: int,
    ) -> Tensor:
        """
        仅压缩 patch tokens（后 P-special_tokens 个），special tokens 保持原样。
        """
        B, H, N, D = kv.shape
        kv_3d = kv.reshape(B, H, S, P, D)

        # 分离 special 和 patch
        kv_special = kv_3d[:, :, :, :special_tokens, :]   # [B, H, S, 5, D]
        kv_patch   = kv_3d[:, :, :, special_tokens:, :]   # [B, H, S, 1369, D]

        # 仅压缩 patch tokens
        keep_top = max(1, int(S * keep_ratio))

        if keep_top < S:
            kv_patch_t = kv_patch.permute(0, 1, 3, 4, 2).contiguous()  # [B,H,P_patch,D,S]
            kv_patch_dct = dct1d(kv_patch_t, dim=-1, norm="ortho")
            kv_patch_dct[..., keep_top:] = 0.0
            kv_patch_idct = idct1d(kv_patch_dct, dim=-1, norm="ortho")
            kv_patch = kv_patch_idct.permute(0, 1, 4, 2, 3).contiguous()

        kv_3d_out = torch.cat([kv_special, kv_patch], dim=3)  # [B, H, S, P, D]
        return kv_3d_out.reshape(B, H, N, D)
