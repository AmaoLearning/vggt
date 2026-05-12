from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple
import torch
from torch import Tensor


@dataclass
class CompressionContext:
    """由 Aggregator 在每次前向前写入 Attention 模块"""
    is_global: bool = False    # 是否为 global attention
    S: int = 1                 # 帧数
    P: int = 1374              # 每帧 token 数（包含 special tokens）
    layer_idx: int = 0         # 当前层编号（0-indexed）
    total_layers: int = 24     # 总层数
    special_tokens: int = 5    # camera(1) + register(4)


class KVReductionHook:
    """
    可插拔压缩 hook 的抽象基类。
    挂载到 Attention 模块后，在 RoPE 之后、SDPA 之前被调用。
    """

    def compress(
        self,
        q: Tensor,  # [B, H, N, D]
        k: Tensor,  # [B, H, N, D]
        v: Tensor,  # [B, H, N, D]
        ctx: CompressionContext,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Callable]]:
        """
        Returns:
            q_new: [B, H, N_q', D]  — 压缩后的 Q（N_q' <= N）
            k_new: [B, H, N_kv', D] — 压缩后的 K
            v_new: [B, H, N_kv', D] — 压缩后的 V
            unmerge_fn: 若 Q 长度被压缩，返回可恢复的 unmerge 函数；否则返回 None
        """
        raise NotImplementedError

    def __call__(self, q, k, v, ctx):
        return self.compress(q, k, v, ctx)
