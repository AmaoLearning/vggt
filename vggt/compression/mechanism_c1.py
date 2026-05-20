import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionContext, KVReductionHook
from .config import CompressionConfig, get_c1_spatial_band_upper
from .utils import dct2d, gather_tokens, idct2d


class FixedBandSpatialCompression(KVReductionHook):
    """方案 C1：使用离线 band 表做固定空间频带截断。"""

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

        q_new = q
        unmerge_fn = None
        if self.config.c1_enable_q_compression:
            q_new, unmerge_fn = self._compress_q(q, ctx)

        k_band_upper = get_c1_spatial_band_upper(ctx.layer_idx, "K", self.config.c1_energy_target)
        v_band_upper = get_c1_spatial_band_upper(ctx.layer_idx, "V", self.config.c1_energy_target)
        kv_output_size = max(k_band_upper, v_band_upper) + 1

        k_new = self._compress_branch(k, ctx, branch_name="K", output_hw=kv_output_size)
        v_new = self._compress_branch(v, ctx, branch_name="V", output_hw=kv_output_size)
        return q_new, k_new, v_new, unmerge_fn

    def _compress_q(self, q: Tensor, ctx: CompressionContext) -> Tuple[Tensor, Callable]:
        if self.config.c1_reconstruct_mode != "lowres_idct":
            raise ValueError("Q compression in mechanism C1 currently requires c1_reconstruct_mode='lowres_idct'.")

        q_new, patch_hw = self._compress_branch(q, ctx, branch_name="Q", return_patch_hw=True)
        patch_h, patch_w, reduced_h, reduced_w = patch_hw
        special_tokens = ctx.special_tokens
        S, P = ctx.S, ctx.P
        mode = self.config.c1_q_unmerge_mode

        def unmerge_fn(x_merged: Tensor) -> Tensor:
            bsz, num_heads, _, dim = x_merged.shape
            tokens_per_frame = special_tokens + reduced_h * reduced_w
            x_frames = x_merged.reshape(bsz, num_heads, S, tokens_per_frame, dim)
            x_special = x_frames[:, :, :, :special_tokens, :]
            x_patch = x_frames[:, :, :, special_tokens:, :].reshape(
                bsz, num_heads, S, reduced_h, reduced_w, dim
            )
            x_patch = x_patch.permute(0, 1, 2, 5, 3, 4).reshape(bsz * num_heads * S, dim, reduced_h, reduced_w)

            interpolate_kwargs = {}
            if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
                interpolate_kwargs["align_corners"] = False

            x_patch = F.interpolate(
                x_patch,
                size=(patch_h, patch_w),
                mode=mode,
                **interpolate_kwargs,
            )
            x_patch = x_patch.reshape(bsz, num_heads, S, dim, patch_h, patch_w)
            x_patch = x_patch.permute(0, 1, 2, 4, 5, 3).reshape(bsz, num_heads, S, patch_h * patch_w, dim)
            x_full = torch.cat([x_special, x_patch], dim=3)
            return x_full.reshape(bsz, num_heads, S * P, dim)

        return q_new, unmerge_fn

    def _compress_branch(
        self,
        tensor: Tensor,
        ctx: CompressionContext,
        branch_name: str,
        output_hw: Optional[int] = None,
        return_patch_hw: bool = False,
    ):
        bsz, num_heads, total_tokens, dim = tensor.shape
        S, P, special_tokens = ctx.S, ctx.P, ctx.special_tokens
        patch_tokens = P - special_tokens
        patch_h = patch_w = math.isqrt(patch_tokens)
        if patch_h * patch_w != patch_tokens:
            raise ValueError(f"Patch token count {patch_tokens} is not a square number.")

        band_upper = get_c1_spatial_band_upper(ctx.layer_idx, branch_name, self.config.c1_energy_target)
        reduced_h = reduced_w = band_upper + 1
        if output_hw is not None:
            reduced_h = reduced_w = max(reduced_h, output_hw)

        tensor_frames = tensor.reshape(bsz, num_heads, S, P, dim)
        special = tensor_frames[:, :, :, :special_tokens, :]
        patch = tensor_frames[:, :, :, special_tokens:, :].reshape(bsz, num_heads, S, patch_h, patch_w, dim)
        patch = self._compress_patch_grid(patch, band_upper + 1, reduced_h)
        patch = patch.reshape(bsz, num_heads, S, reduced_h * reduced_w, dim)

        compressed = torch.cat([special, patch], dim=3)
        compressed = compressed.reshape(bsz, num_heads, S * (special_tokens + reduced_h * reduced_w), dim)
        if return_patch_hw:
            return compressed, (patch_h, patch_w, reduced_h, reduced_w)
        return compressed

    def _compress_patch_grid(self, patch_grid: Tensor, active_h: int, output_h: int) -> Tensor:
        N_spatial = patch_grid.shape[-3]   # 原始空间尺寸 N（VGGT 中为 37）

        coeff = dct2d(patch_grid, dims=(-3, -2), norm="ortho")
        coeff = coeff[:, :, :, :active_h, :active_h, :]
        if output_h > active_h:
            padded = torch.zeros(
                *coeff.shape[:3],
                output_h,
                output_h,
                coeff.shape[-1],
                device=coeff.device,
                dtype=coeff.dtype,
            )
            padded[:, :, :, :active_h, :active_h, :] = coeff
            coeff = padded
        if self.config.c1_reconstruct_mode == "coeff_crop":
            return coeff
        if self.config.c1_reconstruct_mode == "lowres_idct":
            # FreqKV 振幅修正（参考 FreqKV Eq.3）：
            # 对 ortho 归一化的 2D 可分离 DCT，从 N×N 点 DCT 截断到 r×r 点后做 r 点 IDCT，
            # 重建信号的均值被放大了 N/r 倍（推导：Z[0,0] = N·mean(x)，r 点 IDCT 还原后
            # mean(x̂) = Z[0,0]/r = N/r · mean(x)）。
            # 故需乘以 r/N = output_h/N_spatial 来恢复原始幅度。
            scale = output_h / N_spatial
            return idct2d(coeff, dims=(-3, -2), norm="ortho") * scale
        raise ValueError(
            f"Unsupported c1_reconstruct_mode='{self.config.c1_reconstruct_mode}', expected 'lowres_idct' or 'coeff_crop'."
        )