"""
方案 F（重新设计）：FastVGGT Bipartite Token Merging 复现

参考文献：
  Shen et al., "FastVGGT: Training-Free Acceleration of Visual Geometry Transformer"
  ICLR 2026 | arXiv:2509.02560 | https://github.com/mystorm16/FastVGGT/

核心思路（五步流程，直接对应论文 §3.2–3.4）：

  1. Reference-frame anchoring
     Frame 0 的全部 P 个 token 设为 dst（全局坐标系锚点，不参与合并）。

  2. Special-token anchoring
     Frame 1..S-1 中每帧的 special tokens（1 camera + 4 register = 5 个）设为 dst。

  3. Salient-token protection（论文 §3.2 Salient Token Selection）
     Frame 1..S-1 每帧 patch token 中，按固定步长 salient_stride 抽取约 10% 的
     token 标记为"protected"——它们绕过 merge，直接参与 attention。

  4. Region-based random dst（论文 §3.2 Uniform Token Sampling，灵感来自 ToMeSD）
     Frame 1..S-1 的剩余 patch token 被映射到 sy×sx 区域网格；每格随机选 1 个 dst，
     其余全为 src。

  5. Cosine-similarity bipartite merge → attention → unmerge（论文 §3.3–3.4）
     - 每个 src 找余弦最相似的 dst；top-r 个相似度最高的 src 被平均融合进对应 dst。
     - Attention 在压缩后的序列（长度 N'）上运行（复杂度从 O(N²) 降至 O(N'²)）。
     - Unmerge: 将 dst 的注意力输出值复制回其所有合并的 src 位置，恢复 N 长度。

实现差异说明（vs. FastVGGT 官方代码）：
  FastVGGT 在 x（pre-projection）上建立 bipartite 分区；本实现在 k（post-projection）
  上建立分区并同时对 q, k, v 施加相同变换。k 与 x 经过同一 LayerNorm + 线性投影后
  保留了足够的相似度信息，实践中效果等价。
"""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CompressionContext, KVReductionHook
from .config import CompressionConfig


class FastVGGTTokenMerging(KVReductionHook):
    """
    FastVGGT (ICLR 2026) bipartite token merging，适配到本项目 KVReductionHook 框架。

    主要参数（均在 CompressionConfig 中配置）：
      f_merge_ratio   : 目标合并比例（占全序列长度 N），默认 0.9
      f_start_layer   : 从第几个 global block 开始启用 merging，默认 0（全部层）
      f_region_stride : 区域采样步长 sy=sx，默认 2（每 2×2 格取 1 个 dst）
      f_salient_stride: patch token 的 salient 步长，默认 10（约 10% 受保护）
    """

    def __init__(self, config: CompressionConfig):
        self.config = config

    # ──────────────────────────────────────────────────────────────────────────
    # KVReductionHook interface
    # ──────────────────────────────────────────────────────────────────────────

    def compress(
        self,
        q: Tensor,            # [B, H, N, D]
        k: Tensor,            # [B, H, N, D]
        v: Tensor,            # [B, H, N, D]
        ctx: CompressionContext,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Callable]]:
        if not ctx.is_global:
            return q, k, v, None
        if ctx.layer_idx < self.config.f_start_layer:
            return q, k, v, None
        if ctx.S < self.config.min_frames_to_compress:
            return q, k, v, None

        B, H, N, D = q.shape
        S, P = ctx.S, ctx.P
        special = ctx.special_tokens
        patch_tokens = P - special

        grid_h, grid_w = _get_patch_grid(patch_tokens)

        # r = target number of src tokens to merge (FastVGGT: int(N * ratio))
        r = int(N * self.config.f_merge_ratio)
        if r <= 0:
            return q, k, v, None

        # Similarity metric: head-averaged, L2-normalised K
        with torch.no_grad():
            metric = F.normalize(k.mean(dim=1), dim=-1)  # [B, N, D]

        merge_fn, unmerge_fn, N_new = _build_merge_unmerge(
            metric=metric,
            S=S, P=P, special=special,
            grid_h=grid_h, grid_w=grid_w,
            r=r,
            region_stride=self.config.f_region_stride,
            salient_stride=self.config.f_salient_stride,
            device=q.device,
        )

        if merge_fn is None:
            return q, k, v, None

        def _apply_merge(x: Tensor) -> Tensor:
            """[B, H, N, D] → [B, H, N_new, D]"""
            Bx, Hx, Nx, Dx = x.shape
            flat   = x.permute(0, 2, 1, 3).reshape(Bx, Nx, Hx * Dx)
            merged = merge_fn(flat)
            return merged.reshape(Bx, N_new, Hx, Dx).permute(0, 2, 1, 3)

        def do_unmerge(x_out: Tensor) -> Tensor:
            """[B, H, N_new, D] → [B, H, N, D]"""
            Bx, Hx, _, Dx = x_out.shape
            flat = x_out.permute(0, 2, 1, 3).reshape(Bx, N_new, Hx * Dx)
            full = unmerge_fn(flat)
            return full.reshape(Bx, N, Hx, Dx).permute(0, 2, 1, 3)

        return _apply_merge(q), _apply_merge(k), _apply_merge(v), do_unmerge

    def _build_keep_indices(self, reference: Tensor, ctx: CompressionContext) -> Tensor:
        S, P, special_tokens = ctx.S, ctx.P, ctx.special_tokens
        patch_tokens = P - special_tokens
        grid_h = grid_w = math.isqrt(patch_tokens)
        if grid_h * grid_w != patch_tokens:
            raise ValueError(f"Patch token count {patch_tokens} is not a square number.")



# ──────────────────────────────────────────────────────────────────────────────
# Patch grid helper
# ──────────────────────────────────────────────────────────────────────────────

def _get_patch_grid(patch_tokens: int) -> Tuple[int, int]:
    """Return (grid_h, grid_w) for the patch token layout of a single frame."""
    g = int(math.isqrt(patch_tokens))
    if g * g == patch_tokens:
        return g, g
    # Known VGGT default: 518×518 / patch_size=14 → 37×37 = 1369
    if patch_tokens == 1369:
        return 37, 37
    raise ValueError(
        f"Cannot determine patch grid for {patch_tokens} tokens "
        f"(expected a perfect square or the standard VGGT 37×37=1369 config)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core: bipartite partition + merge/unmerge closures
# ──────────────────────────────────────────────────────────────────────────────

def _build_merge_unmerge(
    metric: Tensor,           # [B, N, D]  L2-normalised cosine features
    S: int,
    P: int,                   # tokens per frame (special + patch)
    special: int,             # number of special tokens per frame
    grid_h: int,
    grid_w: int,
    r: int,                   # target merge count (capped at num_src)
    region_stride: int = 2,   # sy = sx
    salient_stride: int = 10, # 1-in-K patch tokens are protected per frame
    device=None,
) -> Tuple[Optional[Callable], Optional[Callable], int]:
    """
    Build (merge_fn, unmerge_fn, N_new).

    Token classification in idx_buffer:
      -2 = salient / protected  →  bypass merge entirely
      -1 = dst                  →  anchor; src tokens merged into these
       0 = src                  →  merged into nearest dst (top-r) or kept unmerged

    Merged sequence layout: [unmerged_src | dst | salient]
    Size: N_new = num_unm + num_dst + num_salient

    Returns (None, None, N) when merging is not applicable.
    """
    B, N, D = metric.shape
    patch_tokens = P - special
    sy = sx = region_stride
    hsy = grid_h // sy
    wsx = grid_w // sx

    if device is None:
        device = metric.device

    # ── Step 1: Build token classification buffer ─────────────────────────────
    #   idx_buffer[i] = -2 (salient) | -1 (dst) | 0 (src, default)
    idx_buffer = torch.zeros(N, dtype=torch.long, device=device)

    # Frame 0: all tokens → dst (global reference anchor)
    idx_buffer[:P] = -1

    if S > 1:
        f_starts = torch.arange(1, S, device=device) * P           # [S-1] frame starts

        # Special tokens (camera + register) per non-reference frame → dst
        spec_off = torch.arange(special, device=device)             # [5]
        idx_buffer[(f_starts.unsqueeze(1) + spec_off.unsqueeze(0)).flatten()] = -1

        # Region-based random dst within each sy×sx cell (vectorised over all frames)
        if hsy > 0 and wsx > 0:
            rand_off = torch.randint(sy * sx, (S - 1, hsy, wsx), device=device)
            row_off  = rand_off // sx                               # [S-1, hsy, wsx]
            col_off  = rand_off % sx
            abs_row  = torch.arange(hsy, device=device).view(1, hsy, 1) * sy + row_off
            abs_col  = torch.arange(wsx, device=device).view(1, 1, wsx) * sx + col_off
            patch_loc  = abs_row * grid_w + abs_col                 # [S-1, hsy, wsx]
            tok_idx    = (f_starts.view(S - 1, 1, 1) + special) + patch_loc
            idx_buffer[tok_idx.flatten()] = -1

        # Salient tokens (protected, overrides any dst set above)
        sal_off  = torch.arange(0, patch_tokens, salient_stride, device=device)
        sal_base = (f_starts + special).unsqueeze(1)                # [S-1, 1]
        idx_buffer[(sal_base + sal_off.unsqueeze(0)).flatten()] = -2

    # ── Step 2: Extract group indices ────────────────────────────────────────
    dst_idx = (idx_buffer == -1).nonzero(as_tuple=False).squeeze(1)  # [num_dst]
    src_idx = (idx_buffer ==  0).nonzero(as_tuple=False).squeeze(1)  # [num_src]
    sal_idx = (idx_buffer == -2).nonzero(as_tuple=False).squeeze(1)  # [num_sal]

    num_dst = dst_idx.shape[0]
    num_src = src_idx.shape[0]
    num_sal = sal_idx.shape[0]

    if num_src == 0 or r <= 0:
        return None, None, N

    # ── Step 3: Cosine-similarity bipartite matching ──────────────────────────
    # Use batch-0 metric (B=1 in standard VGGT inference)
    m0    = metric[0]                  # [N, D]
    src_m = m0[src_idx]                # [num_src, D]
    dst_m = m0[dst_idx]                # [num_dst, D]

    # Chunked matmul to bound peak VRAM (num_src × num_dst can be large)
    CHUNK = 8192
    sim = torch.empty(num_src, num_dst, device=device, dtype=src_m.dtype)
    for i in range(0, num_src, CHUNK):
        sim[i : i + CHUNK] = torch.mm(src_m[i : i + CHUNK], dst_m.T)

    best_sim_val, best_dst_local = sim.max(dim=-1)    # [num_src] local dst index
    del sim

    # Select top-r most-redundant src (highest similarity → safest to merge)
    r_actual = min(r, num_src)
    _, top_src_local = best_sim_val.topk(r_actual, largest=True, sorted=False)

    # Remaining src stay unmerged
    unm_mask = torch.ones(num_src, dtype=torch.bool, device=device)
    unm_mask[top_src_local] = False
    unm_src_local  = unm_mask.nonzero(as_tuple=False).squeeze(1)   # idx into src_idx
    num_unm        = unm_src_local.shape[0]

    # Global token indices for each group
    src_merge_global = src_idx[top_src_local]               # [r_actual]
    unm_src_global   = src_idx[unm_src_local]               # [num_unm]
    dst_local_4src   = best_dst_local[top_src_local]        # [r_actual], local into dst

    # Merged sequence length: [unm_src | dst | salient]
    N_new = num_unm + num_dst + num_sal

    # ── Step 4: merge closure ─────────────────────────────────────────────────

    def merge_fn(x: Tensor) -> Tensor:
        """[B, N, C] → [B, N_new, C]"""
        Bx, _N, Cx = x.shape

        unm_x = torch.gather(x, 1, unm_src_global.view(1, num_unm,   1).expand(Bx, num_unm,   Cx))
        dst_x = torch.gather(x, 1, dst_idx        .view(1, num_dst,   1).expand(Bx, num_dst,   Cx))
        sal_x = torch.gather(x, 1, sal_idx        .view(1, num_sal,   1).expand(Bx, num_sal,   Cx))
        src_x = torch.gather(x, 1, src_merge_global.view(1, r_actual, 1).expand(Bx, r_actual, Cx))

        # Average-merge src into matched dst:
        #   dst_x[j] = mean(dst_x[j], all src_x[i] where dst_local_4src[i] == j)
        dst_x = dst_x.scatter_reduce(
            1,
            dst_local_4src.view(1, r_actual, 1).expand(Bx, r_actual, Cx),
            src_x,
            reduce="mean",
            include_self=True,
        )

        return torch.cat([unm_x, dst_x, sal_x], dim=1)   # [B, N_new, C]

    # ── Step 5: unmerge closure ───────────────────────────────────────────────

    def unmerge_fn(x: Tensor) -> Tensor:
        """[B, N_new, C] → [B, N, C]"""
        Bx, _N_new, Cx = x.shape

        unm_x = x[:, :num_unm, :]
        dst_x = x[:, num_unm : num_unm + num_dst, :]
        sal_x = x[:, num_unm + num_dst :, :]

        # Merged src tokens recover the updated dst value (ToMe "copy" unmerge)
        src_x = torch.gather(
            dst_x, 1,
            dst_local_4src.view(1, r_actual, 1).expand(Bx, r_actual, Cx),
        )

        out = torch.zeros(Bx, N, Cx, device=x.device, dtype=x.dtype)
        out.scatter_(1, dst_idx        .view(1, num_dst,  1).expand(Bx, num_dst,   Cx), dst_x)
        out.scatter_(1, unm_src_global .view(1, num_unm,  1).expand(Bx, num_unm,   Cx), unm_x)
        out.scatter_(1, src_merge_global.view(1, r_actual, 1).expand(Bx, r_actual, Cx), src_x)
        if num_sal > 0:
            out.scatter_(1, sal_idx.view(1, num_sal, 1).expand(Bx, num_sal, Cx), sal_x)

        return out

    return merge_fn, unmerge_fn, N_new