#!/usr/bin/env python3
"""
tests/visualize_attention.py
============================
VGGT 各层 Attention Token 关注度可视化

对 VGGT 的 24 个 Frame Attention 层和 24 个 Global Attention 层，逐层捕获 Key/Query 向量，
以 Key-norm（L2 范数）作为 patch token 被关注程度的代理指标，映射回 37×37 空间网格，
叠加到原始帧图像上：

    高关注度  →  红色        低关注度  →  蓝色

关注度代理指标（--mode 参数）
-------------------------------
    knorm     : Key 向量 L2 范数（默认）
                原理：softmax(Q·Kᵀ/√d) 中，‖K_j‖ 越大，该 token 越容易吸引注意；
                无需计算完整注意力矩阵，对所有序列长度均高效。
    qnorm     : Query 向量 L2 范数，衡量 token "主动寻找信息"的能力。
    cam_attn  : 相机 token → patch 的 softmax attention（仅 Global 层有效）。
                每帧的几何摘要 token（camera token）明确指向哪些 patch。
                成本：S 个 query × S·N_patch 个 key per head，对任意 S 均低开销。

输出结构
---------
    <output_dir>/
        frame_attn/
            layer_{i:02d}/
                frame_grid.png        — S 帧热力图叠加网格
        global_attn/
            layer_{i:02d}/
                frame_grid.png
        summary/
            cross_layer_saliency.png  — 所有层平均关注度的空间分布汇总（16 格）
            entropy_across_layers.png — 关注度空间熵随层深的变化曲线

使用示例
---------
    # 1. 基础：key-norm，所有层，kitchen 场景
    python tests/visualize_attention.py \\
        --image_dir examples/kitchen/images \\
        --output_dir tests/results/attention_vis

    # 2. 只分析全局注意力，相机 token attention 模式，关键层
    python tests/visualize_attention.py \\
        --image_dir examples/kitchen/images \\
        --output_dir tests/results/attention_vis_cam \\
        --block_types global \\
        --mode cam_attn \\
        --layers 0,6,12,18,23

    # 3. 8 帧 room 场景，query-norm 模式，自定义层
    python tests/visualize_attention.py \\
        --image_dir examples/room/images \\
        --output_dir tests/results/attention_vis_room \\
        --mode qnorm \\
        --layers 0,4,8,12,16,20,23 \\
        --num_frames 8
"""

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── 项目根路径 ────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vggt.models.vggt import VGGT
from vggt.compression.base import CompressionContext

# ── 模型架构常量 ──────────────────────────────────────────────────────────────
IMG_SIZE         = 518
PATCH_SIZE       = 14
PATCH_H          = IMG_SIZE // PATCH_SIZE   # 37
PATCH_W          = IMG_SIZE // PATCH_SIZE   # 37
N_PATCHES        = PATCH_H * PATCH_W        # 1369
SPECIAL_TOKENS   = 5                        # camera(1) + register(4)
TOKENS_PER_FRAME = SPECIAL_TOKENS + N_PATCHES   # 1374
N_LAYERS         = 24

# ── 色彩映射：低关注=蓝色，高关注=红色 ──────────────────────────────────────
_ATTN_CMAP = LinearSegmentedColormap.from_list(
    "attn_rb",
    [(0.00, "#2166AC"),   # 深蓝
     (0.25, "#74ADD1"),   # 浅蓝
     (0.50, "#F7F7F7"),   # 白（中性）
     (0.75, "#F4A582"),   # 橙
     (1.00, "#B2182B")],  # 深红
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Attention 捕获 Hook
# ══════════════════════════════════════════════════════════════════════════════

class AttentionCaptureHook:
    """
    插入 Attention._compression_hook 槽位，在 RoPE 之后、SDPA 之前捕获 Q/K，
    **在 GPU 上直接计算** saliency 并存储结果（而非原始张量），避免大量数据搬运。

    支持的 mode：
        knorm    : ‖K_j‖₂ 平均至 [S, 37, 37]
        qnorm    : ‖Q_i‖₂ 平均至 [S, 37, 37]
        cam_attn : camera token → patch softmax attention（仅 global 块有效）
    """

    def __init__(self, mode: str, is_global: bool) -> None:
        self.mode      = mode
        self.is_global = is_global
        # 计算结果：[S, PATCH_H, PATCH_W]，存储于 CPU
        self.saliency: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: "CompressionContext",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        with torch.no_grad():
            if self.mode == "knorm":
                self.saliency = self._vec_norm(k)
            elif self.mode == "qnorm":
                self.saliency = self._vec_norm(q)
            elif self.mode == "cam_attn":
                if self.is_global:
                    self.saliency = self._cam_attn(q, k)
                # 对 frame 块，cam_attn 无意义，跳过
        return q, k, v, None

    # ------------------------------------------------------------------
    def _vec_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 Q 或 K 向量的 L2 范数，映射回空间网格。

        Frame 块 : x 形状 [B*S, H, P, D]，P=1374（含 special tokens）
        Global 块: x 形状 [B,  H, S*P, D]
        返回     : [S, PATCH_H, PATCH_W]（float32, CPU）
        """
        x = x.detach().float()
        if self.is_global:
            B, H, SP, D = x.shape
            S = SP // TOKENS_PER_FRAME
            # [B, H, S, P, D] → 取 patch 部分
            x_patches = x.reshape(B, H, S, TOKENS_PER_FRAME, D)[
                :, :, :, SPECIAL_TOKENS:, :
            ]  # [B, H, S, N_patch, D]
            norms = x_patches.norm(dim=-1)          # [B, H, S, N_patch]
            sal   = norms.mean(dim=(0, 1))           # [S, N_patch]
        else:
            BS, H, P, D = x.shape                   # BS = B*S (通常 B=1)
            x_patches = x[:, :, SPECIAL_TOKENS:, :] # [BS, H, N_patch, D]
            norms = x_patches.norm(dim=-1)           # [BS, H, N_patch]
            sal   = norms.mean(dim=1)                # [BS, N_patch]  ← 即 [S, N_patch]
        return sal.reshape(-1, PATCH_H, PATCH_W).cpu()

    # ------------------------------------------------------------------
    def _cam_attn(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Global 块专用：计算每帧相机 token 对所有帧 patch 的 softmax 注意力，
        聚合为每个 patch 的总接收关注度。

        q, k 形状 : [B, H, S*P, D]
        返回      : [S, PATCH_H, PATCH_W]（float32, CPU）
        """
        q = q.detach().float()
        k = k.detach().float()
        B, H, SP, D = q.shape
        S    = SP // TOKENS_PER_FRAME
        scale = D ** -0.5

        # 相机 token 的 Query 位置：每帧第 0 个 token = f * TOKENS_PER_FRAME
        cam_idx = torch.arange(S, device=q.device) * TOKENS_PER_FRAME   # [S]
        cam_q   = q[:, :, cam_idx, :]                                     # [B, H, S, D]

        # 所有帧的 Patch Key 位置
        frame_starts  = (torch.arange(S, device=k.device) * TOKENS_PER_FRAME
                         + SPECIAL_TOKENS)                                 # [S]
        patch_offsets = torch.arange(N_PATCHES, device=k.device)          # [N_patch]
        patch_idx     = (frame_starts.unsqueeze(1)
                         + patch_offsets.unsqueeze(0)).reshape(-1)         # [S*N_patch]
        patch_k = k[:, :, patch_idx, :]                                    # [B, H, S*N_patch, D]

        # 注意力矩阵：[B, H, S, S*N_patch]
        # S_query=S 个相机 token × S*N_patch 个 patch key，开销极低
        attn = torch.softmax(cam_q @ patch_k.transpose(-2, -1) * scale, dim=-1)

        # 每个 patch 从所有相机 query 累计收到的注意力 → [B, H, S*N_patch]
        attn_recv = attn.sum(dim=2)            # sum over camera queries
        sal = attn_recv.mean(dim=(0, 1))       # [S*N_patch]，平均 B 和 H
        return sal.reshape(S, PATCH_H, PATCH_W).cpu()

    def clear(self) -> None:
        self.saliency = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Hook 安装 / 卸载
# ══════════════════════════════════════════════════════════════════════════════

def install_hooks(
    model:        VGGT,
    layers:       List[int],
    num_frames:   int,
    mode:         str,
    block_types:  List[str],   # 子集，可为 ["frame"], ["global"], 或 ["frame", "global"]
) -> Dict[str, Dict[int, AttentionCaptureHook]]:
    """
    在指定层的 frame_blocks / global_blocks 上安装捕获 hook。

    返回 {"frame": {layer_idx: hook}, "global": {layer_idx: hook}}
    """
    result: Dict[str, Dict[int, AttentionCaptureHook]] = {"frame": {}, "global": {}}

    for layer_idx in layers:
        if "frame" in block_types:
            block = model.aggregator.frame_blocks[layer_idx]
            hook  = AttentionCaptureHook(mode=mode, is_global=False)
            block.attn._compression_hook = hook
            if block.attn._compression_ctx is None:
                block.attn._compression_ctx = CompressionContext(
                    is_global=False,
                    S=num_frames,
                    P=TOKENS_PER_FRAME,
                    layer_idx=layer_idx,
                    total_layers=N_LAYERS,
                    special_tokens=SPECIAL_TOKENS,
                )
            result["frame"][layer_idx] = hook

        if "global" in block_types:
            block = model.aggregator.global_blocks[layer_idx]
            hook  = AttentionCaptureHook(mode=mode, is_global=True)
            block.attn._compression_hook = hook
            if block.attn._compression_ctx is None:
                block.attn._compression_ctx = CompressionContext(
                    is_global=True,
                    S=num_frames,
                    P=TOKENS_PER_FRAME,
                    layer_idx=layer_idx,
                    total_layers=N_LAYERS,
                    special_tokens=SPECIAL_TOKENS,
                )
            result["global"][layer_idx] = hook

    return result


def remove_hooks(model: VGGT, layers: List[int], block_types: List[str]) -> None:
    """清除所有已安装的捕获 hook。"""
    for layer_idx in layers:
        if "frame" in block_types:
            block = model.aggregator.frame_blocks[layer_idx]
            block.attn._compression_hook = None
            block.attn._compression_ctx  = None
        if "global" in block_types:
            block = model.aggregator.global_blocks[layer_idx]
            block.attn._compression_hook = None
            block.attn._compression_ctx  = None


# ══════════════════════════════════════════════════════════════════════════════
# 3. 图像加载
# ══════════════════════════════════════════════════════════════════════════════

def load_images(image_dir: str, num_frames: int) -> Tuple[torch.Tensor, np.ndarray]:
    """
    加载图像，返回：
        images_tensor : [1, S, 3, 518, 518]，float32，[0,1]，用于模型输入
        frames_np     : [S, 518, 518, 3]，float32，[0,1]，用于可视化叠加
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
    paths = sorted(
        p for p in Path(image_dir).iterdir() if p.suffix in exts
    )
    if len(paths) < num_frames:
        raise ValueError(
            f"目录 '{image_dir}' 仅有 {len(paths)} 张图，"
            f"需要至少 {num_frames} 张"
        )

    frames_np = []
    for p in paths[:num_frames]:
        img = Image.open(p).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BICUBIC
        )
        frames_np.append(np.asarray(img, dtype=np.float32) / 255.0)

    frames_np = np.stack(frames_np, axis=0)           # [S, H, W, 3]
    # VGGT 内部归一化，此处传 [0,1] 即可
    tensor = (
        torch.from_numpy(frames_np)
        .permute(0, 3, 1, 2)                          # [S, 3, H, W]
        .unsqueeze(0)                                  # [1, S, 3, H, W]
    )
    return tensor, frames_np


# ══════════════════════════════════════════════════════════════════════════════
# 4. 可视化工具
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_saliency(sal: torch.Tensor) -> np.ndarray:
    """
    将 [S, H, W] saliency 归一化到 [0, 1]。
    使用 1%–99% 分位数裁剪，避免极端值压缩色彩范围。
    """
    s = sal.float().numpy()
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    if hi > lo:
        s = (s - lo) / (hi - lo)
    else:
        s = np.zeros_like(s)
    return np.clip(s, 0.0, 1.0)


def _overlay_heatmap(
    frame_rgb: np.ndarray,   # [H, W, 3]，[0,1]
    sal_map: np.ndarray,     # [37, 37]，[0,1]
    alpha: float = 0.50,
) -> np.ndarray:
    """将 saliency 热力图双线性上采样后与原图叠加，返回 [H, W, 3]，[0,1]。"""
    H, W = frame_rgb.shape[:2]
    sal_t  = torch.from_numpy(sal_map).unsqueeze(0).unsqueeze(0)  # [1,1,37,37]
    sal_up = F.interpolate(
        sal_t.float(), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze().numpy()                                            # [H, W]
    heatmap = _ATTN_CMAP(sal_up)[:, :, :3]                        # [H, W, 3]
    blended = (1.0 - alpha) * frame_rgb + alpha * heatmap
    return np.clip(blended, 0.0, 1.0)


def save_layer_frame_grid(
    frames_np:  np.ndarray,     # [S, H, W, 3]
    saliency:   torch.Tensor,   # [S, 37, 37]
    layer_idx:  int,
    block_type: str,
    mode:       str,
    out_path:   Path,
    alpha:      float = 0.50,
    max_cols:   int   = 8,
) -> None:
    """
    将 S 帧的热力图叠加图拼成网格保存为一张图。
    网格列数 = min(S, max_cols)。
    """
    S    = frames_np.shape[0]
    ncols = min(S, max_cols)
    nrows = (S + ncols - 1) // ncols

    sal_norm = _normalize_saliency(saliency)  # [S, 37, 37]

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.0, nrows * 2.0),
        squeeze=False,
    )
    mode_label = {"knorm": "Key-Norm", "qnorm": "Query-Norm", "cam_attn": "Cam→Patch Attn"}
    fig.suptitle(
        f"Layer {layer_idx:02d}  ·  {block_type.capitalize()} Attention  "
        f"·  {mode_label.get(mode, mode)}  "
        f"  (red=high  blue=low)",
        fontsize=9,
    )

    for flat_idx in range(nrows * ncols):
        ax = axes[flat_idx // ncols, flat_idx % ncols]
        ax.axis("off")
        if flat_idx >= S:
            continue
        blended = _overlay_heatmap(frames_np[flat_idx], sal_norm[flat_idx], alpha)
        ax.imshow(blended)
        ax.set_title(f"f{flat_idx}", fontsize=6, pad=1)

    # 添加色彩条
    sm = plt.cm.ScalarMappable(cmap=_ATTN_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Saliency (normalized)", fontsize=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _spatial_entropy(saliency: torch.Tensor) -> float:
    """
    计算 saliency 的空间熵（比特），值越小表示注意力越集中于少数 patch。
    saliency: [S, H, W] 或 [H, W]
    """
    s = saliency.float().reshape(-1)
    s = s.clamp(min=0.0)
    total = s.sum()
    if total < 1e-9:
        return float(np.log2(s.numel()))   # 最大熵（均匀）
    prob  = s / total
    ent   = -(prob * (prob + 1e-12).log2()).sum()
    return float(ent.item())


def save_summary(
    all_saliency:  Dict[str, Dict[int, torch.Tensor]],  # {"frame"/"global": {layer: [S,37,37]}}
    frames_np:     np.ndarray,                           # [S, H, W, 3]
    layers:        List[int],
    out_dir:       Path,
    mode:          str,
) -> None:
    """
    保存两张汇总图：
    1. cross_layer_saliency.png : 各层平均 saliency 热力图（纯 patch 空间，不叠加图像）
    2. entropy_across_layers.png: 各层 saliency 熵随深度的变化
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_label = {"knorm": "Key-Norm", "qnorm": "Query-Norm", "cam_attn": "Cam→Patch Attn"}
    _save_cross_layer_saliency(all_saliency, layers, out_dir, mode, mode_label)
    _save_entropy_plot(all_saliency, layers, out_dir, mode_label.get(mode, mode))


def _save_cross_layer_saliency(
    all_saliency: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    out_dir: Path,
    mode: str,
    mode_label: dict,
) -> None:
    """
    对每种 block_type，绘制一行 cells，每 cell 是该层在所有帧上的平均 saliency 热力图
    （37×37 分辨率，不叠加图像），方便观察关注区域随层深的演化。
    """
    block_types_present = [bt for bt in ("frame", "global") if bt in all_saliency and all_saliency[bt]]
    if not block_types_present:
        return

    n_layers = len(layers)
    n_rows   = len(block_types_present)
    fig, axes = plt.subplots(
        n_rows, n_layers,
        figsize=(n_layers * 1.6, n_rows * 1.8),
        squeeze=False,
    )
    fig.suptitle(
        f"Cross-Layer Mean Saliency  ·  {mode_label.get(mode, mode)}\n"
        f"(averaged over S frames, 37×37 patch grid)",
        fontsize=9,
    )

    for row_idx, bt in enumerate(block_types_present):
        layer_dict = all_saliency[bt]
        for col_idx, layer_idx in enumerate(layers):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([]); ax.set_yticks([])
            if layer_idx not in layer_dict:
                ax.axis("off")
                continue
            sal = layer_dict[layer_idx]            # [S, 37, 37]
            mean_sal = sal.float().mean(dim=0)     # [37, 37]
            s = mean_sal.numpy()
            lo, hi = s.min(), s.max()
            s_norm = (s - lo) / (hi - lo + 1e-8)
            ax.imshow(s_norm, cmap=_ATTN_CMAP, vmin=0, vmax=1, aspect="equal",
                      interpolation="nearest")
            if col_idx == 0:
                ax.set_ylabel(bt.capitalize(), fontsize=7)
            ax.set_title(f"L{layer_idx:02d}", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "cross_layer_saliency.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 汇总图：{out_dir / 'cross_layer_saliency.png'}")


def _save_entropy_plot(
    all_saliency: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    out_dir: Path,
    mode_str: str,
) -> None:
    """
    绘制各层 saliency 空间熵随层深变化的折线图。
    熵越低 → 注意力越集中于少数 patch（可能代表更有判别性的关注）。
    """
    colors = {"frame": "#E6550D", "global": "#3182BD"}
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.7), 4))

    for bt, color in colors.items():
        if bt not in all_saliency or not all_saliency[bt]:
            continue
        ents = []
        valid_layers = []
        for layer_idx in layers:
            if layer_idx not in all_saliency[bt]:
                continue
            sal = all_saliency[bt][layer_idx]    # [S, 37, 37]
            mean_sal = sal.mean(dim=0)           # [37, 37]，代表性帧平均
            ents.append(_spatial_entropy(mean_sal))
            valid_layers.append(layer_idx)
        if ents:
            ax.plot(valid_layers, ents, "o-", color=color, label=f"{bt.capitalize()} Attention",
                    linewidth=1.5, markersize=4)

    ax.set_xlabel("Layer index", fontsize=10)
    ax.set_ylabel("Spatial entropy (bits)  ←low=focused", fontsize=10)
    ax.set_title(f"Attention Saliency Entropy across Layers  ·  {mode_str}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_across_layers.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 熵曲线：{out_dir / 'entropy_across_layers.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════════════════════════════

def run_visualization(args: argparse.Namespace) -> None:
    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，降级到 CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    layers      = _parse_layers(args.layers)
    block_types = [b.strip() for b in args.block_types.split(",")]
    out_root    = Path(args.output_dir)

    # ── 若 cam_attn 但仅有 frame 块，报警 ────────────────────────────────────
    if args.mode == "cam_attn" and block_types == ["frame"]:
        print("[WARN] cam_attn 仅对 Global Attention 有意义；已自动添加 global。")
        block_types.append("global")

    print(f"\n{'='*60}")
    print(f"  加载模型：{args.model_name}")
    print(f"{'='*60}")
    model: VGGT = VGGT.from_pretrained(args.model_name)
    model.eval().to(device)

    print(f"  加载 {args.num_frames} 帧：{args.image_dir}")
    images, frames_np = load_images(args.image_dir, args.num_frames)
    images = images.to(device)
    S = frames_np.shape[0]

    print(f"  分析层：{layers}  块类型：{block_types}  关注度模式：{args.mode}")
    hooks = install_hooks(model, layers, S, args.mode, block_types)

    # ── 单次前向推理，所有 hook 并行捕获 ────────────────────────────────────
    print("  执行前向推理（单次）...")
    _autocast = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), _autocast:
        _ = model(images)

    remove_hooks(model, layers, block_types)

    # ── 收集 saliency 并生成可视化 ───────────────────────────────────────────
    all_saliency: Dict[str, Dict[int, torch.Tensor]] = {"frame": {}, "global": {}}

    for bt in ("frame", "global"):
        if bt not in hooks or not hooks[bt]:
            continue
        block_out_dir = out_root / f"{bt}_attn"
        for layer_idx in layers:
            hook = hooks[bt].get(layer_idx)
            if hook is None or hook.saliency is None:
                print(f"  [WARN] Layer {layer_idx:02d} {bt}: 未捕获到数据，跳过")
                continue

            sal = hook.saliency                 # [S, 37, 37]
            all_saliency[bt][layer_idx] = sal

            grid_path = block_out_dir / f"layer_{layer_idx:02d}" / "frame_grid.png"
            save_layer_frame_grid(
                frames_np  = frames_np,
                saliency   = sal,
                layer_idx  = layer_idx,
                block_type = bt,
                mode       = args.mode,
                out_path   = grid_path,
                alpha      = args.alpha,
            )
            print(f"  ✓ {bt} Layer {layer_idx:02d} → {grid_path}")
            hook.clear()

    # ── 汇总图 ───────────────────────────────────────────────────────────────
    print("\n  生成汇总图...")
    save_summary(all_saliency, frames_np, layers, out_root / "summary", args.mode)

    print(f"\n  可视化完成，结果保存于：{out_root.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_layers(layers_arg: str) -> List[int]:
    if layers_arg.strip().lower() == "all":
        return list(range(N_LAYERS))
    layers = []
    for item in layers_arg.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if not (0 <= idx < N_LAYERS):
            raise ValueError(f"层编号 {idx} 超出范围 [0, {N_LAYERS - 1}]")
        layers.append(idx)
    if not layers:
        raise ValueError("未指定有效层编号")
    return sorted(set(layers))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VGGT 各层 Attention Token 关注度可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model_name",  type=str, default="facebook/VGGT-1B",
                   help="HuggingFace 模型 ID 或本地路径")
    p.add_argument("--image_dir",   type=str, required=True,
                   help="输入图像目录（至少含 num_frames 张图）")
    p.add_argument("--output_dir",  type=str, default="tests/results/attention_vis",
                   help="输出目录")
    p.add_argument("--layers",      type=str, default="0,4,8,12,16,20,23",
                   help="'all' 或逗号分隔的层编号，如 '0,6,12,18,23'")
    p.add_argument("--block_types", type=str, default="frame,global",
                   help="要分析的块类型：'frame'、'global' 或 'frame,global'（默认）")
    p.add_argument("--mode",        type=str, default="knorm",
                   choices=["knorm", "qnorm", "cam_attn"],
                   help="关注度代理指标（默认 knorm）")
    p.add_argument("--num_frames",  type=int, default=16,
                   help="使用的帧数（默认 16）")
    p.add_argument("--alpha",       type=float, default=0.50,
                   help="热力图叠加透明度，0=仅原图，1=仅热力图（默认 0.50）")
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运算设备（默认 cuda 若可用）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
