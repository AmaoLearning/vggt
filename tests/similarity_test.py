"""
验证 VGGT global_block 前后帧间 token 相似度变化。
新增：K 空间余弦相似度 与 K+V 联合差异指标（面向 P 帧式 KV 压缩）。

用法（从项目根目录）:
    python docs/tests/similarity_test.py \
        --image_dir examples/llff_fern/images \
        --num_frames 8 \
        --output_dir ./similarity_results

要求:
    - 图像目录中有至少 num_frames 张图像（按文件名排序）
    - VGGT 模型权重可从 HuggingFace 下载（facebook/VGGT-1B）
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 项目根路径注入（使脚本从任意工作目录均可运行）
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═════════════════════════════════════════════════════════════════════════════
# 1. 参数解析
# ═════════════════════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(
        description="验证 VGGT global_block 前后帧间 token 相似度变化（含 K 空间分析）"
    )
    parser.add_argument("--image_dir",  type=str, required=True,
                        help="包含输入图像的目录")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="使用的帧数 S（建议 4~16）")
    parser.add_argument("--img_size",   type=int, default=518,
                        help="VGGT 输入图像大小（默认 518）")
    parser.add_argument("--output_dir", type=str, default="./results/similarity/",
                        help="结果输出目录")
    parser.add_argument("--device",     type=str, default="cuda",
                        help="运行设备（cuda/cpu）")
    parser.add_argument("--num_layers", type=int, default=24,
                        help="Aggregator 层数（默认 24）")
    parser.add_argument("--model_name", type=str, default="facebook/VGGT-1B",
                        help="HuggingFace 模型名称")
    parser.add_argument("--layers_to_show", type=int, nargs="+",
                        default=[0, 5, 11, 17, 23],
                        help="热力图中展示的层索引")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="K+V 联合指标中 K 的权重（0~1，默认 0.7）")
    parser.add_argument("--kv_thresholds", type=float, nargs="+",
                        default=[0.80, 0.85, 0.90, 0.95, 0.99],
                        help="P 帧压缩预算图中使用的 K 余弦相似度阈值列表")
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# 2. 数据加载
# ═════════════════════════════════════════════════════════════════════════════

def load_images(image_dir: str, num_frames: int, img_size: int) -> torch.Tensor:
    """从目录加载图像，返回 [1, S, 3, H, W] 张量，值域 [0, 1]。"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths: list[Path] = []
    for ext in exts:
        paths.extend(Path(image_dir).glob(ext))
    paths = sorted(set(paths))[:num_frames]

    if len(paths) < num_frames:
        raise ValueError(
            f"目录中只有 {len(paths)} 张图像，需要 {num_frames} 张。"
        )

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    frames = [tf(Image.open(p).convert("RGB")) for p in paths]
    print(f"  Loaded {len(paths)} frames:")
    for p in paths:
        print(f"    {p.name}")
    return torch.stack(frames).unsqueeze(0)   # [1, S, 3, H, W]


# ═════════════════════════════════════════════════════════════════════════════
# 3a. 特征空间 Hook（frame_blocks / global_blocks 输出）
# ═════════════════════════════════════════════════════════════════════════════

class SimilarityHookManager:
    """截取每层 frame_block 和 global_block 的输出激活，reshape 为 [B, S, P, C]。"""

    def __init__(self, model, B: int, S: int, num_layers: int = 24):
        self.B, self.S, self.num_layers = B, S, num_layers
        self.frame_outputs:  dict[int, torch.Tensor] = {}
        self.global_outputs: dict[int, torch.Tensor] = {}
        self._handles: list = []
        self._register(model)

    def _register(self, model):
        agg = model.aggregator
        for i in range(self.num_layers):
            self._handles.append(
                agg.frame_blocks[i].register_forward_hook(self._make_frame_hook(i))
            )
            self._handles.append(
                agg.global_blocks[i].register_forward_hook(self._make_global_hook(i))
            )

    def _make_frame_hook(self, idx):
        def hook(module, input, output):
            B, S = self.B, self.S
            # output: [B*S, P, C]
            self.frame_outputs[idx] = (
                output.detach().cpu().view(B, S, output.shape[1], output.shape[2])
            )
        return hook

    def _make_global_hook(self, idx):
        def hook(module, input, output):
            B, S = self.B, self.S
            P = output.shape[1] // S
            # output: [B, S*P, C]
            self.global_outputs[idx] = (
                output.detach().cpu().view(B, S, P, output.shape[2])
            )
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ═════════════════════════════════════════════════════════════════════════════
# 3b. K/V 空间 Hook（global_blocks[i].attn 内部）
# ═════════════════════════════════════════════════════════════════════════════

class KVHookManager:
    """
    截取 global_blocks[i] 的 Attention 模块内部 K 和 V。

    Hook 点：
      - K: global_blocks[i].attn.k_norm 的输出
           （K 经过 QK-Norm 后、RoPE 前。RoPE 对同位置 token 施加相同旋转，
             不影响同位置余弦相似度，故此处等价于 RoPE 后比较。）
      - V: global_blocks[i].attn.qkv 线性层输出中的第 3 份
           （V 不经过任何 norm，直接来自 QKV 投影。）

    K/V 输出均 reshape 为 [B, H, S, P, D]，存入 self.k_outputs / self.v_outputs。
    """

    def __init__(self, model, B: int, S: int, num_layers: int = 24,
                 P: int = 1374, H: int = 16, D: int = 64):
        self.B, self.S, self.P = B, S, P
        self.H, self.D = H, D
        self.num_layers = num_layers
        self.k_outputs: dict[int, torch.Tensor] = {}
        self.v_outputs: dict[int, torch.Tensor] = {}
        self._handles: list = []
        self._register(model)

    def _register(self, model):
        for i in range(self.num_layers):
            attn = model.aggregator.global_blocks[i].attn
            self._handles.append(
                attn.k_norm.register_forward_hook(self._make_k_hook(i))
            )
            self._handles.append(
                attn.qkv.register_forward_hook(self._make_v_hook(i))
            )

    def _make_k_hook(self, idx):
        def hook(module, input, output):
            # output: [B, H, S*P, D]  (K after QK-Norm, before RoPE)
            B, H, SP, D = output.shape
            S = self.S
            self.k_outputs[idx] = output.detach().cpu().view(B, H, S, SP // S, D)
        return hook

    def _make_v_hook(self, idx):
        def hook(module, input, output):
            # output: [B, S*P, 3*H*D]  (raw QKV projection)
            B, SP, C3 = output.shape
            S, H, D = self.S, self.H, self.D
            # Extract V (index 2 in the Q/K/V split)
            v = (
                output.detach().cpu()
                .reshape(B, SP, 3, H, D)
                .permute(2, 0, 3, 1, 4)[2]   # [B, H, SP, D]
            )
            self.v_outputs[idx] = v.view(B, H, S, SP // S, D)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ═════════════════════════════════════════════════════════════════════════════
# 4a. 特征空间相似度指标
# ═════════════════════════════════════════════════════════════════════════════

def cosine_sim_adjacent_frames(tokens: torch.Tensor) -> float:
    """相邻帧同位置 token 的平均余弦相似度。tokens: [B, S, P, C]"""
    t_prev = F.normalize(tokens[:, :-1], dim=-1)
    t_next = F.normalize(tokens[:, 1:],  dim=-1)
    return (t_prev * t_next).sum(dim=-1).mean().item()


def cosine_sim_all_frame_pairs(tokens: torch.Tensor) -> float:
    """所有帧对同位置 token 的平均余弦相似度。tokens: [B, S, P, C]"""
    _, S, _, _ = tokens.shape
    tokens_n = F.normalize(tokens, dim=-1)
    total, count = 0.0, 0
    for s1 in range(S):
        for s2 in range(s1 + 1, S):
            total += (tokens_n[:, s1] * tokens_n[:, s2]).sum(dim=-1).mean().item()
            count += 1
    return total / count if count > 0 else 0.0


def inter_frame_variance(tokens: torch.Tensor) -> float:
    """帧维度 token 方差均值（越大 = 帧间差异越大）。tokens: [B, S, P, C]"""
    return tokens.var(dim=1).mean().item()


def spatial_similarity_map(tokens: torch.Tensor, patch_h: int = 37,
                            patch_w: int = 37, patch_start_idx: int = 5) -> np.ndarray:
    """相邻帧同位置 patch token 余弦相似度的 [patch_h, patch_w] 空间热力图。"""
    pt = tokens[:, :, patch_start_idx:]   # [B, S, 1369, C]
    sim = (
        F.normalize(pt[:, :-1], dim=-1) *
        F.normalize(pt[:, 1:],  dim=-1)
    ).sum(dim=-1).mean(dim=(0, 1))        # [1369]
    return sim.numpy().reshape(patch_h, patch_w)


# ═════════════════════════════════════════════════════════════════════════════
# 4b. K 空间 / K+V 联合指标
# ═════════════════════════════════════════════════════════════════════════════

def k_cosine_sim_adjacent_frames(K: torch.Tensor) -> float:
    """
    K 空间相邻帧同位置余弦相似度（头平均标量）。

    Args:
        K: [B, H, S, P, D]  — K after QK-Norm
    Returns:
        scalar float（越高 = K 越相似 = P 帧可压缩）
    """
    K_prev = F.normalize(K[:, :, :-1], dim=-1)
    K_next = F.normalize(K[:, :, 1:],  dim=-1)
    return (K_prev * K_next).sum(dim=-1).mean().item()


def k_cosine_sim_per_position(K: torch.Tensor) -> torch.Tensor:
    """
    K 空间相邻帧同位置余弦相似度，保留空间分辨率（头平均）。

    Args:
        K: [B, H, S, P, D]
    Returns:
        sim: [B, S-1, P]，值越高 = 越相似 = P 帧可舍弃
    """
    K_prev = F.normalize(K[:, :, :-1], dim=-1)
    K_next = F.normalize(K[:, :, 1:],  dim=-1)
    sim_per_head = (K_prev * K_next).sum(dim=-1)   # [B, H, S-1, P]
    return sim_per_head.mean(dim=1)                # [B, S-1, P]


def v_reldelta_adjacent_frames(V: torch.Tensor) -> float:
    """
    V 空间相邻帧同位置相对 L2 变化量均值（头平均标量）。
    ΔV_rel = ||V_s - V_{s-1}|| / (||V_{s-1}|| + ε)

    Args:
        V: [B, H, S, P, D]
    Returns:
        scalar float（越大 = V 变化越大 = 内容差异越大）
    """
    V_prev = V[:, :, :-1]
    V_next = V[:, :, 1:]
    delta = (V_next - V_prev).norm(dim=-1) / (V_prev.norm(dim=-1) + 1e-8)
    return delta.mean().item()


def kv_joint_score_per_position(K: torch.Tensor, V: torch.Tensor,
                                 alpha: float = 0.7) -> torch.Tensor:
    """
    K+V 联合差异指标（P 帧压缩保留决策分数，越高 = 差异越大 = 越应保留）。

    score(s, p) = α·(1 - sim_K(s,p)) + (1-α)·ΔV_rel(s,p)

    Args:
        K, V: [B, H, S, P, D]
        alpha: K 权重（默认 0.7）
    Returns:
        score: [B, S-1, P]
    """
    # K 余弦差异
    K_prev = F.normalize(K[:, :, :-1], dim=-1)
    K_next = F.normalize(K[:, :, 1:],  dim=-1)
    sim_k  = (K_prev * K_next).sum(dim=-1).mean(dim=1)   # [B, S-1, P]
    k_diff = 1.0 - sim_k

    # V 相对 L2 差异
    V_prev = V[:, :, :-1]
    V_next = V[:, :, 1:]
    v_diff = (
        (V_next - V_prev).norm(dim=-1) / (V_prev.norm(dim=-1) + 1e-8)
    ).mean(dim=1)   # [B, S-1, P]

    return alpha * k_diff + (1.0 - alpha) * v_diff


def k_spatial_map(K: torch.Tensor, patch_h: int = 37, patch_w: int = 37,
                  patch_start_idx: int = 5) -> np.ndarray:
    """K 空间相邻帧同位置 patch token 余弦相似度的 [patch_h, patch_w] 热力图。"""
    K_patch = K[:, :, :, patch_start_idx:]        # [B, H, S, 1369, D]
    K_prev  = F.normalize(K_patch[:, :, :-1], dim=-1)
    K_next  = F.normalize(K_patch[:, :, 1:],  dim=-1)
    sim = (K_prev * K_next).sum(dim=-1).mean(dim=(0, 1, 2))  # [1369]
    return sim.numpy().reshape(patch_h, patch_w)


def kv_joint_spatial_map(K: torch.Tensor, V: torch.Tensor, alpha: float = 0.7,
                          patch_h: int = 37, patch_w: int = 37,
                          patch_start_idx: int = 5) -> np.ndarray:
    """K+V 联合差异指标的 [patch_h, patch_w] 空间热力图（score 越高 = 越应保留）。"""
    score = kv_joint_score_per_position(K, V, alpha)     # [B, S-1, P]
    score_patch = score[:, :, patch_start_idx:]          # [B, S-1, 1369]
    return score_patch.mean(dim=(0, 1)).numpy().reshape(patch_h, patch_w)


# ═════════════════════════════════════════════════════════════════════════════
# 5. 可视化
# ═════════════════════════════════════════════════════════════════════════════

def plot_similarity_curves(feat_results: dict, kv_results: dict,
                           output_dir: str) -> None:
    """三合一折线图：相邻帧余弦（feature-frame / feature-global / K-space）、
    全帧对余弦、帧间方差。"""
    n = len(feat_results["frame_adj_sim"])
    layers = list(range(n))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图 1: 相邻帧余弦相似度 ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(layers, feat_results["frame_adj_sim"],  "b-o", ms=4,
            label="feature: after frame_block")
    ax.plot(layers, feat_results["global_adj_sim"], "r-s", ms=4,
            label="feature: after global_block")
    ax.plot(layers, kv_results["k_adj_sim"],        "g-^", ms=4,
            label="K-space: K input to global_block")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Adjacent-frame cosine similarity (feature vs K-space)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── subplot 2: all-pair cosine similarity ─────────────────────────────
    ax = axes[1]
    ax.plot(layers, feat_results["frame_all_sim"],  "b-o", ms=4,
            label="feature: after frame_block")
    ax.plot(layers, feat_results["global_all_sim"], "r-s", ms=4,
            label="feature: after global_block")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean cosine similarity (all pairs)")
    ax.set_title("All-pair cosine similarity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── subplot 3: inter-frame variance ───────────────────────────────────
    ax = axes[2]
    ax.plot(layers, feat_results["frame_var"],  "b-o", ms=4,
            label="feature: after frame_block")
    ax.plot(layers, feat_results["global_var"], "r-s", ms=4,
            label="feature: after global_block")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Inter-frame token variance")
    ax.set_title("Inter-frame token variance (higher = more diverse)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "similarity_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


def plot_kv_curves(kv_results: dict, output_dir: str) -> None:
    """K/V 空间三合一曲线：K 余弦相似度、V 相对 L2 变化量、K+V 联合差异分数。"""
    n = len(kv_results["k_adj_sim"])
    layers = list(range(n))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(layers, kv_results["k_adj_sim"], "g-^", ms=4)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("K cosine similarity (adjacent frames)")
    ax.set_title("K-space adjacent-frame cosine sim (head-avg)\n(higher = more similar K = more compressible)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.plot(layers, kv_results["v_reldelta"], "m-v", ms=4)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("V relative L2 delta (adjacent frames)")
    ax.set_title("V-space adjacent-frame relative L2 delta\n(higher = more content change in V)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(layers, kv_results["kv_joint_score"], "k-D", ms=4)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("K+V joint difference score")
    ax.set_title("K+V joint difference score (higher = less compressible)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "kv_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


def plot_spatial_heatmaps(frame_maps: list[np.ndarray],
                          global_maps: list[np.ndarray],
                          k_maps: list[np.ndarray],
                          kv_maps: list[np.ndarray],
                          output_dir: str,
                          layers_to_show=(0, 5, 11, 17, 23)) -> None:
    """
    4 行热力图：
      行 0: feature  frame_block 后（余弦相似度，绿=高）
      行 1: feature  global_block 后（余弦相似度，绿=高）
      行 2: K 空间余弦相似度（绿=高=可压缩）
      行 3: K+V 联合差异分数（蓝=高=应保留，红=低=可丢弃）
    """
    valid = [i for i in layers_to_show if i < len(frame_maps)]
    n_cols = len(valid)
    if n_cols == 0:
        return

    row_labels = [
        "feature: after frame_block",
        "feature: after global_block",
        "K-space cosine similarity",
        "K+V joint diff score (high=keep)",
    ]
    row_data = [frame_maps, global_maps, k_maps, kv_maps]
    cmaps    = ["RdYlGn", "RdYlGn", "RdYlGn", "RdYlBu"]

    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 16))
    if n_cols == 1:
        axes = axes.reshape(4, 1)

    for row, (label, data, cmap) in enumerate(zip(row_labels, row_data, cmaps)):
        vmin = min(data[i].min() for i in valid)
        vmax = max(data[i].max() for i in valid)
        for col, layer_idx in enumerate(valid):
            im = axes[row, col].imshow(data[layer_idx],
                                       vmin=vmin, vmax=vmax, cmap=cmap)
            axes[row, col].set_title(f"Layer {layer_idx}\n{label}", fontsize=8)
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    plt.suptitle("Spatial heatmaps (37x37 patch grid)", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "spatial_heatmaps.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


def plot_delta_similarity(feat_results: dict, output_dir: str) -> None:
    """每层 Δ = global_adj_sim − frame_adj_sim 柱状图（负值=分化，正值=均质化）。"""
    delta = [g - f for g, f in
             zip(feat_results["global_adj_sim"], feat_results["frame_adj_sim"])]
    layers = list(range(len(delta)))
    colors = ["tomato" if d < 0 else "mediumseagreen" for d in delta]

    fig, ax = plt.subplots(figsize=(max(10, len(delta) // 2), 4))
    ax.bar(layers, delta, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Delta cosine similarity\n(after global_block - after frame_block)")
    ax.set_title("Effect of global_block on inter-frame similarity per layer\n(negative = more diverse; positive = more homogeneous)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "delta_similarity.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


def plot_feature_vs_k_scatter(feat_results: dict, kv_results: dict,
                               output_dir: str) -> None:
    """
    散点图：feature-space global_adj_sim（x）vs K-space k_adj_sim（y）。
    每层一个点，颜色代表层深度。验证两种指标的相关性。
    """
    x = np.array(feat_results["global_adj_sim"])
    y = np.array(kv_results["k_adj_sim"])
    n = len(x)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(n):
        ax.scatter(x[i], y[i], color=colors[i], s=60, zorder=3)
        ax.annotate(str(i), (x[i], y[i]), fontsize=7,
                    textcoords="offset points", xytext=(4, 2))

    lo = min(x.min(), y.min()) - 0.02
    hi = max(x.max(), y.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("Feature-space cosine sim (after global_block)")
    ax.set_ylabel("K-space cosine sim")
    ax.set_title("Feature-space vs K-space adjacent-frame cosine sim\n(labels = layer index, darker = deeper layer)")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(0, n - 1))
    plt.colorbar(sm, ax=ax, label="Layer index")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "feature_vs_k_scatter.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


def plot_p_frame_budget(kv_per_pos: dict, output_dir: str,
                        thresholds=(0.80, 0.85, 0.90, 0.95, 0.99)) -> None:
    """
    P 帧压缩预算图：各层在不同 K 余弦相似度阈值 τ 下，
    可舍弃（sim > τ）的 token 比例。
    """
    n = len(kv_per_pos["k_sim_per_pos"])
    layers = list(range(n))

    fig, ax = plt.subplots(figsize=(12, 5))
    for tau in thresholds:
        ratios = []
        for i in range(n):
            sim = kv_per_pos["k_sim_per_pos"][i]   # [B, S-1, P]
            ratios.append((sim > tau).float().mean().item())
        ax.plot(layers, ratios, "-o", ms=4, label=f"τ = {tau}")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Token drop ratio (K sim > tau)")
    ax.set_title(
        "P-frame compression: token drop ratio per layer at various thresholds\n"
        "(higher = more KV redundancy = higher compression benefit)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "p_frame_budget.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


def plot_summary_boxplot(feat_results: dict, kv_results: dict,
                         output_dir: str) -> None:
    """feature-frame / feature-global / K-space 三组相邻帧相似度分布箱线图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    data   = [feat_results["frame_adj_sim"],
              feat_results["global_adj_sim"],
              kv_results["k_adj_sim"]]
    labels = ["feature: frame_block 后", "feature: global_block 后", "K-space"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["steelblue", "salmon", "mediumseagreen"]
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
    ax.set_ylabel("相邻帧余弦相似度")
    ax.set_title("各指标相似度分布（所有层）")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "boxplot_similarity.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. 统计检验
# ═════════════════════════════════════════════════════════════════════════════

def statistical_test(feat_results: dict, kv_results: dict) -> dict:
    """
    T1: 配对 t 检验 frame_adj_sim > global_adj_sim（H1: global 使帧间分化）
    T2: Pearson r(global_feat_sim, K_sim)（两种指标相关性）
    """
    from scipy import stats

    f_sims = np.array(feat_results["frame_adj_sim"])
    g_sims = np.array(feat_results["global_adj_sim"])
    k_sims = np.array(kv_results["k_adj_sim"])

    t1, p1_two = stats.ttest_rel(f_sims, g_sims)
    p1_one = p1_two / 2 if t1 > 0 else 1.0 - p1_two / 2

    r_gk, p_gk = stats.pearsonr(g_sims, k_sims)

    return {
        "mean_frame_adj_sim":               float(f_sims.mean()),
        "mean_global_adj_sim":              float(g_sims.mean()),
        "mean_k_adj_sim":                   float(k_sims.mean()),
        "mean_delta_feat":                  float((g_sims - f_sims).mean()),
        "t_feat_frame_gt_global":           float(t1),
        "p_feat_frame_gt_global_one_sided": float(p1_one),
        "H1_feat_supported_at_p005":        bool(p1_one < 0.05 and t1 > 0),
        "pearson_r_global_vs_k":            float(r_gk),
        "pearson_p_global_vs_k":            float(p_gk),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. 主流程
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自动回退到 CPU。")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── 1. 加载模型 ──────────────────────────────────────────────────────────
    print(f"\nLoading model '{args.model_name}' ...")
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained(args.model_name)
    model.eval().to(device)
    print("Model loaded.")

    # ── 2. 加载图像 ──────────────────────────────────────────────────────────
    print(f"\nLoading images (num_frames={args.num_frames}) ...")
    images = load_images(args.image_dir, args.num_frames, args.img_size).to(device)
    B, S = images.shape[:2]
    print(f"Input tensor: {tuple(images.shape)}")

    # ── 3. 注册 Hooks（一次推理收集所有数据）──────────────────────────────────
    print(f"\nRegistering hooks ({args.num_layers} layers × 2 types) ...")
    feat_mgr = SimilarityHookManager(model, B=B, S=S, num_layers=args.num_layers)
    kv_mgr   = KVHookManager(model, B=B, S=S, num_layers=args.num_layers)

    # ── 4. 推理 ──────────────────────────────────────────────────────────────
    print("Running inference ...")
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(images)
        else:
            _ = model(images)
    print("Inference done.")

    # ── 5. 计算特征空间指标 ───────────────────────────────────────────────────
    print(f"\nComputing feature-space metrics ...")
    feat_results: dict[str, list] = {
        "frame_adj_sim": [], "global_adj_sim": [],
        "frame_all_sim": [], "global_all_sim": [],
        "frame_var":     [], "global_var":     [],
    }
    frame_spatial_maps:  list[np.ndarray] = []
    global_spatial_maps: list[np.ndarray] = []

    for i in range(args.num_layers):
        f_tok = feat_mgr.frame_outputs[i].float()
        g_tok = feat_mgr.global_outputs[i].float()
        feat_results["frame_adj_sim"].append(cosine_sim_adjacent_frames(f_tok))
        feat_results["global_adj_sim"].append(cosine_sim_adjacent_frames(g_tok))
        feat_results["frame_all_sim"].append(cosine_sim_all_frame_pairs(f_tok))
        feat_results["global_all_sim"].append(cosine_sim_all_frame_pairs(g_tok))
        feat_results["frame_var"].append(inter_frame_variance(f_tok))
        feat_results["global_var"].append(inter_frame_variance(g_tok))
        frame_spatial_maps.append(spatial_similarity_map(f_tok))
        global_spatial_maps.append(spatial_similarity_map(g_tok))

    # ── 6. 计算 K/V 空间指标 ─────────────────────────────────────────────────
    print("Computing K/V-space metrics ...")
    kv_results: dict[str, list] = {
        "k_adj_sim":      [],
        "v_reldelta":     [],
        "kv_joint_score": [],
    }
    kv_per_pos: dict[str, list[torch.Tensor]] = {
        "k_sim_per_pos":    [],
        "kv_score_per_pos": [],
    }
    k_spatial_maps:  list[np.ndarray] = []
    kv_spatial_maps: list[np.ndarray] = []

    for i in range(args.num_layers):
        K = kv_mgr.k_outputs[i].float()   # [B, H, S, P, D]
        V = kv_mgr.v_outputs[i].float()   # [B, H, S, P, D]

        kv_results["k_adj_sim"].append(k_cosine_sim_adjacent_frames(K))
        kv_results["v_reldelta"].append(v_reldelta_adjacent_frames(V))

        joint_per_pos = kv_joint_score_per_position(K, V, args.alpha)  # [B, S-1, P]
        kv_results["kv_joint_score"].append(joint_per_pos.mean().item())

        k_sim_per_pos = k_cosine_sim_per_position(K)   # [B, S-1, P]
        kv_per_pos["k_sim_per_pos"].append(k_sim_per_pos)
        kv_per_pos["kv_score_per_pos"].append(joint_per_pos)

        k_spatial_maps.append(k_spatial_map(K))
        kv_spatial_maps.append(kv_joint_spatial_map(K, V, args.alpha))

        if (i + 1) % 4 == 0 or i == args.num_layers - 1:
            delta = (feat_results["global_adj_sim"][-1]
                     - feat_results["frame_adj_sim"][-1])
            print(
                f"  Layer {i:2d}: "
                f"feat_global={feat_results['global_adj_sim'][-1]:.4f}, "
                f"K_sim={kv_results['k_adj_sim'][-1]:.4f}, "
                f"V_Δ={kv_results['v_reldelta'][-1]:.4f}, "
                f"feat_Δ={delta:+.4f}"
            )

    # ── 7. 统计检验 ──────────────────────────────────────────────────────────
    print("\nRunning statistical tests ...")
    stat = statistical_test(feat_results, kv_results)
    print(f"  mean frame_adj_sim         = {stat['mean_frame_adj_sim']:.4f}")
    print(f"  mean global_adj_sim        = {stat['mean_global_adj_sim']:.4f}")
    print(f"  mean K_adj_sim             = {stat['mean_k_adj_sim']:.4f}")
    print(f"  H1 (feat frame>global) "
          f"p={stat['p_feat_frame_gt_global_one_sided']:.5f}  "
          f"supported={stat['H1_feat_supported_at_p005']}")
    print(f"  Pearson r(global_feat, K)  = {stat['pearson_r_global_vs_k']:.4f}  "
          f"p={stat['pearson_p_global_vs_k']:.5f}")

    # ── 8. 保存数值结果 ───────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "results.json")
    kv_per_pos_json = {
        k: [t.tolist() for t in v] for k, v in kv_per_pos.items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "feat_metrics": feat_results,
            "kv_metrics":   kv_results,
            "statistics":   stat,
            "kv_per_pos":   kv_per_pos_json,
        }, f, indent=2)
    print(f"\n  [saved] {json_path}")

    # ── 9. 可视化 ─────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_similarity_curves(feat_results, kv_results, args.output_dir)
    plot_kv_curves(kv_results, args.output_dir)
    plot_spatial_heatmaps(
        frame_spatial_maps, global_spatial_maps,
        k_spatial_maps, kv_spatial_maps,
        args.output_dir, layers_to_show=args.layers_to_show,
    )
    plot_delta_similarity(feat_results, args.output_dir)
    plot_feature_vs_k_scatter(feat_results, kv_results, args.output_dir)
    plot_p_frame_budget(kv_per_pos, args.output_dir,
                        thresholds=args.kv_thresholds)
    plot_summary_boxplot(feat_results, kv_results, args.output_dir)

    # ── 10. 清理 ──────────────────────────────────────────────────────────────
    feat_mgr.remove()
    kv_mgr.remove()

    print("\n" + "=" * 60)
    print("实验完成。输出文件：")
    for fname in (
        "results.json",
        "similarity_curves.png",
        "kv_curves.png",
        "spatial_heatmaps.png",
        "delta_similarity.png",
        "feature_vs_k_scatter.png",
        "p_frame_budget.png",
        "boxplot_similarity.png",
    ):
        full = os.path.join(args.output_dir, fname)
        status = "✓" if os.path.exists(full) else "✗"
        print(f"  {status}  {full}")
    print("=" * 60)


if __name__ == "__main__":
    main()
