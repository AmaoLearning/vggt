"""
验证 VGGT global_block 前后帧间 token 相似度变化。

实验假设:
  H0: global_block 前后相邻帧同位置 token 的余弦相似度无显著差异。
  H1: global_block 处理前相邻帧同位置 token 的余弦相似度高于处理后，
      即 global attention 使各帧 token 分化，学习了帧间几何差异。

用法:
    # 从项目根目录运行
    python docs/tests/similarity_test.py \
        --image_dir examples/llff_fern/images \
        --num_frames 8 \
        --output_dir ./similarity_results

    # 或从 docs/tests/ 目录运行（自动调整 sys.path）
    python similarity_test.py \
        --image_dir ../../examples/llff_fern/images \
        --num_frames 8 \
        --output_dir ./similarity_results

要求:
    图像目录中至少有 num_frames 张 jpg/png 图像（按文件名排序）。
    VGGT 模型权重从 HuggingFace 自动下载（facebook/VGGT-1B）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── 路径修复：支持从任意目录运行 ──────────────────────────────────────────────
# 将项目根目录加入 sys.path，确保 "vggt" 包可以被找到
_SCRIPT_DIR = Path(__file__).resolve().parent           # tests/
_PROJECT_ROOT = _SCRIPT_DIR.parent              # VGGT/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")          # 无头环境不需要 GUI
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# 1.  参数解析
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(
        description="验证 VGGT global_block 前后帧间 token 相似度变化"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="包含输入图像的目录（jpg / png）"
    )
    parser.add_argument(
        "--num_frames", type=int, default=8,
        help="使用的帧数 S（建议 4~16，默认 8）"
    )
    parser.add_argument(
        "--img_size", type=int, default=518,
        help="VGGT 输入图像大小（默认 518）"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./similarity_results",
        help="结果输出目录（默认 ./similarity_results）"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="运行设备：cuda / cpu（默认 cuda，无 GPU 时自动回退到 cpu）"
    )
    parser.add_argument(
        "--num_layers", type=int, default=24,
        help="Aggregator 层数（默认 24）"
    )
    parser.add_argument(
        "--model_name", type=str, default="facebook/VGGT-1B",
        help="HuggingFace 模型 ID（默认 facebook/VGGT-1B）"
    )
    parser.add_argument(
        "--layers_to_show", type=int, nargs="+", default=[0, 5, 11, 17, 23],
        help="热力图可视化显示的层索引（默认 0 5 11 17 23）"
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_images(image_dir: str, num_frames: int, img_size: int) -> torch.Tensor:
    """
    从目录中按文件名排序加载图像，返回形状 [1, S, 3, H, W]，值域 [0, 1]。

    Args:
        image_dir:  图像目录路径
        num_frames: 最多加载的帧数
        img_size:   目标分辨率

    Returns:
        images: Tensor [1, S, 3, img_size, img_size]
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths: list[Path] = []
    for ext in exts:
        paths.extend(Path(image_dir).glob(ext))
    paths = sorted(set(paths))[:num_frames]

    if len(paths) < num_frames:
        raise ValueError(
            f"目录 '{image_dir}' 中只找到 {len(paths)} 张图像，"
            f"需要至少 {num_frames} 张。"
        )

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),   # → [0, 1]
    ])

    frames = [tf(Image.open(p).convert("RGB")) for p in paths]
    images = torch.stack(frames)        # [S, 3, H, W]
    print(f"  Loaded {len(paths)} frames from '{image_dir}'")
    for p in paths:
        print(f"    {p.name}")
    return images.unsqueeze(0)          # [1, S, 3, H, W]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Hook 管理器
# ══════════════════════════════════════════════════════════════════════════════

class SimilarityHookManager:
    """
    在 aggregator.frame_blocks 和 aggregator.global_blocks 上注册 forward hook，
    截取每层输出，reshape 为 [B, S, P, C] 后存储。

    frame_blocks[i] 以 [B×S, P, C] 形状运行；
    global_blocks[i] 以 [B, S×P, C] 形状运行。
    """

    def __init__(self, model: "VGGT", B: int, S: int, num_layers: int = 24):
        self.B = B
        self.S = S
        self.num_layers = num_layers

        # 存储截取的激活，key = layer index
        self.frame_outputs: dict[int, torch.Tensor] = {}
        self.global_outputs: dict[int, torch.Tensor] = {}

        self._handles: list = []
        self._register_hooks(model)

    # ── 注册 ────────────────────────────────────────────────────────────────

    def _register_hooks(self, model):
        agg = model.aggregator
        for i in range(self.num_layers):
            h1 = agg.frame_blocks[i].register_forward_hook(self._make_frame_hook(i))
            h2 = agg.global_blocks[i].register_forward_hook(self._make_global_hook(i))
            self._handles.extend([h1, h2])

    # ── Hook 工厂 ────────────────────────────────────────────────────────────

    def _make_frame_hook(self, idx: int):
        def hook(module, input, output):
            # output: [B×S, P, C]
            B, S = self.B, self.S
            # 移到 CPU 并脱离计算图，避免显存泄漏
            self.frame_outputs[idx] = (
                output.detach().cpu()
                .view(B, S, output.shape[1], output.shape[2])
            )
        return hook

    def _make_global_hook(self, idx: int):
        def hook(module, input, output):
            # output: [B, S×P, C]
            B, S = self.B, self.S
            P = output.shape[1] // S
            self.global_outputs[idx] = (
                output.detach().cpu()
                .view(B, S, P, output.shape[2])
            )
        return hook

    # ── 清理 ────────────────────────────────────────────────────────────────

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  相似度指标
# ══════════════════════════════════════════════════════════════════════════════

def cosine_sim_adjacent_frames(tokens: torch.Tensor) -> float:
    """
    相邻帧（s, s+1）之间同位置 token 的平均余弦相似度。

    Args:
        tokens: [B, S, P, C]

    Returns:
        scalar float ∈ [-1, 1]，越高越相似
    """
    t_prev = tokens[:, :-1, :, :]   # [B, S-1, P, C]
    t_next = tokens[:, 1:, :, :]    # [B, S-1, P, C]

    t_prev_n = F.normalize(t_prev, dim=-1)
    t_next_n = F.normalize(t_next, dim=-1)

    sim = (t_prev_n * t_next_n).sum(dim=-1)   # [B, S-1, P]
    return sim.mean().item()


def cosine_sim_all_frame_pairs(tokens: torch.Tensor) -> float:
    """
    所有帧对（s, s'）之间同位置 token 的平均余弦相似度。

    Args:
        tokens: [B, S, P, C]

    Returns:
        scalar float
    """
    B, S, P, C = tokens.shape
    tokens_n = F.normalize(tokens, dim=-1)   # [B, S, P, C]

    total_sim = 0.0
    count = 0
    for s1 in range(S):
        for s2 in range(s1 + 1, S):
            sim = (tokens_n[:, s1] * tokens_n[:, s2]).sum(dim=-1)  # [B, P]
            total_sim += sim.mean().item()
            count += 1

    return total_sim / count if count > 0 else 0.0


def inter_frame_variance(tokens: torch.Tensor) -> float:
    """
    帧维度上的 token 方差均值（越大 = 帧间差异越大 = 越分化）。

    Args:
        tokens: [B, S, P, C]

    Returns:
        scalar float
    """
    var = tokens.var(dim=1)    # [B, P, C]
    return var.mean().item()


def spatial_similarity_map(
    tokens: torch.Tensor,
    patch_h: int = 37,
    patch_w: int = 37,
    patch_start_idx: int = 5,
) -> np.ndarray:
    """
    计算每个空间位置上相邻帧的平均余弦相似度，reshape 为 [patch_h, patch_w] 热力图。

    Args:
        tokens:          [B, S, P, C]，P = patch_start_idx + patch_h*patch_w
        patch_h:         patch 网格高度（默认 37 = 518/14）
        patch_w:         patch 网格宽度（默认 37 = 518/14）
        patch_start_idx: camera + register token 数量（默认 5）

    Returns:
        heatmap: ndarray [patch_h, patch_w]，值域 [-1, 1]
    """
    patch_tokens = tokens[:, :, patch_start_idx:, :]   # [B, S, 1369, C]
    t_prev = patch_tokens[:, :-1]   # [B, S-1, 1369, C]
    t_next = patch_tokens[:, 1:]    # [B, S-1, 1369, C]

    t_prev_n = F.normalize(t_prev, dim=-1)
    t_next_n = F.normalize(t_next, dim=-1)

    sim = (t_prev_n * t_next_n).sum(dim=-1)   # [B, S-1, 1369]
    sim_mean = sim.mean(dim=(0, 1))            # [1369]

    return sim_mean.numpy().reshape(patch_h, patch_w)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  统计检验
# ══════════════════════════════════════════════════════════════════════════════

def statistical_test(results: dict) -> dict:
    """
    对逐层 frame_adj_sim 和 global_adj_sim 做配对 t 检验。
    单侧检验 H1: frame > global（即 global attention 使相似度下降）。

    Returns:
        stat_results: dict，包含均值、t 统计量、p 值及结论
    """
    from scipy import stats

    frame_sims  = np.array(results["frame_adj_sim"])
    global_sims = np.array(results["global_adj_sim"])

    # 配对 t 检验（双侧）
    t_stat, p_two = stats.ttest_rel(frame_sims, global_sims)
    # 转换为单侧（H1: frame > global → t > 0）
    p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0

    return {
        "mean_frame_adj_sim":                    float(frame_sims.mean()),
        "mean_global_adj_sim":                   float(global_sims.mean()),
        "mean_delta":                            float((global_sims - frame_sims).mean()),
        "t_statistic":                           float(t_stat),
        "p_value_two_sided":                     float(p_two),
        "p_value_one_sided_frame_gt_global":     float(p_one),
        "hypothesis_H1_supported_at_p005":       bool(p_one < 0.05 and t_stat > 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_similarity_curves(results: dict, output_dir: str) -> None:
    """三合一折线图：相邻帧余弦、全帧对余弦、帧间方差（每层 frame vs global）。"""
    n = len(results["frame_adj_sim"])
    layers = list(range(n))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图 1: 相邻帧余弦相似度 ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(layers, results["frame_adj_sim"],  "b-o", ms=4, label="frame_block 后")
    ax.plot(layers, results["global_adj_sim"], "r-s", ms=4, label="global_block 后")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("相邻帧余弦相似度（同位置 token）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, n - 0.5)

    # ── 子图 2: 全帧对余弦相似度 ────────────────────────────────────────────
    ax = axes[1]
    ax.plot(layers, results["frame_all_sim"],  "b-o", ms=4, label="frame_block 后")
    ax.plot(layers, results["global_all_sim"], "r-s", ms=4, label="global_block 后")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean cosine similarity (all pairs)")
    ax.set_title("全帧对余弦相似度")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, n - 0.5)

    # ── 子图 3: 帧间方差 ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(layers, results["frame_var"],  "b-o", ms=4, label="frame_block 后")
    ax.plot(layers, results["global_var"], "r-s", ms=4, label="global_block 后")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Inter-frame token variance")
    ax.set_title("帧间 token 方差（越大 = 越分化）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, n - 0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "similarity_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


def plot_spatial_heatmaps(
    frame_maps: list[np.ndarray],
    global_maps: list[np.ndarray],
    output_dir: str,
    layers_to_show: tuple | list = (0, 5, 11, 17, 23),
) -> None:
    """选定层的空间相似度热力图（frame vs global 各一行）。"""
    # 过滤掉超出范围的层索引
    valid_layers = [i for i in layers_to_show if i < len(frame_maps)]
    n_cols = len(valid_layers)
    if n_cols == 0:
        print("  [warn] layers_to_show 中没有有效层索引，跳过热力图。")
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    # 若只有一列，axes 维度需统一为二维
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    vmin = min(
        min(frame_maps[i].min() for i in valid_layers),
        min(global_maps[i].min() for i in valid_layers),
    )
    vmax = max(
        max(frame_maps[i].max() for i in valid_layers),
        max(global_maps[i].max() for i in valid_layers),
    )

    for col, layer_idx in enumerate(valid_layers):
        for row, (title_prefix, maps) in enumerate(
            [("frame_block 后", frame_maps), ("global_block 后", global_maps)]
        ):
            im = axes[row, col].imshow(
                maps[layer_idx], vmin=vmin, vmax=vmax, cmap="RdYlGn"
            )
            axes[row, col].set_title(f"Layer {layer_idx}\n{title_prefix}")
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    plt.suptitle(
        "空间位置分辨的相邻帧余弦相似度热力图\n（绿 = 高相似 / 红 = 低相似 / 分化）",
        fontsize=13,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "spatial_heatmaps.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


def plot_delta_similarity(results: dict, output_dir: str) -> None:
    """每层 Δ = global_adj_sim - frame_adj_sim 的柱状图（负值 = 分化，绿色 = 相似度升高）。"""
    delta = [
        g - f
        for g, f in zip(results["global_adj_sim"], results["frame_adj_sim"])
    ]
    layers = list(range(len(delta)))
    colors = ["tomato" if d < 0 else "mediumseagreen" for d in delta]

    fig, ax = plt.subplots(figsize=(max(10, len(delta) // 2), 4))
    ax.bar(layers, delta, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Δ cosine similarity\n(global_block 后 − frame_block 后)")
    ax.set_title(
        "每层 global_block 对帧间相似度的影响\n"
        "（负值 / 红色 = 相似度下降 → token 分化；正值 / 绿色 = 相似度升高）"
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "delta_similarity.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


def plot_summary_boxplot(results: dict, output_dir: str) -> None:
    """
    附加图：frame 与 global 的相邻帧相似度在所有层上的箱线图，
    直观展示两者的分布差异。
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [results["frame_adj_sim"], results["global_adj_sim"]]
    bp = ax.boxplot(data, labels=["frame_block 后", "global_block 后"], patch_artist=True)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][1].set_facecolor("salmon")
    ax.set_ylabel("相邻帧余弦相似度")
    ax.set_title("frame_block vs global_block 相似度分布（所有层）")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "boxplot_similarity.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    if str(device) == "cpu" and args.device == "cuda":
        print("[warn] CUDA 不可用，自动回退到 CPU（推理速度较慢）。")
    print(f"Device: {device}")

    # ── 1. 加载模型 ─────────────────────────────────────────────────────────
    print(f"\nLoading model '{args.model_name}' ...")
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained(args.model_name)
    model.eval().to(device)
    print("Model loaded.")

    # ── 2. 加载图像 ─────────────────────────────────────────────────────────
    print(f"\nLoading images (num_frames={args.num_frames}) ...")
    images = load_images(args.image_dir, args.num_frames, args.img_size)
    images = images.to(device)
    B, S = images.shape[:2]
    print(f"Input tensor: {tuple(images.shape)}")

    # ── 3. 注册 Hook ─────────────────────────────────────────────────────────
    print(f"\nRegistering hooks on {args.num_layers} frame_blocks and global_blocks ...")
    hook_mgr = SimilarityHookManager(model, B=B, S=S, num_layers=args.num_layers)

    # ── 4. 推理 ──────────────────────────────────────────────────────────────
    print("Running inference ...")
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(images)
        else:
            _ = model(images)
    print("Inference done.")

    # ── 5. 逐层计算指标 ──────────────────────────────────────────────────────
    print(f"\nComputing similarity metrics for {args.num_layers} layers ...")
    results: dict[str, list] = {
        "frame_adj_sim":  [],
        "global_adj_sim": [],
        "frame_all_sim":  [],
        "global_all_sim": [],
        "frame_var":      [],
        "global_var":     [],
    }
    frame_spatial_maps:  list[np.ndarray] = []
    global_spatial_maps: list[np.ndarray] = []

    for i in range(args.num_layers):
        f_tok = hook_mgr.frame_outputs[i].float()    # [B, S, P, C]
        g_tok = hook_mgr.global_outputs[i].float()   # [B, S, P, C]

        results["frame_adj_sim"].append(cosine_sim_adjacent_frames(f_tok))
        results["global_adj_sim"].append(cosine_sim_adjacent_frames(g_tok))
        results["frame_all_sim"].append(cosine_sim_all_frame_pairs(f_tok))
        results["global_all_sim"].append(cosine_sim_all_frame_pairs(g_tok))
        results["frame_var"].append(inter_frame_variance(f_tok))
        results["global_var"].append(inter_frame_variance(g_tok))

        # patch_start_idx = 5（camera_token × 1 + register_token × 4）
        frame_spatial_maps.append(spatial_similarity_map(f_tok, patch_start_idx=5))
        global_spatial_maps.append(spatial_similarity_map(g_tok, patch_start_idx=5))

        if (i + 1) % 4 == 0 or i == args.num_layers - 1:
            delta = results["global_adj_sim"][-1] - results["frame_adj_sim"][-1]
            print(
                f"  Layer {i:2d}: "
                f"frame_adj={results['frame_adj_sim'][-1]:.4f}, "
                f"global_adj={results['global_adj_sim'][-1]:.4f}, "
                f"Δ={delta:+.4f}"
            )

    # ── 6. 统计检验 ──────────────────────────────────────────────────────────
    print("\nRunning paired t-test (H1: frame_adj_sim > global_adj_sim) ...")
    stat = statistical_test(results)
    print(f"  mean frame_adj_sim   = {stat['mean_frame_adj_sim']:.4f}")
    print(f"  mean global_adj_sim  = {stat['mean_global_adj_sim']:.4f}")
    print(f"  mean Δ               = {stat['mean_delta']:+.4f}")
    print(f"  t-statistic          = {stat['t_statistic']:.4f}")
    print(f"  p-value (two-sided)  = {stat['p_value_two_sided']:.6f}")
    print(f"  p-value (one-sided)  = {stat['p_value_one_sided_frame_gt_global']:.6f}")
    print(f"  H1 supported (p<0.05): {stat['hypothesis_H1_supported_at_p005']}")

    # ── 7. 保存数值结果 ───────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "statistics": stat}, f, indent=2)
    print(f"\n  [saved] {json_path}")

    # ── 8. 可视化 ────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_similarity_curves(results, args.output_dir)
    plot_spatial_heatmaps(
        frame_spatial_maps, global_spatial_maps,
        args.output_dir, layers_to_show=args.layers_to_show,
    )
    plot_delta_similarity(results, args.output_dir)
    plot_summary_boxplot(results, args.output_dir)

    # ── 9. 清理 Hook ─────────────────────────────────────────────────────────
    hook_mgr.remove()

    print("\n" + "=" * 60)
    print("实验完成。输出文件：")
    for fname in ("results.json", "similarity_curves.png",
                  "spatial_heatmaps.png", "delta_similarity.png",
                  "boxplot_similarity.png"):
        full = os.path.join(args.output_dir, fname)
        status = "✓" if os.path.exists(full) else "✗ (缺失)"
        print(f"  {status}  {full}")
    print("=" * 60)


if __name__ == "__main__":
    main()
