#!/usr/bin/env python3
"""
tests/silence_test.py
=====================
VGGT Layer Silencing Experiment — Camera Pose Focus

动机
----
`visualize_attention.py` 以 cam_attn 模式在 llff_fern 场景（8帧）可视化了
各层 Global Attention 中相机 token 的注意力分布。结果显示：
  - 层 3–4 和 层 10–16 的相机 token 仅对 4–5 个离群位置给予极高权重，
    其余 patch 几乎为零。
  - 相比之下，其他层（如 0–2、5–9、17–23）呈现更连续的空间梯度。

假设
----
注意力极度稀疏的层（3–4, 10–16）对相机位姿预测贡献极小，
可以"静默"（identity bypass）而不显著损害预测质量。

静默机制
--------
通过 PyTorch forward hook 将 Block 的输出替换为输入，使该层成为恒等变换：

    def hook(module, input, output):
        return input[0]   # tokens 不经过 attention+FFN，原样返回

在 Block 执行完计算后，hook 丢弃其结果并返回原始 token，
后续层因此看不到该层的贡献。

实验分组
--------
  A. 逐层全局注意力静默  (global_blocks[0..23])  — 一次静默一层
  B. 逐层帧注意力静默    (frame_blocks[0..23])   — 一次静默一层
  C. 范围静默（用户观测驱动）：
       range_global_3_4   / range_frame_3_4   / range_both_3_4
       range_global_10_16 / range_frame_10_16 / range_both_10_16
       range_both_3_4_10_16  （两段合并）
  D. 边界对照：all_global / all_frame / first_half / second_half

主指标：ATE (m)  on TUM-dynamics walking_xyz
辅指标：acc_mean (m) on 7-Scenes chess

结果输出
--------
  tests/results/silence_test/
    results.json
    summary.csv
    heatmap_per_layer.png       — 每层 ATE 变化量（frame vs global）
    range_ablation.png          — 范围静默柱状图
    sensitivity_curve.png       — 按 ATE 变化排序的敏感度曲线

用法
----
  # 完整实验（全部 48 + 范围 + 边界 = ~60 个配置）
  python tests/silence_test.py --device cuda --max_frames 16

  # 仅测试范围静默（快速，约 15 个配置）
  python tests/silence_test.py --device cuda --groups range boundary

  # 仅测试全局逐层（24 个配置）
  python tests/silence_test.py --device cuda --groups per_layer_global
"""

# ── 标准库 ────────────────────────────────────────────────────────────────────
import sys
import json
import struct
import warnings
import contextlib
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 第三方库 ──────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── 项目路径 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent         # tests/
_PROJECT_ROOT = _SCRIPT_DIR.parent                      # VGGT/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── VGGT 核心 ─────────────────────────────────────────────────────────────────
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# ══════════════════════════════════════════════════════════════════════════════
# 0. 全局常量
# ══════════════════════════════════════════════════════════════════════════════

IMG_SIZE    = 518
DTYPE       = torch.bfloat16
RESULTS_DIR = _SCRIPT_DIR / "results" / "silence_test"

DATASET_ROOT  = _PROJECT_ROOT.parent / "datasets"
DATASET_PATHS = {
    "tum":     DATASET_ROOT / "tum-dynamics" / "walking_xyz",
    "7scenes": DATASET_ROOT / "7-scenes"     / "chess",
}

# 根据 visualize_attention.py 结果的用户观测
OBSERVED_SPARSE_LAYERS = {
    "global_3_4":   ([3, 4],         "global"),
    "global_10_16": (list(range(10, 17)), "global"),
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. 静默配置
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SilenceConfig:
    """
    描述哪些 Aggregator Block 应被静默（identity bypass）。

    Attributes:
        frame_layers : 要静默的 frame_blocks 层索引列表
        global_layers: 要静默的 global_blocks 层索引列表
        label        : 可读描述（用于报告）
    """
    frame_layers:  List[int] = field(default_factory=list)
    global_layers: List[int] = field(default_factory=list)
    label: str = ""

    def is_empty(self) -> bool:
        return len(self.frame_layers) == 0 and len(self.global_layers) == 0

    def __repr__(self) -> str:
        parts = []
        if self.frame_layers:
            parts.append(f"frame={self.frame_layers}")
        if self.global_layers:
            parts.append(f"global={self.global_layers}")
        return f"SilenceConfig({', '.join(parts)})" if parts else "SilenceConfig(baseline)"


def build_silence_configs(groups: Optional[List[str]] = None) -> Dict[str, SilenceConfig]:
    """
    构建所有静默实验配置。

    分组：
      baseline         — 不静默任何层（参照组）
      per_layer_global — 逐层全局注意力静默（24 个配置）
      per_layer_frame  — 逐层帧注意力静默（24 个配置）
      range            — 范围静默（用户观测驱动，7 个配置）
      boundary         — 边界对照（全静默 / 前半 / 后半，4 个配置）
    """
    all_groups = {"baseline", "per_layer_global", "per_layer_frame", "range", "boundary"}
    active = set(groups) if groups else all_groups

    configs: Dict[str, SilenceConfig] = {}

    # ── baseline ──────────────────────────────────────────────────────────────
    if "baseline" in active:
        configs["baseline"] = SilenceConfig(label="Baseline（无静默）")

    # ── per_layer_global：逐层全局注意力静默 ─────────────────────────────────
    if "per_layer_global" in active:
        for i in range(24):
            configs[f"G{i:02d}"] = SilenceConfig(
                global_layers=[i],
                label=f"仅静默 global_blocks[{i}]",
            )

    # ── per_layer_frame：逐层帧注意力静默 ────────────────────────────────────
    if "per_layer_frame" in active:
        for i in range(24):
            configs[f"F{i:02d}"] = SilenceConfig(
                frame_layers=[i],
                label=f"仅静默 frame_blocks[{i}]",
            )

    # ── range：范围静默 ───────────────────────────────────────────────────────
    if "range" in active:
        # 层 3–4：观测到稀疏 cam_attn
        configs["range_global_3_4"] = SilenceConfig(
            global_layers=[3, 4],
            label="全局块 3–4 静默（观测稀疏层）",
        )
        configs["range_frame_3_4"] = SilenceConfig(
            frame_layers=[3, 4],
            label="帧块 3–4 静默",
        )
        configs["range_both_3_4"] = SilenceConfig(
            frame_layers=[3, 4], global_layers=[3, 4],
            label="帧+全局块 3–4 静默",
        )

        # 层 10–16：观测到稀疏 cam_attn
        configs["range_global_10_16"] = SilenceConfig(
            global_layers=list(range(10, 17)),
            label="全局块 10–16 静默（观测稀疏层）",
        )
        configs["range_frame_10_16"] = SilenceConfig(
            frame_layers=list(range(10, 17)),
            label="帧块 10–16 静默",
        )
        configs["range_both_10_16"] = SilenceConfig(
            frame_layers=list(range(10, 17)), global_layers=list(range(10, 17)),
            label="帧+全局块 10–16 静默",
        )

        # 两段合并
        configs["range_both_3_4_10_16"] = SilenceConfig(
            frame_layers=[3, 4] + list(range(10, 17)),
            global_layers=[3, 4] + list(range(10, 17)),
            label="帧+全局块 3–4 + 10–16 合并静默",
        )

    # ── boundary：边界对照 ────────────────────────────────────────────────────
    if "boundary" in active:
        configs["all_global"] = SilenceConfig(
            global_layers=list(range(24)),
            label="全部 24 个全局块静默（破坏上界）",
        )
        configs["all_frame"] = SilenceConfig(
            frame_layers=list(range(24)),
            label="全部 24 个帧块静默（破坏上界）",
        )
        configs["first_half_global"] = SilenceConfig(
            global_layers=list(range(12)),
            label="全局块前半段 0–11 静默",
        )
        configs["second_half_global"] = SilenceConfig(
            global_layers=list(range(12, 24)),
            label="全局块后半段 12–23 静默",
        )

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# 2. 静默机制：forward hook
# ══════════════════════════════════════════════════════════════════════════════

class LayerSilencer:
    """
    上下文管理器：在指定的 Aggregator Block 上安装 identity forward hook。

    用法：
        with LayerSilencer(model, cfg):
            preds, elapsed = run_inference(model, frames, ...)

    Hook 逻辑：
        Block.forward(x, pos=None) → output
        Hook 接收 (module, input, output)，其中 input[0] = x（位置参数）。
        返回 input[0] 使 Block 输出等于输入，跳过 attention + FFN。

    注意：
        pos 以关键字参数传入，不在 input 元组中；Block 的残差结构使得
        返回原始 x 等价于令 Δx = 0（attention 和 MLP 的增量均被清零）。
    """

    def __init__(self, model: VGGT, cfg: SilenceConfig):
        self.model = model
        self.cfg = cfg
        self._handles: List[torch.utils.hooks.RemovableHook] = []

    def _identity_hook(self, module, input, output):
        """返回 Block 的原始输入 x，绕过所有计算。"""
        # input[0] = x（[B*S, P, C] 帧块 或 [B, S*P, C] 全局块）
        return input[0]

    def __enter__(self):
        agg = self.model.aggregator

        for i in self.cfg.frame_layers:
            if 0 <= i < len(agg.frame_blocks):
                h = agg.frame_blocks[i].register_forward_hook(self._identity_hook)
                self._handles.append(h)
            else:
                warnings.warn(f"LayerSilencer: frame_blocks[{i}] 超出范围（depth={len(agg.frame_blocks)}）")

        for i in self.cfg.global_layers:
            if 0 <= i < len(agg.global_blocks):
                h = agg.global_blocks[i].register_forward_hook(self._identity_hook)
                self._handles.append(h)
            else:
                warnings.warn(f"LayerSilencer: global_blocks[{i}] 超出范围（depth={len(agg.global_blocks)}）")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False  # 不吞异常


# ══════════════════════════════════════════════════════════════════════════════
# 3. 数据集加载
# ══════════════════════════════════════════════════════════════════════════════

def _make_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def load_tum_walking_xyz(data_dir: Path, max_frames: int) -> dict:
    """
    加载 TUM RGB-D walking_xyz 序列（相机位姿评估主数据集）。

    目录结构：
        walking_xyz/
          rgb/             timestamped PNG
          groundtruth.txt  ts tx ty tz qx qy qz qw
          rgb.txt          ts path
    """
    gt_file  = data_dir / "groundtruth.txt"
    rgb_file = data_dir / "rgb.txt"

    if not gt_file.exists():
        raise FileNotFoundError(f"未找到 {gt_file}")
    if not rgb_file.exists():
        raise FileNotFoundError(f"未找到 {rgb_file}")

    gt_map: Dict[float, np.ndarray] = {}
    for line in open(gt_file):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        ts = float(parts[0])
        tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:8])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        gt_map[ts] = T
    gt_ts = sorted(gt_map.keys())

    rgb_entries: List[Tuple[float, Path]] = []
    for line in open(rgb_file):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rgb_entries.append((float(parts[0]), data_dir / parts[1]))
    rgb_entries.sort(key=lambda x: x[0])

    tf = _make_transform()
    frames, poses = [], []

    for rgb_ts, img_path in rgb_entries:
        if not img_path.exists():
            continue
        nearest = min(gt_ts, key=lambda t: abs(t - rgb_ts))
        if abs(nearest - rgb_ts) > 0.05:
            continue
        frames.append(tf(Image.open(img_path).convert("RGB")))
        poses.append(gt_map[nearest])
        if len(frames) >= max_frames:
            break

    if not frames:
        raise FileNotFoundError(f"TUM walking_xyz：{data_dir} 中未匹配到 RGB-GT 对")

    return {"frames": torch.stack(frames), "poses_c2w": np.array(poses)}


def load_7scenes_chess(data_dir: Path, max_frames: int) -> dict:
    """
    加载 7-Scenes chess（相机位姿辅助数据集）。

    目录结构：
        chess/seq-01/
          frame-*.color.png
          frame-*.pose.txt
    """
    K = np.array([[525.0, 0.0, 320.0],
                  [0.0, 525.0, 240.0],
                  [0.0, 0.0,   1.0]], dtype=np.float64)

    seq_dir = data_dir / "seq-01"
    if not seq_dir.exists():
        raise FileNotFoundError(f"未找到 {seq_dir}")

    tf = _make_transform()
    frames, poses, depths = [], [], []

    for cf in sorted(seq_dir.glob("frame-*.color.png")):
        stem   = cf.stem.replace(".color", "")
        pose_f = seq_dir / f"{stem}.pose.txt"
        dep_f  = seq_dir / f"{stem}.depth.png"
        if not pose_f.exists():
            continue
        frames.append(tf(Image.open(cf).convert("RGB")))
        poses.append(np.loadtxt(str(pose_f)))
        if dep_f.exists():
            d = np.array(Image.open(dep_f), dtype=np.float32) / 1000.0
            depths.append(d)
        if len(frames) >= max_frames:
            break

    if not frames:
        raise FileNotFoundError(f"7-Scenes chess：{data_dir} 中未找到帧")

    gt_pts_list = []
    if depths:
        H_orig, W_orig = depths[0].shape
        u, v = np.meshgrid(np.arange(W_orig), np.arange(H_orig))
        for depth, pose in zip(depths, poses):
            valid = (depth > 0.1) & (depth < 10.0)
            X = (u[valid] - K[0, 2]) * depth[valid] / K[0, 0]
            Y = (v[valid] - K[1, 2]) * depth[valid] / K[1, 1]
            Z = depth[valid]
            pts_cam   = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)
            pts_world = (pose @ pts_cam.T).T[:, :3]
            gt_pts_list.append(pts_world)

    all_gt = np.empty((0, 3), dtype=np.float32)
    if gt_pts_list:
        all_gt = np.concatenate(gt_pts_list, axis=0)
        if len(all_gt) > 50_000:
            rng = np.random.default_rng(42)
            all_gt = all_gt[rng.choice(len(all_gt), 50_000, replace=False)]

    return {
        "frames":    torch.stack(frames),
        "poses_c2w": np.array(poses),
        "gt_points": all_gt.astype(np.float32),
        "K":         K,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 推理
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model: VGGT, frames: torch.Tensor,
                  dtype: torch.dtype, device: str,
                  warmup: int = 2) -> Tuple[dict, float]:
    """运行 VGGT 推理，返回预测字典和 wall-clock 时间（秒）。"""
    imgs = frames.to(device).float()
    imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE),
                         mode="bilinear", align_corners=False)
    imgs = imgs.unsqueeze(0)  # [1, S, 3, H, W]

    ctx = (torch.cuda.amp.autocast(dtype=dtype)
           if device == "cuda" else contextlib.nullcontext())

    with torch.no_grad():
        for _ in range(warmup):
            with ctx:
                _ = model(imgs)
            if device == "cuda":
                torch.cuda.synchronize()

        if device == "cuda":
            torch.cuda.synchronize()
            s_evt = torch.cuda.Event(enable_timing=True)
            e_evt = torch.cuda.Event(enable_timing=True)
            s_evt.record()
            with ctx:
                preds = model(imgs)
            e_evt.record()
            torch.cuda.synchronize()
            elapsed = s_evt.elapsed_time(e_evt) / 1000.0
        else:
            import time as _time
            t0 = _time.perf_counter()
            with ctx:
                preds = model(imgs)
            elapsed = _time.perf_counter() - t0

    out = {k: v.float().cpu() for k, v in preds.items()
           if isinstance(v, torch.Tensor)}
    if device == "cuda":
        torch.cuda.empty_cache()

    return out, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# 5. 评估指标
# ══════════════════════════════════════════════════════════════════════════════

def _extri_to_cam_pos(extri: np.ndarray) -> np.ndarray:
    """world-to-cam 外参 → 相机中心（世界坐标）。"""
    R = extri[:, :3, :3]   # [S, 3, 3]
    t = extri[:, :3,  3]   # [S, 3]
    return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray
                      ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Umeyama Sim3 对齐：dst ≈ s · R @ src + t"""
    N     = src.shape[0]
    mu_s  = src.mean(0);  mu_d = dst.mean(0)
    sc    = src - mu_s;   dc   = dst - mu_d
    var_s = np.mean(np.sum(sc ** 2, axis=1))
    cov   = (dc.T @ sc) / N
    U, D, Vt = np.linalg.svd(cov)
    sgn  = np.linalg.det(U @ Vt)
    S_d  = np.diag([1.0, 1.0, sgn])
    R    = U @ S_d @ Vt
    scale = np.dot(D, [1.0, 1.0, sgn]) / (var_s + 1e-12)
    t     = mu_d - scale * (R @ mu_s)
    return float(scale), R, t


def compute_ate(pred_extri: np.ndarray, gt_c2w: np.ndarray) -> float:
    """绝对轨迹误差（Sim3 对齐后 RMSE）。"""
    pred_pos = _extri_to_cam_pos(pred_extri)
    gt_pos   = gt_c2w[:, :3, 3]
    s, R_a, t_a = umeyama_alignment(pred_pos, gt_pos)
    aligned = s * (R_a @ pred_pos.T).T + t_a
    return float(np.sqrt(np.mean(np.sum((aligned - gt_pos) ** 2, axis=1))))


def compute_rpe(pred_extri: np.ndarray,
                gt_c2w: np.ndarray) -> Tuple[float, float]:
    """相对位姿误差（平移 m，旋转 °）。"""
    S = pred_extri.shape[0]
    pred_w2c = np.zeros((S, 4, 4))
    pred_w2c[:, :3, :] = pred_extri
    pred_w2c[:, 3,  3] = 1.0
    gt_w2c = np.linalg.inv(gt_c2w)

    rpet, rper = [], []
    for i in range(S - 1):
        pred_rel = pred_w2c[i+1] @ np.linalg.inv(pred_w2c[i])
        gt_rel   = gt_w2c[i+1]  @ np.linalg.inv(gt_w2c[i])
        err      = gt_rel @ np.linalg.inv(pred_rel)
        rpet.append(np.linalg.norm(err[:3, 3]))
        cos = np.clip((np.trace(err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rper.append(np.degrees(np.arccos(cos)))
    return float(np.mean(rpet)), float(np.mean(rper))


def evaluate_camera_pose(model: VGGT, data: dict,
                         dtype: torch.dtype, device: str,
                         warmup: int) -> Tuple[dict, float]:
    """
    在给定数据集上评估相机位姿。

    data 需要包含：
        frames    : Tensor [S, 3, H, W]
        poses_c2w : ndarray [S, 4, 4]

    返回：metrics dict + wall-clock 时间（秒）
    """
    preds, elapsed = run_inference(model, data["frames"], dtype, device, warmup)

    extri, _ = pose_encoding_to_extri_intri(
        preds["pose_enc"],
        image_size_hw=(IMG_SIZE, IMG_SIZE),
        build_intrinsics=False,
    )
    extri_np = extri.squeeze(0).numpy()
    S = min(extri_np.shape[0], data["poses_c2w"].shape[0])
    ate       = compute_ate(extri_np[:S], data["poses_c2w"][:S])
    rpet, rper = compute_rpe(extri_np[:S], data["poses_c2w"][:S])

    return {"ate": ate, "rpet": rpet, "rper": rper}, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# 6. 主评估循环
# ══════════════════════════════════════════════════════════════════════════════

def run_all_evaluations(args) -> dict:
    device = args.device
    dtype  = DTYPE
    warmup = args.warmup

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  加载 VGGT 模型 ({args.model_name}) ...")
    print(f"{'='*60}")
    model: VGGT = VGGT.from_pretrained(args.model_name)
    model.eval().to(device)

    # ── 加载数据集 ────────────────────────────────────────────────────────────
    datasets: Dict[str, Optional[dict]] = {}
    for key, default_path in DATASET_PATHS.items():
        p = Path(args.data_root) / key if args.data_root else default_path
        try:
            if key == "tum":
                print(f"\n  加载 TUM walking_xyz ({p}) ...")
                datasets[key] = load_tum_walking_xyz(p, args.max_frames)
            elif key == "7scenes":
                print(f"  加载 7-Scenes chess  ({p}) ...")
                datasets[key] = load_7scenes_chess(p, args.max_frames)
            S = datasets[key]["frames"].shape[0]
            print(f"    → {S} 帧")
        except (FileNotFoundError, AssertionError) as e:
            print(f"    ⚠ 跳过 {key}：{e}")
            datasets[key] = None

    if all(v is None for v in datasets.values()):
        print("\n✗ 所有数据集均不可用，退出。")
        return {}

    # ── 构建配置 ──────────────────────────────────────────────────────────────
    configs = build_silence_configs(args.groups)
    print(f"\n  将测试 {len(configs)} 个配置：{list(configs.keys())}")

    results: Dict[str, dict] = {}

    for idx, (cfg_name, silence_cfg) in enumerate(configs.items()):
        print(f"\n{'─'*60}")
        print(f"  [{idx+1}/{len(configs)}] {cfg_name}")
        print(f"    {silence_cfg.label or repr(silence_cfg)}")
        print(f"    frame_layers  = {silence_cfg.frame_layers}")
        print(f"    global_layers = {silence_cfg.global_layers}")

        cfg_results: dict = {
            "config":        cfg_name,
            "label":         silence_cfg.label,
            "frame_layers":  silence_cfg.frame_layers,
            "global_layers": silence_cfg.global_layers,
        }

        with LayerSilencer(model, silence_cfg):
            # ── TUM（主指标：ATE）────────────────────────────────────────────
            if datasets.get("tum") is not None:
                print("  TUM walking_xyz …", end=" ", flush=True)
                try:
                    metrics, elapsed = evaluate_camera_pose(
                        model, datasets["tum"], dtype, device, warmup
                    )
                    cfg_results["tum"] = {**metrics, "time_s": elapsed}
                    print(f"ATE={metrics['ate']:.4f} m  "
                          f"RPEt={metrics['rpet']:.4f} m  "
                          f"RPEr={metrics['rper']:.2f}°  "
                          f"t={elapsed:.2f}s")
                except Exception as e:
                    print(f"✗ {e}")
                    cfg_results["tum"] = {"error": str(e)}

            # ── 7-Scenes（辅助：ATE only，快速）──────────────────────────────
            if datasets.get("7scenes") is not None:
                print("  7-Scenes chess  …", end=" ", flush=True)
                try:
                    metrics, elapsed = evaluate_camera_pose(
                        model, datasets["7scenes"], dtype, device, warmup
                    )
                    cfg_results["7scenes"] = {**metrics, "time_s": elapsed}
                    print(f"ATE={metrics['ate']:.4f} m  t={elapsed:.2f}s")
                except Exception as e:
                    print(f"✗ {e}")
                    cfg_results["7scenes"] = {"error": str(e)}

        results[cfg_name] = cfg_results

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7. 结果保存
# ══════════════════════════════════════════════════════════════════════════════

def _to_json_serializable(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    return obj


def save_results_json(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"\n  results.json → {path}")


def save_results_csv(results: dict, out_dir: Path) -> None:
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg_name, res in results.items():
        for task in ("tum", "7scenes"):
            if task not in res or "error" in res[task]:
                continue
            for metric, val in res[task].items():
                if isinstance(val, float):
                    rows.append({
                        "config": cfg_name,
                        "task": task,
                        "metric": metric,
                        "value": val,
                        "frame_layers": str(res.get("frame_layers", [])),
                        "global_layers": str(res.get("global_layers", [])),
                    })
    path = out_dir / "summary.csv"
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  summary.csv   → {path}")


def _safe(results: dict, cfg: str, task: str, metric: str) -> float:
    try:
        v = results[cfg][task][metric]
        return float(v) if (v is not None and np.isfinite(v)) else np.nan
    except (KeyError, TypeError):
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 8. 可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_layer_heatmap(results: dict, out_dir: Path) -> None:
    """
    绘制逐层 ATE 变化量热力图（相对 baseline）。

    布局：2 行（frame / global） × 24 列（层索引）
    颜色：绿色 = 与 baseline 相近（安全静默），红色 = ATE 大幅上升（关键层）
    """
    baseline_tum     = _safe(results, "baseline", "tum",     "ate")
    baseline_7scenes = _safe(results, "baseline", "7scenes", "ate")

    # 收集逐层数据
    n_layers = 24
    ate_delta_global = np.full(n_layers, np.nan)
    ate_delta_frame  = np.full(n_layers, np.nan)
    ate_global       = np.full(n_layers, np.nan)
    ate_frame        = np.full(n_layers, np.nan)

    for i in range(n_layers):
        g_key = f"G{i:02d}"
        f_key = f"F{i:02d}"
        if g_key in results:
            v = _safe(results, g_key, "tum", "ate")
            ate_global[i] = v
            if not np.isnan(baseline_tum) and not np.isnan(v):
                ate_delta_global[i] = v - baseline_tum
        if f_key in results:
            v = _safe(results, f_key, "tum", "ate")
            ate_frame[i] = v
            if not np.isnan(baseline_tum) and not np.isnan(v):
                ate_delta_frame[i] = v - baseline_tum

    # 检查是否有任何有效数据
    has_global = not np.all(np.isnan(ate_delta_global))
    has_frame  = not np.all(np.isnan(ate_delta_frame))

    if not has_global and not has_frame:
        print("  ⚠ 无逐层数据，跳过热力图")
        return

    fig, axes = plt.subplots(2, 1, figsize=(18, 5))
    layer_idx = np.arange(n_layers)

    # 颜色范围：以最大绝对值为准，diverging colormap
    all_deltas = np.concatenate([
        ate_delta_global[~np.isnan(ate_delta_global)],
        ate_delta_frame[~np.isnan(ate_delta_frame)],
    ])
    if len(all_deltas) == 0:
        plt.close(fig)
        return
    vmax = max(abs(all_deltas).max() * 1.1, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r   # 红=ATE上升（有害），绿=ATE下降（无害/有益）

    for ax, deltas, ate_vals, title, color in [
        (axes[0], ate_delta_global, ate_global, "Global Blocks 静默 ATE 变化量 (vs baseline)", "steelblue"),
        (axes[1], ate_delta_frame,  ate_frame,  "Frame Blocks 静默 ATE 变化量 (vs baseline)",  "darkorange"),
    ]:
        valid = ~np.isnan(deltas)
        if not valid.any():
            ax.text(0.5, 0.5, "无数据", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_title(title, fontsize=11)
            continue

        bars = ax.bar(layer_idx[valid], deltas[valid],
                      color=cmap(norm(deltas[valid])), edgecolor="white", linewidth=0.5)

        # 在柱上标注绝对 ATE 值
        for xi, vi, di in zip(layer_idx[valid], ate_vals[valid], deltas[valid]):
            if not np.isnan(vi):
                ax.text(xi, di + np.sign(di) * 0.0005,
                        f"{vi:.3f}", ha="center", va="bottom" if di >= 0 else "top",
                        fontsize=6.5, color="black")

        # 标注用户观测层
        for layer_range, label, linestyle in [
            ([3, 4],         "观测稀疏 3–4",   "--"),
            (list(range(10, 17)), "观测稀疏 10–16", ":"),
        ]:
            for li in layer_range:
                ax.axvline(li, color="purple", linestyle=linestyle,
                           alpha=0.5, linewidth=1.0)
        # 图例说明
        ax.axvline(3, color="purple", linestyle="--", alpha=0.5, linewidth=1.0,
                   label="观测稀疏层 3–4")
        ax.axvline(10, color="purple", linestyle=":", alpha=0.5, linewidth=1.0,
                   label="观测稀疏层 10–16")

        ax.axhline(0, color="black", linewidth=0.8)
        if not np.isnan(baseline_tum):
            ax.set_title(f"{title}  (baseline ATE={baseline_tum:.4f} m)", fontsize=11)
        else:
            ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer Index", fontsize=10)
        ax.set_ylabel("ΔATE (m)", fontsize=10)
        ax.set_xticks(layer_idx)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=8, loc="upper right")

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="ΔATE (m): 红=ATE升（有害），绿=ATE降/稳", shrink=0.6)

    fig.suptitle("VGGT 层静默实验：相机位姿 ATE 变化量（TUM walking_xyz）",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "heatmap_per_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  heatmap_per_layer.png → {path}")


def plot_range_ablation(results: dict, out_dir: Path) -> None:
    """
    范围静默和边界对照的柱状图。
    横轴：配置名；纵轴：ATE (m)；红虚线标注 baseline。
    """
    # 筛选范围配置（排除逐层）
    range_keys = [k for k in results
                  if k not in ("baseline",)
                  and not k.startswith(("G", "F"))
                  or k in ("baseline",)]
    # 更精确的过滤
    range_keys = ["baseline"] + [
        k for k in results
        if k.startswith("range_") or k in (
            "all_global", "all_frame", "first_half_global", "second_half_global"
        )
    ]
    range_keys = [k for k in range_keys if k in results]

    if not range_keys:
        print("  ⚠ 无范围配置数据，跳过 range_ablation 图")
        return

    ate_vals  = [_safe(results, k, "tum", "ate")  for k in range_keys]
    rpet_vals = [_safe(results, k, "tum", "rpet") for k in range_keys]

    baseline_ate = _safe(results, "baseline", "tum", "ate")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, vals, ylabel, title in [
        (axes[0], ate_vals,  "ATE (m)",   "TUM ATE — 范围静默对比"),
        (axes[1], rpet_vals, "RPEt (m)",  "TUM RPEt — 范围静默对比"),
    ]:
        x   = np.arange(len(range_keys))
        clr = []
        for k, v in zip(range_keys, vals):
            if k == "baseline":
                clr.append("steelblue")
            elif np.isnan(v):
                clr.append("lightgray")
            elif v <= baseline_ate * 1.05:
                clr.append("seagreen")   # 与 baseline 相差 < 5%：安全
            elif v <= baseline_ate * 1.20:
                clr.append("gold")       # 5–20%：中等影响
            else:
                clr.append("tomato")     # > 20%：严重影响

        ax.bar(x, [v if not np.isnan(v) else 0 for v in vals],
               color=clr, edgecolor="white", linewidth=0.6)

        if not np.isnan(baseline_ate):
            ax.axhline(baseline_ate, color="red", linestyle="--",
                       linewidth=1.0, label=f"baseline={baseline_ate:.4f} m")
            ax.axhline(baseline_ate * 1.05, color="gold", linestyle=":",
                       linewidth=0.8, label="+5% 容忍阈值")

        ax.set_xticks(x)
        ax.set_xticklabels(range_keys, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("VGGT 范围静默实验 — 相机位姿影响（TUM walking_xyz）",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "range_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  range_ablation.png → {path}")


def plot_sensitivity_curve(results: dict, out_dir: Path) -> None:
    """
    逐层敏感度排序曲线（全局块）。

    将 24 层按 ΔATE 从小到大排序，帮助识别：
      - ΔATE ≈ 0 的层 → 可安全静默（"Low sensitivity"）
      - ΔATE >> 0 的层 → 关键层，不可跳过（"High sensitivity"）

    同时绘制层索引颜色编码，标出用户观测的稀疏层。
    """
    baseline_ate = _safe(results, "baseline", "tum", "ate")
    if np.isnan(baseline_ate):
        print("  ⚠ 无 baseline ATE，跳过敏感度曲线")
        return

    layers, deltas = [], []
    for i in range(24):
        key = f"G{i:02d}"
        if key in results:
            ate = _safe(results, key, "tum", "ate")
            if not np.isnan(ate):
                layers.append(i)
                deltas.append(ate - baseline_ate)

    if not layers:
        print("  ⚠ 无逐层全局块数据，跳过敏感度曲线")
        return

    layers  = np.array(layers)
    deltas  = np.array(deltas)
    order   = np.argsort(deltas)
    s_layers = layers[order]
    s_deltas = deltas[order]

    # 标注用户观测层
    observed_set = set([3, 4] + list(range(10, 17)))

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = []
    for li in s_layers:
        if li in observed_set:
            colors.append("purple")
        elif s_deltas[list(s_layers).index(li)] < 0.001:
            colors.append("seagreen")
        else:
            colors.append("tomato")

    bars = ax.bar(range(len(s_layers)), s_deltas, color=colors,
                  edgecolor="white", linewidth=0.5)

    for xi, (li, di) in enumerate(zip(s_layers, s_deltas)):
        ax.text(xi, di + np.sign(di) * 0.0003,
                str(li), ha="center",
                va="bottom" if di >= 0 else "top",
                fontsize=8, fontweight="bold" if li in observed_set else "normal",
                color="purple" if li in observed_set else "black")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.001, color="gold", linestyle=":", linewidth=0.8,
               label="+1mm 容忍线")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="purple",   label="用户观测稀疏层（3–4, 10–16）"),
        Patch(facecolor="seagreen", label="安全静默层（ΔATE < 1mm）"),
        Patch(facecolor="tomato",   label="敏感层（ΔATE ≥ 1mm）"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("层排序（按 ΔATE 升序）", fontsize=10)
    ax.set_ylabel("ΔATE (m)", fontsize=10)
    ax.set_title(
        f"Global Blocks 逐层敏感度曲线（TUM，baseline={baseline_ate:.4f} m）\n"
        "数字 = 层索引，紫色 = 用户观测稀疏层",
        fontsize=11,
    )
    plt.tight_layout()
    path = out_dir / "sensitivity_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  sensitivity_curve.png → {path}")


def print_summary_table(results: dict) -> None:
    """打印主要结论：哪些配置与 baseline 相差 < 5%（可能可以静默）。"""
    baseline_ate = _safe(results, "baseline", "tum", "ate")
    if np.isnan(baseline_ate):
        return

    print(f"\n{'='*70}")
    print(f"  结论汇总  (baseline TUM ATE = {baseline_ate:.4f} m)")
    print(f"{'='*70}")
    print(f"  {'配置':<35} {'TUM ATE':>9} {'ΔATE':>9} {'影响程度':>10}")
    print(f"  {'─'*35} {'─'*9} {'─'*9} {'─'*10}")

    for cfg_name, res in results.items():
        ate = _safe(results, cfg_name, "tum", "ate")
        if np.isnan(ate):
            continue
        delta = ate - baseline_ate
        if abs(delta) < 0.001:
            impact = "✓ 可静默"
        elif abs(delta) < baseline_ate * 0.05:
            impact = "~ 轻微"
        elif abs(delta) < baseline_ate * 0.20:
            impact = "⚠ 中等"
        else:
            impact = "✗ 显著"
        label = res.get("label", cfg_name)[:34]
        print(f"  {cfg_name:<20} {label:<14} {ate:>9.4f} {delta:>+9.4f} {impact:>10}")

    print()

    # 特别汇报用户关注的范围
    print(f"  {'─'*70}")
    print("  用户观测稀疏层（3–4, 10–16）静默结果：")
    for key in ["range_global_3_4", "range_frame_3_4", "range_both_3_4",
                "range_global_10_16", "range_frame_10_16", "range_both_10_16",
                "range_both_3_4_10_16"]:
        ate = _safe(results, key, "tum", "ate")
        if np.isnan(ate):
            continue
        delta = ate - baseline_ate
        pct   = delta / baseline_ate * 100
        print(f"    {key:<32} ATE={ate:.4f} m  Δ={delta:+.4f} m ({pct:+.1f}%)")


def save_all_plots(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_per_layer_heatmap(results, out_dir)
    plot_range_ablation(results, out_dir)
    plot_sensitivity_curve(results, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="VGGT 层静默实验 — 相机位姿灵敏度分析",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device",      default="cuda",
                   help="推理设备（cuda / cpu）")
    p.add_argument("--model_name",  default="facebook/VGGT-1B",
                   help="HuggingFace 模型名称")
    p.add_argument("--max_frames",  type=int, default=16,
                   help="每个数据集最多使用的帧数")
    p.add_argument("--warmup",      type=int, default=2,
                   help="推理 warmup 次数")
    p.add_argument("--data_root",   type=str, default=None,
                   help="数据集根目录（覆盖默认路径）")
    p.add_argument(
        "--groups",
        nargs="+",
        default=None,
        choices=["baseline", "per_layer_global", "per_layer_frame",
                 "range", "boundary"],
        help=(
            "指定要运行的实验分组（不指定则全跑）。\n"
            "  baseline         — 参照组（无静默）\n"
            "  per_layer_global — 逐层全局块静默（24 配置）\n"
            "  per_layer_frame  — 逐层帧块静默（24 配置）\n"
            "  range            — 范围静默（7 配置，观测驱动）\n"
            "  boundary         — 边界对照（all/first_half/second_half，4 配置）"
        ),
    )
    p.add_argument("--out_dir", type=str, default=None,
                   help="结果输出目录（默认：tests/results/silence_test/）")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_evaluations(args)

    if not results:
        print("无结果，退出。")
        return

    save_results_json(results, out_dir)
    save_results_csv(results, out_dir)
    save_all_plots(results, out_dir)
    print_summary_table(results)

    print(f"\n{'='*60}")
    print(f"  全部完成  →  {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
