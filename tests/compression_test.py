#!/usr/bin/env python3
"""
tests/compression_test.py
==========================
VGGT 压缩机制综合评测脚本

测试所有压缩机制（baseline、A、A-kv-only、B、C、E、A+C、E+C）在三项任务上的
推理速度与预测精度：

  任务 1：点图预测      (Point Map)      — 7-Scenes chess
  任务 2：相机位姿预测  (Camera Pose)    — TUM-dynamics walking_xyz
  任务 3：视频深度预测  (Video Depth)    — Sintel training

指标：
  点图：   Acc_mean/median（m）、Comp_mean/median（m）、NC（余弦相似度）
  相机：   ATE（m）、RPEt（m）、RPEr（°）  [经 Sim3 对齐]
  深度：   AbsRel、δ<1.25

结果保存：tests/results/compression/
  results.json       — 完整数值结果
  summary.csv        — 每行 (mechanism, task, metric, value)
  speed_comparison.png
  metric_comparison_7scenes.png
  metric_comparison_tum.png
  metric_comparison_sintel.png

用法（从项目根目录）：
    python tests/compression_test.py [--max_frames N] [--warmup N] [--device cuda]
"""

# ── 标准库 ────────────────────────────────────────────────────────────────────
import argparse
import contextlib
import csv
import json
import os
import struct
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 第三方库 ──────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 项目路径 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent        # tests/
_PROJECT_ROOT = _SCRIPT_DIR.parent                    # VGGT/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── VGGT 核心 ─────────────────────────────────────────────────────────────────
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# ── 压缩模块（可选，未实现时仅测试 baseline）────────────────────────────────────
try:
    from vggt.compression import (
        CompressionConfig,
        apply_compression_hooks,
        remove_compression_hooks,
    )
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    warnings.warn(
        "[compression_test] vggt.compression 未找到，仅测试 baseline（无压缩）。\n"
        "  请先按 docs/novel/plan_kv_reduction.md 实现 vggt/compression/ 模块。",
        stacklevel=2,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 0. 全局常量 / 默认路径
# ══════════════════════════════════════════════════════════════════════════════

IMG_SIZE    = 518          # VGGT 固定输入分辨率
DTYPE       = torch.bfloat16
RESULTS_DIR = _SCRIPT_DIR / "results" / "compression"

# 数据集路径（相对于项目根目录的上一级）
DATASET_ROOT = _PROJECT_ROOT.parent / "datasets"
DATASET_PATHS = {
    "7scenes": DATASET_ROOT / "7-scenes"     / "chess",
    "tum":     DATASET_ROOT / "tum-dynamics" / "walking_xyz",
    "sintel":  DATASET_ROOT / "sintel"       / "training",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. 压缩配置定义
# ══════════════════════════════════════════════════════════════════════════════

def build_compression_configs() -> Dict[str, Optional[object]]:
    """
    返回 {名称 → CompressionConfig | None} 字典。
    若 vggt.compression 不可用，只返回 baseline。
    """
    configs: Dict[str, Optional[object]] = {"baseline": None}

    if not COMPRESSION_AVAILABLE:
        return configs

    configs.update({
        # 机制 A：KV 时序步长剪枝 + Q 组内合并（完整 Spark3R 对应）
        "A_full": CompressionConfig(
            mechanism="A",
            q_group_size=20,
            kv_insensitive_multiplier=3.0,
            enable_q_compression=True,
        ),
        # 机制 A（仅 KV，用于与 B/C 公平对比）
        "A_kv_only": CompressionConfig(
            mechanism="A",
            q_group_size=20,
            kv_insensitive_multiplier=3.0,
            enable_q_compression=False,
        ),
        # 机制 B：时序 DCT KV 软压缩
        "B": CompressionConfig(
            mechanism="B",
            temporal_keep_ratio_by_zone={
                "shallow":   0.25,
                "sensitive": 0.70,
                "deep":      0.45,
            },
        ),
        # 机制 C：空间 2D-DCT + 异常值保留
        "C": CompressionConfig(
            mechanism="C",
            spatial_low_freq_ratio=0.30,
            spatial_outlier_ratio=0.10,
        ),
        # 机制 E：DCT 代表元 Q merging
        "E": CompressionConfig(
            mechanism="E",
            q_group_size=20,
            kv_insensitive_multiplier=3.0,
            enable_q_compression=True,
        ),
        # 联合：机制 A + C（时序 + 空间双级压缩）
        "A+C": CompressionConfig(
            mechanism="A+C",
            q_group_size=20,
            kv_insensitive_multiplier=3.0,
            spatial_low_freq_ratio=0.30,
            spatial_outlier_ratio=0.10,
        ),
        # 联合：机制 E + C
        "E+C": CompressionConfig(
            mechanism="E+C",
            q_group_size=20,
            kv_insensitive_multiplier=3.0,
            spatial_low_freq_ratio=0.30,
            spatial_outlier_ratio=0.10,
        ),
    })
    return configs


# ══════════════════════════════════════════════════════════════════════════════
# 2. 数据集加载
# ══════════════════════════════════════════════════════════════════════════════

def _make_image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),   # → [0, 1]
    ])


# ── 2.1  7-Scenes chess ───────────────────────────────────────────────────────

def load_7scenes_chess(data_dir: Path, max_frames: int) -> dict:
    """
    加载 7-Scenes chess 序列。

    目录结构：
        chess/
          seq-01/
            frame-000000.color.png   RGB 640×480
            frame-000000.depth.png   uint16 mm
            frame-000000.pose.txt    4×4 camera-to-world

    Returns:
        frames    : Tensor [S, 3, 518, 518]  float32 in [0,1]
        depths    : ndarray [S, 480, 640]    float32 in metres
        poses_c2w : ndarray [S, 4, 4]        camera-to-world
        gt_points : ndarray [N, 3]           sub-sampled GT 点云（世界坐标）
        K         : ndarray [3, 3]           7-Scenes 固定内参
    """
    # 7-Scenes 固定内参（全部场景 640×480）
    K = np.array([[525.0,   0.0, 320.0],
                  [  0.0, 525.0, 240.0],
                  [  0.0,   0.0,   1.0]], dtype=np.float64)

    # 仅使用 seq-01，避免单场景帧数超千帧
    seq_dir = data_dir / "seq-01"
    if not seq_dir.exists():
        raise FileNotFoundError(f"未找到 {seq_dir}（7-Scenes chess seq-01）")

    tf = _make_image_transform()
    frames, depths, poses = [], [], []

    color_files = sorted(seq_dir.glob("frame-*.color.png"))
    for cf in color_files:
        stem = cf.stem.replace(".color", "")
        depth_f = seq_dir / f"{stem}.depth.png"
        pose_f  = seq_dir / f"{stem}.pose.txt"
        if not depth_f.exists() or not pose_f.exists():
            continue

        frames.append(tf(Image.open(cf).convert("RGB")))

        d = np.array(Image.open(depth_f), dtype=np.float32) / 1000.0  # mm → m
        depths.append(d)

        poses.append(np.loadtxt(str(pose_f)))

        if len(frames) >= max_frames:
            break

    if not frames:
        raise FileNotFoundError(f"7-Scenes chess：{data_dir} 中未找到任何帧")

    # 建立 GT 点云
    gt_pts_list = []
    H_orig, W_orig = depths[0].shape
    u, v = np.meshgrid(np.arange(W_orig), np.arange(H_orig))
    for depth, pose in zip(depths, poses):
        valid = (depth > 0.1) & (depth < 10.0)
        X = (u[valid] - K[0, 2]) * depth[valid] / K[0, 0]
        Y = (v[valid] - K[1, 2]) * depth[valid] / K[1, 1]
        Z = depth[valid]
        pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # [N, 4]
        pts_world = (pose @ pts_cam.T).T[:, :3]                  # [N, 3]
        gt_pts_list.append(pts_world)

    all_gt = np.concatenate(gt_pts_list, axis=0)
    # 随机下采样（k-NN 效率）
    if len(all_gt) > 50_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_gt), 50_000, replace=False)
        all_gt = all_gt[idx]

    return {
        "frames":    torch.stack(frames),          # [S, 3, 518, 518]
        "depths":    np.array(depths),             # [S, H, W]
        "poses_c2w": np.array(poses),              # [S, 4, 4]
        "gt_points": all_gt.astype(np.float32),    # [N, 3]
        "K":         K,
        "orig_hw":   (H_orig, W_orig),
    }


# ── 2.2  TUM-dynamics walking_xyz ────────────────────────────────────────────

def load_tum_walking_xyz(data_dir: Path, max_frames: int) -> dict:
    """
    加载 TUM RGB-D walking_xyz 序列。

    目录结构：
        walking_xyz/
          rgb/               timestamped PNG
          depth/             timestamped PNG  (uint16 ÷5000 → m)
          rgb.txt            timestamp path
          groundtruth.txt    timestamp tx ty tz qx qy qz qw  (camera-to-world)

    Returns:
        frames    : Tensor [S, 3, 518, 518]
        poses_c2w : ndarray [S, 4, 4]  camera-to-world
    """
    gt_file  = data_dir / "groundtruth.txt"
    rgb_file = data_dir / "rgb.txt"

    if not gt_file.exists():
        raise FileNotFoundError(f"未找到 {gt_file}")
    if not rgb_file.exists():
        raise FileNotFoundError(f"未找到 {rgb_file}")

    # 解析 GT 位姿
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

    # 解析 RGB 时间戳列表
    rgb_entries: List[Tuple[float, Path]] = []
    for line in open(rgb_file):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        ts   = float(parts[0])
        path = data_dir / parts[1]
        rgb_entries.append((ts, path))
    rgb_entries.sort(key=lambda x: x[0])

    tf = _make_image_transform()
    frames, poses = [], []

    for rgb_ts, img_path in rgb_entries:
        if not img_path.exists():
            continue
        # 最近邻时间戳匹配
        nearest = min(gt_ts, key=lambda t: abs(t - rgb_ts))
        if abs(nearest - rgb_ts) > 0.05:   # >50ms 不采用
            continue

        frames.append(tf(Image.open(img_path).convert("RGB")))
        poses.append(gt_map[nearest])

        if len(frames) >= max_frames:
            break

    if not frames:
        raise FileNotFoundError(
            f"TUM walking_xyz：{data_dir} 中未匹配到任何 RGB-GT 对，"
            "请确认 rgb.txt 与 groundtruth.txt 存在且时间戳对齐。"
        )

    return {
        "frames":    torch.stack(frames),      # [S, 3, 518, 518]
        "poses_c2w": np.array(poses),          # [S, 4, 4]
    }


# ── 2.3  Sintel training ──────────────────────────────────────────────────────

def _read_dpt(path: Path) -> np.ndarray:
    """读取 Sintel .dpt 深度文件 → float32 ndarray [H, W]（单位：m）"""
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        assert abs(magic - 202021.25) < 1.0, f"无效 .dpt magic: {magic}"
        w, h = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(h * w * 4), dtype=np.float32).copy()
    return data.reshape(h, w)


def load_sintel(data_dir: Path, max_frames: int,
                max_scenes: int = 3) -> List[dict]:
    """
    加载 Sintel training 多个场景。

    目录结构：
        training/
          clean/<scene>/frame_0001.png
          depth/<scene>/frame_0001.dpt

    Returns:
        List of {
            scene : str,
            frames : Tensor [S, 3, 518, 518],
            depths : ndarray [S, H_orig, W_orig]   (H=436, W=1024)
        }
    """
    clean_dir = data_dir / "clean"
    depth_dir = data_dir / "depth"

    if not clean_dir.exists():
        raise FileNotFoundError(f"未找到 Sintel clean 目录：{clean_dir}")

    scenes = sorted(d.name for d in clean_dir.iterdir() if d.is_dir())
    scenes = scenes[:max_scenes]

    tf = _make_image_transform()
    result = []

    for scene in scenes:
        rgb_files = sorted((clean_dir / scene).glob("frame_*.png"))[:max_frames]
        frames, depths = [], []

        for rf in rgb_files:
            dpt_f = depth_dir / scene / rf.with_suffix(".dpt").name
            if not dpt_f.exists():
                continue
            frames.append(tf(Image.open(rf).convert("RGB")))
            depths.append(_read_dpt(dpt_f))

        if len(frames) < 2:
            continue

        result.append({
            "scene":  scene,
            "frames": torch.stack(frames),      # [S, 3, 518, 518]
            "depths": np.array(depths),         # [S, 436, 1024]
        })

    if not result:
        raise FileNotFoundError(f"Sintel：{data_dir} 中未找到可用场景")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. 推理 & 计时
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model: VGGT, frames: torch.Tensor,
                  dtype: torch.dtype, device: str,
                  warmup: int = 3) -> Tuple[dict, float]:
    """
    对一组帧运行 VGGT 推理，返回预测字典和 wall-clock 时间（秒）。

    Args:
        model  : 已初始化的 VGGT 模型（已 .eval().to(device)）
        frames : [S, 3, H, W]  float32，值域 [0, 1]
        dtype  : 推理精度（bfloat16 / float16 / float32）
        device : "cuda" | "cpu"
        warmup : 正式计时前的热身轮数

    Returns:
        preds    : VGGT 输出字典
        elapsed  : 推理耗时（秒，不含 warmup）
    """
    # 准备输入：添加 batch 维，resize 到 518×518
    # 模型保持 float32；VGGT 设计：aggregator 可在 autocast(bf16) 下运行，
    # heads 通过内部 autocast(enabled=False) 保持 float32 精度。
    # 不直接转换模型或输入到 bf16，避免 `_apply_pos_embed` 的 pos_embed.float()
    # 与 bf16 特征张量产生 dtype 不一致。
    imgs = frames.to(device).float()      # 保持 float32
    imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE),
                         mode="bilinear", align_corners=False)
    imgs = imgs.unsqueeze(0)              # [1, S, 3, 518, 518], float32

    # CUDA 上使用 autocast 加速 aggregator；CPU 上不需要
    _autocast = (
        torch.cuda.amp.autocast(dtype=dtype)
        if device == "cuda" else contextlib.nullcontext()
    )

    with torch.no_grad():
        # 热身
        for _ in range(warmup):
            with _autocast:
                _ = model(imgs)
            if device == "cuda":
                torch.cuda.synchronize()

        # 正式计时（CUDA event，精度更高）
        if device == "cuda":
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()

            with _autocast:
                preds = model(imgs)

            end_evt.record()
            torch.cuda.synchronize()
            elapsed = start_evt.elapsed_time(end_evt) / 1000.0  # ms → s
        else:
            import time as _time
            t0    = _time.perf_counter()
            with _autocast:
                preds = model(imgs)
            elapsed = _time.perf_counter() - t0

    # 将所有 tensor 移回 CPU，释放 GPU 显存
    preds_cpu = {
        k: (v.float().cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in preds.items()
        if isinstance(v, (torch.Tensor, np.ndarray))
    }
    if device == "cuda":
        torch.cuda.empty_cache()

    return preds_cpu, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# 4. Sim3 对齐工具
# ══════════════════════════════════════════════════════════════════════════════

def umeyama_alignment(src: np.ndarray,
                      dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama (1991) Sim3 对齐：dst ≈ s * R @ src + t。
    src, dst : [N, 3]
    Returns  : scale (float), R (3×3), t (3,)
    """
    N  = src.shape[0]
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    cov_mat = (dst_c.T @ src_c) / N

    U, D, Vt = np.linalg.svd(cov_mat)
    det_sign = np.linalg.det(U @ Vt)
    S_diag   = np.diag([1.0, 1.0, det_sign])

    R     = U @ S_diag @ Vt
    scale = np.dot(D, [1.0, 1.0, det_sign]) / (var_src + 1e-12)
    t     = mu_d - scale * (R @ mu_s)
    return float(scale), R, t


# ══════════════════════════════════════════════════════════════════════════════
# 5. 点图评估（7-Scenes）
# ══════════════════════════════════════════════════════════════════════════════

def _compute_grid_normals(pts_grid: np.ndarray) -> np.ndarray:
    """
    从网格结构点图计算单位法向量。
    pts_grid : [H, W, 3]
    Returns  : [H, W, 3]  单位法向量
    """
    # 右向 / 下向位移
    h_diff = np.concatenate(
        [pts_grid[:, 1:] - pts_grid[:, :-1],
         pts_grid[:, -1:] - pts_grid[:, -2:-1]], axis=1)   # [H, W, 3]
    v_diff = np.concatenate(
        [pts_grid[1:] - pts_grid[:-1],
         pts_grid[-1:] - pts_grid[-2:-1]], axis=0)         # [H, W, 3]

    normals = np.cross(h_diff, v_diff)                      # [H, W, 3]
    norms   = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid   = (norms[..., 0] > 1e-8)
    normals = np.where(valid[..., None],
                       normals / np.where(valid[..., None], norms, 1.0),
                       0.0)
    return normals


def eval_point_map(pred_world_pts: np.ndarray,
                   pred_normals: Optional[np.ndarray],
                   gt_pts: np.ndarray,
                   max_dist_m: float = 0.1) -> dict:
    """
    评估点图预测质量。

    Args:
        pred_world_pts : [N_pred, 3]  预测点云（已经 Sim3 对齐至 GT 坐标系）
        pred_normals   : [N_pred, 3] | None  预测法向量（也经 R 对齐，scale 不影响法向量）
        gt_pts         : [N_gt, 3]   GT 点云
        max_dist_m     : 截断距离（避免离群点主导均值）

    Returns:
        acc_mean, acc_med, comp_mean, comp_med, nc_mean
    """
    # sub-sample 防止 cKDTree OOM
    MAX_PTS = 30_000
    rng = np.random.default_rng(42)

    if len(pred_world_pts) > MAX_PTS:
        idx = rng.choice(len(pred_world_pts), MAX_PTS, replace=False)
        pred_s  = pred_world_pts[idx]
        norm_s  = pred_normals[idx] if pred_normals is not None else None
    else:
        pred_s, norm_s = pred_world_pts, pred_normals

    gt_s = gt_pts
    if len(gt_s) > MAX_PTS:
        idx = rng.choice(len(gt_s), MAX_PTS, replace=False)
        gt_s = gt_s[idx]

    gt_tree   = cKDTree(gt_s)
    pred_tree = cKDTree(pred_s)

    # Accuracy：pred → GT
    acc_d, acc_idx = gt_tree.query(pred_s, k=1)
    acc_d = np.clip(acc_d, 0.0, max_dist_m)

    # Completeness：GT → pred
    comp_d, comp_idx = pred_tree.query(gt_s, k=1)
    comp_d = np.clip(comp_d, 0.0, max_dist_m)

    metrics = {
        "acc_mean":   float(acc_d.mean()),
        "acc_med":    float(np.median(acc_d)),
        "comp_mean":  float(comp_d.mean()),
        "comp_med":   float(np.median(comp_d)),
        "nc_mean":    float("nan"),
    }

    # Normal Consistency（仅在预测法向量可用时计算）
    if norm_s is not None:
        # 找到对应 GT 点位置的 GT 法向量（暂无 GT 法向量，用匹配对的 pred 与 pred_at_gt 对比）
        # 实际实现：GT 侧法向量来自深度图，此处利用 pred 法向量的自一致性近似
        # 即：对每个 GT 点，找最近 pred 点，再找该 pred 点最近的另一个 pred 点，比较法向量
        # 这只是一个近似。正式评估应传入 gt_normals。
        pred_nn_for_gt = pred_s[comp_idx]   # [N_gt, 3]  GT 对应的最近预测点
        _, nn_idx2 = pred_tree.query(pred_nn_for_gt, k=2)  # k=2: [0]自身,[1]次近
        if norm_s.ndim == 2:
            n1 = norm_s[comp_idx]             # [N_gt, 3]
            n2 = norm_s[nn_idx2[:, 1]]        # [N_gt, 3]
            cos = np.clip((n1 * n2).sum(-1), -1.0, 1.0)
            metrics["nc_mean"] = float(cos.mean())

    return metrics


def evaluate_7scenes(model: VGGT, data: dict, dtype: torch.dtype,
                     device: str, warmup: int) -> Tuple[dict, float]:
    """
    对 7-Scenes chess 数据运行推理并返回 (metrics, elapsed)。
    """
    frames    = data["frames"]                # [S, 3, 518, 518]
    gt_pts    = data["gt_points"]             # [N, 3] 世界坐标

    preds, elapsed = run_inference(model, frames, dtype, device, warmup)

    # world_points: [1, S, H, W, 3] → [S, H, W, 3]
    wp = preds["world_points"].squeeze(0).numpy()   # [S, H, W, 3]
    S, H, W, _ = wp.shape

    # 展平为点云
    pred_pts = wp.reshape(-1, 3)

    # 过滤 NaN/Inf
    valid = np.isfinite(pred_pts).all(axis=1)
    pred_pts = pred_pts[valid]

    if len(pred_pts) < 100:
        return {"acc_mean": np.nan, "acc_med": np.nan,
                "comp_mean": np.nan, "comp_med": np.nan,
                "nc_mean": np.nan}, elapsed

    # Sim3 对齐（pred → GT）
    rng = np.random.default_rng(42)
    n_align = min(5000, len(pred_pts), len(gt_pts))
    src_idx  = rng.choice(len(pred_pts), n_align, replace=False)
    dst_idx  = rng.choice(len(gt_pts),  n_align, replace=False)
    scale, R_align, t_align = umeyama_alignment(
        pred_pts[src_idx], gt_pts[dst_idx]
    )
    pred_pts_aligned = scale * (R_align @ pred_pts.T).T + t_align

    # 计算法向量（用旋转对齐，不需要 scale/t）
    normals_list = []
    for s in range(S):
        n = _compute_grid_normals(wp[s])          # [H, W, 3]
        n_flat = n.reshape(-1, 3)
        n_aligned = (R_align @ n_flat.T).T         # 法向量只用旋转
        normals_list.append(n_aligned)
    pred_normals = np.concatenate(normals_list, axis=0)[valid]

    metrics = eval_point_map(pred_pts_aligned, pred_normals, gt_pts)
    return metrics, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# 6. 相机位姿评估（TUM）
# ══════════════════════════════════════════════════════════════════════════════

def _extri_to_cam_position(extri: np.ndarray) -> np.ndarray:
    """
    从 world-to-camera extrinsic [S, 3, 4] 提取相机中心（world 坐标）。
    camera_pos = -R.T @ t = R^{-1} @ (-t)
    """
    R = extri[:, :3, :3]   # [S, 3, 3]
    t = extri[:, :3,  3]   # [S, 3]
    return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)


def compute_ate(pred_extri: np.ndarray, gt_poses_c2w: np.ndarray) -> float:
    """
    ATE（Absolute Trajectory Error，经 Sim3 对齐后）。

    pred_extri   : [S, 3, 4]  VGGT world-to-camera
    gt_poses_c2w : [S, 4, 4]  camera-to-world

    Returns: ATE in metres
    """
    pred_pos = _extri_to_cam_position(pred_extri)   # [S, 3]
    gt_pos   = gt_poses_c2w[:, :3, 3]              # [S, 3]  camera-to-world 平移

    # Sim3 对齐
    _, R_a, t_a = umeyama_alignment(pred_pos, gt_pos)
    # 注意：umeyama_alignment 中 scale 已内嵌；不独立乘以 scale 则只对齐旋转+平移
    # 但标准 ATE 使用完整 Sim3（含 scale），这里用 umeyama 的完整输出
    scale, R_a, t_a = umeyama_alignment(pred_pos, gt_pos)
    pred_aligned = scale * (R_a @ pred_pos.T).T + t_a

    ate = float(np.sqrt(np.mean(np.sum((pred_aligned - gt_pos) ** 2, axis=1))))
    return ate


def compute_rpe(pred_extri: np.ndarray,
                gt_poses_c2w: np.ndarray) -> Tuple[float, float]:
    """
    RPEt（相对平移误差，m）与 RPEr（相对旋转误差，°）。

    pred_extri   : [S, 3, 4]  VGGT world-to-camera
    gt_poses_c2w : [S, 4, 4]  camera-to-world
    """
    S = pred_extri.shape[0]

    # 构建 4×4 world-to-camera
    pred_w2c = np.zeros((S, 4, 4))
    pred_w2c[:, :3, :] = pred_extri
    pred_w2c[:, 3,  3] = 1.0

    gt_w2c = np.linalg.inv(gt_poses_c2w)   # camera-to-world → world-to-camera

    rpet_list, rper_list = [], []

    for i in range(S - 1):
        # 相对位姿：帧 i → 帧 i+1
        pred_rel = pred_w2c[i + 1] @ np.linalg.inv(pred_w2c[i])
        gt_rel   = gt_w2c[i + 1]   @ np.linalg.inv(gt_w2c[i])

        # 误差：err = gt_rel @ inv(pred_rel)
        err = gt_rel @ np.linalg.inv(pred_rel)

        rpet_list.append(np.linalg.norm(err[:3, 3]))

        R_err    = err[:3, :3]
        cos_ang  = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rper_list.append(np.degrees(np.arccos(cos_ang)))

    return float(np.mean(rpet_list)), float(np.mean(rper_list))


def evaluate_tum(model: VGGT, data: dict, dtype: torch.dtype,
                 device: str, warmup: int) -> Tuple[dict, float]:
    """
    对 TUM walking_xyz 数据运行推理并返回 (metrics, elapsed)。
    """
    frames     = data["frames"]       # [S, 3, 518, 518]
    poses_c2w  = data["poses_c2w"]   # [S, 4, 4]

    preds, elapsed = run_inference(model, frames, dtype, device, warmup)

    pose_enc = preds["pose_enc"]                          # [1, S, 9]
    # images shape 仅用于 intrinsics 计算，此处不需要 intrinsics
    extri, _ = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(IMG_SIZE, IMG_SIZE), build_intrinsics=False
    )
    extri_np = extri.squeeze(0).numpy()                   # [S, 3, 4]

    ate  = compute_ate(extri_np, poses_c2w)
    rpet, rper = compute_rpe(extri_np, poses_c2w)

    return {"ate": ate, "rpet": rpet, "rper": rper}, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# 7. 视频深度评估（Sintel）
# ══════════════════════════════════════════════════════════════════════════════

def _scale_shift_align(pred: np.ndarray, gt: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    """
    Per-sequence 最小二乘 scale+shift 对齐：
        aligned = pred * s + b  ≈  gt   (在 mask 区域)
    """
    A = np.stack([pred[mask], np.ones_like(pred[mask])], axis=1)
    b = gt[mask]
    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        s, shift = float(x[0]), float(x[1])
    except np.linalg.LinAlgError:
        return pred
    return pred * s + shift


def eval_depth_sequence(pred_depth: np.ndarray,
                        gt_depth: np.ndarray) -> dict:
    """
    对一个场景（S 帧）计算深度指标。

    pred_depth : [S, 518, 518]  VGGT 输出（已 squeeze 掉最后 1 维）
    gt_depth   : [S, H, W]     GT（Sintel: 436×1024）

    指标（论文约定）：
        AbsRel = mean(|pred - gt| / gt)
        δ<1.25 = mean(max(pred/gt, gt/pred) < 1.25)
    """
    S = min(len(pred_depth), len(gt_depth))
    abs_rels, deltas = [], []

    for s in range(S):
        gt = gt_depth[s].astype(np.float32)   # [H, W]
        # resize pred to GT spatial size
        pred_t = torch.tensor(pred_depth[s]).unsqueeze(0).unsqueeze(0)   # [1,1,518,518]
        pred_r = F.interpolate(pred_t, size=gt.shape, mode="bilinear",
                               align_corners=False).squeeze().numpy()    # [H, W]

        mask = (gt > 0.1) & (gt < 1000.0) & np.isfinite(gt)
        if mask.sum() < 100:
            continue

        pred_aligned = _scale_shift_align(pred_r, gt, mask)
        valid = mask & (pred_aligned > 0)
        if valid.sum() < 100:
            continue

        p, g = pred_aligned[valid], gt[valid]
        abs_rels.append(float(np.mean(np.abs(p - g) / g)))
        deltas.append(float(np.mean(np.maximum(p / g, g / p) < 1.25)))

    if not abs_rels:
        return {"abs_rel": np.nan, "delta_125": np.nan}

    return {
        "abs_rel":   float(np.mean(abs_rels)),
        "delta_125": float(np.mean(deltas)),
    }


def evaluate_sintel(model: VGGT, scenes: List[dict], dtype: torch.dtype,
                    device: str, warmup: int) -> Tuple[dict, float]:
    """
    在 Sintel 多场景上汇总评估结果。
    """
    all_abs_rel, all_delta, all_elapsed = [], [], []

    for scene_data in scenes:
        frames = scene_data["frames"]  # [S, 3, 518, 518]
        depths = scene_data["depths"]  # [S, 436, 1024]

        preds, elapsed = run_inference(model, frames, dtype, device, warmup)
        all_elapsed.append(elapsed)

        # depth: [1, S, 518, 518, 1] → [S, 518, 518]
        pred_d = preds["depth"].squeeze(0).squeeze(-1).numpy()

        m = eval_depth_sequence(pred_d, depths)
        if not np.isnan(m["abs_rel"]):
            all_abs_rel.append(m["abs_rel"])
            all_delta.append(m["delta_125"])

    metrics = {
        "abs_rel":   float(np.nanmean(all_abs_rel)) if all_abs_rel else np.nan,
        "delta_125": float(np.nanmean(all_delta))   if all_delta   else np.nan,
    }
    total_elapsed = float(sum(all_elapsed))
    return metrics, total_elapsed


# ══════════════════════════════════════════════════════════════════════════════
# 8. 主评估循环
# ══════════════════════════════════════════════════════════════════════════════

def run_all_evaluations(args) -> dict:
    """
    遍历所有压缩配置 × 所有任务，收集结果。
    """
    device = args.device
    dtype  = DTYPE
    warmup = args.warmup

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  正在加载 VGGT 模型 ({args.model_name}) ...")
    print(f"{'='*60}")
    model: VGGT = VGGT.from_pretrained(args.model_name)
    model.eval().to(device)
    # 模型保持 float32：VGGT 内部 heads 使用 autocast(enabled=False) 来确保
    # 数值精度，依赖 float32 权重；直接转换到 bf16 会导致 _apply_pos_embed
    # 中 pos_embed.float() 与 bf16 特征产生 dtype 冲突（Input FloatTensor,
    # weight BFloat16Type）。推理中通过 autocast(dtype=bf16) 加速 aggregator。

    # ── 加载数据集 ─────────────────────────────────────────────────────────────
    datasets = {}
    for key, path in DATASET_PATHS.items():
        p = Path(args.data_root) / path.relative_to(DATASET_ROOT) \
            if args.data_root else path
        try:
            if key == "7scenes":
                print(f"\n  加载 7-Scenes chess  ({p}) ...")
                datasets[key] = load_7scenes_chess(p, args.max_frames)
            elif key == "tum":
                print(f"  加载 TUM walking_xyz ({p}) ...")
                datasets[key] = load_tum_walking_xyz(p, args.max_frames)
            elif key == "sintel":
                print(f"  加载 Sintel training ({p}) ...")
                datasets[key] = load_sintel(p, args.max_frames,
                                            max_scenes=args.sintel_scenes)
            S = (datasets[key][0]["frames"].shape[0]
                 if isinstance(datasets[key], list)
                 else datasets[key]["frames"].shape[0])
            print(f"    → {S} 帧 / 场景")
        except (FileNotFoundError, AssertionError) as e:
            print(f"    ⚠ 跳过 {key}：{e}")
            datasets[key] = None

    # ── 压缩配置 ───────────────────────────────────────────────────────────────
    configs = build_compression_configs()
    print(f"\n  将测试 {len(configs)} 个配置：{list(configs.keys())}")

    # ── 结果容器 ───────────────────────────────────────────────────────────────
    results: Dict[str, dict] = {}

    # ── 主循环 ────────────────────────────────────────────────────────────────
    for cfg_name, cfg in configs.items():
        print(f"\n{'─'*60}")
        print(f"  配置：{cfg_name}")
        print(f"{'─'*60}")

        # 挂载/卸载压缩 hook
        if COMPRESSION_AVAILABLE and cfg is not None:
            apply_compression_hooks(model, cfg)

        cfg_results: dict = {"config": cfg_name}

        # ── 任务 1：7-Scenes 点图 ──────────────────────────────────────────────
        if datasets.get("7scenes") is not None:
            print("  [1/3] 7-Scenes chess  点图预测 ...")
            try:
                metrics, elapsed = evaluate_7scenes(
                    model, datasets["7scenes"], dtype, device, warmup
                )
                cfg_results["7scenes"] = {**metrics, "time_s": elapsed}
                print(f"      Acc={metrics['acc_mean']:.4f} m  "
                      f"Comp={metrics['comp_mean']:.4f} m  "
                      f"NC={metrics['nc_mean']:.3f}  "
                      f"time={elapsed:.2f}s")
            except Exception as e:
                print(f"      ✗ 失败：{e}")
                cfg_results["7scenes"] = {"error": str(e)}

        # ── 任务 2：TUM 相机位姿 ───────────────────────────────────────────────
        if datasets.get("tum") is not None:
            print("  [2/3] TUM-dynamics walking_xyz  相机位姿 ...")
            try:
                metrics, elapsed = evaluate_tum(
                    model, datasets["tum"], dtype, device, warmup
                )
                cfg_results["tum"] = {**metrics, "time_s": elapsed}
                print(f"      ATE={metrics['ate']:.4f} m  "
                      f"RPEt={metrics['rpet']:.4f} m  "
                      f"RPEr={metrics['rper']:.2f}°  "
                      f"time={elapsed:.2f}s")
            except Exception as e:
                print(f"      ✗ 失败：{e}")
                cfg_results["tum"] = {"error": str(e)}

        # ── 任务 3：Sintel 视频深度 ────────────────────────────────────────────
        if datasets.get("sintel") is not None:
            print("  [3/3] Sintel training  视频深度 ...")
            try:
                metrics, elapsed = evaluate_sintel(
                    model, datasets["sintel"], dtype, device, warmup
                )
                cfg_results["sintel"] = {**metrics, "time_s": elapsed}
                print(f"      AbsRel={metrics['abs_rel']:.4f}  "
                      f"δ<1.25={metrics['delta_125']:.4f}  "
                      f"time={elapsed:.2f}s")
            except Exception as e:
                print(f"      ✗ 失败：{e}")
                cfg_results["sintel"] = {"error": str(e)}

        results[cfg_name] = cfg_results

        # 卸载 hook
        if COMPRESSION_AVAILABLE and cfg is not None:
            remove_compression_hooks(model)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 9. 结果保存（JSON + CSV + 图表）
# ══════════════════════════════════════════════════════════════════════════════

def save_results_json(results: dict, out_dir: Path) -> None:
    path = out_dir / "results.json"

    def _convert(obj):
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if np.isnan(v) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert(results), f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ 结果已保存：{path}")


def save_results_csv(results: dict, out_dir: Path) -> None:
    """
    每行格式：mechanism, task, metric, value
    """
    path = out_dir / "summary.csv"
    rows = [["mechanism", "task", "metric", "value"]]

    task_metrics = {
        "7scenes": ["acc_mean", "acc_med", "comp_mean", "comp_med",
                    "nc_mean", "time_s"],
        "tum":     ["ate", "rpet", "rper", "time_s"],
        "sintel":  ["abs_rel", "delta_125", "time_s"],
    }

    for cfg_name, cfg_data in results.items():
        for task, metrics in task_metrics.items():
            task_data = cfg_data.get(task, {})
            if "error" in task_data:
                continue
            for m in metrics:
                v = task_data.get(m, np.nan)
                rows.append([cfg_name, task, m,
                              "" if (isinstance(v, float) and np.isnan(v)) else v])

    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ CSV 已保存：{path}")


def _safe_get(results: dict, cfg: str, task: str, metric: str,
              fallback: float = np.nan) -> float:
    try:
        v = results[cfg][task][metric]
        return float(v) if v is not None else fallback
    except (KeyError, TypeError):
        return fallback


def plot_speed_comparison(results: dict, out_dir: Path) -> None:
    cfgs  = list(results.keys())
    tasks = ["7scenes", "tum", "sintel"]
    task_labels = {"7scenes": "点图 (7-Scenes)", "tum": "位姿 (TUM)", "sintel": "深度 (Sintel)"}

    times = {t: [_safe_get(results, c, t, "time_s") for c in cfgs] for t in tasks}

    x  = np.arange(len(cfgs))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(cfgs) * 1.8), 5))

    for i, (task, label) in enumerate(task_labels.items()):
        vals = times[task]
        mask = [not np.isnan(v) for v in vals]
        bars = ax.bar(x[mask] + i * w,
                      [v for v, m in zip(vals, mask) if m],
                      width=w, label=label, alpha=0.85)
        # 标注数值
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * h,
                    f"{h:.1f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(cfgs, rotation=20, ha="right")
    ax.set_ylabel("推理耗时（秒）")
    ax.set_title("各压缩机制推理速度对比")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = out_dir / "speed_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ 速度图：{path}")


def _speedup_bar(ax, cfgs, vals, baseline_val, title, ylabel):
    """辅助：绘制相对 baseline 的加速比（用于速度）或精度比（用于误差）"""
    if np.isnan(baseline_val) or baseline_val == 0:
        return
    ratios = [v / baseline_val if not np.isnan(v) else np.nan for v in vals]
    colors = ["steelblue" if r <= 1.0 else "tomato"
              if r > 1.0 else "gray"
              for r in ratios]
    x = np.arange(len(cfgs))
    bars = ax.bar(x, ratios, color=colors, alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    for bar, r in zip(bars, ratios):
        if np.isnan(r):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{r:.2f}×", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def plot_metric_comparison(results: dict, task: str, metrics: List[str],
                           metric_labels: Dict[str, str],
                           lower_is_better: List[bool],
                           out_path: Path) -> None:
    cfgs = list(results.keys())
    baseline = "baseline"
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric, lib in zip(axes, metrics, lower_is_better):
        vals = [_safe_get(results, c, task, metric) for c in cfgs]
        base_val = _safe_get(results, baseline, task, metric)

        if lib:
            # 误差越低越好：显示相对 baseline 的比值（<1 = 更好 = 绿色）
            colors = ["mediumseagreen" if (not np.isnan(v) and v <= base_val)
                      else "tomato" if (not np.isnan(v) and v > base_val)
                      else "gray"
                      for v in vals]
        else:
            # 指标越高越好（δ<1.25, NC）
            colors = ["mediumseagreen" if (not np.isnan(v) and v >= base_val)
                      else "tomato" if (not np.isnan(v) and v < base_val)
                      else "gray"
                      for v in vals]

        x = np.arange(len(cfgs))
        bars = ax.bar(x, vals, color=colors, alpha=0.85)

        if not np.isnan(base_val):
            ax.axhline(base_val, color="black", linewidth=0.8,
                       linestyle="--", label=f"baseline={base_val:.4f}")
            ax.legend(fontsize=8)

        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=20, ha="right")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(f"{metric_labels.get(metric, metric)}")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"精度对比：{task}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ 指标图：{out_path}")


def save_all_plots(results: dict, out_dir: Path) -> None:
    plot_speed_comparison(results, out_dir)

    # 7-Scenes
    if any("7scenes" in v for v in results.values()):
        plot_metric_comparison(
            results, "7scenes",
            metrics        = ["acc_mean", "comp_mean", "nc_mean"],
            metric_labels  = {"acc_mean": "Acc_mean (m)", "comp_mean": "Comp_mean (m)",
                               "nc_mean": "NC (cos)"},
            lower_is_better = [True, True, False],
            out_path       = out_dir / "metric_comparison_7scenes.png",
        )

    # TUM
    if any("tum" in v for v in results.values()):
        plot_metric_comparison(
            results, "tum",
            metrics        = ["ate", "rpet", "rper"],
            metric_labels  = {"ate": "ATE (m)", "rpet": "RPEt (m)", "rper": "RPEr (°)"},
            lower_is_better = [True, True, True],
            out_path       = out_dir / "metric_comparison_tum.png",
        )

    # Sintel
    if any("sintel" in v for v in results.values()):
        plot_metric_comparison(
            results, "sintel",
            metrics        = ["abs_rel", "delta_125"],
            metric_labels  = {"abs_rel": "AbsRel (↓)", "delta_125": "δ<1.25 (↑)"},
            lower_is_better = [True, False],
            out_path       = out_dir / "metric_comparison_sintel.png",
        )


def print_summary_table(results: dict) -> None:
    """在终端打印三任务汇总表格"""
    sep = "─" * 100
    print(f"\n{'='*100}")
    print("  VGGT 压缩机制综合评测汇总")
    print(f"{'='*100}")

    # ── 7-Scenes ──────────────────────────────────────────────────────────────
    print(f"\n  ■ 任务 1：点图预测（7-Scenes chess）")
    print(f"  {sep}")
    print(f"  {'配置':<14} {'Acc_mean(m)':<14} {'Acc_med(m)':<13} "
          f"{'Comp_mean(m)':<14} {'Comp_med(m)':<13} "
          f"{'NC(cos)':<10} {'Time(s)':<9}")
    print(f"  {sep}")
    for cfg, data in results.items():
        d = data.get("7scenes", {})
        if "error" in d:
            print(f"  {cfg:<14} {'ERROR'}")
            continue
        print(f"  {cfg:<14} "
              f"{_safe_get(results,cfg,'7scenes','acc_mean'):<14.4f}"
              f"{_safe_get(results,cfg,'7scenes','acc_med'):<13.4f}"
              f"{_safe_get(results,cfg,'7scenes','comp_mean'):<14.4f}"
              f"{_safe_get(results,cfg,'7scenes','comp_med'):<13.4f}"
              f"{_safe_get(results,cfg,'7scenes','nc_mean'):<10.3f}"
              f"{_safe_get(results,cfg,'7scenes','time_s'):<9.2f}")

    # ── TUM ───────────────────────────────────────────────────────────────────
    print(f"\n  ■ 任务 2：相机位姿（TUM-dynamics walking_xyz）")
    print(f"  {sep}")
    print(f"  {'配置':<14} {'ATE(m)':<12} {'RPEt(m)':<12} {'RPEr(°)':<12} {'Time(s)':<9}")
    print(f"  {sep}")
    for cfg in results:
        print(f"  {cfg:<14} "
              f"{_safe_get(results,cfg,'tum','ate'):<12.4f}"
              f"{_safe_get(results,cfg,'tum','rpet'):<12.4f}"
              f"{_safe_get(results,cfg,'tum','rper'):<12.2f}"
              f"{_safe_get(results,cfg,'tum','time_s'):<9.2f}")

    # ── Sintel ────────────────────────────────────────────────────────────────
    print(f"\n  ■ 任务 3：视频深度（Sintel training）")
    print(f"  {sep}")
    print(f"  {'配置':<14} {'AbsRel':<12} {'δ<1.25':<12} {'Time(s)':<9}")
    print(f"  {sep}")
    for cfg in results:
        print(f"  {cfg:<14} "
              f"{_safe_get(results,cfg,'sintel','abs_rel'):<12.4f}"
              f"{_safe_get(results,cfg,'sintel','delta_125'):<12.4f}"
              f"{_safe_get(results,cfg,'sintel','time_s'):<9.2f}")

    print(f"\n{'='*100}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 10. CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="VGGT 压缩机制综合评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name", type=str, default="facebook/VGGT-1B",
        help="HuggingFace 模型 ID",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    parser.add_argument(
        "--max_frames", type=int, default=64,
        help="每个数据集最多加载的帧数（越多越慢，但更能体现大序列压缩效果）",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="计时前的热身推理轮数",
    )
    parser.add_argument(
        "--sintel_scenes", type=int, default=3,
        help="Sintel 评估的场景数（每场景约 20-50 帧）",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="数据集根目录（默认 ../datasets 相对于项目根）",
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(RESULTS_DIR),
        help="结果输出目录",
    )
    parser.add_argument(
        "--configs", type=str, nargs="+", default=None,
        help="仅运行指定配置（默认全部），e.g. --configs baseline A_full B",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  输出目录：{out_dir}")
    print(f"  设备：    {args.device}")
    print(f"  最大帧数：{args.max_frames}")
    print(f"  热身轮数：{args.warmup}")

    # 执行全部评估
    results = run_all_evaluations(args)

    # 若指定了部分 configs，过滤
    if args.configs:
        results = {k: v for k, v in results.items() if k in args.configs}

    # 保存结果
    save_results_json(results, out_dir)
    save_results_csv(results, out_dir)
    save_all_plots(results, out_dir)
    print_summary_table(results)

    print(f"  所有结果已保存至：{out_dir}\n")


if __name__ == "__main__":
    main()
