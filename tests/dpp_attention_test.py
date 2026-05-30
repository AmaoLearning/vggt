#!/usr/bin/env python3
"""
tests/dpp_attention_test.py
============================
VGGT DPP Attention 综合评测脚本

对机制 G（DPP Attention）的各参数组合在三项任务上进行
推理速度与预测精度的系统测试：

  任务 1：点图预测      (Point Map)      — 7-Scenes chess
  任务 2：相机位姿预测  (Camera Pose)    — TUM-dynamics walking_xyz
  任务 3：视频深度预测  (Video Depth)    — Sintel training

DPP 参数维度（对照组）：
  keep_ratio      — 保留比例（10% / 30% / 50% / 70% / 90%）
  window_size     — 帧内邻域窗口大小（1 / 3 / 5）
  num_adj_frames  — 参考帧数（全部 / 仅 2 帧）
  relevance_agg   — 跨帧聚合（max / mean）
  on_all_layers   — 逐层重计算 vs 首层缓存

指标：
  点图：   Acc_mean/median（m）、Comp_mean/median（m）、NC（余弦相似度）
  相机：   ATE（m）、RPEt（m）、RPEr（°）  [经 Sim3 对齐]
  深度：   AbsRel、δ<1.25

结果保存：tests/results/dpp_attention/
  results.json                  — 完整数值结果
  summary.csv                   — 每行 (config, task, metric, value)
  speed_comparison.png
  metric_comparison_7scenes.png
  metric_comparison_tum.png
  metric_comparison_sintel.png
  dpp_pareto_7scenes.png        — keep_ratio Pareto 曲线（质量 vs 速度）
  dpp_pareto_sintel.png
  dpp_layer_mode_comparison.png — on_all_layers vs 缓存模式比较

用法（从项目根目录）：
    python tests/dpp_attention_test.py [--max_frames N] [--warmup N] [--device cuda]
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

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    warnings.warn(
        "[dpp_attention_test] open3d not found. ICP refinement and NC for "
        "point map evaluation will be disabled.\n"
        "  Install with: pip install open3d",
        stacklevel=2,
    )

# ── 项目路径 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent        # tests/
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # VGGT/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── VGGT 核心 ─────────────────────────────────────────────────────────────────
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# ── DPP 模块 ──────────────────────────────────────────────────────────────────
try:
    from vggt.compression.mechanism_g import DPPConfig
    DPP_AVAILABLE = True
except ImportError:
    DPP_AVAILABLE = False
    warnings.warn(
        "[dpp_attention_test] vggt.compression.mechanism_g 未找到，仅测试 baseline。\n"
        "  请确认 vggt/compression/mechanism_g.py 已创建。",
        stacklevel=2,
    )
    DPPConfig = None  # type: ignore

# ══════════════════════════════════════════════════════════════════════════════
# 0. 全局常量 / 默认路径
# ══════════════════════════════════════════════════════════════════════════════

IMG_SIZE    = 518
DTYPE       = torch.bfloat16
RESULTS_DIR = _SCRIPT_DIR / "results" / "dpp_attention"

DATASET_ROOT  = _PROJECT_ROOT.parent / "datasets"
DATASET_PATHS = {
    "7scenes": DATASET_ROOT / "7-scenes"     / "chess",
    "tum":     DATASET_ROOT / "tum-dynamics" / "walking_xyz",
    "sintel":  DATASET_ROOT / "sintel"       / "training",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DPP 配置定义
# ══════════════════════════════════════════════════════════════════════════════

def build_dpp_configs() -> Dict[str, Optional["DPPConfig"]]:
    """
    返回 {名称 → DPPConfig | None} 字典，None 表示 baseline（无 DPP）。

    维度说明：
      keep_ratio  : G_kr90 / G_kr70 / G_kr50 / G_kr30 / G_kr10
      window_size : G_win1 / G_win5  （以 G_kr50 为基础变体）
      num_adj_frames : G_adj2         （以 G_kr50 为基础变体）
      relevance_agg  : G_mean         （以 G_kr50 为基础变体）
      on_all_layers  : G_cache        （on_all_layers=False，以 G_kr50 为基础变体）
    """
    configs: Dict[str, Optional[object]] = {"baseline": None}

    if not DPP_AVAILABLE:
        return configs

    # ── keep_ratio 扫描（默认其余参数）──────────────────────────────────────
    for kr, name in [
        (0.90, "G_kr90"),
        (0.70, "G_kr70"),
        (0.50, "G_kr50"),
        (0.30, "G_kr30"),
        (0.10, "G_kr10"),
    ]:
        configs[name] = DPPConfig(
            keep_ratio      = kr,
            window_size     = 3,
            num_adj_frames  = -1,
            relevance_agg   = "max",
            on_all_layers   = True,
        )

    # ── window_size 变体（基于 keep_ratio=0.50）──────────────────────────────
    configs["G_win1"] = DPPConfig(
        keep_ratio     = 0.50,
        window_size    = 1,   # 无邻域窗口（仅单像素自身相似度）
        num_adj_frames = -1,
        relevance_agg  = "max",
        on_all_layers  = True,
    )
    configs["G_win5"] = DPPConfig(
        keep_ratio     = 0.50,
        window_size    = 5,
        num_adj_frames = -1,
        relevance_agg  = "max",
        on_all_layers  = True,
    )

    # ── num_adj_frames 变体（仅使用最近 2 帧）───────────────────────────────
    configs["G_adj2"] = DPPConfig(
        keep_ratio     = 0.50,
        window_size    = 3,
        num_adj_frames = 2,
        relevance_agg  = "max",
        on_all_layers  = True,
    )

    # ── relevance_agg 变体（mean 代替 max）──────────────────────────────────
    configs["G_mean"] = DPPConfig(
        keep_ratio     = 0.50,
        window_size    = 3,
        num_adj_frames = -1,
        relevance_agg  = "mean",
        on_all_layers  = True,
    )

    # ── on_all_layers=False：首层计算，后续层复用缓存索引 ────────────────────
    configs["G_cache"] = DPPConfig(
        keep_ratio     = 0.50,
        window_size    = 3,
        num_adj_frames = -1,
        relevance_agg  = "max",
        on_all_layers  = False,
    )

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# 2. 数据集加载（与 compression_test.py 相同）
# ══════════════════════════════════════════════════════════════════════════════

def _make_image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


# ── 2.1  7-Scenes chess ───────────────────────────────────────────────────────

def load_7scenes_chess(data_dir: Path, max_frames: int) -> dict:
    """
    加载 7-Scenes chess 序列。

    目录结构：
        chess/
          seq-01/
            frame-*.color.png   RGB 640×480
            frame-*.depth.png   uint16 mm
            frame-*.pose.txt    4×4 camera-to-world
    """
    K = np.array([[525.0, 0.0, 320.0],
                  [0.0, 525.0, 240.0],
                  [0.0, 0.0,   1.0]], dtype=np.float64)

    seq_dir = data_dir / "seq-01"
    if not seq_dir.exists():
        raise FileNotFoundError(f"未找到 {seq_dir}（7-Scenes chess seq-01）")

    tf = _make_image_transform()
    frames, depths, poses = [], [], []

    color_files = sorted(seq_dir.glob("frame-*.color.png"))
    for cf in color_files:
        stem    = cf.stem.replace(".color", "")
        depth_f = seq_dir / f"{stem}.depth.png"
        pose_f  = seq_dir / f"{stem}.pose.txt"
        if not depth_f.exists() or not pose_f.exists():
            continue

        frames.append(tf(Image.open(cf).convert("RGB")))
        d = np.array(Image.open(depth_f), dtype=np.float32) / 1000.0
        depths.append(d)
        poses.append(np.loadtxt(str(pose_f)))

        if len(frames) >= max_frames:
            break

    if not frames:
        raise FileNotFoundError(f"7-Scenes chess：{data_dir} 中未找到任何帧")

    gt_pts_list = []
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

    all_gt = np.concatenate(gt_pts_list, axis=0)
    if len(all_gt) > 50_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_gt), 50_000, replace=False)
        all_gt = all_gt[idx]

    return {
        "frames":    torch.stack(frames),
        "depths":    np.array(depths),
        "poses_c2w": np.array(poses),
        "gt_points": all_gt.astype(np.float32),
        "K":         K,
        "orig_hw":   (H_orig, W_orig),
    }


# ── 2.2  TUM-dynamics walking_xyz ─────────────────────────────────────────────

def load_tum_walking_xyz(data_dir: Path, max_frames: int) -> dict:
    """
    加载 TUM RGB-D walking_xyz 序列。

    目录结构：
        walking_xyz/
          rgb/               timestamped PNG
          groundtruth.txt    ts tx ty tz qx qy qz qw
          rgb.txt            ts path
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
        ts   = float(parts[0])
        path = data_dir / parts[1]
        rgb_entries.append((ts, path))
    rgb_entries.sort(key=lambda x: x[0])

    tf = _make_image_transform()
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
        raise FileNotFoundError(
            f"TUM walking_xyz：{data_dir} 中未匹配到任何 RGB-GT 对"
        )

    return {
        "frames":    torch.stack(frames),
        "poses_c2w": np.array(poses),
    }


# ── 2.3  Sintel training ──────────────────────────────────────────────────────

def _read_dpt(path: Path) -> np.ndarray:
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
        frames, dpts = [], []

        for rf in rgb_files:
            dpt_f = depth_dir / scene / rf.with_suffix(".dpt").name
            if not dpt_f.exists():
                continue
            frames.append(tf(Image.open(rf).convert("RGB")))
            dpts.append(_read_dpt(dpt_f))

        if len(frames) < 2:
            continue

        result.append({
            "scene":  scene,
            "frames": torch.stack(frames),
            "depths": np.array(dpts),
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
    """
    imgs = frames.to(device).float()
    imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE),
                         mode="bilinear", align_corners=False)
    imgs = imgs.unsqueeze(0)

    _autocast = (
        torch.cuda.amp.autocast(dtype=dtype)
        if device == "cuda" else contextlib.nullcontext()
    )

    with torch.no_grad():
        for _ in range(warmup):
            with _autocast:
                _ = model(imgs)
            if device == "cuda":
                torch.cuda.synchronize()

        if device == "cuda":
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            with _autocast:
                preds = model(imgs)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed = start_evt.elapsed_time(end_evt) / 1000.0
        else:
            import time as _time
            t0 = _time.perf_counter()
            with _autocast:
                preds = model(imgs)
            elapsed = _time.perf_counter() - t0

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
    """Umeyama Sim3 对齐：dst ≈ s * R @ src + t"""
    N    = src.shape[0]
    mu_s = src.mean(0);  mu_d = dst.mean(0)
    src_c = src - mu_s;  dst_c = dst - mu_d
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

def eval_point_map(pred_world_pts: np.ndarray, gt_pts: np.ndarray,
                   max_dist_m: float = 0.1) -> dict:
    MAX_PTS = 30_000
    rng = np.random.default_rng(42)

    pred_s = pred_world_pts[rng.choice(len(pred_world_pts), min(MAX_PTS, len(pred_world_pts)), replace=False)]
    gt_s   = gt_pts[rng.choice(len(gt_pts), min(MAX_PTS, len(gt_pts)), replace=False)]

    if O3D_AVAILABLE:
        pcd    = o3d.geometry.PointCloud()
        pcd_gt = o3d.geometry.PointCloud()
        pcd.points    = o3d.utility.Vector3dVector(pred_s.astype(np.float64))
        pcd_gt.points = o3d.utility.Vector3dVector(gt_s.astype(np.float64))
        reg = o3d.pipelines.registration.registration_icp(
            pcd, pcd_gt, 0.1, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pcd = pcd.transform(reg.transformation)
        pred_s = np.asarray(pcd.points, dtype=np.float32)

    gt_tree   = cKDTree(gt_s)
    pred_tree = cKDTree(pred_s)
    acc_d,  acc_idx  = gt_tree.query(pred_s, k=1)
    comp_d, comp_idx = pred_tree.query(gt_s, k=1)
    acc_d  = np.clip(acc_d,  0.0, max_dist_m)
    comp_d = np.clip(comp_d, 0.0, max_dist_m)

    metrics = {
        "acc_mean":  float(acc_d.mean()),
        "acc_med":   float(np.median(acc_d)),
        "comp_mean": float(comp_d.mean()),
        "comp_med":  float(np.median(comp_d)),
        "nc_mean":   float("nan"),
    }

    if O3D_AVAILABLE:
        search = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        pcd.estimate_normals(search_param=search)
        pcd_gt.estimate_normals(search_param=search)
        pred_n = np.asarray(pcd.normals,    dtype=np.float32)
        gt_n   = np.asarray(pcd_gt.normals, dtype=np.float32)
        if len(pred_n) > 0 and len(gt_n) > 0:
            nc1 = float(np.abs((pred_n * gt_n[acc_idx]).sum(-1)).mean())
            nc2 = float(np.abs((gt_n * pred_n[comp_idx]).sum(-1)).mean())
            metrics["nc_mean"] = (nc1 + nc2) / 2.0

    return metrics


def _extri_to_cam_position(extri: np.ndarray) -> np.ndarray:
    R = extri[:, :3, :3]
    t = extri[:, :3,  3]
    return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)


def evaluate_7scenes(model: VGGT, data: dict, dtype: torch.dtype,
                     device: str, warmup: int) -> Tuple[dict, float]:
    frames = data["frames"]
    gt_pts = data["gt_points"]

    preds, elapsed = run_inference(model, frames, dtype, device, warmup)

    wp = preds["world_points"].squeeze(0).numpy()
    S, H, W, _ = wp.shape
    pred_pts = wp.reshape(-1, 3)
    valid = np.isfinite(pred_pts).all(axis=1)
    pred_pts = pred_pts[valid]

    if len(pred_pts) < 100:
        return {"acc_mean": np.nan, "acc_med": np.nan,
                "comp_mean": np.nan, "comp_med": np.nan, "nc_mean": np.nan}, elapsed

    pose_enc_pred = preds["pose_enc"]
    extri_pred, _ = pose_encoding_to_extri_intri(
        pose_enc_pred, image_size_hw=(IMG_SIZE, IMG_SIZE), build_intrinsics=False
    )
    extri_pred_np    = extri_pred.squeeze(0).numpy()
    pred_cam_centers = _extri_to_cam_position(extri_pred_np)
    gt_cam_centers   = data["poses_c2w"][:S, :3, 3]
    scale, R_align, t_align = umeyama_alignment(pred_cam_centers, gt_cam_centers)
    pred_pts_aligned = scale * (R_align @ pred_pts.T).T + t_align

    metrics = eval_point_map(pred_pts_aligned, gt_pts)
    return metrics, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# 6. 相机位姿评估（TUM）
# ══════════════════════════════════════════════════════════════════════════════

def compute_ate(pred_extri: np.ndarray, gt_poses_c2w: np.ndarray) -> float:
    pred_pos = _extri_to_cam_position(pred_extri)
    gt_pos   = gt_poses_c2w[:, :3, 3]
    scale, R_a, t_a = umeyama_alignment(pred_pos, gt_pos)
    pred_aligned = scale * (R_a @ pred_pos.T).T + t_a
    return float(np.sqrt(np.mean(np.sum((pred_aligned - gt_pos) ** 2, axis=1))))


def compute_rpe(pred_extri: np.ndarray,
                gt_poses_c2w: np.ndarray) -> Tuple[float, float]:
    S = pred_extri.shape[0]
    pred_w2c = np.zeros((S, 4, 4))
    pred_w2c[:, :3, :] = pred_extri
    pred_w2c[:, 3,  3] = 1.0
    gt_w2c = np.linalg.inv(gt_poses_c2w)

    rpet_list, rper_list = [], []
    for i in range(S - 1):
        pred_rel = pred_w2c[i + 1] @ np.linalg.inv(pred_w2c[i])
        gt_rel   = gt_w2c[i + 1]   @ np.linalg.inv(gt_w2c[i])
        err      = gt_rel @ np.linalg.inv(pred_rel)
        rpet_list.append(np.linalg.norm(err[:3, 3]))
        cos_ang = np.clip((np.trace(err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rper_list.append(np.degrees(np.arccos(cos_ang)))

    return float(np.mean(rpet_list)), float(np.mean(rper_list))


def evaluate_tum(model: VGGT, data: dict, dtype: torch.dtype,
                 device: str, warmup: int) -> Tuple[dict, float]:
    frames    = data["frames"]
    poses_c2w = data["poses_c2w"]

    preds, elapsed = run_inference(model, frames, dtype, device, warmup)

    extri, _ = pose_encoding_to_extri_intri(
        preds["pose_enc"], image_size_hw=(IMG_SIZE, IMG_SIZE), build_intrinsics=False
    )
    extri_np = extri.squeeze(0).numpy()
    ate  = compute_ate(extri_np, poses_c2w)
    rpet, rper = compute_rpe(extri_np, poses_c2w)

    return {"ate": ate, "rpet": rpet, "rper": rper}, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# 7. 视频深度评估（Sintel）
# ══════════════════════════════════════════════════════════════════════════════

def _scale_shift_align(pred: np.ndarray, gt: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    valid_mask = mask & np.isfinite(pred) & (pred > 0)
    if valid_mask.sum() < 10:
        return pred
    p = pred[valid_mask].astype(np.float64)
    g = gt[valid_mask].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    try:
        x, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
        a, b = float(x[0]), float(x[1])
    except np.linalg.LinAlgError:
        return pred
    return np.clip(a * pred.astype(np.float64) + b, 1e-9, None).astype(np.float32)


def eval_depth_sequence(pred_depth: np.ndarray,
                        gt_depth: np.ndarray) -> dict:
    S = min(len(pred_depth), len(gt_depth))
    abs_rels, deltas = [], []

    for s in range(S):
        gt   = gt_depth[s].astype(np.float32)
        pred_t = torch.tensor(pred_depth[s]).unsqueeze(0).unsqueeze(0)
        pred_r = F.interpolate(pred_t, size=gt.shape, mode="bilinear",
                               align_corners=False).squeeze().numpy()
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
    return {"abs_rel": float(np.mean(abs_rels)),
            "delta_125": float(np.mean(deltas))}


def evaluate_sintel(model: VGGT, scenes: List[dict], dtype: torch.dtype,
                    device: str, warmup: int) -> Tuple[dict, float]:
    all_abs_rel, all_delta, all_elapsed = [], [], []

    for scene_data in scenes:
        frames = scene_data["frames"]
        depths = scene_data["depths"]
        preds, elapsed = run_inference(model, frames, dtype, device, warmup)
        all_elapsed.append(elapsed)

        pred_d = preds["depth"].squeeze(0).squeeze(-1).numpy()
        m = eval_depth_sequence(pred_d, depths)
        if not np.isnan(m["abs_rel"]):
            all_abs_rel.append(m["abs_rel"])
            all_delta.append(m["delta_125"])

    metrics = {
        "abs_rel":   float(np.nanmean(all_abs_rel)) if all_abs_rel else np.nan,
        "delta_125": float(np.nanmean(all_delta))   if all_delta   else np.nan,
    }
    return metrics, float(sum(all_elapsed))


# ══════════════════════════════════════════════════════════════════════════════
# 8. 主评估循环
# ══════════════════════════════════════════════════════════════════════════════

def _enable_dpp(model: VGGT, cfg: "DPPConfig") -> None:
    """为模型 aggregator 启用 DPP Attention。"""
    model.aggregator.enable_dpp(cfg)


def _disable_dpp(model: VGGT) -> None:
    """关闭 DPP Attention，恢复原始 Global Attention。"""
    model.aggregator.disable_dpp()


def run_all_evaluations(args) -> dict:
    device = args.device
    dtype  = DTYPE
    warmup = args.warmup

    print(f"\n{'='*60}")
    print(f"  正在加载 VGGT 模型 ({args.model_name}) ...")
    print(f"{'='*60}")
    model: VGGT = VGGT.from_pretrained(args.model_name)
    model.eval().to(device)

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

    configs = build_dpp_configs()
    if args.configs:
        configs = {k: v for k, v in configs.items() if k in args.configs}
    print(f"\n  将测试 {len(configs)} 个配置：{list(configs.keys())}")

    results: Dict[str, dict] = {}

    for cfg_name, dpp_cfg in configs.items():
        print(f"\n{'─'*60}")
        print(f"  配置：{cfg_name}")
        if dpp_cfg is not None:
            print(f"    keep_ratio={dpp_cfg.keep_ratio}  "
                  f"window_size={dpp_cfg.window_size}  "
                  f"num_adj_frames={dpp_cfg.num_adj_frames}  "
                  f"relevance_agg={dpp_cfg.relevance_agg}  "
                  f"on_all_layers={dpp_cfg.on_all_layers}")
        print(f"{'─'*60}")

        # 启用/禁用 DPP
        if DPP_AVAILABLE and dpp_cfg is not None:
            _enable_dpp(model, dpp_cfg)

        cfg_results: dict = {"config": cfg_name}

        # ── 任务 1：7-Scenes ──────────────────────────────────────────────────
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

        # ── 任务 2：TUM ───────────────────────────────────────────────────────
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

        # ── 任务 3：Sintel ────────────────────────────────────────────────────
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

        # 禁用 DPP
        if DPP_AVAILABLE and dpp_cfg is not None:
            _disable_dpp(model)

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
    path = out_dir / "summary.csv"
    rows = [["config", "task", "metric", "value"]]
    task_metrics = {
        "7scenes": ["acc_mean", "acc_med", "comp_mean", "comp_med", "nc_mean", "time_s"],
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
    task_labels = {
        "7scenes": "Point Map (7-Scenes)",
        "tum":     "Camera Pose (TUM)",
        "sintel":  "Depth (Sintel)",
    }
    times = {t: [_safe_get(results, c, t, "time_s") for c in cfgs] for t in tasks}
    x = np.arange(len(cfgs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(cfgs) * 1.8), 5))
    for i, (task, label) in enumerate(task_labels.items()):
        vals = times[task]
        mask = [not np.isnan(v) for v in vals]
        bars = ax.bar(x[mask] + i * w,
                      [v for v, m in zip(vals, mask) if m],
                      width=w, label=label, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * h,
                    f"{h:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(cfgs, rotation=25, ha="right")
    ax.set_ylabel("Inference Time (s)")
    ax.set_title("DPP Attention: Inference Speed Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "speed_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ 速度图：{path}")


def plot_metric_comparison(results: dict, task: str, metrics: List[str],
                           metric_labels: Dict[str, str],
                           lower_is_better: List[bool],
                           out_path: Path) -> None:
    cfgs      = list(results.keys())
    baseline  = "baseline"
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric, lib in zip(axes, metrics, lower_is_better):
        vals     = [_safe_get(results, c, task, metric) for c in cfgs]
        base_val = _safe_get(results, baseline, task, metric)
        if lib:
            colors = ["mediumseagreen" if (not np.isnan(v) and v <= base_val)
                      else "tomato" if (not np.isnan(v) and v > base_val)
                      else "gray" for v in vals]
        else:
            colors = ["mediumseagreen" if (not np.isnan(v) and v >= base_val)
                      else "tomato" if (not np.isnan(v) and v < base_val)
                      else "gray" for v in vals]
        x    = np.arange(len(cfgs))
        bars = ax.bar(x, vals, color=colors, alpha=0.85)
        if not np.isnan(base_val):
            ax.axhline(base_val, color="black", linewidth=0.8, linestyle="--",
                       label=f"baseline={base_val:.4f}")
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=25, ha="right")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"DPP Attention Accuracy Comparison: {task}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ 指标图：{out_path}")


# ── DPP 专有图 1：keep_ratio Pareto 曲线 ─────────────────────────────────────

def plot_pareto_curve(results: dict, out_dir: Path) -> None:
    """
    绘制 keep_ratio 变体（G_kr*）的 Pareto 曲线：
        X 轴：推理时间（秒）
        Y 轴：精度指标（7-Scenes Acc_mean / Sintel AbsRel）
    越靠左下方越好。
    """
    kr_cfgs = [k for k in results if k.startswith("G_kr") or k == "baseline"]

    for task, metric, lower_is_better, fname in [
        ("7scenes", "acc_mean",  True,  "dpp_pareto_7scenes.png"),
        ("sintel",  "abs_rel",   True,  "dpp_pareto_sintel.png"),
    ]:
        xs = [_safe_get(results, c, task, "time_s") for c in kr_cfgs]
        ys = [_safe_get(results, c, task, metric)   for c in kr_cfgs]
        valid = [(x, y, c) for x, y, c in zip(xs, ys, kr_cfgs)
                 if not (np.isnan(x) or np.isnan(y))]
        if len(valid) < 2:
            continue

        x_v, y_v, labels = zip(*valid)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_v, y_v, s=80, zorder=3)
        for x, y, lbl in zip(x_v, y_v, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.plot(x_v, y_v, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Inference Time (s)")
        ylabel = f"{metric} (↓ lower=better)" if lower_is_better else f"{metric} (↑ higher=better)"
        ax.set_ylabel(ylabel)
        ax.set_title(f"DPP keep_ratio Pareto — {task}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  ✓ Pareto 曲线：{out_path}")


# ── DPP 专有图 2：on_all_layers vs cache 模式对比 ─────────────────────────────

def plot_layer_mode_comparison(results: dict, out_dir: Path) -> None:
    """
    对比 G_kr50（on_all_layers=True）vs G_cache（on_all_layers=False）
    在精度和速度上的差异。
    """
    compare_cfgs = [c for c in ["baseline", "G_kr50", "G_cache"] if c in results]
    if len(compare_cfgs) < 2:
        return

    tasks_metrics = [
        ("7scenes", "acc_mean", "Acc_mean (m)", True),
        ("sintel",  "abs_rel",  "AbsRel",       True),
        ("7scenes", "time_s",   "Time (s)",     False),
    ]
    n = len(tasks_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (task, metric, ylabel, lib) in zip(axes, tasks_metrics):
        vals   = [_safe_get(results, c, task, metric) for c in compare_cfgs]
        colors = ["steelblue", "mediumseagreen", "coral"][:len(compare_cfgs)]
        bars   = ax.bar(compare_cfgs, vals, color=colors[:len(vals)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} ({'lower=better' if lib else 'higher=better'})")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("DPP: on_all_layers vs Cache Mode", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / "dpp_layer_mode_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ 层模式对比图：{out_path}")


def save_all_plots(results: dict, out_dir: Path) -> None:
    plot_speed_comparison(results, out_dir)

    if any("7scenes" in v for v in results.values()):
        plot_metric_comparison(
            results, "7scenes",
            metrics         = ["acc_mean", "comp_mean", "nc_mean"],
            metric_labels   = {"acc_mean": "Acc_mean (m)", "comp_mean": "Comp_mean (m)",
                                "nc_mean": "NC (cos)"},
            lower_is_better = [True, True, False],
            out_path        = out_dir / "metric_comparison_7scenes.png",
        )

    if any("tum" in v for v in results.values()):
        plot_metric_comparison(
            results, "tum",
            metrics         = ["ate", "rpet", "rper"],
            metric_labels   = {"ate": "ATE (m)", "rpet": "RPEt (m)", "rper": "RPEr (deg)"},
            lower_is_better = [True, True, True],
            out_path        = out_dir / "metric_comparison_tum.png",
        )

    if any("sintel" in v for v in results.values()):
        plot_metric_comparison(
            results, "sintel",
            metrics         = ["abs_rel", "delta_125"],
            metric_labels   = {"abs_rel": "AbsRel (↓)", "delta_125": "delta<1.25 (↑)"},
            lower_is_better = [True, False],
            out_path        = out_dir / "metric_comparison_sintel.png",
        )

    # DPP 专有图
    plot_pareto_curve(results, out_dir)
    plot_layer_mode_comparison(results, out_dir)


def print_summary_table(results: dict) -> None:
    sep = "─" * 105
    print(f"\n{'='*105}")
    print("  VGGT DPP Attention 综合评测汇总")
    print(f"{'='*105}")

    print(f"\n  ■ 任务 1：点图预测（7-Scenes chess）")
    print(f"  {sep}")
    print(f"  {'配置':<14} {'Acc_mean(m)':<14} {'Acc_med(m)':<13} "
          f"{'Comp_mean(m)':<14} {'Comp_med(m)':<13} {'NC(cos)':<10} {'Time(s)':<9}")
    print(f"  {sep}")
    for cfg in results:
        d = results[cfg].get("7scenes", {})
        if "error" in d:
            print(f"  {cfg:<14} ERROR: {d['error'][:40]}")
            continue
        print(f"  {cfg:<14} "
              f"{_safe_get(results,cfg,'7scenes','acc_mean'):<14.4f}"
              f"{_safe_get(results,cfg,'7scenes','acc_med'):<13.4f}"
              f"{_safe_get(results,cfg,'7scenes','comp_mean'):<14.4f}"
              f"{_safe_get(results,cfg,'7scenes','comp_med'):<13.4f}"
              f"{_safe_get(results,cfg,'7scenes','nc_mean'):<10.3f}"
              f"{_safe_get(results,cfg,'7scenes','time_s'):<9.2f}")

    print(f"\n  ■ 任务 2：相机位姿（TUM-dynamics walking_xyz）")
    print(f"  {sep}")
    print(f"  {'配置':<14} {'ATE(m)':<12} {'RPEt(m)':<12} {'RPEr(°)':<12} {'Time(s)':<9}")
    print(f"  {sep}")
    for cfg in results:
        d = results[cfg].get("tum", {})
        if "error" in d:
            print(f"  {cfg:<14} ERROR: {d['error'][:40]}")
            continue
        print(f"  {cfg:<14} "
              f"{_safe_get(results,cfg,'tum','ate'):<12.4f}"
              f"{_safe_get(results,cfg,'tum','rpet'):<12.4f}"
              f"{_safe_get(results,cfg,'tum','rper'):<12.2f}"
              f"{_safe_get(results,cfg,'tum','time_s'):<9.2f}")

    print(f"\n  ■ 任务 3：视频深度（Sintel training）")
    print(f"  {sep}")
    print(f"  {'配置':<14} {'AbsRel':<12} {'δ<1.25':<12} {'Time(s)':<9}")
    print(f"  {sep}")
    for cfg in results:
        d = results[cfg].get("sintel", {})
        if "error" in d:
            print(f"  {cfg:<14} ERROR: {d['error'][:40]}")
            continue
        print(f"  {cfg:<14} "
              f"{_safe_get(results,cfg,'sintel','abs_rel'):<12.4f}"
              f"{_safe_get(results,cfg,'sintel','delta_125'):<12.4f}"
              f"{_safe_get(results,cfg,'sintel','time_s'):<9.2f}")

    print(f"\n{'='*105}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 10. CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="VGGT DPP Attention 综合评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames", type=int, default=64,
                        help="每个数据集最多加载的帧数")
    parser.add_argument("--warmup", type=int, default=3,
                        help="计时前的热身推理轮数")
    parser.add_argument("--sintel_scenes", type=int, default=3,
                        help="Sintel 评估的场景数")
    parser.add_argument("--data_root", type=str, default=None,
                        help="数据集根目录（默认 ../datasets 相对于项目根）")
    parser.add_argument("--out_dir", type=str, default=str(RESULTS_DIR),
                        help="结果输出目录")
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        help="仅运行指定配置（默认全部），e.g. --configs baseline G_kr50")
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  输出目录：{out_dir}")
    print(f"  设备：    {args.device}")
    print(f"  最大帧数：{args.max_frames}")
    print(f"  热身轮数：{args.warmup}")

    results = run_all_evaluations(args)

    save_results_json(results, out_dir)
    save_results_csv(results, out_dir)
    save_all_plots(results, out_dir)
    print_summary_table(results)

    print(f"  所有结果已保存至：{out_dir}\n")


if __name__ == "__main__":
    main()
