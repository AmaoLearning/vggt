#!/usr/bin/env python3
"""
VGGT Layer Silencing Experiment - Depth Focus
"""

import argparse
import contextlib
import csv
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vggt.models.vggt import VGGT

IMG_SIZE = 518
DTYPE = torch.bfloat16
RESULTS_DIR = _SCRIPT_DIR / "results" / "depth_silence_test"
DATASET_ROOT = _PROJECT_ROOT.parent / "datasets"
DEFAULT_SINTEL_DIR = DATASET_ROOT / "sintel" / "training"


@dataclass
class SilenceConfig:
    frame_layers: List[int] = field(default_factory=list)
    global_layers: List[int] = field(default_factory=list)
    label: str = ""

    def is_empty(self) -> bool:
        return len(self.frame_layers) == 0 and len(self.global_layers) == 0


def build_silence_configs(groups: Optional[List[str]] = None) -> Dict[str, SilenceConfig]:
    """
    Build silence configs.

    Groups:
      - baseline
      - per_layer_global
      - per_layer_frame
      - range
      - boundary
    """
    all_groups = {
        "baseline",
        "per_layer_global",
        "per_layer_frame",
        "range",
        "boundary",
    }
    active = set(groups) if groups else all_groups

    configs: Dict[str, SilenceConfig] = {}

    if "baseline" in active:
        configs["baseline"] = SilenceConfig(label="Baseline (no silencing)")

    if "per_layer_global" in active:
        for i in range(24):
            configs[f"G{i:02d}"] = SilenceConfig(
                global_layers=[i],
                label=f"Global block {i}",
            )

    if "per_layer_frame" in active:
        for i in range(24):
            configs[f"F{i:02d}"] = SilenceConfig(
                frame_layers=[i],
                label=f"Frame block {i}",
            )

    if "range" in active:
        configs["range_global_3_4"] = SilenceConfig(
            global_layers=[3, 4],
            label="global 3-4",
        )
        configs["range_frame_3_4"] = SilenceConfig(
            frame_layers=[3, 4],
            label="frame 3-4",
        )
        configs["range_both_3_4"] = SilenceConfig(
            global_layers=[3, 4],
            frame_layers=[3, 4],
            label="global+frame 3-4",
        )

        configs["range_global_4_8"] = SilenceConfig(
            global_layers=list(range(4, 9)),
            label="global 4-8",
        )
        configs["range_frame_4_8"] = SilenceConfig(
            frame_layers=list(range(4, 9)),
            label="frame 4-8",
        )
        configs["range_both_4_8"] = SilenceConfig(
            global_layers=list(range(4, 9)),
            frame_layers=list(range(4, 9)),
            label="global+frame 4-8",
        )

        configs["range_global_10_16"] = SilenceConfig(
            global_layers=list(range(10, 17)),
            label="global 10-16",
        )
        configs["range_frame_10_16"] = SilenceConfig(
            frame_layers=list(range(10, 17)),
            label="frame 10-16",
        )
        configs["range_both_10_16"] = SilenceConfig(
            global_layers=list(range(10, 17)),
            frame_layers=list(range(10, 17)),
            label="global+frame 10-16",
        )

        configs["range_global_20_23"] = SilenceConfig(
            global_layers=list(range(20, 24)),
            label="global 20-23",
        )
        configs["range_frame_20_23"] = SilenceConfig(
            frame_layers=list(range(20, 24)),
            label="frame 20-23",
        )
        configs["range_both_20_23"] = SilenceConfig(
            global_layers=list(range(20, 24)),
            frame_layers=list(range(20, 24)),
            label="global+frame 20-23",
        )

        configs["range_global_4_8_20_23"] = SilenceConfig(
            global_layers=list(range(4, 9)) + list(range(20, 24)),
            label="global 4-8 + 20-23",
        )
        configs["range_both_3_4_10_16"] = SilenceConfig(
            global_layers=[3, 4] + list(range(10, 17)),
            frame_layers=[3, 4] + list(range(10, 17)),
            label="global+frame 3-4 + 10-16",
        )

        configs["range_keep_global_4_8_14_20"] = SilenceConfig(
            global_layers=list(range(0, 4)) + list(range(9, 14)) + list(range(21, 24)),
            label="keep global 4-8 + 14-20, keep all frame",
        )
        configs["range_keep_frame_1_9"] = SilenceConfig(
            frame_layers=[0] + list(range(10, 24)),
            label="keep all global, keep frame 1-9",
        )
        configs["range_keep_global_4_8_14_20_frame_1_9"] = SilenceConfig(
            global_layers=list(range(0, 4)) + list(range(9, 14)) + list(range(21, 24)),
            frame_layers=[0] + list(range(10, 24)),
            label="keep global 4-8 + 14-20, keep frame 1-9",
        )

    if "boundary" in active:
        configs["all_global"] = SilenceConfig(
            global_layers=list(range(24)),
            label="all global blocks",
        )
        configs["all_frame"] = SilenceConfig(
            frame_layers=list(range(24)),
            label="all frame blocks",
        )
        configs["first_half_global"] = SilenceConfig(
            global_layers=list(range(12)),
            label="global first half",
        )
        configs["second_half_global"] = SilenceConfig(
            global_layers=list(range(12, 24)),
            label="global second half",
        )

    return configs


class LayerSilencer:
    """Context manager that applies identity forward hook on selected blocks."""

    def __init__(self, model: VGGT, cfg: SilenceConfig):
        self.model = model
        self.cfg = cfg
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _identity_hook(self, module, input, output):
        return input[0]

    def __enter__(self):
        agg = self.model.aggregator

        for i in self.cfg.frame_layers:
            if 0 <= i < len(agg.frame_blocks):
                self._handles.append(agg.frame_blocks[i].register_forward_hook(self._identity_hook))
            else:
                print(f"[WARN] frame block index out of range: {i}")

        for i in self.cfg.global_layers:
            if 0 <= i < len(agg.global_blocks):
                self._handles.append(agg.global_blocks[i].register_forward_hook(self._identity_hook))
            else:
                print(f"[WARN] global block index out of range: {i}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def _make_image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def _read_dpt(path: Path) -> np.ndarray:
    """Read Sintel .dpt depth file as float32 in meters."""
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        if abs(magic - 202021.25) >= 1.0:
            raise ValueError(f"Invalid .dpt magic: {magic}")
        w, h = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(h * w * 4), dtype="<f4").copy()
    return data.reshape(h, w)


def load_sintel(data_dir: Path, max_frames: int,
                max_scenes: int = 3) -> List[dict]:
    """Load Sintel scenes for depth evaluation."""
    clean_dir = data_dir / "clean"
    depth_dir = data_dir / "depth"

    if not clean_dir.exists():
        raise FileNotFoundError(f"Sintel clean dir not found: {clean_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Sintel depth dir not found: {depth_dir}")

    scenes = sorted(d.name for d in clean_dir.iterdir() if d.is_dir())[:max_scenes]

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
            "scene": scene,
            "frames": torch.stack(frames),
            "depths": np.array(depths),
        })

    if not result:
        raise FileNotFoundError(f"No usable scenes in {data_dir}")

    return result


def run_inference(model: VGGT, frames: torch.Tensor,
                  dtype: torch.dtype, device: str,
                  warmup: int = 1) -> Tuple[dict, float]:
    """Run inference and return predictions + wall-clock time in seconds."""
    imgs = frames.to(device).float()
    imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    imgs = imgs.unsqueeze(0)

    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else contextlib.nullcontext()

    with torch.no_grad():
        for _ in range(warmup):
            with autocast_ctx:
                _ = model(imgs)
            if device == "cuda":
                torch.cuda.synchronize()

        if device == "cuda":
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            with autocast_ctx:
                preds = model(imgs)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed = start_evt.elapsed_time(end_evt) / 1000.0
        else:
            import time as _time
            t0 = _time.perf_counter()
            with autocast_ctx:
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

    return np.clip((a * pred.astype(np.float64) + b), 1e-9, None).astype(np.float32)


def eval_depth_sequence(pred_depth: np.ndarray,
                        gt_depth: np.ndarray) -> dict:
    """Compute AbsRel and delta_125 for one scene sequence."""
    S = min(len(pred_depth), len(gt_depth))
    abs_rels, deltas = [], []

    for s in range(S):
        gt = gt_depth[s].astype(np.float32)
        pred_t = torch.tensor(pred_depth[s], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pred_r = F.interpolate(pred_t, size=gt.shape, mode="bilinear", align_corners=False).squeeze().numpy()

        mask = (gt > 0.1) & (gt < 1000.0) & np.isfinite(gt)
        if mask.sum() < 100:
            continue

        pred_aligned = _scale_shift_align(pred_r, gt, mask)
        valid = mask & (pred_aligned > 0)
        if valid.sum() < 100:
            continue

        p = pred_aligned[valid]
        g = gt[valid]
        abs_rels.append(float(np.mean(np.abs(p - g) / g)))
        deltas.append(float(np.mean(np.maximum(p / g, g / p) < 1.25)))

    if not abs_rels:
        return {"abs_rel": np.nan, "delta_125": np.nan}

    return {
        "abs_rel": float(np.mean(abs_rels)),
        "delta_125": float(np.mean(deltas)),
    }


def evaluate_sintel_depth(model: VGGT, scenes: List[dict], dtype: torch.dtype,
                         device: str, warmup: int) -> Tuple[dict, float, list]:
    """Evaluate Sintel clean/depth. Returns (metrics, time_s, scene_metrics)."""
    all_abs_rel, all_delta = [], []
    scene_metrics = []
    total_elapsed = 0.0

    for scene_data in scenes:
        frames = scene_data["frames"]
        depths = scene_data["depths"]

        preds, elapsed = run_inference(model, frames, dtype, device, warmup)
        total_elapsed += elapsed

        if "depth" not in preds:
            raise KeyError("Model output missing 'depth'")

        pred_d = preds["depth"]
        pred_d = pred_d.squeeze()
        if pred_d.ndim == 4 and pred_d.shape[0] == 1:
            pred_d = pred_d[0]
        if pred_d.ndim == 3 and pred_d.shape[-1] == 1:
            pred_d = pred_d[:, :, 0]
        if pred_d.ndim != 3:
            raise RuntimeError(f"Unexpected depth shape: {pred_d.shape}")

        metrics = eval_depth_sequence(pred_d, depths)

        scene_metrics.append({
            "scene": scene_data["scene"],
            "frames": int(min(len(pred_d), len(depths))),
            "abs_rel": metrics["abs_rel"],
            "delta_125": metrics["delta_125"],
        })

        if not np.isnan(metrics["abs_rel"]):
            all_abs_rel.append(metrics["abs_rel"])
            all_delta.append(metrics["delta_125"])

    return {
        "abs_rel": float(np.nanmean(all_abs_rel)) if all_abs_rel else np.nan,
        "delta_125": float(np.nanmean(all_delta)) if all_delta else np.nan,
    }, float(total_elapsed), scene_metrics


def run_all_evaluations(model: VGGT, scenes: List[dict],
                       dtype: torch.dtype, device: str,
                       warmup: int, groups: Optional[List[str]]) -> dict:
    configs = build_silence_configs(groups)
    results: Dict[str, dict] = {}

    for cfg_name, cfg in configs.items():
        print(f"\n{'='*60}\nConfig: {cfg_name} ({cfg.label})\n{'='*60}")

        cfg_result = {
            "config": cfg_name,
            "label": cfg.label,
            "frame_layers": cfg.frame_layers,
            "global_layers": cfg.global_layers,
        }

        if cfg.is_empty():
            metrics, elapsed, scene_metrics = evaluate_sintel_depth(model, scenes, dtype, device, warmup)
        else:
            with LayerSilencer(model, cfg):
                metrics, elapsed, scene_metrics = evaluate_sintel_depth(model, scenes, dtype, device, warmup)

        cfg_result["sintel_depth"] = {
            **metrics,
            "time_s": elapsed,
            "scene_metrics": scene_metrics,
        }
        results[cfg_name] = cfg_result

    return results


def _to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        return None if (obj != obj) else obj
    return obj


def save_results_json(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"  results.json: {path}")


def save_results_csv(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for cfg_name, data in results.items():
        task = data.get("sintel_depth", {})
        if not task or "error" in task:
            continue
        for metric in ("abs_rel", "delta_125", "time_s"):
            rows.append({
                "config": cfg_name,
                "task": "sintel_depth",
                "metric": metric,
                "value": task.get(metric, np.nan),
                "frame_layers": str(data.get("frame_layers", [])),
                "global_layers": str(data.get("global_layers", [])),
            })

    path = out_dir / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config", "task", "metric", "value", "frame_layers", "global_layers"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  summary.csv: {path}")


def _safe(results: dict, cfg: str, metric: str) -> float:
    try:
        v = results[cfg]["sintel_depth"][metric]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return float(v)
    except (KeyError, TypeError):
        return np.nan


def plot_range_ablation(results: dict, out_dir: Path) -> None:
    keys = [
        "baseline",
        "range_global_3_4", "range_frame_3_4", "range_both_3_4",
        "range_global_4_8", "range_frame_4_8", "range_both_4_8",
        "range_global_10_16", "range_frame_10_16", "range_both_10_16",
        "range_global_20_23", "range_frame_20_23", "range_both_20_23",
        "range_both_3_4_10_16", "range_global_4_8_20_23",
        "range_keep_global_4_8_14_20",
        "range_keep_frame_1_9",
        "range_keep_global_4_8_14_20_frame_1_9",
        "all_global", "all_frame", "first_half_global", "second_half_global",
    ]
    keys = [k for k in keys if k in results]
    if not keys:
        print("[WARN] No range/boundary configs to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(keys) * 0.75), 5))

    base_abs = _safe(results, "baseline", "abs_rel")
    base_delta = _safe(results, "baseline", "delta_125")

    for ax, metric, is_lower_better in zip(axes, ["abs_rel", "delta_125"], [True, False]):
        vals = [_safe(results, k, metric) for k in keys]
        x = np.arange(len(keys))

        colors = []
        for i, v in enumerate(vals):
            if np.isnan(v):
                colors.append("lightgray")
            elif keys[i] == "baseline":
                colors.append("steelblue")
            elif is_lower_better:
                colors.append("seagreen" if (not np.isnan(base_abs) and v <= base_abs) else "tomato")
            else:
                colors.append("seagreen" if (not np.isnan(base_delta) and v >= base_delta) else "tomato")

        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
        if keys:
            ax.set_xticks(x)
            ax.set_xticklabels(keys, rotation=40, ha="right", fontsize=7)

        if not np.isnan(base_abs if is_lower_better else base_delta):
            base_v = base_abs if is_lower_better else base_delta
            ax.axhline(base_v, color="black", linestyle="--", label="baseline")

        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
        if keys:
            ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = out_dir / "depth_range_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  depth_range_ablation.png: {out}")


def plot_sensitivity_curve(results: dict, out_dir: Path) -> None:
    baseline = _safe(results, "baseline", "abs_rel")
    if np.isnan(baseline):
        print("[WARN] No baseline for sensitivity curve")
        return

    layers, deltas = [], []
    for i in range(24):
        k = f"G{i:02d}"
        if k in results:
            v = _safe(results, k, "abs_rel")
            if not np.isnan(v):
                layers.append(i)
                deltas.append(v - baseline)

    if not layers:
        print("[WARN] No per-layer global config for sensitivity curve")
        return

    layers = np.array(layers)
    deltas = np.array(deltas)
    order = np.argsort(deltas)
    layers = layers[order]
    deltas = deltas[order]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(layers))
    ax.bar(x, deltas, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_title("Depth sensitivity curve: global blocks")
    ax.set_xlabel("Global layer (sorted by abs_rel delta)")
    ax.set_ylabel("AbsRel delta")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "depth_sensitivity_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  depth_sensitivity_curve.png: {out}")


def print_summary_table(results: dict) -> None:
    base_abs = _safe(results, "baseline", "abs_rel")
    base_delta = _safe(results, "baseline", "delta_125")
    if np.isnan(base_abs) and np.isnan(base_delta):
        print("[WARN] Baseline unavailable")
        return

    print("\n" + "=" * 90)
    print("Depth Silencing Summary")
    print("=" * 90)
    print(f"{'Config':<24} {'AbsRel':>10} {'AbsRel change':>14} {'delta_125':>12} {'delta_125 change':>18}")
    print("-" * 90)

    for cfg_name, data in results.items():
        m = data.get("sintel_depth", {})
        abs_v = m.get("abs_rel", np.nan)
        dlt_v = m.get("delta_125", np.nan)
        if np.isnan(abs_v) or np.isnan(dlt_v):
            continue

        da = abs_v - base_abs if not np.isnan(base_abs) else np.nan
        dd = dlt_v - base_delta if not np.isnan(base_delta) else np.nan
        print(f"{cfg_name:<24} {abs_v:>10.4f} {da:>+14.4f} {dlt_v:>12.4f} {dd:>+18.4f}")

    print("=" * 90)


def save_all_plots(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_range_ablation(results, out_dir)
    plot_sensitivity_curve(results, out_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="VGGT Layer Silencing Experiment - Depth Focus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", default="cuda", help="Device to use: cuda / cpu")
    p.add_argument("--model_name", default="facebook/VGGT-1B", help="Model name")
    p.add_argument("--max_frames", type=int, default=16, help="Max frames per scene")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    p.add_argument("--sintel_max_scenes", type=int, default=3, help="Max Sintel scenes")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="If set, uses <data_root>/sintel/training",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: tests/results/depth_silence_test)",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=["baseline", "range", "boundary"],
        choices=["baseline", "per_layer_global", "per_layer_frame", "range", "boundary"],
        help="Config groups to run",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"output: {out_dir}")
    print(f"device: {args.device}")
    print(f"model: {args.model_name}")

    model: VGGT = VGGT.from_pretrained(args.model_name)
    model.eval().to(args.device)

    sintel_dir = Path(args.data_root) / "sintel" / "training" if args.data_root else DEFAULT_SINTEL_DIR

    scenes = load_sintel(sintel_dir, args.max_frames, max_scenes=args.sintel_max_scenes)
    print(f"loaded {len(scenes)} scene(s)")

    results = run_all_evaluations(
        model=model,
        scenes=scenes,
        dtype=DTYPE,
        device=args.device,
        warmup=args.warmup,
        groups=args.groups,
    )

    save_results_json(results, out_dir)
    save_results_csv(results, out_dir)
    save_all_plots(results, out_dir)
    print_summary_table(results)

    print(f"Done. Output saved to {out_dir}")


if __name__ == "__main__":
    main()
