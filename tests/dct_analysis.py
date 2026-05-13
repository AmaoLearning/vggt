#!/usr/bin/env python3
"""
VGGT Global Attention Q/K/V Patch Token DCT analysis.

Usage:
    python tests/dct_analysis.py \
        --model_path facebook/VGGT-1B \
        --image_dir /path/to/images \
        --output_dir tests/dct_analysis_output \
        --layers all \
        --num_frames 20 \
        --device cuda
"""

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import curve_fit
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vggt.models.vggt import VGGT
from vggt.compression.base import CompressionContext
from vggt.compression.utils import dct1d, dct2d


IMG_SIZE = 518
PATCH_SIZE = 14
PATCH_H = IMG_SIZE // PATCH_SIZE
PATCH_W = IMG_SIZE // PATCH_SIZE
N_PATCHES = PATCH_H * PATCH_W
SPECIAL_TOKENS = 5
TOKENS_PER_FRAME = SPECIAL_TOKENS + N_PATCHES
N_GLOBAL_LAYERS = 24
ENERGY_RATIO_TARGETS = [0.95, 0.90, 0.80, 0.50]
COLORS = {"Q": "#1f77b4", "K": "#ff7f0e", "V": "#2ca02c"}


class AnalysisHook:
    """Capture RoPE-applied Q/K/V from the compression hook entrypoint."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def __call__(self, q, k, v, ctx):
        if ctx.is_global:
            self.q = q.detach().float().cpu()
            self.k = k.detach().float().cpu()
            self.v = v.detach().float().cpu()
        return q, k, v, None

    def get(self, which: str) -> Optional[torch.Tensor]:
        return {"Q": self.q, "K": self.k, "V": self.v}[which]

    def clear(self) -> None:
        self.q = None
        self.k = None
        self.v = None


def install_hooks(model: VGGT, layers: List[int], num_frames: int) -> Dict[int, AnalysisHook]:
    """Install analysis hooks on selected global attention blocks."""
    hooks: Dict[int, AnalysisHook] = {}
    for layer_idx in layers:
        block = model.aggregator.global_blocks[layer_idx]
        hook = AnalysisHook(layer_idx)
        block.attn._compression_hook = hook
        if block.attn._compression_ctx is None:
            block.attn._compression_ctx = CompressionContext(
                is_global=True,
                S=num_frames,
                P=TOKENS_PER_FRAME,
                layer_idx=layer_idx,
                total_layers=N_GLOBAL_LAYERS,
                special_tokens=SPECIAL_TOKENS,
            )
        hooks[layer_idx] = hook
    return hooks


def remove_hooks(model: VGGT, layers: List[int]) -> None:
    for layer_idx in layers:
        block = model.aggregator.global_blocks[layer_idx]
        block.attn._compression_hook = None
        block.attn._compression_ctx = None


def parse_layers(layers_arg: str) -> List[int]:
    if layers_arg == "all":
        return list(range(N_GLOBAL_LAYERS))

    layers = []
    for item in layers_arg.split(","):
        item = item.strip()
        if not item:
            continue
        layer_idx = int(item)
        if layer_idx < 0 or layer_idx >= N_GLOBAL_LAYERS:
            raise ValueError(f"Invalid layer index {layer_idx}; expected 0..{N_GLOBAL_LAYERS - 1}")
        layers.append(layer_idx)

    if not layers:
        raise ValueError("No layers selected")
    return sorted(set(layers))


def load_images(image_dir: str, num_frames: int = 20) -> torch.Tensor:
    """
    Load images as [S, 3, 518, 518] in [0, 1].

    VGGT normalizes internally in Aggregator.forward(), so this loader must not
    apply ImageNet normalization again.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    image_dir_path = Path(image_dir)
    paths = sorted([path for path in image_dir_path.iterdir() if path.suffix in exts])
    if len(paths) < num_frames:
        raise ValueError(f"Found only {len(paths)} images in '{image_dir}', expected at least {num_frames}")

    frames = []
    for path in paths[:num_frames]:
        image = Image.open(path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(image_np).permute(2, 0, 1))

    return torch.stack(frames, dim=0)


def extract_patch_tokens(qkv_full: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Extract patch tokens as [H, S, PATCH_H, PATCH_W, D]."""
    _, num_heads, num_tokens, head_dim = qkv_full.shape
    expected_tokens = num_frames * TOKENS_PER_FRAME
    if num_tokens != expected_tokens:
        raise ValueError(f"Expected N={expected_tokens}, got {num_tokens}")

    tokens = qkv_full[0].reshape(num_heads, num_frames, TOKENS_PER_FRAME, head_dim)
    patch = tokens[:, :, SPECIAL_TOKENS:, :]
    return patch.reshape(num_heads, num_frames, PATCH_H, PATCH_W, head_dim)


def extract_patch_tokens_flat(qkv_full: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Extract patch tokens as [H, S, N_PATCHES, D]."""
    tokens = qkv_full[0].reshape(qkv_full.shape[1], num_frames, TOKENS_PER_FRAME, qkv_full.shape[-1])
    return tokens[:, :, SPECIAL_TOKENS:, :]


def compute_spatial_2d_dct_energy(patch_spatial: torch.Tensor) -> torch.Tensor:
    """Compute per-frame 2D-DCT energy maps: [S, PATCH_H, PATCH_W]."""
    dct_coeff = dct2d(patch_spatial, dims=(-3, -2))
    return (dct_coeff ** 2).mean(dim=(0, -1))


def compute_temporal_1d_dct_energy(patch_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-patch temporal DCT energy and mean spectrum."""
    temporal = patch_flat.permute(0, 2, 1, 3)
    dct_coeff = dct1d(temporal, dim=-2)
    energy = (dct_coeff ** 2).mean(dim=(0, -1))
    mean_spectrum = energy.mean(dim=0)
    return energy, mean_spectrum


def _exp_decay(k, a, b):
    return a * np.exp(-b * k)


def compute_spatial_stats(energy: torch.Tensor) -> dict:
    energy_np = energy.numpy()
    dc_values = energy_np[:, 0, 0]
    ac_total = energy_np.reshape(energy_np.shape[0], -1).sum(axis=1) - dc_values
    mean_energy = energy_np.mean(axis=0)
    threshold = mean_energy[0, 0] * 0.01
    effective_bandwidth = int((mean_energy > threshold).sum())
    spatial_concentration = float(
        energy_np.reshape(energy_np.shape[0], -1).max(axis=1).mean()
        / (energy_np.reshape(energy_np.shape[0], -1).sum(axis=1).mean() + 1e-8)
    )

    return {
        "dc_mean": float(dc_values.mean()),
        "dc_std": float(dc_values.std()),
        "dc_max": float(dc_values.max()),
        "dc_min": float(dc_values.min()),
        "ac_total_mean": float(ac_total.mean()),
        "dc_ac_ratio": float(dc_values.mean() / (ac_total.mean() + 1e-8)),
        "effective_bandwidth": effective_bandwidth,
        "spatial_concentration": spatial_concentration,
    }


def compute_temporal_stats(energy: torch.Tensor, mean_spectrum: torch.Tensor) -> dict:
    energy_np = energy.numpy()
    spectrum_np = mean_spectrum.numpy()
    dc_values = energy_np[:, 0]
    cumulative = np.cumsum(spectrum_np) / (spectrum_np.sum() + 1e-8)
    effective_k = int(np.searchsorted(cumulative, 0.90)) + 1

    ks = np.arange(1, len(spectrum_np))
    ac_spectrum = spectrum_np[1:]
    try:
        popt, _ = curve_fit(
            _exp_decay,
            ks,
            ac_spectrum + 1e-12,
            p0=[ac_spectrum[0] + 1e-6, 0.1],
            maxfev=2000,
        )
        decay_rate = float(popt[1])
    except Exception:
        decay_rate = float("nan")

    return {
        "temporal_dc_mean": float(dc_values.mean()),
        "temporal_dc_std": float(dc_values.std()),
        "temporal_dc_max": float(dc_values.max()),
        "temporal_dc_min": float(dc_values.min()),
        "temporal_low_freq_ratio": float(energy_np[:, :3].sum() / (energy_np.sum() + 1e-8)),
        "temporal_effective_k": effective_k,
        "temporal_ac_decay_rate": decay_rate,
        "patch_homogeneity": float(1.0 - dc_values.std() / (dc_values.mean() + 1e-8)),
    }


def compute_spatial_energy_retention_curve(energy: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute retained-energy ratio while shrinking the kept 2D low-frequency block
    from full 37x37 down to 1x1 DC.

    Returns:
        sizes_desc: [37] retained square size r, ordered 37 -> 1
        retention_desc: [37] retained energy ratio for top-left r x r block
    """
    mean_energy = energy.mean(dim=0).numpy()
    total_energy = float(mean_energy.sum()) + 1e-8
    sizes = np.arange(1, PATCH_H + 1)
    retention = np.array([
        float(mean_energy[:size, :size].sum() / total_energy)
        for size in sizes
    ])
    return sizes[::-1], retention[::-1]


def compute_temporal_energy_retention_curve(mean_spectrum: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute retained-energy ratio while shrinking the kept temporal low-frequency
    coefficients from all 20 down to only the DC component.

    Returns:
        keep_desc: [S] retained coefficient count k, ordered S -> 1
        retention_desc: [S] retained energy ratio for first k coefficients
    """
    spectrum = mean_spectrum.numpy()
    total_energy = float(spectrum.sum()) + 1e-8
    keep_counts = np.arange(1, len(spectrum) + 1)
    retention = np.array([
        float(spectrum[:keep_count].sum() / total_energy)
        for keep_count in keep_counts
    ])
    return keep_counts[::-1], retention[::-1]


def compute_band_upper_limits_from_retention(
    kept_sizes_desc: np.ndarray,
    retention_desc: np.ndarray,
    targets: List[float],
) -> List[int]:
    """
    Convert a retention curve into the minimal retained band upper limit that
    reaches each target energy ratio.

    For 2D, kept_sizes_desc is r in [37..1] and the returned upper limit is r-1
    in [36..0]. For 1D, kept_sizes_desc is k in [20..1] and the returned upper
    limit is k-1 in [19..0].
    """
    kept_sizes_asc = kept_sizes_desc[::-1]
    retention_asc = retention_desc[::-1]
    upper_limits = []
    for target in targets:
        hit_indices = np.where(retention_asc >= target)[0]
        if len(hit_indices) == 0:
            upper_limits.append(int(kept_sizes_asc[-1] - 1))
            continue
        kept_size = int(kept_sizes_asc[hit_indices[0]])
        upper_limits.append(kept_size - 1)
    return upper_limits


def plot_spatial_2d_dct(energy: torch.Tensor, layer_idx: int, which: str, out_path: Path) -> None:
    """Plot per-frame spatial 2D-DCT heatmaps."""
    num_frames = energy.shape[0]
    energy_log = torch.log1p(energy).numpy()
    vmin, vmax = energy_log.min(), energy_log.max()

    figure, axes = plt.subplots(1, num_frames, figsize=(num_frames * 2.2, 2.5), squeeze=False)
    figure.suptitle(f"Layer {layer_idx:02d} · {which} Spatial 2D-DCT Energy (log1p)", fontsize=11)
    for frame_idx in range(num_frames):
        axis = axes[0, frame_idx]
        image = axis.imshow(energy_log[frame_idx], cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        axis.set_title(f"f{frame_idx}", fontsize=7)
        axis.axis("off")
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    figure.tight_layout()
    figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def plot_spatial_dc_ac_bar(energy: torch.Tensor, layer_idx: int, which: str, out_path: Path) -> None:
    """Plot DC and total AC energy per frame."""
    num_frames = energy.shape[0]
    energy_np = energy.numpy()
    dc_values = energy_np[:, 0, 0]
    ac_values = energy_np.reshape(num_frames, -1).sum(axis=1) - dc_values
    x_axis = np.arange(num_frames)

    figure, axes = plt.subplots(2, 1, figsize=(max(8, num_frames * 0.6), 5), sharex=True)
    axes[0].bar(x_axis, dc_values, color=COLORS[which], alpha=0.85)
    axes[0].set_ylabel("DC energy")
    axes[0].set_title(f"Layer {layer_idx:02d} · {which} DC energy per frame")
    axes[1].bar(x_axis, ac_values, color=COLORS[which], alpha=0.65)
    axes[1].set_ylabel("AC total energy")
    axes[1].set_title("AC total energy per frame")
    axes[1].set_xlabel("Frame index")
    figure.tight_layout()
    figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def plot_temporal_freq_spatial(freq_spatial: np.ndarray, layer_idx: int, which: str, out_path: Path) -> None:
    """Plot 20 spatial maps, one per temporal DCT frequency."""
    num_freqs = freq_spatial.shape[0]
    nrows, ncols = 4, 5
    figure, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    figure.suptitle(
        f"Layer {layer_idx:02d} · {which} Temporal 1D-DCT Energy per Frequency (spatial)",
        fontsize=11,
    )

    for freq_idx in range(num_freqs):
        axis = axes[freq_idx // ncols, freq_idx % ncols]
        vmax = max(float(freq_spatial[freq_idx].max()), 1e-9)
        axis.imshow(freq_spatial[freq_idx], cmap="plasma", vmin=0, vmax=vmax, aspect="equal")
        axis.set_title(f"k={freq_idx}" + (" (DC)" if freq_idx == 0 else ""), fontsize=7)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def plot_temporal_patch_freq(energy: np.ndarray, layer_idx: int, which: str, out_path: Path) -> None:
    """Plot patch x temporal-frequency heatmap."""
    figure, axis = plt.subplots(figsize=(8, 10))
    vmax = max(float(np.percentile(energy, 99)), 1e-9)
    image = axis.imshow(
        energy,
        cmap="magma",
        aspect="auto",
        vmin=0,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
    )
    axis.set_xlabel("Temporal DCT freq k", fontsize=10)
    axis.set_ylabel("Patch index (raster scan)", fontsize=10)
    axis.set_title(f"Layer {layer_idx:02d} · {which} Temporal 1D-DCT Energy [patch × freq]", fontsize=11)
    axis.set_xticks(range(energy.shape[1]))
    axis.set_xticklabels([str(freq_idx) for freq_idx in range(energy.shape[1])], fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.02, pad=0.02)
    figure.tight_layout()
    figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def plot_temporal_spectrum(
    mean_spectrum: np.ndarray,
    layer_idx: int,
    which: str,
    stats: dict,
    out_path: Path,
) -> None:
    """Plot mean temporal spectrum with optional exponential fit."""
    num_freqs = len(mean_spectrum)
    freqs = np.arange(num_freqs)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(freqs, mean_spectrum, "o-", color=COLORS[which], linewidth=1.5, label=f"{which} mean spectrum")

    decay_rate = stats.get("temporal_ac_decay_rate", float("nan"))
    if not np.isnan(decay_rate) and num_freqs > 1:
        start_value = max(float(mean_spectrum[1]), 1e-12)
        fit_freqs = np.linspace(1, num_freqs - 1, 200)
        fit_values = start_value * np.exp(-decay_rate * (fit_freqs - 1))
        axis.plot(fit_freqs, fit_values, "--", color="gray", linewidth=1, label=f"exp fit b={decay_rate:.3f}")

    axis.axvline(x=0, color="red", linestyle=":", alpha=0.6, label="DC (k=0)")
    axis.set_xlabel("Temporal DCT frequency k", fontsize=10)
    axis.set_ylabel("Mean energy (over all patches)", fontsize=10)
    axis.set_title(f"Layer {layer_idx:02d} · {which} Mean Temporal DCT Spectrum", fontsize=11)
    axis.legend(fontsize=8)
    axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def plot_spatial_energy_retention(
    sizes_desc: np.ndarray,
    retention_desc: np.ndarray,
    layer_idx: int,
    which: str,
    out_path: Path,
) -> None:
    """Plot spatial retained-energy ratio while reducing 2D-DCT resolution."""
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(sizes_desc, retention_desc, "o-", color=COLORS[which], linewidth=1.5)
    axis.set_xlabel("Retained low-frequency block size r (top-left r×r)", fontsize=10)
    axis.set_ylabel("Retained energy / total energy", fontsize=10)
    axis.set_title(f"Layer {layer_idx:02d} · {which} Spatial Energy Retention", fontsize=11)
    axis.set_xlim(PATCH_H, 1)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def plot_temporal_energy_retention(
    keep_desc: np.ndarray,
    retention_desc: np.ndarray,
    layer_idx: int,
    which: str,
    out_path: Path,
) -> None:
    """Plot temporal retained-energy ratio while reducing 1D-DCT coefficients."""
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(keep_desc, retention_desc, "o-", color=COLORS[which], linewidth=1.5)
    axis.set_xlabel("Retained temporal low-frequency coefficient count k", fontsize=10)
    axis.set_ylabel("Retained energy / total energy", fontsize=10)
    axis.set_title(f"Layer {layer_idx:02d} · {which} Temporal Energy Retention", fontsize=11)
    axis.set_xlim(len(keep_desc), 1)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def plot_summary_trend(all_stats: Dict[int, Dict[str, dict]], metric_key: str, ylabel: str, title: str, out_path: Path) -> None:
    """Plot a cross-layer trend for one metric."""
    layers = sorted(all_stats.keys())
    figure, axis = plt.subplots(figsize=(10, 4))
    for which in ["Q", "K", "V"]:
        values = [all_stats[layer_idx][which].get(metric_key, float("nan")) for layer_idx in layers]
        axis.plot(layers, values, "o-", color=COLORS[which], label=which, linewidth=1.5, markersize=4)
    axis.set_xlabel("Layer index", fontsize=10)
    axis.set_ylabel(ylabel, fontsize=10)
    axis.set_title(title, fontsize=11)
    axis.legend(fontsize=9)
    axis.set_xticks(layers)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def analyze_layer(layer_idx: int, hook: AnalysisHook, num_frames: int, out_dir: Path) -> Dict[str, dict]:
    """Run the full analysis for one layer and save all artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_stats: Dict[str, dict] = {}

    for which in ["Q", "K", "V"]:
        raw = hook.get(which)
        if raw is None:
            print(f"  [WARN] Layer {layer_idx:02d} {which}: no tensor captured, skipping")
            continue

        patch_spatial = extract_patch_tokens(raw, num_frames)
        patch_flat = extract_patch_tokens_flat(raw, num_frames)

        spatial_energy = compute_spatial_2d_dct_energy(patch_spatial)
        spatial_stats = compute_spatial_stats(spatial_energy)
        spatial_keep_sizes, spatial_retention = compute_spatial_energy_retention_curve(spatial_energy)
        spatial_band_upper_limits = compute_band_upper_limits_from_retention(
            spatial_keep_sizes,
            spatial_retention,
            ENERGY_RATIO_TARGETS,
        )
        plot_spatial_2d_dct(spatial_energy, layer_idx, which, out_dir / f"spatial_2d_dct_{which}.png")
        plot_spatial_dc_ac_bar(spatial_energy, layer_idx, which, out_dir / f"spatial_dc_ac_bar_{which}.png")
        plot_spatial_energy_retention(
            spatial_keep_sizes,
            spatial_retention,
            layer_idx,
            which,
            out_dir / f"spatial_energy_retention_{which}.png",
        )

        temporal_energy, mean_spectrum = compute_temporal_1d_dct_energy(patch_flat)
        temporal_stats = compute_temporal_stats(temporal_energy, mean_spectrum)
        temporal_keep_counts, temporal_retention = compute_temporal_energy_retention_curve(mean_spectrum)
        temporal_band_upper_limits = compute_band_upper_limits_from_retention(
            temporal_keep_counts,
            temporal_retention,
            ENERGY_RATIO_TARGETS,
        )
        freq_spatial = temporal_energy.T.reshape(num_frames, PATCH_H, PATCH_W).numpy()
        plot_temporal_freq_spatial(freq_spatial, layer_idx, which, out_dir / f"temporal_freq_spatial_{which}.png")
        plot_temporal_patch_freq(temporal_energy.numpy(), layer_idx, which, out_dir / f"temporal_patch_freq_{which}.png")
        plot_temporal_spectrum(
            mean_spectrum.numpy(),
            layer_idx,
            which,
            temporal_stats,
            out_dir / f"temporal_spectrum_{which}.png",
        )
        plot_temporal_energy_retention(
            temporal_keep_counts,
            temporal_retention,
            layer_idx,
            which,
            out_dir / f"temporal_energy_retention_{which}.png",
        )

        layer_stats[which] = {
            **spatial_stats,
            **temporal_stats,
            "energy_ratio_targets": list(ENERGY_RATIO_TARGETS),
            "spatial_band_upper_limits": spatial_band_upper_limits,
            "temporal_band_upper_limits": temporal_band_upper_limits,
        }
        print(
            f"  Layer {layer_idx:02d} {which} "
            f"dc_ac_ratio={spatial_stats['dc_ac_ratio']:.3f} "
            f"temporal_low_freq_ratio={temporal_stats['temporal_low_freq_ratio']:.3f} "
            f"effective_k={temporal_stats['temporal_effective_k']} "
            f"spatial_bands={spatial_band_upper_limits} "
            f"temporal_bands={temporal_band_upper_limits}"
        )

    hook.clear()
    return layer_stats


def save_summary(all_stats: Dict[int, Dict[str, dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_summary_trend(
        all_stats,
        "dc_ac_ratio",
        "DC / AC energy ratio",
        "Spatial DC/AC Ratio across Layers (Q/K/V)",
        out_dir / "dc_ac_ratio_across_layers.png",
    )
    plot_summary_trend(
        all_stats,
        "temporal_low_freq_ratio",
        "Low-freq (k<=2) energy / total",
        "Temporal Low-Freq Ratio across Layers",
        out_dir / "temporal_low_freq_ratio.png",
    )
    plot_summary_trend(
        all_stats,
        "temporal_effective_k",
        "# DCT freqs for 90% energy",
        "Temporal Effective-k (90% energy) across Layers",
        out_dir / "temporal_effective_k.png",
    )
    plot_summary_trend(
        all_stats,
        "temporal_ac_decay_rate",
        "Exponential decay rate b",
        "Temporal AC Decay Rate across Layers",
        out_dir / "temporal_ac_decay_rate.png",
    )

    rows = []
    for layer_idx in sorted(all_stats.keys()):
        for which in ["Q", "K", "V"]:
            row = {"layer": layer_idx, "which": which}
            row.update(all_stats[layer_idx].get(which, {}))
            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        csv_path = out_dir / "stats_table.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Summary] stats saved to {csv_path}")

    json_path = out_dir / "stats_table.json"
    json_stats = {
        str(layer_idx): {which: all_stats[layer_idx].get(which, {}) for which in ["Q", "K", "V"]}
        for layer_idx in sorted(all_stats.keys())
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_stats, handle, indent=2)
    print(f"[Summary] stats saved to {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT Global Attention DCT Analysis")
    parser.add_argument("--model_path", type=str, default="facebook/VGGT-1B", help="HuggingFace model ID or local path")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with at least num_frames images")
    parser.add_argument("--output_dir", type=str, default="tests/results/dct_analysis")
    parser.add_argument("--layers", type=str, default="all", help="'all' or comma-separated indices such as '0,6,12,18,23'")
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    if args.num_frames <= 0:
        raise ValueError("--num_frames must be positive")

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

    output_root = Path(args.output_dir)
    print(f"Analyzing layers: {layers}")
    print(f"Loading model: {args.model_path}")
    model = VGGT.from_pretrained(args.model_path)
    model.eval().to(device)

    print(f"Loading {args.num_frames} frames from {args.image_dir}")
    images = load_images(args.image_dir, args.num_frames).unsqueeze(0).to(device)

    hooks = install_hooks(model, layers, args.num_frames)
    print(f"Installed {len(hooks)} analysis hooks")

    print("Running forward pass...")
    autocast_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad():
        with autocast_context:
            _ = model(images)

    all_stats: Dict[int, Dict[str, dict]] = {}
    for layer_idx in tqdm(layers, desc="Analyzing layers"):
        print(f"\n[Layer {layer_idx:02d}]")
        layer_stats = analyze_layer(layer_idx, hooks[layer_idx], args.num_frames, output_root / f"layer_{layer_idx:02d}")
        all_stats[layer_idx] = layer_stats

    print("\nGenerating summary plots...")
    save_summary(all_stats, output_root / "summary")
    remove_hooks(model, layers)
    print(f"\nAnalysis complete. Results saved to: {output_root.resolve()}")


if __name__ == "__main__":
    main()