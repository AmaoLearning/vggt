from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

# VGGT 的敏感层（来自 Spark3R Figure 5，>1.05x 降质阈值）
VGGT_SENSITIVE_LAYERS = frozenset(range(11, 17))  # layers 11-16

# 各层的帧间相似度分区（来自我们的实验）
# 浅层（0-10）：高帧间相似度，适合激进时序压缩
# 中层（11-16）：中等相似度 + 高敏感度，双重保守
# 深层（17-23）：低帧间相似度，用空间压缩而非时序
LAYER_ZONE = {
    "shallow": range(0, 11),    # 高帧间相似，低敏感
    "sensitive": range(11, 17), # 中帧间相似，高敏感
    "deep": range(17, 24),      # 低帧间相似，低敏感
}


@dataclass
class CompressionConfig:
    """
    压缩机制的总配置类。
    每个 CompressionConfig 对应一种压缩策略，在 apply_compression_hooks() 中使用。

    mechanism: "A" | "B" | "C" | "E" | "A+C" | "E+C"
        A:   KV 时序步长剪枝 + Q 组内 token merging（Spark3R 基线）
        B:   KV 时序 DCT 压缩
        C:   KV 空间 2D-DCT + 异常值保留（叠加在 A/B 之上）
        E:   DCT 代表元 Q merging（替代机制 A 的 Q 路径）
        A+C: 机制 A（KV时序） + 机制 C（KV空间），联合压缩
        E+C: 机制 E（Q路径） + 机制 A KV路径 + 机制 C（KV空间）
    """
    mechanism: str = "A"

    # ── 机制 A / E 的 Q 路径参数 ──────────────────────────────────────────
    enable_q_compression: bool = True   # 是否启用 Q 路径压缩（False = 仅压缩 KV，便于与机制 B/C 公平对比）
    q_group_size: int = 20          # 每组帧数 G（Spark3R 默认值）
    q_reduction_factor_table: Dict[str, int] = field(default_factory=lambda: {
        # S → rQ 的映射（来自 Spark3R Eq.4）
        # 由 get_rQ(S) 使用
    })

    # ── 机制 A 的 KV 路径参数 ─────────────────────────────────────────────
    kv_base_reduction_factor: Optional[int] = None   # None = 自动按 S 计算
    kv_sensitive_multiplier: float = 1.0   # 敏感层的 rKV 乘数（默认不变）
    kv_insensitive_multiplier: float = 3.0 # 非敏感层的 rKV 乘数（Spark3R 默认 l=3）
    always_keep_special_tokens: bool = True # camera + register 始终保留完整帧

    # ── 机制 B 的 DCT 参数 ────────────────────────────────────────────────
    temporal_keep_ratio_by_zone: Dict[str, float] = field(default_factory=lambda: {
        "shallow": 0.25,    # 激进：帧间高相似，低频足够
        "sensitive": 0.70,  # 保守：敏感层
        "deep": 0.45,       # 中等
    })

    # ── 机制 C 的空间 DCT 参数 ───────────────────────────────────────────
    spatial_low_freq_ratio: float = 0.30   # 2D DCT 保留的低频区域比例（面积比）
    spatial_outlier_ratio: float = 0.10    # 在低频基础上额外保留的异常值比例
    spatial_keep_special: bool = True      # special tokens 始终保留

    # ── 通用 ─────────────────────────────────────────────────────────────
    only_global: bool = True    # 仅压缩 global_blocks（推荐）
    min_frames_to_compress: int = 8   # S < 此值时跳过压缩（短序列无需压缩）


def get_rQ(S: int) -> int:
    """按 Spark3R Eq.4 计算 Q 路径的 reduction factor"""
    if S <= 100:
        return 1
    elif S <= 300:
        return 2
    elif S <= 500:
        return 3
    else:
        return 4


def get_rKV_base(S: int) -> int:
    """按 Spark3R Eq.5 计算 KV 路径的基础 reduction factor"""
    if S <= 100:
        return 1
    else:
        return max(1, round(S / 40))


def get_layer_rKV(S: int, layer_idx: int, multiplier_insensitive: float = 3.0) -> int:
    """
    计算某一层的实际 KV reduction factor（层自适应调度）。
    - 敏感层（11-16）：base rKV
    - 非敏感层：base rKV × multiplier_insensitive（默认 3×）
    """
    base = get_rKV_base(S)
    if layer_idx in VGGT_SENSITIVE_LAYERS:
        return base
    else:
        return max(1, round(base * multiplier_insensitive))


def get_temporal_keep_ratio(layer_idx: int, config: "CompressionConfig") -> float:
    """根据层编号返回机制 B 的 keep_ratio"""
    if layer_idx in LAYER_ZONE["shallow"]:
        return config.temporal_keep_ratio_by_zone["shallow"]
    elif layer_idx in LAYER_ZONE["sensitive"]:
        return config.temporal_keep_ratio_by_zone["sensitive"]
    else:
        return config.temporal_keep_ratio_by_zone["deep"]
