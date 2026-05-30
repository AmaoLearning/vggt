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

C1_ALLOWED_ENERGY_TARGETS = (0.95, 0.90, 0.80, 0.50)

C1_SPATIAL_BAND_TABLES = {
    0.95: {
        "Q": [22, 25, 29, 30, 30, 31, 31, 31, 31, 31, 32, 32, 31, 31, 31, 31, 27, 11, 7, 8, 12, 3, 6, 6],
        "K": [27, 25, 30, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 30, 32, 31, 31, 15, 18, 24, 20, 11, 9],
        "V": [29, 26, 31, 33, 33, 33, 33, 33, 33, 33, 34, 34, 33, 33, 26, 34, 34, 32, 14, 14, 29, 25, 23, 2],
    },
    0.90: {
        "Q": [12, 14, 20, 24, 23, 26, 24, 24, 24, 25, 27, 27, 26, 26, 25, 25, 17, 2, 3, 3, 5, 2, 4, 2],
        "K": [17, 14, 24, 27, 28, 27, 27, 27, 27, 28, 28, 28, 28, 27, 23, 27, 26, 25, 5, 7, 13, 8, 7, 4],
        "V": [22, 16, 26, 29, 28, 29, 29, 30, 29, 30, 31, 30, 30, 30, 15, 31, 32, 28, 2, 4, 21, 12, 12, 1],
    },
    0.80: {
        "Q": [7, 9, 9, 12, 11, 14, 12, 12, 13, 12, 16, 17, 15, 15, 13, 13, 3, 1, 1, 1, 2, 1, 2, 1],
        "K": [8, 7, 12, 17, 18, 17, 16, 17, 17, 18, 18, 19, 18, 17, 11, 17, 13, 12, 2, 2, 5, 2, 3, 2],
        "V": [11, 3, 15, 21, 20, 21, 22, 22, 21, 23, 25, 24, 23, 23, 2, 24, 26, 18, 1, 1, 10, 2, 3, 1],
    },
    0.50: {
        "Q": [1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        "K": [2, 2, 2, 2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 1, 1, 0, 0, 1, 1, 1, 1],
        "V": [2, 2, 2, 4, 3, 4, 4, 5, 4, 5, 6, 4, 3, 3, 1, 2, 4, 3, 1, 1, 1, 1, 1, 1],
    },
}


@dataclass
class CompressionConfig:
    """
    压缩机制的总配置类。
    每个 CompressionConfig 对应一种压缩策略，在 apply_compression_hooks() 中使用。

    mechanism: "A" | "B" | "C" | "C1" | "D2" | "E" | "F" | "A+C" | "E+C"
        A:   KV 时序步长剪枝 + Q 组内 token merging（Spark3R 基线）
        B:   KV 时序 DCT 压缩
        C:   KV 空间 2D-DCT + 异常值保留（叠加在 A/B 之上）
        C1:  依据离线 DCT 统计表的固定空间频带截断（Q/K/V 可分支配置）
        D2:  相邻帧小窗口局部冗余对删除
        E:   DCT 代表元 Q merging（替代机制 A 的 Q 路径）
        F:   FastVGGT (ICLR 2026) bipartite token merging，对 Global Attention 全序列（Q/K/V）做 merge/unmerge
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
    # NOTE: always_keep_special_tokens 已移除。Spark3R 论文未提及对 special tokens 做
    # 任何豁免，机制 A 的 _build_dst_mask 现在对所有位置（含 camera/register）
    # 采用相同的循环偏移规则。

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

    # ── 方案 C1 的固定频带表参数 ─────────────────────────────────────────
    c1_energy_target: float = 0.90         # 仅允许 {0.95, 0.90, 0.80, 0.50}
    c1_enable_q_compression: bool = False  # 首版建议先做 KV-only，Q 压缩放在第二阶段
    c1_reconstruct_mode: str = "lowres_idct"  # "lowres_idct" | "coeff_crop"
    c1_q_unmerge_mode: str = "bilinear"   # Q 路径恢复到 37×37 的上采样方式

    # ── 方案 D2 的局部匹配参数 ───────────────────────────────────────────
    d2_window_radius: int = 2              # 小窗口半径，2 表示 5×5 邻域
    d2_drop_ratio: float = 0.50            # 每个相邻帧对计划删除的冗余 pair 比例
    d2_pair_stride: int = 1                # 使用 (t,t+1) 还是 (t,t+2) 等相邻帧对
    d2_similarity_policy: str = "lowest"  # "lowest"=删最低相似配对（保留可靠对应关系，用于位姿估计）
    d2_apply_to_q: bool = False            # 首版建议仅删 KV，避免 Q 路径过敏感

    # ── 方案 D3 的 K 相似度阈值参数 ──────────────────────────────────────
    d3_threshold: float = 0.90            # K 余弦相似度超过此值则删除（I/P 帧式压缩）
    d3_reference: str = "adjacent"        # "adjacent"=s vs s-1；"first"=所有帧 vs 帧 0

    # ── 方案 F（FastVGGT Token Merging）参数 ────────────────────────────────
    f_merge_ratio: float = 0.9       # 目标合并比例（占全序列长度 N）；实际 = min(r, num_src)
    f_start_layer: int = 0           # 从第几个 global block 开始启用 merging（0=全部层）
    f_region_stride: int = 2         # region-based dst 采样步长 sy=sx（FastVGGT 默认 2）
    f_salient_stride: int = 10       # 每 K 个 patch token 取 1 个 salient（≈10% per frame）

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



def get_c1_spatial_band_upper(layer_idx: int, branch: str, target_ratio: float) -> int:
    """根据离线统计表返回 C1 的空间频带上限。"""
    matched_ratio = None
    for allowed in C1_ALLOWED_ENERGY_TARGETS:
        if abs(float(target_ratio) - allowed) < 1e-6:
            matched_ratio = allowed
            break
    if matched_ratio is None:
        raise ValueError(
            f"Unsupported c1_energy_target={target_ratio}, expected one of {C1_ALLOWED_ENERGY_TARGETS}."
        )
    branch = branch.upper()
    if branch not in ("Q", "K", "V"):
        raise ValueError(f"Unsupported branch '{branch}', expected one of ('Q', 'K', 'V').")
    return C1_SPATIAL_BAND_TABLES[matched_ratio][branch][layer_idx]



