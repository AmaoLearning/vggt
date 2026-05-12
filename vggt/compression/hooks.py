import copy
from .base import CompressionContext
from .config import CompressionConfig, get_layer_rKV
from .mechanism_a import TemporalStridePruning
from .mechanism_b import TemporalDCTKVCompression
from .mechanism_c import Spatial2DDCTCompression
from .mechanism_e import QueryDCTMerging


def apply_compression_hooks(model, config: CompressionConfig) -> None:
    """
    将指定的压缩 hook 挂载到模型的 global_blocks 中的每个 Attention 模块。

    Args:
        model: VGGT 模型实例（包含 model.aggregator）
        config: CompressionConfig 实例
    """
    aggregator = model.aggregator if hasattr(model, "aggregator") else model

    # 根据 mechanism 选择 hook
    mechanism = config.mechanism.upper()
    if mechanism == "A":
        hook = TemporalStridePruning(config)
    elif mechanism == "B":
        hook = TemporalDCTKVCompression(config)
    elif mechanism == "C":
        hook = Spatial2DDCTCompression(config)
    elif mechanism == "E":
        hook = QueryDCTMerging(config)
    elif mechanism == "A+C":
        hook = CombinedHook(
            TemporalStridePruning(config),
            Spatial2DDCTCompression(config),
        )
    elif mechanism == "E+C":
        hook = CombinedHook(
            QueryDCTMerging(config),
            Spatial2DDCTCompression(config),
        )
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    # 挂载到每个 global_block 的 Attention 模块
    for i, block in enumerate(aggregator.global_blocks):
        attn = block.attn
        attn._compression_hook = hook
        # 初始化 ctx（S, P, layer_idx 将在每次 forward 前动态更新）
        attn._compression_ctx = CompressionContext(
            is_global=True,
            S=1,       # 占位，动态更新
            P=1374,    # 默认值
            layer_idx=i,
            total_layers=len(aggregator.global_blocks),
            special_tokens=aggregator.patch_start_idx,
        )

    print(
        f"[Compression] Mechanism '{mechanism}' applied to "
        f"{len(aggregator.global_blocks)} global attention blocks."
    )


def remove_compression_hooks(model) -> None:
    """移除所有压缩 hook（恢复原始 VGGT 行为）"""
    aggregator = model.aggregator if hasattr(model, "aggregator") else model
    for block in aggregator.global_blocks:
        block.attn._compression_hook = None
        block.attn._compression_ctx = None
    print("[Compression] All hooks removed.")


class CombinedHook:
    """
    顺序组合两个 KVReductionHook：先执行 hook1（时序压缩 + Q），再执行 hook2（空间压缩）。
    hook1 处理 Q 路径；hook2 仅在 hook1 的 k_new, v_new 基础上进一步压缩 K/V 空间维。
    """

    def __init__(self, hook1, hook2):
        self.hook1 = hook1  # 负责 Q 路径 + KV 时序
        self.hook2 = hook2  # 负责 KV 空间

    def __call__(self, q, k, v, ctx):
        # Step 1: 时序压缩（含 Q 路径）
        q, k, v, unmerge_fn = self.hook1(q, k, v, ctx)

        # Step 2: 空间压缩（仅 KV，Q 不变）
        # 此时 k, v 的 S 维已被时序剪枝压缩，需更新 ctx.S 为压缩后的帧数
        rKV = get_layer_rKV(ctx.S, ctx.layer_idx, self.hook1.config.kv_insensitive_multiplier)
        n_anchor = max(1, (ctx.S + rKV - 1) // rKV)

        ctx_updated = copy.copy(ctx)
        ctx_updated.S = n_anchor  # 时序压缩后的帧数

        _, k, v, _ = self.hook2(q, k, v, ctx_updated)

        return q, k, v, unmerge_fn
