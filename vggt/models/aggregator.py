# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from vggt.compression.mechanism_g import DPPConfig, DPPTokenSelector

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

        # DPP Attention（默认禁用；调用 enable_dpp() 后启用）
        self.dpp_selector: Optional["DPPTokenSelector"] = None
        self._dpp_cfg: Optional["DPPConfig"] = None
        self._dpp_cached_idx: Optional[torch.Tensor] = None

    def enable_dpp(self, cfg: "DPPConfig") -> None:
        """
        启用 DPP Attention（推理前调用）。

        Args:
            cfg: DPPConfig 实例，控制 keep_ratio / window_size 等超参。
        """
        from vggt.compression.mechanism_g import DPPTokenSelector
        self.dpp_selector    = DPPTokenSelector(cfg)
        self._dpp_cfg        = cfg
        self._dpp_cached_idx = None  # 重置跨层缓存

    def disable_dpp(self) -> None:
        """关闭 DPP Attention，恢复原始全序列 Global Attention。"""
        self.dpp_selector    = None
        self._dpp_cfg        = None
        self._dpp_cached_idx = None

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        # 每次 forward 重置 DPP 缓存（on_all_layers=False 模式下每序列只算一次）
        self._dpp_cached_idx = None

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        When dpp_selector is set, each block uses DPP token reduction before attention.
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.dpp_selector is not None:
                # ── DPP 压缩路径 ──────────────────────────────────────────
                tokens, pos = self._dpp_global_block(
                    tokens, B, S, P, C, global_idx, pos
                )
            else:
                # ── 原始全序列路径 ────────────────────────────────────────
                # 写入压缩上下文（其他 hook 可能需要）
                attn_module = self.global_blocks[global_idx].attn
                if hasattr(attn_module, '_compression_ctx') and attn_module._compression_ctx is not None:
                    attn_module._compression_ctx.S = S
                    attn_module._compression_ctx.P = P
                    attn_module._compression_ctx.layer_idx = global_idx
                    attn_module._compression_ctx.is_global = True

                if self.training:
                    tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = self.global_blocks[global_idx](tokens, pos=pos)

            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def _dpp_global_block(
        self,
        tokens:     torch.Tensor,            # [B, S*P, C]
        B:          int,
        S:          int,
        P:          int,
        C:          int,
        global_idx: int,
        pos:        Optional[torch.Tensor],  # [B, S*P, 2] 或 None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        单层 DPP 压缩 Global Attention。

        流程：
            1. 计算（或复用）每帧 patch token 的 DPP 子集索引
            2. Gather：将压缩后的序列 [B, S*(N_sp+M), C] 送入 global block
            3. Scatter Back：将 global 输出写回原始全序列位置

        Returns:
            tokens : [B, S*P, C]  更新后的完整 token 序列
            pos    : [B, S*P, 2] 或 None（形状保持不变）
        """
        cfg  = self._dpp_cfg
        N_sp = self.patch_start_idx        # 5（1 camera + 4 register）
        N    = P - N_sp                    # 1369 patch token / 帧

        # patch_h / patch_w：假设 patch token 排布为正方形网格
        patch_h = patch_w = int(round(N ** 0.5))
        # 若 N 不是完全平方数（极少见），退化回全序列，避免崩溃
        if patch_h * patch_w != N:
            logger.warning(
                f"DPP Attention: N={N} 不是完全平方数，跳过压缩（层 {global_idx}）"
            )
            if self.training:
                return checkpoint(
                    self.global_blocks[global_idx], tokens, pos,
                    use_reentrant=self.use_reentrant
                ), pos
            return self.global_blocks[global_idx](tokens, pos=pos), pos

        tokens_4d = tokens.view(B, S, P, C)

        # ── 1. 计算或复用 DPP 索引 ───────────────────────────────────────
        if cfg.on_all_layers or self._dpp_cached_idx is None:
            # 取 batch=0 的 patch token 作为代表（推理时 B=1；训练时共享索引）
            patch_tok = tokens_4d[0, :, N_sp:, :].detach()   # [S, N, C]
            select_idx = self.dpp_selector.select(
                patch_tok, patch_h, patch_w
            )                                                   # [S, M]
            if not cfg.on_all_layers:
                # on_all_layers=False：缓存供后续层复用
                self._dpp_cached_idx = select_idx
        else:
            select_idx = self._dpp_cached_idx                  # [S, M]

        M = select_idx.shape[1]

        # ── 2. Gather 压缩序列 ────────────────────────────────────────────
        # 目标：[B, S*(N_sp+M), C]；每帧布局 [special(N_sp) | selected_patch(M)]
        pos_4d = pos.view(B, S, P, 2) if pos is not None else None

        comp_tok_list: List[torch.Tensor] = []
        comp_pos_list: Optional[List[torch.Tensor]] = [] if pos_4d is not None else None

        for s in range(S):
            sp_tok  = tokens_4d[:, s, :N_sp, :]              # [B, N_sp, C]
            pt_tok  = tokens_4d[:, s, N_sp:, :]              # [B, N, C]
            sel_tok = pt_tok[:, select_idx[s], :]             # [B, M, C]
            comp_tok_list.append(torch.cat([sp_tok, sel_tok], dim=1))  # [B, N_sp+M, C]

            if pos_4d is not None:
                sp_pos  = pos_4d[:, s, :N_sp, :]              # [B, N_sp, 2]
                pt_pos  = pos_4d[:, s, N_sp:, :]              # [B, N, 2]
                sel_pos = pt_pos[:, select_idx[s], :]          # [B, M, 2]
                comp_pos_list.append(torch.cat([sp_pos, sel_pos], dim=1))

        comp_tokens = torch.cat(comp_tok_list, dim=1)          # [B, S*(N_sp+M), C]
        comp_pos    = torch.cat(comp_pos_list, dim=1) if comp_pos_list is not None else None

        # ── 3. 压缩序列经 Global Attention Block ─────────────────────────
        if self.training:
            comp_out = checkpoint(
                self.global_blocks[global_idx], comp_tokens, comp_pos,
                use_reentrant=self.use_reentrant
            )
        else:
            comp_out = self.global_blocks[global_idx](comp_tokens, pos=comp_pos)
        # comp_out: [B, S*(N_sp+M), C]

        # ── 4. Scatter Back ───────────────────────────────────────────────
        comp_out_4d = comp_out.view(B, S, N_sp + M, C)

        # 以 frame attention 输出（tokens_4d）为基础，仅覆盖被选 token
        out_4d = tokens_4d.clone()

        for s in range(S):
            idx_s = select_idx[s]                              # [M]
            # 更新特殊 token
            out_4d[:, s, :N_sp, :]         = comp_out_4d[:, s, :N_sp, :]
            # 更新被选 patch token
            # 使用 index_put 保证梯度正确流动
            out_4d[:, s, N_sp:, :] = out_4d[:, s, N_sp:, :].clone().index_put(
                (torch.arange(B, device=tokens.device).unsqueeze(-1),
                 idx_s.unsqueeze(0).expand(B, -1)),
                comp_out_4d[:, s, N_sp:, :],
            )
            # 未被选的 patch token 保持 frame attention 输出不变（已通过 clone 保留）

        # pos 保持原始全序列形状（下层 frame block 需要完整 pos）
        return out_4d.view(B, S * P, C), pos


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
