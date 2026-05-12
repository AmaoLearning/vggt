import torch
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# DCT 工具（基于 torch.fft 实现，不依赖第三方库）
# ──────────────────────────────────────────────────────────────────────────────

def dct1d(x: Tensor, dim: int = -1, norm: str = "ortho") -> Tensor:
    """
    1D DCT-II（标准 DCT）沿指定维度计算。
    使用 FFT 镜像扩展法，复杂度 O(N log N)。

    Args:
        x:    任意形状的实数 Tensor
        dim:  沿哪个维度做 DCT
        norm: "ortho" 使用正交归一化（与 scipy.fft.dct 的 norm='ortho' 一致）
    Returns:
        与 x 相同形状的 DCT 系数
    """
    N = x.shape[dim]
    # 1. 镜像扩展: [x0, x1, ..., xN-1, xN-1, ..., x1]
    v = torch.cat([x, x.flip([dim])], dim=dim)  # [..., 2N, ...]

    # 2. FFT
    Vc = torch.fft.rfft(v, n=2 * N, dim=dim)   # [..., N+1, ...]

    # 3. 相位旋转因子 W_k = exp(-j * pi * k / (2N))
    k = torch.arange(N, dtype=x.dtype, device=x.device)
    shape = [1] * x.dim()
    shape[dim] = N
    k = k.view(shape)
    W = torch.exp(-1j * torch.pi * k / (2.0 * N)).to(Vc.dtype)

    # 4. 取前 N 分量，乘以旋转因子，取实部
    X = (Vc.narrow(dim, 0, N) * W).real

    # 5. 正交归一化
    if norm == "ortho":
        X = X * (2.0 / N) ** 0.5
        # 将 DC 分量额外除以 sqrt(2) 使其与 AC 系数在同一能量尺度
        idx = [slice(None)] * x.dim()
        idx[dim] = slice(0, 1)
        X[idx] = X[idx] / (2.0 ** 0.5)

    return X


def idct1d(X: Tensor, dim: int = -1, norm: str = "ortho") -> Tensor:
    """
    1D IDCT-II（对应 dct1d 的逆变换）。
    """
    N = X.shape[dim]

    if norm == "ortho":
        X = X.clone()
        idx = [slice(None)] * X.dim()
        idx[dim] = slice(0, 1)
        X[idx] = X[idx] * (2.0 ** 0.5)
        X = X / (2.0 / N) ** 0.5

    # 逆 DCT = DCT-III，等价于：IDCT(X) = DCT(X)[reversed and scaled]
    # 利用 DCT-II 的对称性：IDCT = (1/2N) * real(IFFT(W_inv * [X, 0, -X_flip]))
    W_inv = torch.exp(1j * torch.pi * torch.arange(N, dtype=X.dtype, device=X.device) / (2.0 * N))
    shape = [1] * X.dim()
    shape[dim] = N
    W_inv = W_inv.view(shape).to(torch.complex64)

    # 构造 IFFT 输入
    X_c = (X * W_inv)
    # 补零扩展到 2N（利用 IFFT 的对称性）
    pad_shape = list(X.shape)
    pad_shape[dim] = 1
    zeros = torch.zeros(pad_shape, dtype=X_c.dtype, device=X.device)
    neg_flip = -X_c.flip([dim])
    Vc = torch.cat([X_c, zeros, neg_flip], dim=dim)   # [..., 2N+1 → 取前 2N]

    x = torch.fft.irfft(Vc.narrow(dim, 0, N + 1), n=2 * N, dim=dim)
    return x.narrow(dim, 0, N) * N


def dct2d(x: Tensor, dims: tuple = (-2, -1), norm: str = "ortho") -> Tensor:
    """2D DCT：先沿 dim[0] 再沿 dim[1]"""
    return dct1d(dct1d(x, dim=dims[0], norm=norm), dim=dims[1], norm=norm)


def idct2d(X: Tensor, dims: tuple = (-2, -1), norm: str = "ortho") -> Tensor:
    """2D IDCT：先沿 dim[1] 再沿 dim[0]"""
    return idct1d(idct1d(X, dim=dims[1], norm=norm), dim=dims[0], norm=norm)


# ──────────────────────────────────────────────────────────────────────────────
# Token 操作工具
# ──────────────────────────────────────────────────────────────────────────────

def gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
    """
    从 tokens 中按 indices 提取子集。
    tokens:  [B, H, N, D]
    indices: [B, H, M] — 要保留的 token 位置
    Returns: [B, H, M, D]
    """
    B, H, M = indices.shape
    idx = indices.unsqueeze(-1).expand(B, H, M, tokens.shape[-1])
    return tokens.gather(2, idx)


def scatter_tokens(src: Tensor, dst: Tensor, indices: Tensor) -> Tensor:
    """
    将 src 中的 token 按 indices 散射回 dst 的对应位置（in-place 修改 dst 副本）。
    src:     [B, H, M, D]
    dst:     [B, H, N, D]
    indices: [B, H, M]
    Returns: [B, H, N, D]（dst 的副本，indices 位置替换为 src）
    """
    B, H, M, D = src.shape
    out = dst.clone()
    idx = indices.unsqueeze(-1).expand(B, H, M, D)
    out.scatter_(2, idx, src)
    return out


def cosine_similarity_batched(a: Tensor, b: Tensor) -> Tensor:
    """
    计算 a 和 b 中每个向量对的余弦相似度。
    a: [B, H, M, D]
    b: [B, H, N, D]
    Returns: [B, H, M, N]
    """
    a_norm = F.normalize(a, dim=-1)  # [B, H, M, D]
    b_norm = F.normalize(b, dim=-1)  # [B, H, N, D]
    return torch.einsum("bhmd,bhnd->bhmn", a_norm, b_norm)
