import torch
from einops import rearrange
from torch import Tensor

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, weight: float) -> Tensor:
    """
    Hybrid PAG + NAG attention:
    - Use standard (NAG-style) scaled dot-product to compute x_pos.
    - For PAG-style negative features, use identity-based attention.
    """
    q, k = apply_rope(q, k, pe)
    x_pos = torch.nn.functional .scaled_dot_product_attention(q, k, v)  # positive features

    B, H, L, D = q.shape

   
    x_neg = v

    
    x = x_pos + weight * (x_pos - x_neg)
    x = torch.nn.functional.normalize(x, dim=-1)

    
    x = x.transpose(1, 2).reshape(B, L, H * D)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
