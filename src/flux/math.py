import torch
from einops import rearrange
from torch import Tensor

def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pe: torch.Tensor,
    pag_weight: float = 0.0,  
    tau: float = 1.5
) -> torch.Tensor:
    """
    Hybrid PAG + NAG Attention:
    - Like NAG: Extrapolate between positive and negative features.
    - Like PAG: Use identity attention for negative (instead of a prompt).
    - Normalize the extrapolated feature if its norm grows too large.
    """
    B, H, L, D = q.shape

    
    q, k = apply_rope(q, k, pe) 

    
    z_pos = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # [B, H, L, D]

    
    identity_attn = torch.eye(L, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
    identity_attn = identity_attn.expand(B, H, L, L)
    z_neg = torch.matmul(identity_attn, v)

    
    z_ex = z_pos + pag_weight * (z_pos - z_neg)

    
    norm_pos = torch.norm(z_pos, dim=-1, keepdim=True)
    norm_ex = torch.norm(z_ex, dim=-1, keepdim=True)

    ratio = norm_ex / (norm_pos + 1e-6)
    scale = torch.where(ratio > tau, tau / ratio, torch.ones_like(ratio))
    z_scaled = z_ex * scale

    return rearrange(z_scaled, "b h l d -> b l (h d)")

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
