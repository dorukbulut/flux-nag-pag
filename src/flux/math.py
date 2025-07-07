import torch
from einops import rearrange
from torch import Tensor

import torch
from einops import rearrange

def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    pe: torch.Tensor, guidance_weight: float,
    tau: float = 1.5, alpha: float = 0.6
) -> torch.Tensor:
    """
    Hybrid PAG + NAG Attention:
    - PAG: Perturb attention weights with identity for structural guidance.
    - NAG: Normalize and refine extrapolated features.
    """
    B, H, L, D = q.shape
    
    
    q, k = apply_rope(q, k, pe)
    
    
    z_pos = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    
    if guidance_weight > 1.0:
        
        identity_weights = torch.eye(L, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        identity_weights = identity_weights.expand(B, H, -1, -1)
        z_neg = torch.matmul(identity_weights, v)
        
        z_ex = z_pos + guidance_weight * (z_pos - z_neg)
        
        
        norm_pos = torch.norm(z_pos, dim=-1, keepdim=True)
        norm_ex = torch.norm(z_ex, dim=-1, keepdim=True)
        
        
        ratio = norm_ex / (norm_pos + 1e-8)
        scale_factor = torch.where(
            ratio > tau,
            tau / ratio,
            torch.ones_like(ratio)
        )
        z_norm = z_ex * scale_factor
        

        z_final = alpha * z_norm + (1 - alpha) * z_pos
    else:

        z_final = z_pos
    

    out = rearrange(z_final, "b h l d -> b l (h d)")
    return out

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
