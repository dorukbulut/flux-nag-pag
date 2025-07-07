import torch
from einops import rearrange
from torch import Tensor

import torch
from einops import rearrange

def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    pe: torch.Tensor, guidance_weight: float,
    tau: float = 1.2, alpha: float = 0.3
) -> torch.Tensor:
    """
    Hybrid PAG + NAG Attention:
    - PAG: Perturb attention weights with identity for structural guidance.
    - NAG: Normalize and refine extrapolated features.

    Args:
      guidance_weight: extrapolation scale φ (typically 2.5-7.5)
      tau: max norm ratio for normalization (reduced from 2.0)
      alpha: blend factor between normalized and original positive features (reduced)
    """
    B, H, L, D = q.shape
    
    # 1. Apply rotary position embeddings
    q, k = apply_rope(q, k, pe)
    
    # 2. Compute positive attention (normal attention)
    z_pos = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # 3. Only apply guidance if guidance_weight > 1.0 (indicating guidance is active)
    if guidance_weight > 1.0:
        # PAG-style negative: identity attention (attend to same position)
        # Create identity attention weights and apply to values
        identity_weights = torch.eye(L, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        identity_weights = identity_weights.expand(B, H, -1, -1)
        z_neg = torch.matmul(identity_weights, v)
        
        # 4. NAG-style extrapolation
        guidance_scale = guidance_weight - 1.0  # Convert to scale factor
        z_ex = z_pos + guidance_scale * (z_pos - z_neg)
        
        # 5. Gentle L2-based normalization (more stable than L1)
        norm_pos = torch.norm(z_pos, dim=-1, keepdim=True)
        norm_ex = torch.norm(z_ex, dim=-1, keepdim=True)
        
        # Only normalize if the expansion is significant
        ratio = norm_ex / (norm_pos + 1e-8)
        scale_factor = torch.where(
            ratio > tau,
            tau / ratio,
            torch.ones_like(ratio)
        )
        z_norm = z_ex * scale_factor
        
        # 6. Conservative feature refinement
        # Blend more conservatively to maintain stability
        z_final = alpha * z_norm + (1 - alpha) * z_pos
    else:
        # No guidance, use standard attention
        z_final = z_pos
    
    # 7. Merge heads
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
