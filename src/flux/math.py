import torch
from einops import rearrange
from torch import Tensor

def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    pe: torch.Tensor, guidance_weight: float,
    tau: float = 2.0, alpha: float = 0.5
) -> torch.Tensor:
    """
    Hybrid PAG + NAG Attention:
    - PAG: Perturb attention with identity for structural guidance.
    - NAG: Normalize and refine extrapolated features.

    Args:
      guidance_weight: extrapolation scale φ
      tau: max L1-magnitude ratio for normalization
      alpha: blend factor between normalized and original positive features
    """

    # 1. Apply rotary embeddings
    q, k = apply_rope(q, k, pe)

    # 2. Positive attention output
    z_pos = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # 3. PAG-style negative: identity-attended features
    z_neg = v.clone()

    # 4. Extrapolate along feature difference (NAG)
    z_ex = z_pos + guidance_weight * (z_pos - z_neg)

    # 5. L1‑based normalization
    # Compute L1 norms per token across feature dim
    norm_pos = z_pos.abs().sum(dim=-1, keepdim=True)  # [B,H,L,1]
    norm_ex = z_ex.abs().sum(dim=-1, keepdim=True)
    ratio = norm_ex / (norm_pos + 1e-6)
    scale = torch.clamp(ratio, max=tau) / (ratio + 1e-6)
    z_norm = z_ex * scale

    # 6. Feature refinement (blend with original positive features)
    z_ref = alpha * z_norm + (1 - alpha) * z_pos

    # 7. Merge heads
    B, H, L, D = q.shape
    out = rearrange(z_ref, "b h l d -> b l (h d)")
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
