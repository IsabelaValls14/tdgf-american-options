"""
TDGF loss version 3.0 (time-discrete energy functional) 
- 1D (masked-safe). (use full-batch proximity + masked energy)
"""
import torch
from tDGF.operators import A_1d, b_1d  # b_1d reserved for future use

def tdgf_loss_1d(u_curr, u_prev, S, r, sigma, h, mask=None, lambda_full_prox=0.1):
    """
    Discrete TDGF loss:
      - full-batch proximity keeps u_k near u_{k-1} globally,
      - masked energy (continuation region) drives PDE descent.
    """
    if mask is None:
        mask = torch.ones_like(u_curr, dtype=torch.bool)

    # gradient on full batch (use .sum trick)
    du_dS_full = torch.autograd.grad(u_curr.sum(), S, create_graph=True)[0]

    # full-batch proximity (stabilizer)
    prox_full = 0.5 * torch.mean((u_curr - u_prev) ** 2)

    # mask continuation region for energy terms
    u_c = u_curr[mask]
    u_p = u_prev[mask]
    S_m = S[mask]
    du  = du_dS_full[mask]

    prox_masked = 0.5 * torch.mean((u_c - u_p) ** 2) if u_c.numel() > 0 else 0.0

    A = A_1d(S_m, sigma)
    grad_energy = 0.5 * torch.mean(A * (du ** 2)) if du.numel() > 0 else 0.0

    r_term = r * torch.mean(u_c ** 2) if u_c.numel() > 0 else 0.0

    return lambda_full_prox * prox_full + prox_masked + h * (grad_energy + r_term)
