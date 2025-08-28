# --- TDGF Phase-3 demo: trains 1D American put and compares vs LS baseline ---
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import torch

from tDGF.sampling import uniform_S_sampler
from tDGF.trainer import train_tdgf_1d
from baseline_mc.longstaff_schwartz import price_american_put_ls

# ----- problem params -----
STRIKE = 100.0
r      = 0.05
sigma  = 0.20
T      = 1.0

# TDGF hyperparams (keep modest for a first run)
K_steps      = 50         # number of time steps in TDGF (not the strike!)
h            = T / K_steps
steps_per_k  = 60         # optimizer steps per time step
lr           = 3e-4

# device
device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

# payoff as a torch function (captures STRIKE)
def payoff_put_torch(S):
    return torch.clamp(STRIKE - S, min=0.0)

# sampler over moneyness (cover deep ITM/OTM)
S_low, S_high = 0.01 * STRIKE, 3.0 * STRIKE
S_sampler = uniform_S_sampler(S_low, S_high)

# ----- train TDGF -----
nets = train_tdgf_1d(
    payoff_fn = payoff_put_torch,
    S_sampler = S_sampler,
    r=r, sigma=sigma,
    K=K_steps, h=h, lr=lr, steps_per_k=steps_per_k,
    device=device
)

# helper to evaluate final network at t=0 (nets[-1])
def tdgf_price_at_S0(net, S0):
    S = torch.tensor([[S0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        u = net(S, payoff_put_torch(S))
    return float(u.item())

final_net = nets[-1]

# ----- compare vs LS baseline on a grid -----
S0_grid = np.linspace(60, 140, 17)
tdgf_prices = [tdgf_price_at_S0(final_net, s) for s in S0_grid]
ls_prices   = [price_american_put_ls(s, STRIKE, r, sigma, T, paths=80_000, steps=100, basis_degree=2)[0]
               for s in S0_grid]

# print a focal comparison
S0_star = 100.0
p_tdgf  = tdgf_price_at_S0(final_net, S0_star)
p_ls, _ = price_american_put_ls(S0_star, STRIKE, r, sigma, T, paths=120_000, steps=120, basis_degree=2)
rel_err = abs(p_tdgf - p_ls) / max(1e-8, p_ls)
print(f"S0={S0_star:.1f}  TDGF={p_tdgf:.4f}  LS={p_ls:.4f}  rel_error={100*rel_err:.2f}%")

# ----- shape sanity (monotonicity) -----
is_nonincreasing = all(tdgf_prices[i] >= tdgf_prices[i+1] for i in range(len(tdgf_prices)-1))
print(f"TDGF price monotone non-increasing in S0? {'YES' if is_nonincreasing else 'NO'}")

# ----- plot & save -----
plt.figure()
plt.plot(S0_grid, ls_prices,   label="American (LS baseline)")
plt.plot(S0_grid, tdgf_prices, label="American (TDGF)")
plt.xlabel("S0"); plt.ylabel("Option price")
plt.title("American Put under Black–Scholes: TDGF vs Longstaff–Schwartz")
plt.legend(); plt.grid(True); plt.tight_layout()

out_dir = "experiments/img"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "tdgf_vs_ls.png")
plt.savefig(out_path, dpi=150)
print(f"✅ Plot saved to {out_path}")

plt.show()
