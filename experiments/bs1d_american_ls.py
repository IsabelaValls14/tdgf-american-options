import numpy as np
import matplotlib.pyplot as plt
import os

from baseline_mc.mc_european import price_euro_mc
from baseline_mc.longstaff_schwartz import price_american_put_ls
from payoffs import euro_put
from math import erf

# Closed-form European put (Black–Scholes)
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def bs_put(S0,K,r,sigma,T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*(1-norm_cdf(d2)) - S0*(1-norm_cdf(d1))

if __name__ == "__main__":
    K=100; r=0.05; sigma=0.2; T=1.0
    S0_grid = np.linspace(60, 140, 17)

    euro_cf = [bs_put(S0,K,r,sigma,T) for S0 in S0_grid]
    euro_mc = [price_euro_mc(S0,K,r,sigma,T,paths=40_000,steps=252,payoff_fn=euro_put)
               for S0 in S0_grid]
    amer_ls = [price_american_put_ls(S0,K,r,sigma,T,paths=80_000,steps=100,basis_degree=2)[0]
               for S0 in S0_grid]

    plt.figure()
    plt.plot(S0_grid, euro_cf,  label="European (Black–Scholes)")
    plt.plot(S0_grid, euro_mc,  "o", ms=3, label="European (MC)")
    plt.plot(S0_grid, amer_ls,  label="American (Longstaff–Schwartz)")
    plt.xlabel("S0")
    plt.ylabel("Option price")
    plt.title("European vs American Put (Longstaff–Schwartz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save in experiments/img/
    out_dir = "experiments/img"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ls_vs_euro.png")
    plt.savefig(out_path, dpi=150)
    print(f"✅ Plot saved to {out_path}")

    plt.show()
