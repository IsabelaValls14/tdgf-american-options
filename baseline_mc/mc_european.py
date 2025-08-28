"""
Monte Carlo pricing for European options under Black–Scholes.
"""
import numpy as np
from models.black_scholes import gbm_step

def price_euro_mc(S0, K, r, sigma, T, paths=50_000, steps=252, payoff_fn=None, seed=42):
    """
    Vectorized Monte Carlo pricer for European options.
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    S = np.full(paths, float(S0))
    for _ in range(steps):
        Z = rng.standard_normal(paths)
        S = gbm_step(S, dt, r, sigma, Z)
    if payoff_fn is None:
        raise ValueError("Provide a payoff_fn(S_T, K) -> payoff array")
    payoff = payoff_fn(S, K)
    disc = np.exp(-r * T)
    return disc * payoff.mean()
