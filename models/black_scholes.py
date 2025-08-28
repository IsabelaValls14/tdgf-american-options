"""
Black–Scholes GBM simulator utilities.
"""
import numpy as np

def gbm_step(S, dt, r, sigma, Z):
    """
    Exact lognormal step for GBM under risk-neutral measure.
    """
    S = np.asarray(S, dtype=float)
    return S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
