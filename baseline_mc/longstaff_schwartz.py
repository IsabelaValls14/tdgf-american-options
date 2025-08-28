"""
Longstaff–Schwartz (LS) for American options under Black–Scholes.
"""
import numpy as np
from models.black_scholes import gbm_step

def price_american_put_ls(S0, K, r, sigma, T, paths=50_000, steps=100, basis_degree=2, seed=123):
    """
    Prices an American put via LS with polynomial basis [1, S, S^2, ...].
    Returns (price, details_dict).
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    # simulate paths
    S = np.empty((steps + 1, paths), dtype=float)
    S[0] = S0
    for t in range(1, steps + 1):
        Z = rng.standard_normal(paths)
        S[t] = gbm_step(S[t-1], dt, r, sigma, Z)

    # intrinsic payoff
    def intrinsic(s): return np.maximum(K - s, 0.0)

    CF = intrinsic(S[-1])          # cashflows at maturity
    disc = np.exp(-r * dt)

    for t in range(steps - 1, 0, -1):
        itm = intrinsic(S[t]) > 0
        X = S[t, itm]
        if X.size == 0:
            CF *= disc
            continue
        Y = CF[itm] * disc
        Phi = np.vstack([X**k for k in range(basis_degree + 1)]).T
        reg = 1e-8 * np.eye(Phi.shape[1])
        beta = np.linalg.lstsq(Phi.T @ Phi + reg, Phi.T @ Y, rcond=None)[0]
        C_hat = Phi @ beta
        ex_now = intrinsic(X) > C_hat
        cont_idx = np.where(itm)[0][~ex_now]
        ex_idx   = np.where(itm)[0][ex_now]
        CF[cont_idx] = CF[cont_idx] * disc
        CF[ex_idx]   = intrinsic(S[t, ex_idx])
        CF[~itm] *= disc

    price = CF.mean()
    details = {"paths": paths, "steps": steps, "basis_degree": basis_degree}
    return price, details
