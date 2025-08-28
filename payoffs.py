"""
Payoff functions.
"""
import numpy as np

def euro_call(S, K):
    S = np.asarray(S, dtype=float)
    return np.maximum(S - K, 0.0)

def euro_put(S, K):
    S = np.asarray(S, dtype=float)
    return np.maximum(K - S, 0.0)

def american_put(S, K):
    # same intrinsic as euro put; early exercise handled by LS/TDGF logic
    return euro_put(S, K)
