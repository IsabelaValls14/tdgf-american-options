"""
Operators for Black–Scholes in 1D (A(S), b(S)).
"""
import torch

def A_1d(S, sigma):
    return 0.5 * (sigma**2) * (S**2)

def b_1d(S, r, sigma):
    return (sigma**2 - r) * S
