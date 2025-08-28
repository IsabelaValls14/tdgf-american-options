"""
Sampling utilities.
"""
import torch

def uniform_S_sampler(low, high):
    def _sample(n):
        return (low + (high - low) * torch.rand(n, 1))
    return _sample
