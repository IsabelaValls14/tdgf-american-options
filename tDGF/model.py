"""
TDGF model: MLP with tanh hidden, softplus output + payoff skip.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PayoffPreservingMLP(nn.Module):
    def __init__(self, hidden=50, layers=3):
        super().__init__()
        dims = [1] + [hidden]*layers + [1]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

    def forward_g(self, x):
        h = x
        for L in self.linears[:-1]:
            h = torch.tanh(L(h))
        return self.linears[-1](h)

    def forward(self, x, payoff):
        g = self.forward_g(x)
        return payoff + F.softplus(g)
