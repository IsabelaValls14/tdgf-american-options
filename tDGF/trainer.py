# """
# TDGF trainer loop (Algorithm 1) - 1D skeleton.
# """
# import torch
# from torch import optim
# from tDGF.model import PayoffPreservingMLP
# from tDGF.loss import tdgf_loss_1d

# def train_tdgf_1d(payoff_fn, S_sampler, r, sigma, K=100, h=0.01, lr=3e-4, steps_per_k=100, device="cpu"):
#     nets = []
#     net0 = PayoffPreservingMLP().to(device)
#     opt0 = optim.Adam(net0.parameters(), lr=lr)
#     for _ in range(500):
#         S = S_sampler(2048).to(device)
#         S.requires_grad_(True)
#         payoff = payoff_fn(S)
#         u = net0(S, payoff)
#         loss0 = torch.mean((u - payoff)**2)
#         opt0.zero_grad(); loss0.backward(); opt0.step()
#     nets.append(net0)

#     for k in range(1, K+1):
#         net = PayoffPreservingMLP().to(device)
#         opt = optim.Adam(net.parameters(), lr=lr)
#         for _ in range(steps_per_k):
#             S = S_sampler(2048).to(device)
#             S.requires_grad_(True)
#             payoff = payoff_fn(S)
#             u_prev = nets[-1](S, payoff)
#             u_curr = net(S, payoff)
#             mask = (u_curr > payoff).detach().squeeze()
#             if mask.sum() == 0: continue
#             loss = tdgf_loss_1d(u_curr[mask], u_prev[mask], S[mask], r, sigma, h)
#             opt.zero_grad(); loss.backward(); opt.step()
#         nets.append(net)
#     return nets

# """
# TDGF trainer loop (Algorithm 1) - 1D skeleton.
# """
# import torch
# from torch import optim
# from tDGF.model import PayoffPreservingMLP
# from tDGF.loss import tdgf_loss_1d

# def train_tdgf_1d(payoff_fn, S_sampler, r, sigma, K=100, h=0.01, lr=3e-4, steps_per_k=100, device="cpu"):
#     nets = []
#     # Step 0: fit payoff
#     net0 = PayoffPreservingMLP().to(device)
#     opt0 = optim.Adam(net0.parameters(), lr=lr)
#     for _ in range(500):
#         S = S_sampler(2048).to(device)
#         S.requires_grad_(True)
#         payoff = payoff_fn(S)
#         u = net0(S, payoff)
#         loss0 = torch.mean((u - payoff)**2)
#         opt0.zero_grad(); loss0.backward(); opt0.step()
#     nets.append(net0)

#     for k in range(1, K+1):
#         net = PayoffPreservingMLP().to(device)
#         opt = optim.Adam(net.parameters(), lr=lr)
#         for _ in range(steps_per_k):
#             S = S_sampler(2048).to(device)
#             S.requires_grad_(True)
#             payoff = payoff_fn(S)
#             u_prev = nets[-1](S, payoff)
#             u_curr = net(S, payoff)
#             # continuation region mask
#             mask = (u_curr > payoff).detach().squeeze()
#             if mask.sum() == 0:
#                 continue
#             loss = tdgf_loss_1d(u_curr, u_prev, S, r, sigma, h, mask=mask)
#             opt.zero_grad(); loss.backward(); opt.step()
#         nets.append(net)
#     return nets


"""
(warm-start + previous-step mask)
TDGF trainer loop (Algorithm 1) - 1D improved.
Needs to be improved 
"""
import copy
import torch
from torch import optim
from tDGF.model import PayoffPreservingMLP
from tDGF.loss import tdgf_loss_1d

def train_tdgf_1d(payoff_fn, S_sampler, r, sigma, K=100, h=0.01, lr=3e-4,
                  steps_per_k=100, device="cpu"):
    nets = []

    # Step 0: fit payoff
    net0 = PayoffPreservingMLP().to(device)
    opt0 = optim.Adam(net0.parameters(), lr=lr)
    for _ in range(500):
        S = S_sampler(2048).to(device)
        S.requires_grad_(True)
        payoff = payoff_fn(S)
        u = net0(S, payoff)
        loss0 = torch.mean((u - payoff)**2)  # enforce u≈payoff at t=0
        opt0.zero_grad(); loss0.backward(); opt0.step()
    nets.append(net0)

    # k = 1..K: warm-start from previous net and descend energy
    for k in range(1, K + 1):
        # warm-start from previous (important!)
        net = PayoffPreservingMLP().to(device)
        net.load_state_dict(copy.deepcopy(nets[-1]).state_dict())

        opt = optim.Adam(net.parameters(), lr=lr)

        for _ in range(steps_per_k):
            S = S_sampler(2048).to(device)
            S.requires_grad_(True)
            payoff = payoff_fn(S)

            u_prev = nets[-1](S, payoff)
            u_curr = net(S, payoff)

            # define continuation from previous step (not current!)
            mask = (u_prev > payoff + 1e-6).detach().squeeze()
            if mask.sum() == 0:
                # still update with full-batch proximity to keep continuity
                mask = None

            loss = tdgf_loss_1d(u_curr, u_prev, S, r, sigma, h, mask=mask,
                                lambda_full_prox=0.1)
            opt.zero_grad(); loss.backward(); opt.step()

        nets.append(net)

    return nets
