# TDGF American Options

A compact project to price American options under Black–Scholes using:

- **Baseline**: Monte Carlo (MC) + Longstaff–Schwartz (LS)
- **Neural PDE**: Time Deep Gradient Flow (TDGF) with a payoff-preserving network

## Phases
- **Phase 1** — European option (MC baseline)
- **Phase 2** — American option (LS baseline)
- **Phase 3** — TDGF (1D American put) (needs tailoring)
- **Phase 4** — 2D Basket (stretch)

# References:
- https://arxiv.org/search/?query=Time+Deep+Gradient+Flow+Method+for+Pricing+American+Options&searchtype=all&source=header
- https://arxiv.org/abs/2507.17606


## Quickstart
```bash

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn torch
python3 experiments/bs1d_euro_quickstart.py
```


## Repo Structure

tdgf-american-options/
├─ models/black_scholes.py (exact GBM step)
├─ payoffs.py (call/put intrinsic payoffs)
├─ baseline_mc/
│ ├─ mc_european.py (European MC pricer)
│ └─ longstaff_schwartz.py (American LS pricer)
├─ tDGF/
│ ├─ model.py (payoff-preserving MLP: tanh + softplus skip)
│ ├─ operators.py (BS coefficients A(S)=½σ²S², b(S))
│ ├─ loss.py (TDGF energy; proximity + gradient energy + r term)
│ ├─ trainer.py (time-stepping loop; warm-start + prev-step mask)
│ └─ sampling.py (S samplers over moneyness)
└─ experiments/
├─ bs1d_euro_quickstart.py (Phase 1: MC vs Black–Scholes)
├─ bs1d_american_ls.py (Phase 2: LS vs European; saves plot)
└─ bs1d_tdgf_demo.py (Phase 3: TDGF vs LS; saves plot)