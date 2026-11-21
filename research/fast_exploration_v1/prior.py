"""Gaussian reward-model prior fitted for absolute post-update utility.

The teacher and student are not trained here.  The training focal actions are
uniform, allowing an inverse-propensity (IPS) estimate of action utility.  No
simulator oracle values or held-out rewards enter the training objective.
"""
import numpy as np
import torch

from research.opponent_adaptation.study import histories
from research.structured_scale.run import D

BUDGETS = (0, 2, 4, 8, 16, 32)


def auc_weights(budgets=BUDGETS):
    """Normalized trapezoidal weights for absolute reward as a function of k."""
    if len(budgets) < 2 or budgets[0] != 0 or any(b <= a for a, b in zip(budgets, budgets[1:])):
        raise ValueError('budgets must be increasing and start at zero')
    distances = np.diff(budgets).astype(float)
    weights = np.zeros(len(budgets), dtype=float)
    weights[:-1] += distances / (2 * budgets[-1])
    weights[1:] += distances / (2 * budgets[-1])
    return weights


def absolute_auc(budgets, rewards):
    if len(budgets) != len(rewards):
        raise ValueError('rewards and budgets must have equal length')
    return float(np.dot(auc_weights(tuple(budgets)), np.asarray(rewards)))


def fit_return_prior(data, *, steps=160, learning_rate=.025, temperature=5.,
                     ridge=.01, seed=0, device='auto'):
    """Fit N(mu,I) via IPS absolute-reward AUC under identical random probes.

    At each probe k, the Gaussian posterior mean for candidate action a is an
    affine function of mu, computed by ``histories``. Train a softmax surrogate
    for greedy selection from the observed uniform arm reward (3*pi(a)*r).
    AUC uses absolute utility, never improvement minus a frozen baseline.
    Only factual randomly-selected rewards appear in the optimization target.
    """
    if steps < 1 or learning_rate <= 0 or temperature <= 0 or ridge < 0:
        raise ValueError('invalid prior training configuration')
    from .study import resolve_device
    device=resolve_device(device)
    n_games = len(data['a'])
    if n_games == 0:
        raise ValueError('empty training games')
    n_trajectories = 2 * n_games
    xp, offsets, chosen, observed = [], [], [], []
    used = tuple(k for k in BUDGETS if k < 32)
    budget_weights = dict(zip(used, auc_weights(used)))
    actions = np.repeat(data['a'], 2, axis=0)
    reward = data['y'].reshape(n_trajectories, 32)
    for t, (_, z, offset) in enumerate(histories(data)):
        if t in budget_weights and t < 32:  # k=32 has no subsequent observed reward
            xp.append(z.copy())
            offsets.append(offset.copy())
            chosen.append(actions[:, t].copy())
            observed.append(reward[:, t].copy())
    # Integrating up to k=32 requires a reward at that state; the probe ends at
    # t=32.  Use the strictly available 0..16 budgets instead of pretending
    # that an unobserved reward is a label.
    weights = auc_weights(used)
    # Consumer NVIDIA cards commonly have low FP64 throughput; use FP32
    # for the differentiable surrogate on GPU, retain FP64 CPU baseline.
    dtype=torch.float32 if device.type=='cuda' else torch.float64
    X = torch.as_tensor(np.stack(xp), dtype=dtype, device=device)
    off = torch.as_tensor(np.stack(offsets), dtype=dtype, device=device)
    a = torch.as_tensor(np.stack(chosen), dtype=torch.int64, device=device)
    y = torch.as_tensor(np.stack(observed), dtype=dtype, device=device)
    wt = torch.as_tensor(weights, dtype=dtype, device=device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        mu = torch.nn.Parameter(torch.zeros(D, dtype=dtype, device=device))
        optimizer = torch.optim.Adam([mu], lr=learning_rate)
        for _ in range(steps):
            scores = torch.einsum('ktad,d->kta', X, mu) + off
            probs = torch.softmax(temperature * scores, dim=-1)
            selected = probs.gather(-1, a[..., None]).squeeze(-1)
            # A random action has probability 1/3, hence the factor 3.
            utility = (3 * selected * y).mean(dim=1)
            objective = (wt * utility).sum()
            loss = -objective + ridge * mu.square().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return mu.detach().cpu().numpy().astype(np.float64,copy=True)
