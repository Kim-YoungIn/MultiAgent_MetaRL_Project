"""Amortized two-decision control with a shared, frozen terminal responder."""
import numpy as np
import torch
from torch import nn

N = 3
WINDOW = 4


def opening_features(data):
    """Per-candidate public information available before choosing the opening."""
    a, b, r = (data[key] for key in ('prefix_a', 'prefix_b', 'prefix_reward'))
    samples, k = a.shape
    candidate = np.arange(N)[None, :]
    cols = []

    def add(x):
        cols.append(np.broadcast_to(x, (samples, N)))

    for x in data['prefix_public'][:, -1].T:
        add(x[:, None] / 2)
    # Bounded joint history includes the seeded reset history at short prefixes.
    for actions, player in [(a, 0), (b, 1)]:
        history = np.concatenate([data['init_hist'][:, ::-1, player], actions], axis=1)
        for lag in (1, 2, 3):
            add(history[:, -lag, None] == candidate)
    for lag in range(1, WINDOW+1):
        if lag > k:
            for _ in range(8):
                add(0.)
            continue
        own, other, reward = a[:, -lag, None], b[:, -lag, None], 4*r[:, -lag, None]
        add(1.)
        add(own == candidate)
        add(other == candidate)
        add(own == other)
        add(reward)
        add(reward * (own == candidate))
        for x in data['prefix_public'][:, k-lag].T:
            add(x[:, None] / 2)
    return np.stack(cols, axis=-1).astype(np.float32)


def terminal_features(data):
    """Build (history, opening, channel, terminal action, features) from public fields."""
    base = opening_features(data)
    s, n, d = base.shape
    shape = (s, n, 2, n)
    own = np.arange(n)[None, :, None, None]
    channel = np.arange(2)[None, None, :, None]
    target = np.arange(n)[None, None, None, :]
    other = data['first_b'][:, None, None, None]
    rest = [own == target, other == target, other == own, channel,
            4 * channel * data['first_reward'][:, :, None, None]]
    rest.extend(x[:, :, None, None]/2 for x in data['post_public'].transpose(2, 0, 1))
    scalars = np.stack([np.broadcast_to(x, shape) for x in rest], axis=-1)
    return np.concatenate([np.broadcast_to(base[:, :, None, None, :], shape+(d,)),
                           np.broadcast_to(base[:, None, None, :, :], shape+(d,)),
                           scalars], axis=-1).astype(np.float32)


class Regressor(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 64), nn.SiLU(), nn.Linear(64, 64),
                                    nn.SiLU(), nn.Linear(64, outputs))
        self.register_buffer('center', torch.zeros(outputs))
        self.register_buffer('scale', torch.ones(outputs))

    def forward(self, x):
        return self.layers(x)*self.scale + self.center


def fit(features, targets, seed, steps, batch):
    x = torch.from_numpy(features.reshape(-1, features.shape[-1]))
    y = torch.from_numpy(np.ascontiguousarray(targets.reshape(len(x), -1), dtype=np.float32))
    torch.manual_seed(seed)
    model = Regressor(x.shape[1], y.shape[1])
    model.center.copy_(y.mean(0))
    model.scale.copy_(y.std(0, unbiased=False).clamp_min(1e-4))
    standardized = (y-model.center)/model.scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    losses = []
    for t in range(steps):
        idx = torch.randint(len(x), (batch,))
        pred = model.layers(x[idx])
        loss = (pred-standardized[idx]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if t == 0 or (t+1) % 100 == 0 or t+1 == steps:
            losses.append([t+1, loss.item()])
    return model.eval(), losses


@torch.no_grad()
def predict(model, x):
    flat = torch.from_numpy(x.reshape(-1, x.shape[-1]))
    out = torch.cat([model(part) for part in flat.split(8192)])
    return out.numpy().reshape(*x.shape[:-1], -1)


def fit_terminal(data, seed):
    x = terminal_features(data)
    # Values are labels only, for each candidate response, with or without feedback.
    y = np.broadcast_to(data['next_value'].transpose(0, 1, 3, 2)[:, :, None], x.shape[:-1]+(2,))
    return fit(x, y, seed, 800, 384)


def terminal_actions(model, data):
    pred = predict(model, terminal_features(data))
    return pred.argmax(axis=3).transpose(0, 1, 3, 2)  # S, opening, regime, channel


def achieved(data, terminal):
    return np.take_along_axis(data['next_value'], terminal, axis=3)


def fit_opening(data, terminal_model, seed):
    pay = achieved(data, terminal_actions(terminal_model, data))
    y = np.stack([data['first_value'], pay[:, :, 0, 0], pay[:, :, 0, 1]-pay[:, :, 0, 0],
                  pay[:, :, 1, 0], pay[:, :, 1, 1]-pay[:, :, 1, 0]], axis=-1)
    return fit(opening_features(data), y, seed, 600, 256)


def opening_actions(scores):
    decisions = []
    for j in (1, 3):
        immediate = scores[:, :, 0]
        blind = immediate + scores[:, :, j]
        decisions.append(np.stack([immediate.argmax(1), blind.argmax(1),
                                   (blind + scores[:, :, j+1]).argmax(1)], axis=-1))
    return np.stack(decisions, axis=1)  # S, regime, myopic/masked/visible


def empirical_actions(data):
    """Online-only greedy sample-mean baseline, initialized at one zero reward."""
    a = data['prefix_a'][:, :, None] == np.arange(N)
    total = (a*data['prefix_reward'][:, :, None]).sum(1)
    count = 1+a.sum(1)
    opening = (total/count).argmax(1)
    i = np.arange(len(opening))
    visible_total = total.copy()
    visible_count = count.copy()
    visible_total[i, opening] += data['first_reward'][i, opening]
    visible_count[i, opening] += 1
    return opening, np.stack([opening, (visible_total/visible_count).argmax(1)], axis=1)
