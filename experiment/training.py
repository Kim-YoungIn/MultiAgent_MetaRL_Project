"""PPO training for the three D-031 exploration objectives."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
import resource
import time
import torch

from mamrl.meta_batch import collect_batch, replay_batch
from mamrl.meta_exploration import decode_rule
from mamrl.meta_training_state import context_digest, save_training_state, restore_training_state
from mamrl.ppo import ppo_loss
from mamrl.rules import compile_rule
from .model import ContextActor
from .artifacts import atomic_path


def train(directory, entries, *, method, seed, profiles, updates=800, trials=32,
          k=24, h=24, device='cpu', stop=None):
    if method not in ('M', 'P', 'O') or not entries or not profiles:
        raise ValueError('invalid method, rules or opponent profiles')
    if min(updates, trials, k, h) <= 0:
        raise ValueError('training budgets must be positive')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model = ContextActor(seed=seed, masked=method == 'O').to(device)
    initial = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    sampler = random.Random(seed)
    config = dict(method=method, seed=seed, profiles=list(profiles), updates=updates,
                  trials=trials, k=k, h=h)
    context = context_digest(config, entries, 'd031', device)
    state_path = root / 'training_state.pt'
    config_path = root / 'model_config.pt'
    if not config_path.exists():
        with atomic_path(config_path) as tmp:
            torch.save({'hidden_size': 32, 'masked': method == 'O'}, tmp)
    history = []
    if state_path.exists():
        history = restore_training_state(state_path, model=model, initial=initial,
                                         optimizer=optimizer, sampler=sampler, context=context)
    tensors = [compile_rule(decode_rule(entry['rule'])) for entry in entries]
    for update in range(len(history), min(updates, stop if stop is not None else updates)):
        started = time.monotonic()
        sampled = [(sampler.randrange(len(tensors)), sampler.getrandbits(63), sampler.choice(profiles))
                   for _ in range(trials)]
        jobs = [(tensors[index], trial_seed, profile) for index, trial_seed, profile in sampled]
        batch = collect_batch(model, jobs, steps=k, assessment_steps=h)
        targets = torch.tensor([r.trial['exploration_mean'] if method == 'P' else r.meta_reward
                                for r in batch for _ in r.actions], dtype=torch.float32, device=device)
        values = torch.tensor([v for r in batch for v in r.old_values], device=device)
        advantages = targets - values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        old_logs = torch.tensor([v for r in batch for v in r.old_log_probs], device=device)
        for _ in range(3):
            logs, predictions, entropy = replay_batch(model, batch)
            loss, _ = ppo_loss(logs, old_logs, predictions, targets, advantages, entropy,
                               clip_ratio=.2, value_coefficient=.5, entropy_coefficient=.01)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
        history.append({'update': update + 1, 'target': float(targets.mean()),
                        'assessment': sum(r.trial['assessment_mean'] for r in batch) / len(batch),
                        'G': sum(r.meta_reward for r in batch)/len(batch),
                        'exploration_reward': sum(r.trial['exploration_mean'] for r in batch)/len(batch),
                        'rule_exposures': dict(Counter(entries[i]['id'] for i, _, _ in sampled)),
                        'opponent_exposures': dict(Counter(p for _, _, p in sampled)),
                        'seconds': time.monotonic() - started,
                        'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'cuda_peak_bytes': (torch.cuda.max_memory_allocated(device)
                                            if device.type == 'cuda' else None)})
        save_training_state(state_path, model=model, initial=initial, optimizer=optimizer,
                            sampler=sampler, history=history, context=context)
    return model


def load_model(directory, *, initial=False, device='cpu'):
    root = Path(directory)
    state = torch.load(root / 'training_state.pt', map_location='cpu', weights_only=True)
    # The mask is stored separately because it is not a trainable parameter.
    config = torch.load(root / 'model_config.pt', map_location='cpu', weights_only=True)
    model = ContextActor(hidden_size=config['hidden_size'], masked=config['masked']).to(device)
    model.load_state_dict(state['initial' if initial else 'model'])
    return model
