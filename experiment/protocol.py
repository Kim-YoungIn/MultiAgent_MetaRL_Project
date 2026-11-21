"""Predeclared training cells and compute budgets of D-031."""
from itertools import product

BLOCKS = {
    'primary': ('primary',),
    'rules': ('rules_low', 'rules_high'),
    'opponents': ('q_only', 'sarsa_only'),
    'families': ('focused',),
    'actions': ('n5',),
}


def budget(dev):
    if dev:
        return dict(updates=2, trials=3, k=6, h=6, ks=(0, 1, 3, 6), reps=1)
    return dict(updates=800, trials=32, k=24, h=24, ks=(0, 3, 6, 12, 24), reps=8)


def plan(dev=False):
    cells = []
    indices = range(1, 2 if dev else 6)
    for block, conditions in BLOCKS.items():
        for condition, catalog, replica in product(conditions, indices, indices):
            counts = {'rules_low': 2 if dev else 8, 'rules_high': 8 if dev else 128}
            profiles = {'q_only': ('steady',), 'sarsa_only': ('sarsa',)}
            cells.append(dict(block=block, condition=condition, catalog=catalog,
                              replica=replica, train_n=5 if condition == 'n5' else 3,
                              diversity='focused' if condition == 'focused' else 'broad',
                              count=counts.get(condition, 4 if dev else 32),
                              methods=('M', 'P', 'O') if condition == 'primary' else ('M', 'P'),
                              profiles=profiles.get(condition, ('steady', 'sarsa')),
                              seed=22001 if dev else 41000 + 100*catalog + replica))
    return cells
