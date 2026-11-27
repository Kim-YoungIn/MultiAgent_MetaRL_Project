"""Explore repeatability and distribution of paired 96-turn returns on opened panels."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('runs')
HORIZON = ROOT / 'fast_exploration_v2_horizon_final'
UCB = ROOT / 'fast_exploration_v2_horizon_ucb'
DYNAMICS = ROOT / 'dynamics_transfer_v1'
SOURCE = ROOT / 'fast_exploration_v2_final'
OUTPUT = Path('research/retrospective/evidence')
FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def metric(x):
    """Each row is a game and columns are the two paired resets."""
    game = x.mean(axis=1)
    sign = np.where(np.abs(x) <= 1e-12, 0, np.sign(x))
    same_nonzero = np.mean((sign[:, 0] == sign[:, 1]) & (sign[:, 0] != 0))
    selected = (sign != 0).all(axis=1)
    both_pos = np.mean((sign > 0).all(axis=1))
    both_neg = np.mean((sign < 0).all(axis=1))
    lo, hi = np.quantile(game, [.05, .95])
    return dict(mean=game.mean(), median=np.median(game),
                q10=np.quantile(game, .1), q90=np.quantile(game, .9),
                wins=np.mean(game > 1e-12), losses=np.mean(game < -1e-12),
                ties=np.mean(np.abs(game) <= 1e-12), both_reset_win=both_pos,
                both_reset_loss=both_neg, same_nonzero=same_nonzero,
                second_win_given_first_win=np.mean(x[x[:,0] > 1e-12,1] > 1e-12),
                sign_agreement_nonzero=(same_nonzero / selected.mean() if selected.any() else np.nan),
                trim5_mean=game[(game >= lo) & (game <= hi)].mean(),
                top_decile_contribution=game[game >= np.quantile(game, .9)].sum() / len(game),
                bottom_decile_contribution=game[game <= np.quantile(game, .1)].sum() / len(game))


def read():
    report = json.loads((UCB/'report.json').read_text())
    src_manifest = digest(SOURCE/'manifest.json')
    assert json.loads((HORIZON/'manifest.json').read_text())['trained_source_sha256'] == src_manifest
    manifest = json.loads((DYNAMICS/'manifest.json').read_text())
    assert manifest['source_manifest_sha256'] == src_manifest
    assert report['parent_manifest_sha256'] == digest(HORIZON/'manifest.json')
    outputs = {('horizon', comp): [] for comp in ('student','ucb')}
    outputs.update({('dynamics', comp): [] for comp in ('student','ucb')})
    families, masters = {}, []
    for i in range(36):
        name = f'block_{i:03d}'
        train = json.loads((SOURCE/f'{name}.json').read_text())
        horizon = json.loads((HORIZON/f'{name}.json').read_text())
        dynamics = json.loads((DYNAMICS/f'{name}.json').read_text())
        assert horizon['raw_sha256'] == digest(HORIZON/f'{name}.npz')
        assert dynamics['raw_sha256'] == digest(DYNAMICS/f'{name}.npz')
        assert manifest['source_receipts'][str(i)] == digest(SOURCE/f'{name}.json')
        assert report['blocks'][f'{name}.npz'] == digest(UCB/f'{name}.npz')
        assert horizon['teacher_sha256'] == dynamics['teacher_sha256'] == train['weights_sha256']
        assert digest(SOURCE/f'{name}.pt') == train['weights_sha256']
        masters.append(train['master_seed'])
        with np.load(HORIZON/f'{name}.npz') as h, np.load(UCB/f'{name}.npz') as u, np.load(DYNAMICS/f'{name}.npz') as d:
            for key in ('game_seeds','reset_seeds','families'):
                np.testing.assert_array_equal(h[key],u[key])
            np.testing.assert_array_equal(h['methods'],['pooled_greedy','pooled_fixed_value','pooled_fixed_total','pooled_fixed_value','pooled_fixed_total'])
            np.testing.assert_array_equal(h['ks'],[0,8,8,24,24])
            np.testing.assert_array_equal(u['ks'],[8,24])
            np.testing.assert_array_equal(d['live_methods'],['pooled_fixed_total','pooled_ucb','pooled_greedy'])
            np.testing.assert_allclose(d['distances'],[0,.3,.6])
            np.testing.assert_array_equal(d['lags'],[0,1])
            for panel, fs, seeds in [('horizon',h['families'],h['game_seeds']),('dynamics',d['families'],d['game_seeds'])]:
                assert len(fs)==2*len(seeds) and np.array_equal(fs[::2],fs[1::2])
                families.setdefault(panel, []).append(fs[::2].copy())
            hr = h['rewards'][...,0].sum(axis=-1)
            ur = u['rewards'][...,0].sum(axis=-1)
            dr = d['live_rewards'][0,0,...,0].sum(axis=-1)
            for comp, delta in [('student',hr[2]-hr[0]),('ucb',hr[2]-ur[0])]:
                outputs['horizon',comp].append(delta.reshape(-1,2))
            for comp, delta in [('student',dr[0]-dr[2]),('ucb',dr[0]-dr[1])]:
                outputs['dynamics',comp].append(delta.reshape(-1,2))
    return {k:np.stack(v) for k,v in outputs.items()}, {k:np.stack(v) for k,v in families.items()}, np.array(masters)


def plot_distribution(values, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True, layout='constrained')
    for ax, (baseline, title) in zip(axes, [('student', 'Teacher − greedy Student'),
                                             ('ucb', 'Teacher − UCB')], strict=True):
        for panel, label, color in [('horizon', 'First panel', '#21608a'),
                                    ('dynamics', 'Second panel', '#b15b32')]:
            game = np.sort(values[panel, baseline].mean(axis=-1).ravel())
            ax.step(game, np.arange(1, len(game) + 1) / len(game), where='post',
                    color=color, label=label)
            ax.plot(np.median(game), .5, 'o', color=color, markersize=4)
        ax.axhline(.5, color='0.65', ls=':', lw=1, zorder=0)
        ax.axvline(0, color='0.45', lw=1, zorder=0)
        ax.set(xscale='symlog', title=title, xlabel='96-turn paired reward difference')
        ax.set_xticks([-10, -1, 0, 1, 10], labels=['−10', '−1', '0', '1', '10'])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Fraction of games at or below difference')
    axes[0].legend(frameon=False, loc='lower right')
    fig.savefig(output / 'distribution.png', dpi=155)
    plt.close(fig)


def main():
    values, families, masters = read()
    assert len(set(masters)) == 3 and all((masters == s).sum() == 12 for s in set(masters))
    rng = np.random.default_rng(17592036)
    indices = np.concatenate([rng.choice(np.flatnonzero(masters==s),(20000,12),replace=True)
                              for s in sorted(set(masters))], axis=1)
    rows = []
    block_values = {}
    for (panel, comp), arr in values.items():
        for family in ('all', *FAMILIES):
            select = np.ones(families[panel].shape, dtype=bool) if family == 'all' else families[panel] == family
            assert (select.sum(axis=1)==select.sum(axis=1)[0]).all()
            subset = np.stack([x[m] for x,m in zip(arr,select,strict=True)])
            aggregate = metric(subset.reshape(-1,2))
            per_block = np.array([x.mean() for x in subset])
            win_blocks = np.array([(x.mean(axis=1)>1e-12).mean() for x in subset])
            both_blocks = np.array([((x[:,0]>1e-12)&(x[:,1]>1e-12)).mean() for x in subset])
            for field, vals in [('mean',per_block),('wins',win_blocks),('both_reset_win',both_blocks)]:
                aggregate[field+'_low'],aggregate[field+'_high'] = np.quantile(vals[indices].mean(axis=1),[.025,.975])
            # Recompute the pooled median after resampling whole training blocks.
            # Family composition is balanced within each block.
            median_draws = []
            games = subset.mean(axis=2)
            for start in range(0, len(indices), 100):
                sample = games[indices[start:start+100]].reshape(-1, games.size)
                median_draws.extend(np.median(sample, axis=1))
            aggregate['median_low'],aggregate['median_high'] = np.quantile(median_draws,[.025,.975])
            rows.append(dict(panel=panel, comparison='teacher_minus_'+comp, family=family,
                             n_blocks=len(subset), n_games=subset.shape[0]*subset.shape[1], **aggregate))
            if family=='all':
                block_values[panel,comp] = per_block
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plot_distribution(values, OUTPUT)
    with (OUTPUT/'distribution.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with (OUTPUT/'distribution_blocks.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['block','master_seed','horizon_vs_student','dynamics_vs_student','horizon_vs_ucb','dynamics_vs_ucb'])
        w.writeheader()
        for i in range(36):
            w.writerow(dict(block=i,master_seed=masters[i],horizon_vs_student=block_values['horizon','student'][i],
                            dynamics_vs_student=block_values['dynamics','student'][i],
                            horizon_vs_ucb=block_values['horizon','ucb'][i], dynamics_vs_ucb=block_values['dynamics','ucb'][i]))
    cross_rows = []
    for comp in ('student','ucb'):
        x,y=block_values['horizon',comp],block_values['dynamics',comp]
        corr = np.corrcoef(x,y)[0,1]
        ci = np.quantile([np.corrcoef(x[idx],y[idx])[0,1] for idx in indices[:5000]], [.025,.975])
        cross_rows.append(dict(comparison='teacher_minus_'+comp, blocks=36, correlation=corr,
                               low=ci[0], high=ci[1], same_sign=np.mean(np.sign(x)==np.sign(y))))
        print(comp,'cross_panel block Pearson',corr, 'interval',ci, 'same sign', np.mean(np.sign(x)==np.sign(y)))
    with (OUTPUT/'distribution_cross_panel.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cross_rows[0].keys())
        w.writeheader()
        w.writerows(cross_rows)
    for row in rows:
        if row['family']=='all':
            print(row)


if __name__ == '__main__':
    main()
