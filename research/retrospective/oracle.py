"""Cross-tabulate the value of seeing a reward and of changing the opening move."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from research.structured_oracle.run import TOL


def analyze(rows):
    gaps = []
    for row in rows:
        immediate, masked, visible = (np.asarray(row[key]) for key in ('first', 'masked', 'visible'))
        greedy = immediate >= immediate.max() - TOL
        best_masked = masked >= masked.max() - TOL
        # Favor ties among masked-optimal openings when scoring them with visible rewards.
        gaps.append([visible.max() - masked.max(), visible.max() - visible[greedy].max(),
                     masked.max() - masked[greedy].max(), visible.max() - visible[best_masked].max()])
        np.testing.assert_allclose(gaps[-1][:3], row['gaps'], atol=TOL, rtol=0)
    gains = np.asarray(gaps)
    info, nonmyopic, _, switch = (gains > TOL).T
    assert not np.any(switch & ~info)
    return dict(pairs=len(rows), info=int(info.sum()), nonmyopic=int(nonmyopic.sum()),
                info_and_nonmyopic=int((info & nonmyopic).sum()), info_only=int((info & ~nonmyopic).sum()),
                nonmyopic_only=int((~info & nonmyopic).sum()), neither=int((~info & ~nonmyopic).sum()),
                must_change_opening=int(switch.sum()), mean_info=gains[:, 0].mean(),
                mean_loss_keeping_masked_opening=gains[:, 3].mean())


def main():
    source = Path('research/evidence/two_turn_pairs.json')
    raw = Path('runs/structured_oracle_v1/pairs.json')
    if raw.exists() and hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(raw.read_bytes()).digest():
        raise ValueError(f'different versions of {source} and {raw}')
    rows = json.loads(source.read_text())
    summary = []
    for n, depth in sorted({(r['n'], r['depth']) for r in rows}):
        group = [r for r in rows if (r['n'], r['depth']) == (n, depth)]
        assert len({(r['family'], r['pair']) for r in group}) == len(group) == 96
        for family in ('all', *sorted({r['family'] for r in group})):
            chosen = [r for r in group if family == 'all' or r['family'] == family]
            summary.append(dict(actions=n, depth=depth, family=family, **analyze(chosen)))
    out = Path('research/retrospective/evidence/oracle.csv')
    out.parent.mkdir(exist_ok=True, parents=True)
    with out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == '__main__':
    main()
