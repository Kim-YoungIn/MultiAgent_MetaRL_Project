"""Count completed D-031 fits and paired evaluations from atomic artifacts."""
import argparse
import json
from pathlib import Path
from experiment.protocol import plan


def status(root):
    cells = plan(False)
    progress = {}
    for cell in cells:
        condition = cell['condition']
        folder = root/'cells'/condition/f"c{cell['catalog']}"/f"j{cell['replica']}"
        count = progress.setdefault(condition,dict(fits=0,planned_fits=0,home_evaluated=0,
                                                    planned_cells=0,cross_evaluated=0))
        count['planned_cells'] += 1
        count['planned_fits'] += len(cell['methods'])
        count['fits'] += sum((folder/'training'/m/'ledger.json').is_file() for m in cell['methods'])
        n = cell['train_n']
        expected = (*cell['methods'],'random','initial','epsilon','balanced')
        if all((folder/f'n{n}'/f'{m}.receipt.json').is_file() for m in expected):
            count['home_evaluated'] += 1
        if condition in ('primary','n5') and all(
            (folder/f'n{target}'/f'{m}.receipt.json').is_file()
            for target in (3,4,5) if target!=n for m in ('M','P')
        ):
            count['cross_evaluated'] += 1
    return progress


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    args=parser.parse_args()
    if not (args.study/'study.json').is_file():
        parser.error('expected a sealed D-031 study directory')
    print(json.dumps(status(args.study),indent=2))
