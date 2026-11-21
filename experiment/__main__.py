"""Command line for the research experiment."""
import argparse
import json
from pathlib import Path

from .protocol import BLOCKS, budget, plan


def main():
    parser = argparse.ArgumentParser(description='D-031 multi-agent adaptation experiment')
    parser.add_argument('command', choices=('plan', 'prepare', 'audit', 'run', 'report'))
    parser.add_argument('--output', type=Path, help='dedicated study directory')
    parser.add_argument('--dev', action='store_true', help='use separate development seeds and short budgets')
    parser.add_argument('--block', choices=(*BLOCKS, 'all'), default='primary')
    parser.add_argument('--catalog', type=int, choices=range(1, 6))
    parser.add_argument('--replica', type=int, choices=range(1, 6))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--defer-report', action='store_true', help='skip aggregate reporting after a training shard')
    args = parser.parse_args()
    if args.command == 'plan':
        cells = plan(args.dev)
        limits = budget(args.dev)
        fits = sum(len(c['methods']) for c in cells)
        per_fit = limits['updates'] * limits['trials'] * (limits['k'] + 2*limits['h'])
        print(json.dumps({'development': args.dev, 'training_runs': fits, 'budget': limits,
                          'training_simulation_steps': per_fit*fits, 'cells': cells}, indent=2))
        return
    if args.output is None:
        parser.error('--output is required for this command')
    import torch
    from .runner import audit, exclusive, make_report, run, seal
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('PyTorch cannot access CUDA in this process')
    torch.set_num_threads(1)
    root = args.output.resolve()
    if args.command == 'report' and not (root / 'panel_sha256.json').exists():
        parser.error('study has no sealed panel')
    with exclusive(root, wait=True):
        panels = seal(root, args.dev, args.device)
    if args.command == 'audit':
        with exclusive(root / 'locks' / 'audit', wait=True):
            audit(root, panels)
    elif args.command == 'run':
        run(root, panels, args.dev, args.block, args.catalog, args.replica, args.device)
    if args.command == 'report' or (args.command == 'run' and not args.defer_report):
        with exclusive(root / 'locks' / 'report', wait=True):
            make_report(root, args.dev)
        print(root / 'summary.md')


if __name__ == '__main__':
    main()
