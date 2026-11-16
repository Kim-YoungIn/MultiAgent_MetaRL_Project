"""Resolve requested training devices without silently ignoring CUDA requests."""
import torch


def resolve_device(request='auto'):
    if request == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if request not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu, cuda, or auto')
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA is unavailable in this process (torch={torch.__version__}, runtime={torch.version.cuda}). '
                           'Run python -m mamrl.diagnose_device --smoke in the training terminal.')
    return torch.device(request)
