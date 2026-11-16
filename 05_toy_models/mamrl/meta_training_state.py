"""Atomic, update-boundary restart state for meta-exploration training."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch


def context_digest(config, manifest, suite, device):
    config = dict(config)
    config.pop('device', None)  # auto and an explicit equivalent device may resume.
    return hashlib.sha256(json.dumps({'config':config,'manifest':manifest,'suite':suite,
                                     'device':str(device)}, sort_keys=True).encode()).hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value, list): return [cpu_tree(item) for item in value]
    if isinstance(value, tuple): return tuple(cpu_tree(item) for item in value)
    return value


def save_training_state(path, *, model, initial, optimizer, sampler, history, context):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'version':1,'context':context,'torch_version':str(torch.__version__),
             'model':cpu_tree(model.state_dict()),'initial':cpu_tree(initial.state_dict()),
             'optimizer':cpu_tree(optimizer.state_dict()),'sampler':sampler.getstate(),
             'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all() if model.device.type=='cuda' else [],
             'history':history,'completed_updates':len(history)}
    fd, temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def restore_training_state(path, *, model, initial, optimizer, sampler, context):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('version')!=1 or state.get('context')!=context:
        raise ValueError('resume configuration/catalog/device mismatch')
    if state.get('torch_version')!=str(torch.__version__):
        raise ValueError('resume requires the same PyTorch version')
    if state['completed_updates']!=len(state['history']):
        raise ValueError('resume update/history mismatch')
    model.load_state_dict(state['model']);initial.load_state_dict(state['initial'])
    optimizer.load_state_dict(state['optimizer'])
    # Adam step counters stay on CPU; moment tensors follow parameters.
    for parameter, values in optimizer.state.items():
        for key,value in values.items():
            if isinstance(value,torch.Tensor) and key!='step':values[key]=value.to(parameter.device)
    sampler.setstate(state['sampler']);torch.set_rng_state(state['torch_rng'])
    if model.device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['history']
