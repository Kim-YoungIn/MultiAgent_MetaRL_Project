"""Compare training and evaluation at an identical reduced sample budget."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.structured_scale.run import FAMILIES
from . import batch, study


def run(output, use_cuda):
    torch.set_num_threads(1)
    study.KS = (0, 4, 8, 16, 24)
    cfg = dict(teacher_updates=16,batch=32,ks=study.KS,h=32)
    prior = np.zeros(20)
    seeds = np.arange(411000,411032)
    plan = batch.prepare(seeds,[FAMILIES[i%4] for i in range(len(seeds))],24,32)
    # Warm Python imports, GEMM, GPU initialization and the optimizer once.
    actor = study.BeliefActor(seed=216199)
    torch.optim.Adam(actor.parameters())
    if use_cuda:
        study.resolve_device('cuda')
        batch.train(prior,216199,{**cfg,'teacher_updates':1},device='cuda',sampling='cuda',belief=True)
    methods = batch.METHODS[:2]

    start = perf_counter()
    original = {m:study.train_teacher(prior,216199,cfg,device='cpu',belief=m=='learned') for m in methods}
    original_train = perf_counter()-start
    start = perf_counter()
    reference = study.evaluate(prior,original['learned'],seeds,h=32,blind_teacher=original['blind'])
    reference_control = study.evaluate(prior,original['learned'],seeds,h=32,blind_teacher=original['blind'],controlled=True)
    original_eval = perf_counter()-start
    measurements = [dict(training='scalar CPU',sampling='scalar CPU',
                         seconds=original_train)]
    candidates = [('cpu','cpu')]
    if use_cuda:
        candidates.extend([('cuda','cpu'),('cuda','cuda')])
    batched_models = None
    for device,sampling in candidates:
        start=perf_counter()
        trained=batch.train_group(prior,216199,cfg,device=device,sampling=sampling,methods=methods)
        if device=='cuda':
            torch.cuda.synchronize()
        duration=perf_counter()-start
        cpu_trained={m:model.cpu() for m,model in trained.items()}
        delta=max(float((tensor-original[m].state_dict()[name]).abs().max().detach())
                  for m,v in cpu_trained.items() for name,tensor in v.state_dict().items())
        measurements.append(dict(training=device,sampling=sampling,seconds=duration,
                                 maximum_weight_absolute_difference=delta))
        if device=='cpu':
            batched_models=cpu_trained
    start=perf_counter()
    results,_=batch.evaluate({'reward_auc':prior},original,plan,cfg)
    batched_eval=perf_counter()-start
    for name,matches in [('reward_auc',reference),('controlled_auc',reference_control)]:
        for field in ('curves','auc','total_return'):
            for method in batch.METHODS:
                np.testing.assert_allclose(results[name][field][method],matches[field][method],
                                           rtol=0.,atol=1e-12)
    summary=dict(config=cfg,reset_samples=len(seeds),training=measurements,
                 scalar_evaluation_seconds=original_eval,batch_evaluation_seconds=batched_eval,
                 corresponding_scalar_and_batch_eval_agree=True,torch=str(torch.__version__),
                 gpu=torch.cuda.get_device_name() if use_cuda else None)
    output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--cuda',action='store_true')
    args=p.parse_args()
    run(args.output,args.cuda)
