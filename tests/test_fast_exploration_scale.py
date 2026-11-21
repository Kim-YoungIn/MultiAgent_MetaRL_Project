"""Regression coverage for the batched rollout and resumable policy updates."""
import copy
import unittest

import numpy as np
import torch

from research.fast_exploration_v1 import batch, study
from research.structured_scale.run import FAMILIES


class BatchedStudyTests(unittest.TestCase):
    def test_scalar_equivalence_across_regimes_and_budgets(self):
        seeds = list(range(58021,58029))
        families = [FAMILIES[i%4] for i in range(len(seeds))]
        actor = study.BeliefActor(seed=51)
        prior = np.random.default_rng(12).normal(0,.02,20)
        plan = batch.prepare(seeds,families,24,32)
        warmed = batch.warm_up(plan)
        for k in (0,8,24):
            for m in batch.METHODS:
                result, explore, _, actions = batch.collect(
                    plan,warmed,prior,k,32,method=m,teacher=actor,controlled=True)
                for i,(seed,family) in enumerate(zip(seeds,families,strict=True)):
                    for regime in result:
                        frozen = regime=='controlled'
                        ref = study.rollout(seed,family,prior,k,method=m,teacher=actor,h=32,
                            freeze_assessment=frozen,neutralize_opponent=frozen,
                            opponent_learning=not frozen)
                        np.testing.assert_allclose(result[regime][i],ref['assessment_rewards'],atol=1e-12,rtol=0)
                        self.assertAlmostEqual(explore[i].sum(),ref['explore_sum'])
                        if m in batch.METHODS[:2]:
                            np.testing.assert_equal(actions[i],ref['actions'])

    def test_resets_reuse_hidden_rule(self):
        plan = batch.prepare([712,713],['mixed']*2,24,32,rule_seeds=[3199]*2)
        self.assertEqual(plan.rules[0].fingerprint(),plan.rules[1].fingerprint())
        self.assertNotEqual(plan.explore_seed[0],plan.explore_seed[1])

    def test_checkpoint_reproduces_joint_training(self):
        plan = dict(teacher_updates=4,batch=6,ks=(0,4,8,16,24),h=32)
        run = dict(device='cpu',sampling='cpu',methods=batch.METHODS[:2])
        saved = {}
        def capture(update, models, optimizers, rng, metrics):
            saved.update(update=update,models=copy.deepcopy({k:v.state_dict() for k,v in models.items()}),
                         optimizers=copy.deepcopy({k:v.state_dict() for k,v in optimizers.items()}),
                         rng=copy.deepcopy(rng.bit_generator.state))
        prior = np.arange(20)/1000
        expected = batch.train_group(prior,9264,plan,**run)
        batch.train_group(prior,9264,{**plan,'teacher_updates':2},history=capture,**run)
        resumed = batch.train_group(prior,9264,plan,resume=saved,**run)
        for name,model in expected.items():
            for key,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,resumed[name].state_dict()[key]), (name,key))


if __name__=='__main__':
    unittest.main()
