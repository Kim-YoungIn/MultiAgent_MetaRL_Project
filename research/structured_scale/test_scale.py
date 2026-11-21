"""Regression tests for scalable features, kernel, and online updates."""
import copy
import dataclasses
import unittest
import numpy as np
from mamrl.structured_games import StructuredEnv
from research.structured_scale.run import D, Model, RewardKernel, features, game, models


class ScaleContracts(unittest.TestCase):
    def test_kernel_matches_environment(self):
        rng=np.random.default_rng(884)
        for family in ('resource','sequence','reciprocity','mixed'):
            for n,depth in ((3,1),(3,3),(6,5)):
                g=game(100+depth,family,n,depth)
                env=StructuredEnv(g,max_steps=14)
                env.reset(seed=31)
                kernel=RewardKernel(g)
                for _ in range(8):
                    expected=kernel(env)
                    for a in range(n):
                        for b in range(n):
                            self.assertAlmostEqual(expected[a,b],copy.deepcopy(env).step(a,b).rewards[0],places=12)
                    env.step(int(rng.integers(n)),int(rng.integers(n)))

    def test_action_equivariance_and_padding(self):
        g=game(71,'resource',6,5)
        obs=StructuredEnv(g,max_steps=4).reset(seed=77)[0]
        perm=np.array([2,5,1,4,0,3])
        new=dataclasses.replace(obs,
            previous_own_action=int(perm[obs.previous_own_action]),
            previous_opponent_action=int(perm[obs.previous_opponent_action]),
            action_history=tuple((int(perm[a]),int(perm[b])) for a,b in obs.action_history))
        old=features(obs,6); transformed=features(new,6)
        np.testing.assert_equal(old,transformed[perm])
        self.assertEqual(old[0,2],0)
        g=game(71,'sequence',3,1)
        obs=StructuredEnv(g,max_steps=4).reset(seed=77)[0]
        np.testing.assert_equal(features(obs,3)[:,8:],0)

    def test_sequential_posterior_and_feature_ablation(self):
        rng=np.random.default_rng(12)
        prior=rng.normal(size=D)
        x=rng.normal(size=(10,D)); y=rng.normal(size=10)
        model=Model(3,prior)
        for row,r in zip(x,y):
            model.observe(np.array([row,row,row]),1,r)
        closed=np.linalg.solve(np.eye(D)+x.T@x,prior+x.T@y)
        np.testing.assert_allclose(model.mean[1],closed,atol=1e-10)
        no_state,no_history=models(3,prior,prior)[4:]
        self.assertTrue(np.all(no_state.mask[1:3]==0))
        self.assertTrue(np.all(no_history.mask[8:]==0))


if __name__=='__main__': unittest.main()
