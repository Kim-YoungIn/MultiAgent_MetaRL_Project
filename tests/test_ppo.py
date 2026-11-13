"""Numerical PPO and finite-trial GAE contract checks."""
import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from mamrl.ppo import compute_gae, ppo_loss


@unittest.skipIf(torch is None, 'optional training dependency torch is not installed')
class PPOContract(unittest.TestCase):
    def test_gae_carries_across_intermediate_episode_end(self):
        # Two transitions straddle the episode reset. There is no episode mask.
        # d1 = 2 - .25 = 1.75, A1 = 1.75
        # d0 = 1 + .9*.25 - .5 = .725; A0 = .725 + .9*.8*1.75 = 1.985
        advantage, returns = compute_gae([1,2], [.5,.25], gamma=.9, gae_lambda=.8, bootstrap_value=0)
        self.assertEqual(tuple(advantage.shape), (2,))
        self.assertEqual(advantage.dtype, torch.float32)
        torch.testing.assert_close(advantage, torch.tensor([1.985,1.75]))
        torch.testing.assert_close(returns, torch.tensor([2.485,2.0]))

    def test_gae_nonzero_bootstrap_and_invalid_inputs(self):
        adv, ret = compute_gae([1], [0.5], gamma=.9, gae_lambda=.2, bootstrap_value=2)
        torch.testing.assert_close(adv, torch.tensor([2.3]))
        torch.testing.assert_close(ret, torch.tensor([2.8]))
        for rewards, values in (([], []), ([1], []), ([float('nan')], [1]), ([1], [float('inf')])):
            with self.subTest(rewards=rewards, values=values), self.assertRaises((ValueError, TypeError)):
                compute_gae(rewards, values)
        for gamma, lam in ((-0.1,.9),(1.1,.9),(.9,-.1),(.9,1.1),(float('nan'),.9)):
            with self.subTest(gamma=gamma, lam=lam), self.assertRaises((ValueError, TypeError)):
                compute_gae([1], [0], gamma=gamma, gae_lambda=lam)

    def test_positive_negative_clip_math_and_target_detach(self):
        # min terms: [1.2, .5, -1.5, -.8], so policy loss = +.15.
        new_log = torch.tensor([math.log(1.5), math.log(.5), math.log(1.5), math.log(.5)], requires_grad=True)
        old_log = torch.zeros(4, requires_grad=True)
        values = torch.tensor([0.,1.,2.,3.], requires_grad=True)
        returns = torch.ones(4, requires_grad=True)
        advantages = torch.tensor([1.,1.,-1.,-1.], requires_grad=True)
        entropies = torch.ones(4, requires_grad=True)
        loss, metrics = ppo_loss(new_log, old_log, values, returns, advantages, entropies,
                                 clip_ratio=.2, value_coefficient=.5, entropy_coefficient=.01)
        self.assertAlmostEqual(loss.item(), .515, places=6)
        self.assertAlmostEqual(float(metrics['policy_loss']), .15, places=6)
        self.assertAlmostEqual(float(metrics['value_loss']), .75, places=6)
        self.assertAlmostEqual(float(metrics['entropy']), 1, places=6)
        self.assertAlmostEqual(float(metrics['clip_fraction']), 1, places=6)
        expected_kl = sum((r-1)-math.log(r) for r in (1.5,.5,1.5,.5))/4
        self.assertAlmostEqual(float(metrics['approx_kl']), expected_kl, places=6)
        loss.backward()
        self.assertIsNone(old_log.grad)
        self.assertIsNone(returns.grad)
        self.assertIsNone(advantages.grad)
        self.assertEqual(new_log.grad[0].item(), 0)
        self.assertNotEqual(new_log.grad[1].item(), 0)
        self.assertNotEqual(new_log.grad[2].item(), 0)
        self.assertEqual(new_log.grad[3].item(), 0)
        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(entropies.grad)

    def test_invalid_shapes_nonfinite_and_overflow_rejected(self):
        good = torch.zeros(2)
        for new_log in (torch.tensor([float('nan'),0]), torch.tensor([1000.,0]), torch.zeros(2,1)):
            with self.subTest(new_log=new_log), self.assertRaises((ValueError, TypeError)):
                ppo_loss(new_log, good, good, good, good, good)
        with self.assertRaises((ValueError, TypeError)):
            ppo_loss(torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0))
        for clip in (-1, float('nan')):
            with self.subTest(clip=clip), self.assertRaises((ValueError, TypeError)):
                ppo_loss(good, good, good, good, good, good, clip_ratio=clip)


if __name__ == '__main__':
    unittest.main()
