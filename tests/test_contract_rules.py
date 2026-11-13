"""Contract-first checks for tensor axes and logical reward rules."""
import itertools
import math
import os
import random
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError

from mamrl.tensor import RewardTensor
from mamrl.rules import RuleTerm, RuleSpec, generate_rule, evaluate_rule, compile_rule, matrix_to_tensor


class TensorContract(unittest.TestCase):
    def test_global_axes_shape_and_immutable_snapshot(self):
        for n in (3, 4, 5):
            values = [10000*p + 1000*x + 100*y + 10*a + b
                      for p, x, y, a, b in itertools.product(range(2), range(n), range(n), range(n), range(n))]
            tensor = RewardTensor(n, values)
            self.assertEqual(tensor.shape, (2, n, n, n, n))
            values[0] = -999
            for p, x, y, a, b in itertools.product(range(2), range(n), range(n), range(n), range(n)):
                self.assertEqual(tensor[p, x, y, a, b], 10000*p + 1000*x + 100*y + 10*a + b)
            with self.assertRaises((AttributeError, FrozenInstanceError, TypeError)):
                tensor.n = n + 1
            for key in ((-1, 0, 0, 0, 0), (2, 0, 0, 0, 0), (0, 0, 0, 0, n)):
                with self.assertRaises(IndexError):
                    _ = tensor[key]

    def test_malformed_tensor(self):
        for n in (True, 3.5, '3'):
            with self.subTest(n=n), self.assertRaises(TypeError):
                RewardTensor(n, [])
        for n in (-1, 0, 1):
            with self.subTest(n=n), self.assertRaises(ValueError):
                RewardTensor(n, [])
        with self.assertRaises(ValueError):
            RewardTensor(3, [0] * 161)
        for invalid in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                RewardTensor(3, [invalid] + [0] * 161)
        for invalid in (True, '1', 1j):
            with self.subTest(value=invalid), self.assertRaises(TypeError):
                RewardTensor(3, [invalid] + [0] * 161)

    def test_static_matrix_is_independent_of_previous_actions(self):
        for n in (3, 4, 5):
            matrices = [[[100*p + 10*a + b for b in range(n)] for a in range(n)] for p in range(2)]
            tensor = matrix_to_tensor(matrices)
            matrices[0][0][0] = 9999
            for p, x, y, a, b in itertools.product(range(2), range(n), range(n), range(n), range(n)):
                self.assertEqual(tensor[p, x, y, a, b], 100*p + 10*a + b)
        for invalid in ([], [[[0, 1], [1, 0]]], [[[0, 1], [1, 0]], [[0], [1]]]):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                matrix_to_tensor(invalid)


class RuleContract(unittest.TestCase):
    def test_malformed_rule_inputs_fail_clearly(self):
        with self.assertRaises(ValueError):
            RuleTerm('unknown')
        with self.assertRaises(ValueError):
            generate_rule(3, seed=10, family='unknown')
        with self.assertRaises(ValueError):
            RuleSpec(3, (RuleTerm('action_bonus', 1, 3),))
        with self.assertRaises(TypeError):
            RuleSpec(3, ('not-a-term',))
        for invalid in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(coefficient=invalid), self.assertRaises(ValueError):
                RuleTerm('repeat_own', invalid)
        for invalid in (True, '3', 1j):
            with self.subTest(coefficient=invalid), self.assertRaises(TypeError):
                RuleTerm('repeat_own', invalid)
        spec = RuleSpec(3, (RuleTerm('repeat_own', 1),))
        for previous, actions in (((0,), (0,1)), ((0,1), (0,1,2)),
                                  ((-1,1), (0,1)), ((0,1), (0,3))):
            with self.subTest(previous=previous, actions=actions), self.assertRaises(ValueError):
                evaluate_rule(spec, previous, actions)
        for previous, actions in ((None, (0,1)), ((0,1), (True,1))):
            with self.subTest(previous=previous, actions=actions), self.assertRaises(TypeError):
                evaluate_rule(spec, previous, actions)

    def test_fixed_composition_against_independent_formula(self):
        for n in (3, 4, 5):
            terms = [RuleTerm('coordination', 2), RuleTerm('repeat_own', -1),
                     RuleTerm('copy_opponent', 3), RuleTerm('action_bonus', 4, 1)]
            spec = RuleSpec(n, terms)
            terms.clear()  # A rule is fixed after creation, even if its input list changes.
            tensor = compile_rule(spec)
            for p0, p1, a0, a1 in itertools.product(range(n), repeat=4):
                # These expressions are derived from the written contract, not evaluate_rule.
                expected0 = (2*(a0 == a1) - (a0 == p0) + 3*(a0 == p1) + 4*(a0 == 1))
                expected1 = (2*(a0 == a1) - (a1 == p1) + 3*(a1 == p0) + 4*(a1 == 1))
                self.assertEqual((tensor[0,p0,p1,a0,a1], tensor[1,p0,p1,a0,a1]), (expected0, expected1))
                self.assertEqual(evaluate_rule(spec, (p0,p1), (a0,a1)), (expected0, expected1))
            with self.assertRaises((AttributeError, FrozenInstanceError, TypeError)):
                spec.n = n + 1

    def test_cyclic_zero_sum_tables_include_even_tie(self):
        signs = {3: (0, 1, -1), 4: (0, 1, 0, -1), 5: (0, 1, 1, -1, -1)}
        for n in (3, 4, 5):
            tensor = compile_rule(RuleSpec(n, (RuleTerm('cyclic_zero_sum', 2),)))
            for x, y, a, b in itertools.product(range(n), repeat=4):
                expected = 2 * signs[n][(a-b) % n]
                self.assertEqual(tensor[0,x,y,a,b], expected)
                self.assertEqual(tensor[1,x,y,a,b], -expected)

    def test_generated_rules_are_reproducible_and_leave_global_rng_alone(self):
        before = random.getstate()
        for n in (3, 4, 5):
            for family in ('zero_sum', 'coordination', 'memory_one', 'mixed'):
                a = generate_rule(n, seed=9876, family=family)
                b = generate_rule(n, seed=9876, family=family)
                self.assertEqual(a, b)
                self.assertEqual(compile_rule(a), compile_rule(b))
                self.assertEqual(compile_rule(a).shape, (2,n,n,n,n))
        self.assertEqual(random.getstate(), before)
        mixed = [generate_rule(3, seed=s, family='mixed') for s in range(20)]
        self.assertGreater(len({tuple(t.kind for t in spec.terms) for spec in mixed}), 1)
        self.assertTrue(any(len(spec.terms) > 1 for spec in mixed))
        for bad in (-1, True, 1.2, '3'):
            with self.subTest(seed=bad), self.assertRaises((ValueError, TypeError)):
                generate_rule(3, seed=bad)

    def test_seed_is_reproducible_across_process_hash_seeds(self):
        script = ("from mamrl.rules import generate_rule, compile_rule; "
                  "s = generate_rule(4, seed=101, family='mixed'); "
                  "print(repr(s), repr(compile_rule(s).values))")
        results = []
        for hash_seed in ('1', '2'):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            results.append(subprocess.check_output([sys.executable, '-c', script], env=env, text=True))
        self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
