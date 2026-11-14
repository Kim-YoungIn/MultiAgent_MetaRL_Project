"""Contract-based checks of generated logical game expressions and global payoff axes."""
from dataclasses import FrozenInstanceError
import copy
import itertools
import json
import math
import os
import random
import subprocess
import sys
import unittest

from mamrl.logical_rules import (Predicate, Payoff, LogicalRule, evaluate_logical_rule,
                                 compile_logical_rule, generate_logical_rule,
                                 rule_to_dict, rule_from_dict)
from mamrl.rules import compile_rule


def eq(left, right):
    return Predicate('eq', left=left, right=right)


def ne(left, right):
    return Predicate('ne', left=left, right=right)


def const(value):
    return Payoff('const', value=value)


def conditional(predicate, yes, no):
    return Payoff('if', condition=predicate, children=(yes, no))


def fixture(n):
    # +2 if players match now after distinct previous actions, else -1;
    # an additional -2 when own action copies the previous opposing action.
    condition = Predicate('and', children=(eq('own','other'), ne('prev_own','prev_other')))
    main = conditional(condition, const(2), const(-1))
    cost = Payoff('scale', value=-.5, children=(conditional(eq('own','prev_other'), const(4), const(0)),))
    return LogicalRule(n, Payoff('add', children=(main,cost)))


class GeneratedGrammarContract(unittest.TestCase):
    def test_hand_formula_for_every_tensor_entry_n3_n4_n5(self):
        for n in (3,4,5):
            rule = fixture(n)
            tensor = compile_logical_rule(rule)
            self.assertEqual(tensor, compile_rule(rule))
            self.assertEqual(tensor.shape, (2,n,n,n,n))
            for x,y,a,b in itertools.product(range(n), repeat=4):
                match = a == b and x != y
                expected0 = (2 if match else -1) - 2*(a == y)
                expected1 = (2 if match else -1) - 2*(b == x)
                self.assertEqual((tensor[0,x,y,a,b],tensor[1,x,y,a,b]), (expected0,expected1))
                self.assertEqual(evaluate_logical_rule(rule,(x,y),(a,b)), (expected0,expected1))
                self.assertEqual(tensor[1,x,y,a,b], tensor[0,y,x,b,a])
            with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
                rule.n = n+1

    def test_negation_disjunction_and_conditional_priority(self):
        # if NOT (own==other OR own==prev_own), return 3; otherwise -1.
        branch = Predicate('not', children=(Predicate('or', children=(eq('own','other'),eq('own','prev_own'))),))
        rule = LogicalRule(3, conditional(branch,const(3),const(-1)))
        for x,y,a,b in itertools.product(range(3), repeat=4):
            got = evaluate_logical_rule(rule,(x,y),(a,b))
            self.assertEqual(got, (3 if a!=b and a!=x else -1,
                                   3 if b!=a and b!=y else -1))

    def test_json_roundtrip_and_strict_envelope(self):
        rule = fixture(3)
        record = rule_to_dict(rule)
        loaded = json.loads(json.dumps(record, allow_nan=False))
        self.assertEqual(rule_from_dict(loaded),rule)
        self.assertEqual(compile_rule(rule_from_dict(loaded)), compile_rule(rule))
        invalid = copy.deepcopy(loaded)
        invalid['extra'] = 'not permitted'
        with self.assertRaises((TypeError,ValueError)):
            rule_from_dict(invalid)
        for version in (True,2,1.0,'1'):
            invalid = copy.deepcopy(loaded)
            invalid['version'] = version
            with self.subTest(version=version), self.assertRaises((TypeError,ValueError)):
                rule_from_dict(invalid)
        invalid = copy.deepcopy(loaded)
        invalid['root']['children'][0]['condition']['children'][0]['unexpected'] = 1
        with self.assertRaises((TypeError,ValueError)):
            rule_from_dict(invalid)
        for bad in (None, [], 'rule'):
            with self.subTest(bad=bad), self.assertRaises((TypeError,ValueError)):
                rule_from_dict(bad)

    def test_bad_operand_operator_arity_and_numeric_values_rejected(self):
        bad_conditions = [
            lambda: Predicate('eq', left='own', right='time'),
            lambda: Predicate('eq', left='own', right=True),
            lambda: Predicate('eq', left='own'),
            lambda: Predicate('and', children=(eq('own','other'),)),
            lambda: Predicate('not', children=(eq('own','other'),eq('own','prev_own'))),
            lambda: Predicate('mystery',left='own',right='other'),
        ]
        for build in bad_conditions:
            with self.subTest(build=build), self.assertRaises((TypeError,ValueError)):
                LogicalRule(3, conditional(build(),const(1),const(0)))
        for value in (True, float('nan'), float('inf'), float('-inf'), '1'):
            with self.subTest(value=value), self.assertRaises((TypeError,ValueError)):
                LogicalRule(3,const(value))
        for bad in (-1,3):
            with self.subTest(action_constant=bad), self.assertRaises((TypeError,ValueError)):
                LogicalRule(3,conditional(eq('own',bad),const(1),const(0)))
        for args in (dict(op='if',condition=eq('own','other'),children=(const(1),)),
                     dict(op='add',children=(const(1),)),
                     dict(op='scale',value=2,children=()), dict(op='unknown')):
            with self.subTest(args=args), self.assertRaises((TypeError,ValueError)):
                LogicalRule(3,Payoff(**args))

    def test_complexity_bound(self):
        condition = eq('own','other')
        for _ in range(8):
            condition = Predicate('not',children=(condition,))
        with self.assertRaises((TypeError,ValueError)):
            LogicalRule(3,conditional(condition,const(1),const(0)))
        # A tree can exceed 63 nodes while every path remains within depth 6.
        def full(level):
            return const(0) if level == 0 else conditional(eq('own','other'),full(level-1),full(level-1))
        with self.assertRaises((TypeError,ValueError)):
            LogicalRule(3,full(5))

    def test_generator_seed_determinism_and_global_rng_isolation(self):
        before = random.getstate()
        for structure in ('atom','and','or','nested'):
            for n in (3,4,5):
                x = generate_logical_rule(n=n,seed=123,structure=structure)
                y = generate_logical_rule(n=n,seed=123,structure=structure)
                self.assertEqual(x,y)
                self.assertEqual(compile_rule(x),compile_rule(y))
                self.assertEqual(rule_from_dict(rule_to_dict(x)),x)
        self.assertEqual(random.getstate(),before)
        for bad in (-1,True,1.2,'3',None):
            with self.subTest(seed=bad), self.assertRaises((ValueError,TypeError)):
                generate_logical_rule(3,seed=bad)
        with self.assertRaises((TypeError,ValueError)):
            generate_logical_rule(3,seed=0,structure='unknown')

    def test_seed_reproducible_across_process_hash_seeds(self):
        program = ("import json; from mamrl.logical_rules import generate_logical_rule,rule_to_dict; "
                   "print(json.dumps(rule_to_dict(generate_logical_rule(3,seed=17,structure='nested')),sort_keys=True))")
        outputs = [subprocess.check_output([sys.executable,'-c',program],text=True,
                     env=dict(os.environ,PYTHONHASHSEED=h)) for h in ('1','2')]
        self.assertEqual(outputs[0],outputs[1])


if __name__ == '__main__':
    unittest.main()
