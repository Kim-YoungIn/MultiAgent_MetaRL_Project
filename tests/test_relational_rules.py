"""Hand-derived semantic-rule, payoff-axis and boundary contracts."""
from dataclasses import FrozenInstanceError
from itertools import product
import copy
import json
import random
import unittest

from mamrl.environment import MemoryOneEnv
from mamrl.rules import compile_rule


class RelationalRuleContract(unittest.TestCase):
    def test_social_rule_is_exact_embedded_pd_and_symmetric_for_every_axis(self):
        from mamrl.relational_rules import social_rule, compile_relational_rule, evaluate_relational_rule

        for n in (3, 4, 5):
            rule = social_rule(n=n, T=5, R=3, P=1, S=0)
            tensor = compile_relational_rule(rule)
            self.assertEqual(tensor, compile_rule(rule))
            self.assertEqual(tensor.shape, (2, n, n, n, n))
            for p, q in product(range(n), repeat=2):
                quadrants = set()
                for a, b in product(range(n), repeat=2):
                    own_c, other_c = a == p, b == q
                    quadrants.add((own_c, other_c))
                    expected0 = 3 if own_c and other_c else 0 if own_c else 5 if other_c else 1
                    expected1 = 3 if own_c and other_c else 5 if own_c else 0 if other_c else 1
                    self.assertEqual((tensor[0,p,q,a,b], tensor[1,p,q,a,b]), (expected0, expected1))
                    self.assertEqual(evaluate_relational_rule(rule, (p,q), (a,b)), (expected0, expected1))
                    self.assertEqual(tensor[1,p,q,a,b], tensor[0,q,p,b,a])
                self.assertEqual(quadrants, {(False,False), (False,True), (True,False), (True,True)})
                # Every unilateral change dominates repeat for fixed opposing action.
                for b in range(n):
                    repeated = tensor[0,p,q,p,b]
                    for a in range(n):
                        if a != p:
                            self.assertGreater(tensor[0,p,q,a,b], repeated)
            with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
                rule.n = n + 1

    def test_joint_losing_to_previous_opponent_and_hidden_bonus(self):
        from mamrl.relational_rules import Relation, Reward, RelationalRule, evaluate_relational_rule

        # Explicit, user-defined table, independent of any numeric label convention.
        outcomes = ((0, 1, -1), (-1, 0, 1), (1, -1, 0))
        loses_own = Relation('lose', left='own', right='prev_other')
        loses_other = Relation('lose', left='other', right='prev_own')
        both_lose = Relation('and', children=(loses_own, loses_other))
        own_zero = Relation('eq', left='own', right=0)
        base = Reward('outcome', left='own', right='other')
        bonus = Reward('if', condition=both_lose,
                       children=(Reward('const', value=2), Reward('const', value=0)))
        hidden = Reward('if', condition=own_zero,
                        children=(Reward('const', value=4), Reward('const', value=0)))
        rule = RelationalRule(3, outcomes, Reward('add', children=(Reward('add', children=(base, bonus)), hidden)))
        tensor = compile_rule(rule)
        triggers = 0
        for p, q, a, b in product(range(3), repeat=4):
            condition = outcomes[a][q] == -1 and outcomes[b][p] == -1
            triggers += condition
            expected0 = outcomes[a][b] + 2*condition + 4*(a == 0)
            expected1 = outcomes[b][a] + 2*condition + 4*(b == 0)
            self.assertEqual(evaluate_relational_rule(rule, (p,q), (a,b)), (expected0,expected1))
            self.assertEqual((tensor[0,p,q,a,b],tensor[1,p,q,a,b]), (expected0,expected1))
        self.assertEqual(triggers, 9)

    def test_ordered_exception_overrides_result_only_when_condition_holds(self):
        from mamrl.relational_rules import Relation, Reward, RelationalRule, evaluate_relational_rule

        outcomes = ((0, 1, -1), (-1, 0, 1), (1, -1, 0))
        first = Relation('eq', left='own', right='prev_own')
        second = Relation('eq', left='other', right='prev_other')
        exception = Relation('and', children=(first, second))
        base = Reward('outcome', left='own', right='other')
        reversed_outcome = Reward('scale', value=-1, children=(base,))
        rule = RelationalRule(3, outcomes, Reward('if', condition=exception, children=(reversed_outcome, base)))
        for p,q,a,b in product(range(3), repeat=4):
            sign = -1 if a == p and b == q else 1
            self.assertEqual(evaluate_relational_rule(rule,(p,q),(a,b)),
                             (sign*outcomes[a][b], sign*outcomes[b][a]))

    def test_draw_uses_table_including_even_n_opposite_actions(self):
        from mamrl.relational_rules import Relation, Reward, RelationalRule, cyclic_outcomes, evaluate_relational_rule

        outcomes = cyclic_outcomes(4)
        condition = Relation('draw', left='own', right='other')
        rule = RelationalRule(4,outcomes,Reward('if',condition=condition,
                             children=(Reward('const',value=8),Reward('const',value=-2))))
        for a,b in product(range(4),repeat=2):
            expected = 8 if (a-b)%4 in (0,2) else -2
            self.assertEqual(evaluate_relational_rule(rule,(0,1),(a,b)), (expected,expected))

    def test_codec_roundtrip_is_strict_and_scale_is_positive(self):
        from mamrl.relational_rules import social_rule, relational_to_dict, relational_from_dict

        rule=social_rule(n=3,T=5,R=3,P=1,S=0)
        encoded=relational_to_dict(rule)
        self.assertEqual(relational_from_dict(json.loads(json.dumps(encoded))),rule)
        for field,value in (('format','unknown'),('version',True),('version',2),('n',True),
                            ('scale',0),('scale',-1),('scale',float('nan')),('shift',float('inf'))):
            bad=copy.deepcopy(encoded);bad[field]=value
            with self.subTest(field=field,value=value), self.assertRaises((TypeError,ValueError)):
                relational_from_dict(bad)
        bad=copy.deepcopy(encoded);bad['unexpected']='not permitted'
        with self.assertRaises((TypeError,ValueError)):
            relational_from_dict(bad)
        for matrix in (((0,1,-1),(-1,0,1),(1,-1,1)),
                       ((0,1,-1),(1,0,1),(1,-1,0)),
                       ((0,1),(-1,0))):
            bad=copy.deepcopy(encoded);bad['outcomes']=[list(row) for row in matrix]
            with self.subTest(matrix=matrix), self.assertRaises((TypeError,ValueError)):
                relational_from_dict(bad)

    def test_generation_is_seeded_and_preserves_global_rng(self):
        from mamrl.relational_rules import generate_relational_rule, relational_to_dict, relational_from_dict

        before=random.getstate()
        for family in ('social','contest','reciprocity','hidden','exception'):
            for n in (3,4,5):
                first=generate_relational_rule(n=n,seed=294,family=family)
                second=generate_relational_rule(n=n,seed=294,family=family)
                self.assertEqual(first,second)
                self.assertEqual(compile_rule(first),compile_rule(second))
                self.assertEqual(relational_from_dict(relational_to_dict(first)), first)
        self.assertEqual(before,random.getstate())
        for seed in (None,-1,True,1.2):
            with self.subTest(seed=seed), self.assertRaises((TypeError,ValueError)):
                generate_relational_rule(n=3,seed=seed,family='social')
        with self.assertRaises((TypeError,ValueError)):
            generate_relational_rule(n=3,seed=1,family='invalid')

    def test_reset_observation_is_random_without_rule_or_horizon_leak(self):
        from mamrl.relational_rules import social_rule
        n=3
        rule=social_rule(n=n)
        env=MemoryOneEnv(compile_rule(rule),max_steps=2)
        for seed in (0,7,331):
            observations=env.reset(seed=seed)
            rng=random.Random(seed)
            p,q=rng.randrange(n),rng.randrange(n)
            self.assertEqual((observations[0].previous_own_action,observations[0].previous_opponent_action),(p,q))
            self.assertEqual((observations[1].previous_own_action,observations[1].previous_opponent_action),(q,p))
            self.assertEqual(observations[0].previous_own_reward,0.0)
            self.assertTrue(observations[0].episode_start_flag)
            result=env.step(p,(q+1)%n)
            self.assertEqual(result.rewards,(0,5))
            self.assertFalse(result.truncated)
            self.assertEqual(set(result.observations[0].__dataclass_fields__),
                             {'previous_own_action','previous_opponent_action','previous_own_reward','episode_start_flag'})
            self.assertTrue(env.step(p,q).truncated)
            with self.assertRaises(RuntimeError): env.step(0,0)


if __name__ == '__main__':
    unittest.main()
