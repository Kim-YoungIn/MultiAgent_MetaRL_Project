"""Record consistency checks from hand-calculated reward terms and schema."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mamrl.environment import MemoryOneEnv
from mamrl.policy import PolicyOutput
from mamrl.recording import build_record, load_record, save_run, validate_record
from mamrl.rollout import run_episode
from mamrl.rules import RuleSpec, RuleTerm, compile_rule


class ScriptedPolicy:
    def __init__(self, actions):
        self.actions = actions

    def initial_state(self, *, n, seed=None):
        return 0

    def act(self, observation, state, *, n):
        return PolicyOutput(self.actions[state], state + 1)


def fixture(policy_names=('ScriptedPolicy', 'ScriptedPolicy')):
    spec = RuleSpec(3, (RuleTerm('coordination', 2), RuleTerm('repeat_own', -1),
                        RuleTerm('copy_opponent', 3), RuleTerm('action_bonus', 4, 1)))
    tensor = compile_rule(spec)
    policies = (ScriptedPolicy([1, 2, 0]), ScriptedPolicy([1, 1, 2]))
    episode = run_episode(MemoryOneEnv(tensor, max_steps=3), policies,
                          seed=8, policy_seeds=(9, 10))
    record = build_record(spec, tensor, episode, family='mixed', seed=8,
                          policy_seeds=(9, 10), policy_names=policy_names)
    return spec, tensor, episode, record


def expected_terms(p0, p1, a0, a1):
    return [[2*(a0 == a1), 2*(a0 == a1)],
            [-(a0 == p0), -(a1 == p1)],
            [3*(a0 == p1), 3*(a1 == p0)],
            [4*(a0 == 1), 4*(a1 == 1)]]


class RecordContract(unittest.TestCase):
    def test_record_contents_contributions_summary_hash_and_observations(self):
        spec, tensor, episode, record = fixture()
        self.assertIsNone(validate_record(record))
        self.assertEqual(record['schema_version'], 1)
        self.assertEqual(record['metadata']['n'], 3)
        self.assertEqual(record['metadata']['family'], 'mixed')
        self.assertEqual(record['metadata']['seed'], 8)
        self.assertEqual(record['metadata']['policy_seeds'], [9,10])
        self.assertEqual(record['metadata']['policies'], ['ScriptedPolicy','ScriptedPolicy'])
        self.assertTrue(record['metadata']['python_version'])
        self.assertTrue(record['metadata']['semantics_version'])
        self.assertEqual(record['rule']['n'], 3)
        self.assertEqual(len(record['rule_descriptions']), 4)
        self.assertEqual(record['tensor']['shape'], [2,3,3,3,3])
        payload = {'shape': record['tensor']['shape'], 'values': record['tensor']['values']}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                            allow_nan=False).encode('utf-8')).hexdigest()
        self.assertEqual(record['tensor']['sha256'], digest)
        cumulative = [0, 0]
        action_counts = [[0]*3 for _ in range(2)]
        self.assertEqual(len(record['transitions']), 3)
        for t, source in zip(record['transitions'], episode.transitions):
            p0, p1 = source.observations[0].previous_own_action, source.observations[1].previous_own_action
            a0, a1 = source.actions
            terms = expected_terms(p0, p1, a0, a1)
            self.assertEqual(t['turn'], 1 + sum(action_counts[0]))
            self.assertEqual(t['previous_joint_action'], [p0,p1])
            self.assertEqual(t['actions'], [a0,a1])
            self.assertEqual(t['observations'], [asdict(o) for o in source.observations])
            self.assertEqual(t['next_observations'], [asdict(o) for o in source.next_observations])
            self.assertEqual([r['contributions'] for r in t['reward_terms']], terms)
            for i in (0,1):
                action_counts[i][source.actions[i]] += 1
                reward = sum(term[i] for term in terms)
                self.assertEqual(t['rewards'][i], reward)
                cumulative[i] += reward
            self.assertEqual(t['cumulative_rewards'], cumulative)
        self.assertEqual(record['summary']['steps'], 3)
        self.assertEqual(record['summary']['total_rewards'], cumulative)
        self.assertEqual(record['summary']['mean_rewards'], [v/3 for v in cumulative])
        self.assertEqual(record['summary']['action_counts'], action_counts)
        self.assertFalse(record['summary']['terminated'])
        self.assertTrue(record['summary']['truncated'])
        # Recording cannot alter the source rollout, its policy observations, or tensor.
        self.assertEqual([t.actions for t in episode.transitions], [(1,1), (2,1), (0,2)])
        record['transitions'][0]['observations'][0]['previous_own_action'] = 99
        self.assertNotEqual(episode.transitions[0].observations[0].previous_own_action, 99)
        self.assertEqual(tensor.shape, (2,3,3,3,3))

    def test_save_load_roundtrip_and_refuse_to_overwrite(self):
        _, _, _, record = fixture()
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'run'
            json_path, html_path = save_run(record, target)
            self.assertEqual(json_path, target / 'run.json')
            self.assertEqual(html_path, target / 'report.html')
            self.assertEqual(load_record(json_path), record)
            self.assertTrue(html_path.is_file())
            before = (json_path.read_bytes(), html_path.read_bytes())
            with self.assertRaises((FileExistsError, ValueError)):
                save_run(record, target)
            self.assertEqual((json_path.read_bytes(), html_path.read_bytes()), before)
            other = Path(tmp) / 'existing-file'
            other.write_text('preserve', encoding='utf-8')
            with self.assertRaises((FileExistsError, ValueError, NotADirectoryError)):
                save_run(record, other)
            self.assertEqual(other.read_text(encoding='utf-8'), 'preserve')

    def test_validate_rejects_tampered_records(self):
        _, _, _, original = fixture()
        def check(name, mutate):
            candidate = deepcopy(original)
            mutate(candidate)
            with self.subTest(name=name), self.assertRaises((ValueError, TypeError)):
                validate_record(candidate)
        check('version', lambda r: r.__setitem__('schema_version', 2))
        check('schema-type', lambda r: r.__setitem__('schema_version', True))
        check('n mismatch', lambda r: r['metadata'].__setitem__('n', 4))
        check('tensor shape', lambda r: r['tensor']['shape'].__setitem__(1, 4))
        check('tensor hash', lambda r: r['tensor'].__setitem__('sha256', 'wrong'))
        check('tensor reward', lambda r: r['tensor']['values'].__setitem__(0, 999))
        check('tensor hash representation', lambda r: r['tensor']['values'].__setitem__(0, int(r['tensor']['values'][0])))
        check('nonfinite tensor', lambda r: r['tensor']['values'].__setitem__(0, float('nan')))
        check('nonfinite coefficient', lambda r: r['rule']['terms'][0].__setitem__('coefficient', float('inf')))
        check('step number', lambda r: r['transitions'][0].__setitem__('turn', 2))
        check('observed local ordering', lambda r: r['transitions'][0]['observations'][0].__setitem__('previous_opponent_action', 999))
        check('reward', lambda r: r['transitions'][0]['rewards'].__setitem__(0, 999))
        check('per-term contribution', lambda r: r['transitions'][0]['reward_terms'][0]['contributions'].__setitem__(0, 999))
        check('running total', lambda r: r['transitions'][-1]['cumulative_rewards'].__setitem__(0, 999))
        check('false early termination', lambda r: r['transitions'][0].__setitem__('truncated', True))
        check('false final truncation', lambda r: r['transitions'][-1].__setitem__('truncated', False))
        check('summary total', lambda r: r['summary']['total_rewards'].__setitem__(0, 999))
        check('summary mean', lambda r: r['summary']['mean_rewards'].__setitem__(0, 999))
        check('unrepresentable summary mean', lambda r: r['summary']['mean_rewards'].__setitem__(0, 10**400))
        check('summary action counts', lambda r: r['summary']['action_counts'][0].__setitem__(0, 999))
        check('missing transitions', lambda r: r.__delitem__('transitions'))

    def test_load_rejects_invalid_json_and_nonfinite_literals(self):
        _, _, _, record = fixture()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.json'
            path.write_text('{invalid', encoding='utf-8')
            with self.assertRaises((ValueError, TypeError)):
                load_record(path)
            path.write_text(json.dumps(record).replace('"schema_version": 1', '"schema_version": NaN'), encoding='utf-8')
            with self.assertRaises((ValueError, TypeError)):
                load_record(path)

    def test_equivalent_integer_value_with_its_own_payload_hash_is_valid(self):
        _, _, _, record = fixture()
        record['tensor']['values'][0] = int(record['tensor']['values'][0])
        payload = {'shape': record['tensor']['shape'], 'values': record['tensor']['values']}
        record['tensor']['sha256'] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        ).hexdigest()
        self.assertIsNone(validate_record(record))


if __name__ == '__main__':
    unittest.main()
