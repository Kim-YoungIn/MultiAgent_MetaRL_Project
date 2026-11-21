"""Contracts for variable-size logical game distributions."""
import copy
import random
import unittest

from mamrl.diverse_catalog import build_diverse_manifest, validate_diverse_manifest
from mamrl.relational_rules import relational_from_dict
from mamrl.rules import compile_rule


class DiverseCatalogContract(unittest.TestCase):
    def test_repeatability_split_prefix_and_global_rng_isolation(self):
        before = random.getstate()
        small = build_diverse_manifest(seed=19, train_rules=8, fresh_rules=8, structural_rules=8)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(small, build_diverse_manifest(seed=19, train_rules=8, fresh_rules=8, structural_rules=8))
        large = build_diverse_manifest(seed=19, train_rules=32, fresh_rules=8, structural_rules=8)
        focused = build_diverse_manifest(seed=19, train_rules=32, fresh_rules=8, structural_rules=8, diversity='focused')
        for split in ('fresh_rules', 'structural'):
            select = lambda m: [r for r in m['rules'] if r['split'] == split]
            self.assertEqual(select(small), select(large))
            self.assertEqual(select(large), select(focused))
        self.assertEqual([r for r in small['rules'] if r['split'] == 'train'],
                         [r for r in large['rules'] if r['split'] == 'train'][:8])
        self.assertEqual({r['family'] for r in focused['rules'] if r['split'] == 'train'}, {'contest'})
        self.assertEqual({r['family'] for r in large['rules'] if r['split'] == 'train'},
                         {'social', 'contest', 'reciprocity', 'hidden'})
        for manifest in (small, large, focused):
            fingerprints = [r['canonical_fingerprint'] for r in manifest['rules']]
            self.assertEqual(len(fingerprints), len(set(fingerprints)))
            self.assertIs(validate_diverse_manifest(manifest), manifest)

    def test_focus_supports_128_distinct_games(self):
        manifest = build_diverse_manifest(seed=20261118, train_rules=128, fresh_rules=4,
                                         structural_rules=4, diversity='focused')
        self.assertEqual(len(manifest['rules']), 136)
        self.assertEqual(len({r['canonical_fingerprint'] for r in manifest['rules']}), 136)
        self.assertEqual(manifest['summary']['train']['families'], {'contest': 128})
        validate_diverse_manifest(manifest)

    def test_action_count_and_compiler_symmetry(self):
        for n in (3, 4, 5):
            manifest = build_diverse_manifest(n=n, seed=41, train_rules=2, fresh_rules=2,
                                              structural_rules=2)
            validate_diverse_manifest(manifest)
            for entry in manifest['rules']:
                rule = relational_from_dict(entry['rule'])
                tensor = compile_rule(rule)
                self.assertEqual(rule.n, n)
                self.assertEqual(len(entry['common_payoffs']), n ** 4)
                self.assertAlmostEqual(min(entry['common_payoffs']), -1.)
                self.assertAlmostEqual(max(entry['common_payoffs']), 1.)
                for p, q, a, b in ((0, 1, 2, 0), (n-1, 0, 1, n-1)):
                    self.assertEqual(tensor[0, p, q, a, b], tensor[1, q, p, b, a])

    def test_manifest_detects_payoff_and_provenance_tampering(self):
        manifest = build_diverse_manifest(seed=24, train_rules=2, fresh_rules=2, structural_rules=2)
        for field in ('seed', 'common_payoffs', 'normalization', 'attempt'):
            changed = copy.deepcopy(manifest)
            row = changed['rules'][0]
            if field == 'seed': row[field] += 1
            elif field == 'attempt': row[field] += 1
            elif field == 'common_payoffs': row[field][0] += .1
            else: row[field]['shift'] += .1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_diverse_manifest(changed)
        with self.assertRaises(ValueError): build_diverse_manifest(n=6)
        with self.assertRaises(ValueError): build_diverse_manifest(diversity='other')
        with self.assertRaises(ValueError): build_diverse_manifest(train_rules=0)


if __name__ == '__main__':
    unittest.main()
