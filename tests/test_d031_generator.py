import tempfile
import unittest
from pathlib import Path

from experiment.generator import candidate, canonical_values, catalog_rows
from experiment.panels import prepare, evaluation_entries, training_entries
from mamrl.rule_metrics import canonical_values as legacy_canonical


class GeneratorRevisionTest(unittest.TestCase):
    def test_canonical_fingerprints_match_legacy_all_n(self):
        for n in (3, 4, 5):
            for family in ('social', 'contest', 'reciprocity', 'hidden', 'exception'):
                item = candidate(n, 321, family)
                if item:
                    self.assertEqual(canonical_values(item[-2], n), legacy_canonical(item[-2], n))

    def test_hidden_coefficient_is_seeded_and_nonconstant(self):
        fingerprints = set()
        coefficients = set()
        for seed in range(40):
            one = candidate(3, seed, 'hidden')
            self.assertEqual(one, candidate(3, seed, 'hidden'))
            if one is not None:
                coefficients.add(one[2])
                fingerprints.add(one[-1])
                self.assertGreaterEqual(one[2], .5)
                self.assertLessEqual(one[2], 1.5)
        self.assertGreater(len(coefficients), 30)
        self.assertGreater(len(fingerprints), 25)

    def test_training_prefix(self):
        common = dict(n=3, seed=31001, diversity='broad', fresh_count=4, structural_count=2)
        small = catalog_rows(**common, train_count=8)
        large = catalog_rows(**common, train_count=32)
        self.assertEqual([r for r in small if r['split'] == 'train'],
                         [r for r in large if r['split'] == 'train'][:8])

    def test_dev_panel_preserves_split_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = Path(tmp) / 'panel'
            prepare(panel, dev=True)
            for n in (3, 4, 5):
                evaluate = evaluation_entries(panel, n)
                self.assertEqual(len(evaluate), 10)
                fingerprints = {r['canonical_fingerprint'] for r in evaluate}
                self.assertEqual(len(fingerprints), 10)
                for level in ('focused', 'broad'):
                    train = training_entries(panel, n, level, 1, 8)
                    self.assertFalse(fingerprints & {r['canonical_fingerprint'] for r in train})


if __name__ == '__main__':
    unittest.main()
