"""End-to-end inference checks for the predeclared 5-by-5 paired study."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy
except ImportError:
    numpy = None
else:
    from experiment.statistics import report


@unittest.skipIf(numpy is None, 'optional NumPy')
class CompleteInferenceContract(unittest.TestCase):
    def test_positive_effect_and_missing_method(self):
        cells = [dict(condition='primary', catalog=c, replica=j, train_n=3, count=32)
                 for c in range(1, 6) for j in range(1, 6)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for cell in cells:
                directory = root / 'cells' / 'primary' / f"c{cell['catalog']}" / f"j{cell['replica']}" / 'n3'
                directory.mkdir(parents=True)
                for method, value in (('M', .5), ('P', .1), ('O', .2)):
                    rows = []
                    for i in range(64):
                        for profile in ('steady', 'sarsa'):
                            rows.append({
                                'rule': f'fresh-{i}', 'family': f'family-{i % 4}',
                                'split': 'fresh_rules', 'profile': profile, 'auc': value,
                                'zero_hash': f'{i}-{profile}',
                                'curve': [{'k': 0, 'return': 0.0},
                                          {'k': 24, 'return': value * 2}],
                            })
                    with gzip.open(directory / f'{method}.json.gz', 'wt') as stream:
                        json.dump(rows, stream)

            complete = report(root, cells, sealed=False)
            primary = complete['primary']
            self.assertTrue(primary['complete'])
            self.assertEqual(primary['claim'], 'positive')
            self.assertEqual(primary['n_models'], 25)
            self.assertAlmostEqual(primary['mean'], .4)
            self.assertAlmostEqual(primary['confidence_interval'][0], .4)
            self.assertAlmostEqual(primary['confidence_interval'][1], .4)
            self.assertEqual(set(primary['catalog_means']), {'1', '2', '3', '4', '5'})
            self.assertEqual(set(primary['leave_one_catalog_out']), {'1', '2', '3', '4', '5'})
            self.assertAlmostEqual(primary['rule_sampling_sensitivity_interval'][0], .4)
            self.assertAlmostEqual(primary['rule_sampling_sensitivity_interval'][1], .4)

            (root / 'cells/primary/c3/j2/n3/O.json.gz').unlink()
            incomplete = report(root, cells, sealed=False)
            self.assertFalse(incomplete['primary']['complete'])
            self.assertEqual(incomplete['primary']['claim'], 'incomplete')
            self.assertTrue(any('c3/j2/n3/O.json.gz' in name
                                for name in incomplete['missing_required_files']))


if __name__ == '__main__':
    unittest.main()
