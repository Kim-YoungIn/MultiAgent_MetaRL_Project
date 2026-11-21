import gzip
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy
except ImportError:
    numpy = None
else:
    from experiment.statistics import report


@unittest.skipIf(numpy is None, 'optional NumPy')
class StatisticsContract(unittest.TestCase):
    def test_paired_difference_and_incomplete_primary(self):
        cells = [dict(condition='primary', catalog=1, replica=1, train_n=3, count=32)]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'cells/primary/c1/j1/n3'
            directory.mkdir(parents=True)
            for method, score in [('M', 2.), ('P', 1.), ('O', .5)]:
                rows = []
                for family in ('social', 'contest'):
                    for profile in ('steady', 'sarsa'):
                        rows.append(dict(rule=family, family=family, split='fresh_rules', profile=profile,
                                         auc=score, zero_hash='same', curve=[dict(k=0, **{'return': 0.}),
                                         dict(k=3, **{'return': score})]))
                with gzip.open(directory/(method+'.json.gz'), 'wt') as stream:
                    json.dump(rows, stream)
            unsigned = report(tmp, cells, dev=True)
            self.assertFalse(unsigned['primary']['complete'])
            self.assertIsNone(unsigned['primary']['mean'])
            result = report(tmp, cells, dev=True, sealed=False)
            self.assertEqual(result['primary']['mean'], 1.)
            self.assertEqual(result['primary']['claim'], 'development only')
            self.assertEqual(result['groups']['primary/n3/fresh_rules/mixed/M-P/return/k3']['mean'], 1.)
            self.assertEqual(report(tmp, cells, dev=False, sealed=False)['primary']['claim'], 'incomplete')

    def test_zero_hash_mismatch_fails(self):
        cells = [dict(condition='primary', catalog=1, replica=1, train_n=3, count=32)]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'cells/primary/c1/j1/n3'
            directory.mkdir(parents=True)
            for method in ('M', 'P', 'O'):
                rows = [dict(rule='r', family='social', split='fresh_rules', profile='steady', auc=0,
                             zero_hash=method, curve=[])]
                with gzip.open(directory/(method+'.json.gz'), 'wt') as stream:
                    json.dump(rows, stream)
            with self.assertRaisesRegex(ValueError, 'k=0 mismatch'):
                report(tmp, cells, dev=True, sealed=False)


if __name__ == '__main__':
    unittest.main()
