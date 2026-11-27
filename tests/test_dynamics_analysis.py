import unittest

import numpy as np

from research.dynamics_transfer.analyze import (
    CONTROLLED, LIVE, PROBE, _contrast_rows, block_metrics,
)


class TestDynamicsAnalysis(unittest.TestCase):
    def test_family_weighting_and_paired_shift(self):
        # Resource appears twice. Equal family weighting differs from episode weighting.
        families = np.array(['resource', 'resource', 'sequence', 'reciprocity', 'mixed'])
        count = len(families)
        raw = dict(distances=[0, .3, .6], lags=[0, 1], live_methods=LIVE,
                   controlled_methods=CONTROLLED, probe_methods=PROBE,
                   families=families, action_to_role=np.tile([2, 0, 1], (count, 1)))
        for name, methods, turns in (('live', LIVE, 96), ('controlled', CONTROLLED, 32),
                                     ('probe', PROBE, 8)):
            shape = (2, 3, len(methods), count, turns, 2)
            raw[name + '_actions'] = np.ones(shape, dtype=np.int8)
            raw[name + '_rewards'] = np.zeros(shape)
        # Full feedback gains increase by 1 on all families at distance .6.
        # Resource contributes 4 at distance 0, other families contribute 0.
        raw['controlled_rewards'][:, :, 0, :2, :, 0] = 4
        raw['controlled_rewards'][:, 2, 0, :, :, 0] += 1
        raw['live_rewards'][:, :, 0, :, :, 0] = 2
        values = block_metrics(raw)
        self.assertEqual(values['controlled_mean:full_minus_masked'][0, 0, 0], 1)
        self.assertEqual(values['controlled_mean:full_minus_masked'][0, 2, 0], 2)
        self.assertEqual(values['total_96:teacher_minus_ucb'][0, 0, 0], 192)
        # Action 1 maps to invest for both agents on every turn.
        self.assertEqual(values['joint_invest_probe:pooled_fixed_total'][0, 0, 0], 1)
        self.assertEqual(values['current_and_both_lags_invest_tail:pooled_fixed_total'][0, 0, 0], 1)
        columns = {name: np.stack([value, value]) for name, value in values.items()}
        primary = next(v for name, family, comparison, v in _contrast_rows(columns)
                       if name == 'controlled_mean:full_minus_masked' and family == 'all'
                       and comparison == 'lag0_distance0.6_minus_0')
        np.testing.assert_array_equal(primary, [1, 1])


if __name__ == '__main__':
    unittest.main()
