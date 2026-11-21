import json
import tempfile
import unittest
from pathlib import Path

from experiment.math_audit import pair_value, strategy_losses
from experiment.panels import prepare, training_entries, evaluation_entries
from mamrl.relational_rules import social_rule, relational_to_dict


class PanelContracts(unittest.TestCase):
    def test_constant_game_has_zero_information_value(self):
        row = {'id': 'constant', 'rule': relational_to_dict(social_rule(n=3, T=1, R=1, P=1, S=1))}
        for opponent in ('uniform', 'steady', 'sarsa'):
            value = pair_value(row, row, opponent)
            self.assertAlmostEqual(value['observed'], 2)
            self.assertAlmostEqual(value['masked'], 2)
            self.assertAlmostEqual(value['active_value'], 0)
        self.assertAlmostEqual(strategy_losses([row], [row])['constant'], 0)

    def test_informative_pair_distinguishes_observation_value(self):
        from mamrl.relational_rules import RelationalRule, Relation, Reward, cyclic_outcomes
        def rule(action):
            condition = Relation('eq', 'own', action)
            root = Reward('if', condition=condition,
                          children=(Reward('const', value=1), Reward('const', value=0)))
            return {'id': str(action), 'rule': relational_to_dict(RelationalRule(3, cyclic_outcomes(3), root))}
        value = pair_value(rule(0), rule(1))
        self.assertAlmostEqual(value['observed'], 1.5)
        self.assertAlmostEqual(value['masked'], 1.0)
        self.assertAlmostEqual(value['observation_value'], 0.5)
        self.assertAlmostEqual(value['active_value'], 0.0)

    def test_past_exclusion_recovers_both_catalog_naming_conventions(self):
        from experiment import panels
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / 'runs' / 'old'
            runs.mkdir(parents=True)
            for filename in ('catalog_n3_broad_128.json', 'manifest.json'):
                (runs / filename).write_text(json.dumps(
                    {'n': 3, 'rules': [{'canonical_fingerprint': filename}]}))
            with patch.object(panels, '__file__', str(root / 'experiment' / 'panels.py')):
                found, sources, unreadable, missing = panels._past_fingerprints(3)
            self.assertEqual(found, {'catalog_n3_broad_128.json', 'manifest.json'})
            self.assertEqual(len(sources), 2)
            self.assertFalse(unreadable or missing)

    def test_development_panel_integrity_and_disjointness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'panel'
            prepare(path, dev=True)
            for n in (3, 4, 5):
                test = evaluation_entries(path, n)
                for level in ('focused', 'broad'):
                    train = training_entries(path, n, level, 1, 8)
                    self.assertFalse({x['canonical_fingerprint'] for x in test} &
                                     {x['canonical_fingerprint'] for x in train})
            source = path/'sources'/'tensor.py'
            original = source.read_bytes()
            source.write_bytes(original + b'\n# changed\n')
            with self.assertRaises(ValueError):
                evaluation_entries(path, 3)
            source.write_bytes(original)
            file = path/'panel.json'
            payload = json.loads(file.read_text())
            payload['evaluation']['3'][0]['family'] = 'tampered'
            file.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                evaluation_entries(path, 3)

if __name__ == '__main__':
    unittest.main()
