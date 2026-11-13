"""Standalone report and CLI checks without a browser or network."""
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree

from mamrl.environment import MemoryOneEnv
from mamrl.recording import build_record, load_record
from mamrl.report import render_report
from mamrl.rollout import run_episode
from mamrl.rules import RuleSpec, RuleTerm, compile_rule
from test_recording import ScriptedPolicy, fixture


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.content = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.links.extend(value for name, value in attrs if name in ('href','src') and value is not None)

    def handle_data(self, data):
        self.content.append(data)


def negative_or_zero_record(coefficient):
    spec = RuleSpec(3, (RuleTerm('coordination', coefficient),))
    tensor = compile_rule(spec)
    policies = (ScriptedPolicy([0,0,0]), ScriptedPolicy([0,0,0]))
    episode = run_episode(MemoryOneEnv(tensor, max_steps=3), policies,
                          seed=8, policy_seeds=(9,10))
    return build_record(spec, tensor, episode, family='coordination', seed=8,
                        policy_seeds=(9,10), policy_names=('ScriptedPolicy','ScriptedPolicy'))


class ReportContract(unittest.TestCase):
    def test_displayed_turns_and_global_axis_matrices_match_record(self):
        _, tensor, _, record = fixture()
        html = render_report(record)
        for field, label in (('total_rewards', '총 보상 (P0, P1)'),
                             ('mean_rewards', '평균 보상 (P0, P1)')):
            match = re.search(r'<dt>' + re.escape(label) + r'</dt>\s*<dd>\(([^,]+),\s*([^\)]+)\)</dd>', html)
            self.assertIsNotNone(match, field)
            self.assertEqual(tuple(map(float, match.groups())), tuple(record['summary'][field]))
        for turn in record['transitions']:
            match = re.search(r'<tr id="turn-' + str(turn['turn']) + r'">(.*?)<td><details>', html, re.DOTALL)
            self.assertIsNotNone(match, turn['turn'])
            cells = re.findall(r'<(?:th|td)(?:\s+[^>]*)?>(.*?)</(?:th|td)>', match.group(1), re.DOTALL)
            self.assertEqual(int(cells[0]), turn['turn'])
            def numbers(value):
                return tuple(map(float, re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', value)))
            self.assertEqual(numbers(cells[1]), tuple(turn['previous_joint_action']))
            self.assertEqual(numbers(cells[2]), tuple(turn['actions']))
            self.assertEqual(numbers(cells[3]), tuple(turn['rewards']))
            self.assertEqual(numbers(cells[4]), tuple(turn['cumulative_rewards']))
        matrices = re.findall(r'<details class="matrix">\s*<summary>이전 행동 \((\d+),\s*(\d+)\)</summary>(.*?)</details>', html, re.DOTALL)
        self.assertEqual(len(matrices), 9)
        for prev0, prev1, content in matrices:
            p0, p1 = int(prev0), int(prev1)
            rows = re.findall(r'<tr><th scope="row">P0:\s*(\d+)</th>(.*?)</tr>', content, re.DOTALL)
            self.assertEqual(len(rows), 3)
            self.assertEqual(re.findall(r'<th scope="col">P1:\s*(\d+)</th>', content), ['0','1','2'])
            for action0, cells in rows:
                pairs = re.findall(r'<td>\(([^,]+),\s*([^\)]+)\)</td>', cells)
                self.assertEqual(len(pairs), 3)
                for action1, pair in enumerate(pairs):
                    expected = (tensor[0,p0,p1,int(action0),action1], tensor[1,p0,p1,int(action0),action1])
                    self.assertEqual(tuple(map(float, pair)), expected)

    def test_html_escapes_names_and_has_no_active_or_external_content(self):
        hostile = '<img src=x onerror=alert(123)>'
        _, _, _, record = fixture(policy_names=(hostile, 'Text & More'))
        html = render_report(record)
        page = PageParser()
        page.feed(html)
        self.assertNotIn('script', page.tags)
        self.assertNotIn('iframe', page.tags)
        self.assertNotIn('img', page.tags)
        self.assertNotIn(hostile, html)
        self.assertIn(hostile, ''.join(page.content))
        self.assertIn('Text & More', ''.join(page.content))
        self.assertNotRegex(html.lower(), r'@import\b')
        self.assertFalse(any(re.match(r'(?i)\s*(?:https?:)?//', url) for url in page.links))
        self.assertIn('details', page.tags)
        self.assertIn('svg', page.tags)
        self.assertIn('table', page.tags)

    def test_graph_svg_is_parseable_and_finite_for_negative_and_zero_rewards(self):
        for coefficient in (-2, 0, 2):
            with self.subTest(coefficient=coefficient):
                record = negative_or_zero_record(coefficient)
                self.assertEqual(record['summary']['total_rewards'], [3*coefficient]*2)
                html = render_report(record)
                svg_match = re.search(r'<svg\b[^>]*>.*?</svg>', html, re.DOTALL | re.IGNORECASE)
                self.assertIsNotNone(svg_match)
                svg = ElementTree.fromstring(svg_match.group(0))
                self.assertGreater(len(list(svg.iter())), 2)
                self.assertNotRegex(svg_match.group(0).lower(), r'(?<![a-z])(?:nan|[+-]?inf(?:inity)?)(?![a-z])')
                lines = [node for node in svg.iter() if node.tag.endswith('polyline')]
                self.assertEqual(len(lines), 2)
                points = [tuple(map(float, pair.split(','))) for pair in lines[0].attrib['points'].split()]
                self.assertEqual(len(points), 4)
                self.assertEqual([x for x, _ in points], sorted(x for x, _ in points))
                ys = [y for _, y in points]
                if coefficient < 0:
                    self.assertEqual(ys, sorted(ys))
                elif coefficient > 0:
                    self.assertEqual(ys, sorted(ys, reverse=True))
                else:
                    self.assertTrue(all(y == ys[0] for y in ys))
                for element in svg.iter():
                    for attr in ('x','y','x1','x2','y1','y2','cx','cy','r','width','height'):
                        value = element.attrib.get(attr)
                        if value is not None and re.fullmatch(r'[-+\d.eE]+', value):
                            self.assertTrue(math.isfinite(float(value)), (element.tag, attr, value))

    def test_cli_output_and_rebuild_refuse_overwrite(self):
        env = dict(os.environ)
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'episode'
            cmd = [sys.executable, '-m', 'mamrl', '--n', '3', '--steps', '3', '--seed', '8',
                   '--family', 'mixed', '--output-dir', str(target)]
            first = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            output = json.loads(first.stdout)
            self.assertEqual(output['steps'], 3)
            json_path, report_path = target / 'run.json', target / 'report.html'
            self.assertTrue(json_path.is_file())
            self.assertTrue(report_path.is_file())
            record = load_record(json_path)
            self.assertEqual(record['summary']['steps'], 3)
            before = (json_path.read_bytes(), report_path.read_bytes())
            second = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((json_path.read_bytes(), report_path.read_bytes()), before)
            rebuilt = Path(tmp) / 'rebuilt.html'
            rebuild_cmd = [sys.executable, '-m', 'mamrl.report', str(json_path), '--output', str(rebuilt)]
            built = subprocess.run(rebuild_cmd, env=env, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertTrue(rebuilt.is_file())
            before_rebuild = rebuilt.read_bytes()
            again = subprocess.run(rebuild_cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(rebuilt.read_bytes(), before_rebuild)


if __name__ == '__main__':
    unittest.main()
