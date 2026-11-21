"""Human-readable study status and mechanism measurements."""
from collections import defaultdict
import gzip
import json
from statistics import mean

from .artifacts import atomic_path
from .protocol import plan


def _number(value):
    return '—' if value is None else f'{value:+.4f}'


def write_summary(root, report, dev):
    title = '개발 실행' if dev else '본실험'
    expected = sum(len(cell['methods']) for cell in plan(dev))
    trained = len(list(root.glob('cells/*/c*/j*/training/*/ledger.json')))
    primary = report['primary']
    lines = [f'# D-031 {title}', '',
             f'완료한 학습 {trained}/{expected}회. 주 효과의 모델 쌍 {primary["n_models"]}개.', '',
             '## 주 평가', '',
             '같은 미학습 규칙·학습 상대에서 측정한 탐색 경험 예산별 학생 성능 차이다. '
             '적응 곡선은 각 방법의 탐색 0턴 성능을 기준으로 적분했다.', '',
             '| 지표 | 추정치 | 95% 구간 |', '|---|---:|---|']
    def line(name, group):
        interval = group.get('confidence_interval')
        ci = '—' if interval is None else ' … '.join(_number(v) for v in interval)
        lines.append(f"| {name} | {_number(group.get('mean'))} | {ci} |")
    line('적응 곡선 면적 M−P', primary)
    prefix = 'primary/n3/fresh_rules/mixed/'
    for key, name in [('M', 'M'), ('P', 'P'), ('O', 'O'), ('M-O', 'M−O')]:
        group = report['groups'].get(prefix + key)
        if group:
            line('적응 곡선 면적 ' + name, group)
    groups = report['groups']
    for k in ((0, 1, 3, 6) if dev else (0, 3, 6, 12, 24)):
        for method in ('M', 'P', 'M-P'):
            key = prefix + f'{method}/return/k{k}'
            if key in groups:
                line(f'{k}턴 탐색 후 {method}', groups[key])
    interpretation = {'development only': '개발용 seed와 축소 예산으로 실행 경로를 확인했다.',
                      'incomplete': '사전 지정된 모델 쌍이 완성될 때까지 주 판정을 보류한다.',
                      'positive': '이 분포에서 후속 적응을 목표로 학습한 탐색자의 평균 경험 효율이 높았다.',
                      'negative': '이 분포에서 후속 적응을 목표로 학습한 탐색자의 평균 경험 효율이 낮았다.',
                      'inconclusive': '주 효과의 구간이 0을 포함한다.'}
    lines.extend(['', interpretation[primary['claim']], '',
                  '## 훈련 조건별 결과', '',
                  '각 조건의 학습 행동 수에서 fresh 규칙과 두 학습 상대에 대한 M−P 적응 곡선 면적이다.', '',
                  '| 훈련 조건 | 추정치 | 95% 구간 |', '|---|---:|---|'])
    for cell in plan(dev):
        if cell['catalog'] != 1 or cell['replica'] != 1:
            continue
        key = f"{cell['condition']}/n{cell['train_n']}/fresh_rules/mixed/M-P"
        if key in groups:
            line(cell['condition'], groups[key])
    lines.extend(['', '## 경험과 상대 상태 교차 평가', ''])
    values = defaultdict(list)
    for path in sorted(root.glob('cells/primary/c*/j1/n3/diagnostics.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for text in stream:
                row = json.loads(text)
                if row['kind'] == 'cross' and row['mode'] == 'online':
                    values[row['history'], row['reference']].append(row['assessment_mean'])
    if values:
        lines.extend(['행은 학생에게 제공한 경험의 수집 정책이고 열은 평가 시작 시 상대 상태를 만든 정책이다. '
                      '값은 학생이 제어권을 받은 뒤의 평균 보상이다.', '',
                      '| 경험 ∖ 상대 | M | P | random |', '|---|---:|---:|---:|'])
        for method in ('M', 'P', 'random'):
            values_in_row = [_number(mean(values[method, ref])) for ref in ('M', 'P', 'random')]
            lines.append('| ' + ' | '.join([method, *values_in_row]) + ' |')
    else:
        lines.append('수집된 교차 평가가 없다.')
    lines.extend(['', '상대별·가족별 대비와 규칙 재표집 구간은 `report.json`, '
                  '개별 시행은 각 평가 폴더의 `*.trials.jsonl.gz`, 훈련 노출과 계산량은 `ledger.json`에 있다.', ''])
    with atomic_path(root / 'summary.md') as path:
        path.write_text('\n'.join(lines), encoding='utf-8')
