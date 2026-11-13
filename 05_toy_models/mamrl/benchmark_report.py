"""Self-contained Korean report for the action-set functional benchmark."""

from __future__ import annotations

from html import escape
import json
from typing import Any


_FAMILIES = (
    ("hidden_action", "A · 숨겨진 행동"),
    ("probe_memory", "B · 탐색과 기억"),
    ("opponent_response", "C · 상대 반응"),
)
_ARCHITECTURES = {"action_set_v1": "행동 집합 recurrent", "action_set_ff_v1": "행동 집합 reactive (매 행동 hidden 초기화)"}
_REFERENCES = (("random", "Random"), ("reactive", "가족 지식 기반 reactive"),
               ("q_learning", "온라인 Q 학습"), ("informed", "가족 지식 기반 belief"),
               ("informed_reset", "belief · episode마다 기억 초기화"))


def _h(value: object) -> str:
    return escape(str(value), quote=True)


def _v(value: object) -> str:
    if value is None:
        return "미정의"
    return _h(format(value, '.6g') if isinstance(value, float) else value)


def _task_rows(source: str, metrics: dict[str, Any]) -> str:
    rows = []
    for task, item in metrics["by_task"].items():
        rows.append(f'<tr><th scope="row">{_h(source)}</th><td>{_h(task)}</td><td>{_v(item["trials"])}</td>'
                    f'<td>{_v(item["mean_reward_per_step"])}</td><td>{_v(item["best_response_bound"])}</td>'
                    f'<td>{_v(item["mean_regret_per_step"])}</td></tr>')
    return "".join(rows)


def _family_section(family: str, title: str, record: dict[str, Any]) -> str:
    refs = record["references"].get(family, {})
    runs = [run for run in record["runs"] if run["family"] == family]
    aggregates = record["aggregates"].get(family, {})
    aggregate_rows = []
    for architecture, metrics in aggregates.items():
        aggregate_rows.append(f'<tr><th scope="row">{_h(_ARCHITECTURES.get(architecture, architecture))}</th>'
                              f'<td>{_v(metrics["mean_reward"])}</td><td>{_v(metrics["sd_across_learning_seeds"])}</td>'
                              f'<td>{_v(metrics["mean_memory_benefit"])}</td></tr>')
    if not aggregate_rows:
        aggregate_rows.append('<tr><td colspan="4">학습 실행 없음</td></tr>')
    ref_rows, task_rows = [], []
    for key, label in _REFERENCES:
        if key not in refs:
            continue
        metrics = refs[key]
        ref_rows.append(f'<tr><th scope="row">{_h(label)}</th><td>{_v(metrics["mean_reward_per_step"])}</td>'
                        f'<td>{_v(metrics["standard_error"])}</td><td>{_v(metrics["mean_regret_per_step"])}</td>'
                        f'<td>{_v(metrics["early_reward"])}</td><td>{_v(metrics["late_reward"])}</td>'
                        f'<td>{_v(metrics["episode2_first_action"])}</td><td>{_v(metrics["post_identification_accuracy"])}</td>'
                        f'<td>{_v(metrics["post_identification_opportunities"])}</td>'
                        f'<td>{_v(metrics["probe_rate"]) if family == "probe_memory" else "—"}</td></tr>')
        task_rows.append(_task_rows(label, metrics))
    if not ref_rows:
        ref_rows.append('<tr><td colspan="10">비교 기준 없음</td></tr>')
    run_rows = []
    for run in runs:
        trained = run["comparisons"]["trained"]
        initial = run["comparisons"]["initial"]
        reset = run["comparisons"]["episode_reset"]
        architecture = _ARCHITECTURES.get(run["architecture"], run["architecture"])
        name = f'{architecture} · seed {run["seed"]}'
        task_rows.append(_task_rows(name, trained))
        run_rows.append(f'''<tr><th scope="row">{_h(run["seed"])}</th><td>{_h(architecture)}</td>
          <td>{_v(initial["mean_reward_per_step"])}</td><td>{_v(trained["mean_reward_per_step"])}</td>
          <td>{_v(trained["standard_error"])}</td><td>{_v(reset["mean_reward_per_step"])}</td>
          <td>{_v(run["memory_benefit"]["mean"])} ± {_v(run["memory_benefit"]["standard_error"])}</td>
          <td>{_v(trained["early_reward"])}</td><td>{_v(trained["late_reward"])}</td>
          <td>{_v(trained["episode2_first_action"])}</td><td>{_v(trained["post_identification_accuracy"])}</td>
          <td>{_v(trained["post_identification_opportunities"])}</td><td>{_v(trained["probe_rate"]) if family == "probe_memory" else "—"}</td>
          <td>{_v(run["parameter_count"])}</td><td>{_v(run["hidden_state_elements"])}</td></tr>''')
    if not run_rows:
        run_rows.append('<tr><td colspan="15">학습 실행 없음</td></tr>')
    if not task_rows:
        task_rows.append('<tr><td colspan="6">계층별 결과 없음</td></tr>')
    special = ("상대별 상한과 regret을 분리합니다. avoid_previous의 상한은 1/(n−1)이며, "
               "개별 유한 표본의 regret은 음수일 수 있습니다." if family == "opponent_response" else
               "각 target 또는 모드의 결과를 분리합니다. 표본 regret은 음수일 수 있습니다.")
    return f'''<section class="card" id="{_h(family)}"><h2>{_h(title)}</h2>
<p class="muted">이 가족 안에서만 비교합니다. {_h(special)}</p>
<div class="scroll"><table><caption>모델별 독립 학습 seed 요약</caption><thead><tr><th scope="col">모델</th><th scope="col">평균 보상</th><th scope="col">학습 seed 간 SD</th><th scope="col">평균 기억 이득</th></tr></thead><tbody>{''.join(aggregate_rows)}</tbody></table></div>
<h3>비교 기준</h3><div class="scroll"><table><caption>같은 가족의 기준 정책 평가 결과</caption><thead><tr><th scope="col">기준</th><th scope="col">평균 보상</th><th scope="col">평가 SE</th><th scope="col">평균 regret</th><th scope="col">초기 3행동</th><th scope="col">마지막 3행동</th><th scope="col">두 번째 episode 첫 행동</th><th scope="col">식별 이후 정답률</th><th scope="col">기회 수</th><th scope="col">probe 비율</th></tr></thead><tbody>{''.join(ref_rows)}</tbody></table></div>
<details><summary>학습 seed별 결과와 모델 크기</summary><div class="scroll"><table><caption>같은 가족의 학습 seed별 평가</caption><thead><tr>
<th scope="col">seed</th><th scope="col">모델</th><th scope="col">초기</th><th scope="col">학습</th><th scope="col">평가 SE</th><th scope="col">episode reset</th><th scope="col">기억 이득 ± 평가 SE</th><th scope="col">초기 3행동</th><th scope="col">마지막 3행동</th><th scope="col">episode 2 첫 행동</th><th scope="col">식별 이후 정답률</th><th scope="col">기회 수</th><th scope="col">probe 비율</th><th scope="col">파라미터 수</th><th scope="col">hidden 원소 수</th>
</tr></thead><tbody>{''.join(run_rows)}</tbody></table></div></details>
<details><summary>target·모드·상대 유형별 결과</summary><div class="scroll"><table><caption>계층별 행동당 보상과 task-aware 기준 상한</caption><thead><tr><th scope="col">정책</th><th scope="col">target / 모드 / 상대</th><th scope="col">trials</th><th scope="col">평균 보상</th><th scope="col">상한</th><th scope="col">평균 regret</th></tr></thead><tbody>{''.join(task_rows)}</tbody></table></div></details></section>'''


_CSS = """
:root{--ink:#142334;--muted:#5d6b78;--line:#dce4e8;--bg:#f2f5f3}*{box-sizing:border-box}
body{margin:0;background:var(--bg);font:15px/1.6 system-ui,-apple-system,sans-serif;color:var(--ink)}
.shell{width:min(1280px,calc(100% - 40px));margin:auto}header{background:#102d34;color:white;padding:48px 0 40px}
h1{font-size:clamp(2rem,5vw,3rem);line-height:1.12;margin:10px 0}h2{margin:0 0 .4rem}header p{color:#d2e0dc;max-width:850px}.eyebrow{color:#a5d9c8;font-weight:700;letter-spacing:.13em;font-size:.78rem}
a{color:#096b64;text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #e29b38;outline-offset:3px}
nav{display:flex;gap:20px;flex-wrap:wrap;padding:17px 0;border-bottom:1px solid var(--line);font-weight:650}main{padding:26px 0 60px}
.card{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin-bottom:18px}.muted{color:var(--muted)}
.warning{background:#fff5e8;border-left:3px solid #da9a3b;padding:10px 14px;color:#684a1d}.note{background:#eef7f4;border-left:3px solid #31a98f;padding:10px 14px;color:#31564d}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}caption{text-align:left;color:var(--muted);padding:8px 0}
th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 11px;vertical-align:top;white-space:nowrap}th{font-size:.85rem;color:#465967}tbody th{color:var(--ink)}
details{margin-top:20px}summary{cursor:pointer;color:#096b64;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f8f6;padding:16px;border-radius:8px}
@media(max-width:600px){.shell{width:calc(100% - 24px)}header{padding:34px 0}.card{padding:17px}nav{gap:12px}}
"""


def render_benchmark_report(record: dict[str, Any]) -> str:
    """Render already-assembled researcher results without external assets."""
    sections = "".join(_family_section(family, title, record) for family, title in _FAMILIES)
    checks = _h(json.dumps(record["task_checks"], ensure_ascii=False, indent=2, sort_keys=True))
    protocol = _h(json.dumps(record["protocol"], ensure_ascii=False, indent=2, sort_keys=True))
    static_control = record.get('posthoc_controls', {}).get('opponent_response_always_repeat')
    static_note = (f'<p class="warning">상대 과제의 추가 진단: 상대 종류를 식별하지 않고 자기 직전 행동만 반복하는 정책도 '
                   f'평균 {_v(static_control["mean_reward_per_step"])}를 얻습니다. Random 대비 상승만으로 상대 식별을 주장하지 않습니다. '
                   '이 기준은 결과 확인 후 추가했으며 기존 학습은 바꾸지 않았습니다.</p>' if static_control else '')
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAMRL 기능 벤치마크</title><style>{_CSS}</style></head><body>
<header><div class="shell"><div class="eyebrow">MAMRL · FUNCTIONAL BENCHMARKS</div><h1>행동 추적과 기억, 상대 반응</h1>
<p>서로 다른 세 게임 가족에서 행동 집합 정책을 평가합니다. 가족마다 보상 척도가 다르므로 가족 간 전체 순위를 만들지 않습니다.</p></div></header>
<div class="shell"><nav aria-label="리포트 목차"><a href="#hidden_action">숨겨진 행동</a><a href="#probe_memory">탐색과 기억</a><a href="#opponent_response">상대 반응</a><a href="#interpretation">조건과 해석</a><a href="#checks">과제 성립 확인</a></nav><main>
<section class="card"><h2>읽는 방법</h2><p>각 가족 안에서 학습 결과와 같은 task 조건의 기준 정책을 비교합니다. 학습 seed 간 SD는 독립 학습 실행 간 편차이고, 평가 SE는 고정된 정책의 평가 trial 변동입니다. 기억 이득은 paired 평가에서 episode 경계 hidden 유지 − 초기화의 행동당 평균 보상 차이입니다.</p>
<p class="note">식별 이후 정답률은 기록 분석자가 이 가족의 정확한 likelihood와 관찰된 이력을 이용해 식별된 시점을 판정한 값입니다. 정책이 해당 시점을 알고 있다는 의미는 아닙니다. 초기/마지막 값은 trial 내 첫/마지막 최대 세 행동의 평균 보상입니다.</p></section>
{sections}{static_note}
<section class="card" id="interpretation"><h2>비교 조건과 한계</h2><ul>
<li>Recurrent 정책은 trial 중 hidden을 유지합니다. 학습된 reactive 비교는 같은 행동 집합 구조에서 <strong>매 행동마다 hidden을 0으로 설정</strong>한 모델입니다. episode reset은 그와 별개로 episode 경계에서만 기억을 지우는 평가입니다.</li>
<li>Q 학습은 평가 중 표를 온라인 갱신합니다. 학습 정책의 가중치는 평가 중 고정됩니다. Belief와 reactive 기준에는 게임 <strong>가족</strong>에 대한 사전 지식이 하드코딩되어 있어 학습 정책과 같은 사전 지식의 비교가 아닙니다.</li>
<li>새 정책은 임의의 행동 ID 순서를 점수에 직접 입력하지 않습니다. 이전 ordinal 정책의 hidden 크기는 H, 새 정책은 nH이며 파라미터 수도 다를 수 있어 구조 비교를 단일 요인 인과 효과로 해석할 수 없습니다.</li>
<li>규칙 ID·모드·상대 유형은 정책 관측에 제공하지 않습니다. 소프트웨어 계약 검증과 과제의 기억 필요성 확인은 학습 성공을 뜻하지 않습니다. 이 결과는 H1/H2 또는 새로운 규칙 일반화의 증거가 아닙니다.</li></ul>
<details><summary>실행 프로토콜</summary><pre>{protocol}</pre></details></section>
<section class="card" id="checks"><h2>과제 성립 확인</h2><p class="muted">동일 현재 관측에 도달하는 이력과 task-aware 상한 등 연구자용 확인 결과입니다. 정책에 전달한 정보가 아닙니다.</p>
<details><summary>기억 witness와 상한 원문</summary><pre>{checks}</pre></details>
<p><a href="benchmark.json">원본 벤치마크 기록 JSON</a></p></section>
</main></div></body></html>'''
