"""Render a validated single-episode record as a self-contained HTML report."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any


def _text(value: object) -> str:
    """Escape all values before putting them into HTML or SVG."""
    return escape(str(value), quote=True)


def _seed(value: int | None) -> str:
    return "자동" if value is None else _text(value)


def _pair(values: list[Any] | tuple[Any, ...]) -> str:
    return f"({_text(values[0])}, {_text(values[1])})"


def _chart(transitions: list[dict[str, Any]]) -> str:
    """Plot rewards on a shared axis; normalize first to avoid float overflow."""
    points = [(0.0, 0.0)] + [tuple(t["cumulative_rewards"]) for t in transitions]
    values = [x for point in points for x in point]
    scale = max(1.0, *(abs(x) for x in values))
    low = min(0.0, *(x / scale for x in values))
    high = max(0.0, *(x / scale for x in values))
    if low == high:
        low, high = -1.0, 1.0
    left, right, top, bottom = 80.0, 740.0, 24.0, 238.0
    width, height = right - left, bottom - top
    count = len(points) - 1

    def y(value: float) -> float:
        return bottom - (value / scale - low) / (high - low) * height

    def x(index: int) -> float:
        return left + index / max(count, 1) * width

    grid: list[str] = []
    for i in range(5):
        fraction = i / 4
        ordinate = top + height * fraction
        tick = (high * (1 - fraction) + low * fraction) * scale
        grid.append(
            f'<line x1="{left:.2f}" y1="{ordinate:.2f}" x2="{right:.2f}" y2="{ordinate:.2f}" class="gridline"/>'
            f'<text x="{left-10:.2f}" y="{ordinate+4:.2f}" text-anchor="end" class="axis-text">{_text(format(tick, ".4g"))}</text>'
        )
    paths = []
    for player in (0, 1):
        coordinates = " ".join(f"{x(i):.2f},{y(point[player]):.2f}" for i, point in enumerate(points))
        paths.append(f'<polyline points="{coordinates}" class="line line-{player}"/>')
    return f'''<svg viewBox="0 0 780 300" role="img" aria-labelledby="chart-title chart-desc">
      <title id="chart-title">플레이어별 누적 보상</title>
      <desc id="chart-desc">가로축은 턴, 세로축은 누적 보상입니다. 0턴의 보상은 0입니다.</desc>
      {''.join(grid)}
      <line x1="{left:.2f}" y1="{y(0):.2f}" x2="{right:.2f}" y2="{y(0):.2f}" class="zero-line"/>
      {''.join(paths)}
      <text x="{left:.2f}" y="260" text-anchor="middle" class="axis-text">0</text>
      <text x="{right:.2f}" y="260" text-anchor="middle" class="axis-text">{_text(count)}</text>
      <text x="410" y="286" text-anchor="middle" class="axis-text">턴</text>
      <text x="16" y="130" text-anchor="middle" transform="rotate(-90 16 130)" class="axis-text">누적 보상</text>
    </svg>'''


def _turn_rows(transitions: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for turn in transitions:
        terms = []
        for term in turn["reward_terms"]:
            terms.append(
                "<tr>"
                f'<td>{_text(term["label"])}</td>'
                f'<td>{_text(term["coefficient"])}</td>'
                f'<td>{_text(term["contributions"][0])}</td>'
                f'<td>{_text(term["contributions"][1])}</td>'
                "</tr>"
            )
        if not terms:
            terms.append('<tr><td colspan="4">규칙 항 없음</td></tr>')
        reason = "시간 제한" if turn["truncated"] else ("게임 종료" if turn["terminated"] else "—")
        turn_no = _text(turn["turn"])
        previous = turn["previous_joint_action"]
        rows.append(f'''<tr id="turn-{turn_no}">
            <th scope="row">{turn_no}</th>
            <td>{_pair(previous)}</td><td>{_pair(turn["actions"])}</td>
            <td>{_pair(turn["rewards"])}</td><td>{_pair(turn["cumulative_rewards"])}</td>
            <td><details><summary>항별 근거</summary>
              <table class="nested"><caption>턴 {turn_no} 규칙 항별 기여</caption>
                <thead><tr><th scope="col">규칙</th><th scope="col">계수</th><th scope="col">P0</th><th scope="col">P1</th></tr></thead>
                <tbody>{''.join(terms)}</tbody></table></details></td>
            <td>{_text(reason)}</td>
          </tr>''')
    return "".join(rows)


def _rule_matrices(record: dict[str, Any]) -> str:
    n = record["metadata"]["n"]
    values = record["tensor"]["values"]

    def reward(player: int, p0: int, p1: int, a0: int, a1: int) -> Any:
        return values[((((player * n + p0) * n + p1) * n + a0) * n + a1)]

    matrices: list[str] = []
    for p0 in range(n):
        for p1 in range(n):
            cols = "".join(f'<th scope="col">P1: {_text(a1)}</th>' for a1 in range(n))
            rows: list[str] = []
            for a0 in range(n):
                cells = "".join(
                    f'<td>{_pair((reward(0,p0,p1,a0,a1), reward(1,p0,p1,a0,a1)))}</td>'
                    for a1 in range(n)
                )
                rows.append(f'<tr><th scope="row">P0: {_text(a0)}</th>{cells}</tr>')
            matrices.append(f'''<details class="matrix"><summary>이전 행동 ({_text(p0)}, {_text(p1)})</summary>
              <div class="table-wrap"><table><caption>이전 joint action ({_text(p0)}, {_text(p1)})의 보상 (P0, P1)</caption>
              <thead><tr><th scope="col">현재 행동</th>{cols}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>''')
    return "".join(matrices)


_CSS = """
:root{color-scheme:light;--ink:#142334;--muted:#5d6b78;--line:#dce4e8;--bg:#f2f5f3;--panel:#fff;--teal:#087d71;--orange:#c35d26}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font:15px/1.6 system-ui,-apple-system,sans-serif;color:var(--ink)}
a{color:#096b64;text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #e29b38;outline-offset:3px}
.shell{width:min(1180px,calc(100% - 40px));margin:auto}.hero{background:#102d34;color:#fff;padding:50px 0 42px}
.eyebrow{color:#a5d9c8;font-size:.78rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase}
h1{font-size:clamp(2rem,5vw,3.4rem);line-height:1.1;margin:12px 0}h2{font-size:1.45rem;margin:0 0 .3rem}h3{margin:.6rem 0}
.hero p{max-width:760px;color:#d2e0dc;margin:12px 0 0}.hero a{color:#b5ebdc}.meta{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}
.pill{border:1px solid #5d8380;border-radius:99px;padding:4px 11px;font-size:.83rem}.nav{display:flex;gap:20px;flex-wrap:wrap;padding:17px 0;border-bottom:1px solid var(--line);font-weight:650}
main{padding:28px 0 60px}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px;margin-bottom:18px;box-shadow:0 4px 20px #1b3c350a}
.section-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:16px}.muted{color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.stat{background:#f4f8f6;border-radius:10px;padding:15px}
.stat dt{color:var(--muted);font-size:.82rem}.stat dd{font-size:1.3rem;font-weight:750;margin:5px 0 0;overflow-wrap:anywhere}
.note{background:#eef7f4;border-left:3px solid #31a98f;padding:10px 14px;border-radius:4px;color:#31564d}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}caption{text-align:left;color:var(--muted);padding:8px 0}
th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 11px;vertical-align:top}th{font-size:.85rem;color:#465967}tbody th{color:var(--ink)}
.turns{min-width:860px}details summary{cursor:pointer;color:#086b64;font-weight:650}details[open]>summary{margin-bottom:10px}.nested{min-width:300px;font-size:.88rem}.nested th,.nested td{padding:6px}
.matrix-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}.matrix{background:#f7faf9;border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
.matrix[open]{grid-column:1/-1}.matrix table{min-width:320px}.rules li{padding:4px 0}.legend{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:.88rem}
.legend span::before{content:'';display:inline-block;width:22px;border-top:3px solid var(--teal);vertical-align:middle;margin-right:7px}.legend span:nth-child(2)::before{border-color:var(--orange);border-top-style:dashed}
svg{width:100%;height:auto;display:block}.gridline{stroke:#dfe7e4;stroke-width:1}.zero-line{stroke:#90a6a0;stroke-width:1.2}.axis-text{fill:#526762;font-size:12px}.line{fill:none;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}.line-0{stroke:var(--teal)}.line-1{stroke:var(--orange);stroke-dasharray:7 5}
footer{color:var(--muted);font-size:.85rem}@media(max-width:600px){.shell{width:calc(100% - 24px)}.hero{padding:34px 0}.card{padding:17px}.nav{gap:12px}}
"""


def render_report(record: dict[str, Any]) -> str:
    """Render a previously validated v1 record; never accesses the network."""
    meta = record["metadata"]
    summary = record["summary"]
    transitions = record["transitions"]
    descriptions = record["rule_descriptions"]
    rule_terms = record["rule"]["terms"]
    rule_items = "".join(
        f'<li>{_text(description)} <span class="muted">(계수: {_text(term["coefficient"])})</span></li>'
        for description, term in zip(descriptions, rule_terms)
    ) or "<li>규칙 항 없음</li>"
    counts = summary["action_counts"]
    action_rows = "".join(
        f'<tr><th scope="row">{_text(action)}</th><td>{_text(counts[0][action])}</td><td>{_text(counts[1][action])}</td></tr>'
        for action in range(meta["n"])
    )
    policies = meta["policies"]
    reason = "시간 제한 절단" if summary["truncated"] else ("게임 종료" if summary["terminated"] else "진행 중")
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAMRL 실행 리포트</title><style>{_CSS}</style></head><body>
<header class="hero"><div class="shell"><div class="eyebrow">MAMRL · EPISODE REPORT</div>
<h1>한 번의 게임, 자세히 보기</h1>
<p>두 에이전트가 논리 규칙으로 정의된 게임을 플레이한 단일 episode의 기록입니다. 수치는 학습 또는 적응 성능의 증거가 아닙니다.</p>
<div class="meta"><span class="pill">규칙 계열: {_text(meta["family"])}</span><span class="pill">행동 수: {_text(meta["n"])}</span><span class="pill">{_text(summary["steps"])}턴</span></div>
</div></header>
<div class="shell"><nav class="nav" aria-label="리포트 목차"><a href="#summary">실행 요약</a><a href="#rewards">누적 보상</a><a href="#turns">턴별 기록</a><a href="#rules">게임 규칙</a><a href="#actions">행동 빈도</a></nav>
<main><section class="card" id="summary"><div class="section-head"><div><h2>실행 요약</h2><span class="muted">두 플레이어의 결과를 글로벌 순서로 표시합니다.</span></div></div>
<dl class="stats"><div class="stat"><dt>총 보상 (P0, P1)</dt><dd>{_pair(summary["total_rewards"])}</dd></div>
<div class="stat"><dt>평균 보상 (P0, P1)</dt><dd>{_pair(summary["mean_rewards"])}</dd></div>
<div class="stat"><dt>종료 사유</dt><dd>{_text(reason)}</dd></div><div class="stat"><dt>진행 턴</dt><dd>{_text(summary["steps"])}</dd></div></dl>
<div class="table-wrap"><table><caption>실행 설정</caption><tbody>
<tr><th scope="row">정책 P0 / P1</th><td>{_text(policies[0])} / {_text(policies[1])}</td></tr>
<tr><th scope="row">규칙 / 환경 seed</th><td>{_seed(meta["seed"])}</td></tr>
<tr><th scope="row">정책 seed P0 / P1</th><td>{_seed(meta["policy_seeds"][0])} / {_seed(meta["policy_seeds"][1])}</td></tr>
<tr><th scope="row">Python / 해석 버전</th><td>{_text(meta["python_version"])} / {_text(meta["semantics_version"])}</td></tr>
</tbody></table></div>
<p class="note">이 보고서는 연구자용입니다. 규칙·계수·episode 길이는 정책 관측에 직접 제공되지 않습니다. 고정 길이는 경험을 통해 추정될 수 있습니다.</p></section>
<section class="card" id="rewards"><div class="section-head"><div><h2>누적 보상</h2><span class="muted">0턴에서 시작해 각 턴의 보상을 더합니다.</span></div><div class="legend"><span>P0</span><span>P1</span></div></div>{_chart(transitions)}</section>
<section class="card" id="turns"><div class="section-head"><div><h2>턴별 기록</h2><span class="muted">행동과 보상은 (P0, P1) 순서입니다. 항별 근거를 펼칠 수 있습니다.</span></div></div>
<div class="table-wrap"><table class="turns"><caption>이전 행동, 이번 행동, 보상과 규칙 항별 기여</caption><thead><tr><th scope="col">턴</th><th scope="col">이전 행동</th><th scope="col">현재 행동</th><th scope="col">보상</th><th scope="col">누적 보상</th><th scope="col">근거</th><th scope="col">종료</th></tr></thead><tbody>{_turn_rows(transitions)}</tbody></table></div></section>
<section class="card" id="rules"><div class="section-head"><div><h2>게임 규칙</h2><span class="muted">각 항의 기여를 더해 보상을 계산합니다.</span></div></div><ol class="rules">{rule_items}</ol>
<h3>이전 상태별 보상표</h3><p class="muted">각 표의 행은 현재 P0 행동, 열은 현재 P1 행동이며 셀은 (P0 보상, P1 보상)입니다.</p><div class="matrix-grid">{_rule_matrices(record)}</div></section>
<section class="card" id="actions"><div class="section-head"><div><h2>행동 빈도</h2><span class="muted">관찰된 선택 횟수이며 적응 지표가 아닙니다.</span></div></div>
<div class="table-wrap"><table><caption>플레이어별 행동 선택 횟수</caption><thead><tr><th scope="col">행동</th><th scope="col">P0</th><th scope="col">P1</th></tr></thead><tbody>{action_rows}</tbody></table></div></section>
</main><footer><p>단일 episode 기록 · 데이터 스키마 {_text(record["schema_version"])}</p></footer></div></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description="저장된 MAMRL 실행 기록으로 오프라인 HTML 리포트를 만듭니다.")
    parser.add_argument("input", type=Path, help="검증할 run.json 경로")
    parser.add_argument("--output", required=True, type=Path, help="생성할 HTML 경로 (기존 파일 덮어쓰기 금지)")
    args = parser.parse_args()
    # Lazy import avoids a cycle when recording.save_run uses this renderer.
    from .recording import load_record

    try:
        record = load_record(args.input)
        html = render_report(record)
        with args.output.open("x", encoding="utf-8") as out:
            out.write(html)
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(args.output)


if __name__ == "__main__":
    main()
