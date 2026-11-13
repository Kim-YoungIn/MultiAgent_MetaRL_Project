"""Offline Korean HTML report for one recurrent PPO engineering run."""

from __future__ import annotations

from html import escape
from typing import Any


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _plot(series: list[tuple[str, list[tuple[int, float]]]], *, title: str, x_label: str) -> str:
    """Accessible SVG; scaled before subtraction to avoid overflow in the y range."""
    all_points = [point for _, points in series for point in points]
    if not all_points:
        return '<p class="muted">표시할 데이터가 없습니다.</p>'
    values = [point[1] for point in all_points]
    scale = max(1.0, *(abs(value) for value in values))
    low = min(0.0, *(value / scale for value in values))
    high = max(0.0, *(value / scale for value in values))
    if low == high:
        low, high = -1.0, 1.0
    xmin = min(point[0] for point in all_points)
    xmax = max(point[0] for point in all_points)
    left, right, top, bottom = 88, 740, 25, 235

    def x(value: int) -> float:
        return (left + right) / 2 if xmin == xmax else left + (value-xmin)/(xmax-xmin)*(right-left)

    def y(value: float) -> float:
        return bottom - (value / scale - low) / (high-low) * (bottom-top)

    grid = []
    for i in range(5):
        fraction = i / 4
        ordinate = top + (bottom-top) * fraction
        tick = (high * (1-fraction) + low * fraction) * scale
        grid.append(f'<line x1="{left}" y1="{ordinate:.2f}" x2="{right}" y2="{ordinate:.2f}" class="grid"/>'
                    f'<text x="{left-9}" y="{ordinate+4:.2f}" text-anchor="end" class="axis">{_e(format(tick,".4g"))}</text>')
    lines = []
    legend = []
    for index, (name, points) in enumerate(series):
        coords = " ".join(f"{x(px):.2f},{y(py):.2f}" for px, py in points)
        if coords:
            lines.append(f'<polyline points="{coords}" class="series s{index}"/>')
            for px, py in points:
                lines.append(f'<circle cx="{x(px):.2f}" cy="{y(py):.2f}" r="2.7" class="point p{index}"/>')
        legend.append(f'<span class="legend-{index}">{_e(name)}</span>')
    return f'''<div class="legend">{''.join(legend)}</div>
    <svg viewBox="0 0 780 300" role="img" aria-label="{_e(title)}">
      <title>{_e(title)}</title><desc>가로축 {_e(x_label)}, 세로축 행동당 평균 보상. 세로축에 0을 포함합니다.</desc>
      {''.join(grid)}<line x1="{left}" y1="{y(0):.2f}" x2="{right}" y2="{y(0):.2f}" class="zero"/>
      {''.join(lines)}
      <text x="{left}" y="259" text-anchor="middle" class="axis">{_e(xmin)}</text>
      <text x="{right}" y="259" text-anchor="middle" class="axis">{_e(xmax)}</text>
      <text x="414" y="287" text-anchor="middle" class="axis">{_e(x_label)}</text>
      <text x="17" y="130" text-anchor="middle" transform="rotate(-90 17 130)" class="axis">행동당 평균 보상</text>
    </svg>'''


_CSS = """
:root{--ink:#142334;--muted:#5d6b78;--line:#dce4e8;--bg:#f2f5f3;--teal:#087d71;--orange:#c35d26;--blue:#315f9c;--purple:#79529c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font:15px/1.6 system-ui,-apple-system,sans-serif;color:var(--ink)}
a{color:#096b64;text-underline-offset:3px}a:focus-visible{outline:3px solid #e29b38;outline-offset:3px}
.shell{width:min(1180px,calc(100% - 40px));margin:auto}.hero{background:#102d34;color:#fff;padding:48px 0 40px}
.eyebrow{color:#a5d9c8;font-size:.78rem;font-weight:700;letter-spacing:.13em}h1{font-size:clamp(2rem,5vw,3.2rem);line-height:1.12;margin:12px 0}
h2{font-size:1.45rem;margin:0 0 .4rem}.hero p{color:#d2e0dc;max-width:850px}.nav{display:flex;gap:20px;flex-wrap:wrap;padding:17px 0;border-bottom:1px solid var(--line);font-weight:650}
main{padding:26px 0 60px}.card{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin-bottom:18px;box-shadow:0 4px 20px #1b3c350a}.muted{color:var(--muted)}
.note{background:#eef7f4;border-left:3px solid #31a98f;padding:10px 14px;border-radius:4px;color:#31564d}.warning{background:#fff5e8;border-left:3px solid #da9a3b;padding:10px 14px;color:#684a1d}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.stat{background:#f4f8f6;border-radius:10px;padding:15px}.stat dt{color:var(--muted);font-size:.82rem}.stat dd{font-size:1.25rem;font-weight:750;margin:5px 0 0;overflow-wrap:anywhere}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}caption{text-align:left;color:var(--muted);padding:8px 0}
th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 11px;vertical-align:top}th{font-size:.85rem;color:#465967}tbody th{color:var(--ink)}.history{min-width:900px}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:.87rem;margin:12px 0}.legend span::before{content:'';display:inline-block;width:20px;border-top:3px solid var(--teal);vertical-align:middle;margin-right:7px}
.legend .legend-1::before{border-color:var(--orange);border-top-style:dashed}.legend .legend-2::before{border-color:var(--blue);border-top-style:dotted}.legend .legend-3::before{border-color:var(--purple);border-top-style:dashed}
svg{width:100%;height:auto;display:block}.grid{stroke:#dfe7e4;stroke-width:1}.zero{stroke:#90a6a0;stroke-width:1.2}.axis{fill:#526762;font-size:12px}.series{fill:none;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}.s0{stroke:var(--teal)}.s1{stroke:var(--orange);stroke-dasharray:8 5}.s2{stroke:var(--blue);stroke-dasharray:2 4}.s3{stroke:var(--purple);stroke-dasharray:12 4 2 4}.p0{fill:var(--teal)}.p1{fill:var(--orange)}.p2{fill:var(--blue)}.p3{fill:var(--purple)}
footer{color:var(--muted);font-size:.85rem}@media(max-width:600px){.shell{width:calc(100% - 24px)}.hero{padding:34px 0}.card{padding:17px}.nav{gap:12px}}
"""


def render_training_report(record: dict[str, Any]) -> str:
    """Render an already constructed training record without external assets."""
    config = record["config"]
    opponent_label = ('고정 Random 상대' if config.get('opponent_mode','random') == 'random'
                      else f"상대 조건 {config['opponent_mode']}")
    runtime = record["runtime"]
    history = record["history"]
    evaluations = record["evaluations"]
    comparisons = record["comparisons"]
    conditions = (("initial", "초기 정책"), ("random", "Random 정책"),
                  ("trained", "학습 정책"), ("hidden_reset", "학습 정책 · episode마다 hidden 초기화"))
    episode_count = max(len(comparisons[key]["mean_reward_by_episode"]) for key, _ in conditions)
    headers = "".join(f'<th scope="col">Episode {_e(i+1)}</th>' for i in range(episode_count))
    comparison_rows = []
    for key, name in conditions:
        item = comparisons[key]
        episodes = item["mean_reward_by_episode"]
        cells = "".join(f'<td>{_e(episodes[i]) if i < len(episodes) else "—"}</td>' for i in range(episode_count))
        comparison_rows.append(f'<tr><th scope="row">{_e(name)}</th><td>{_e(item["mean_reward_per_step"])}</td>'
                               f'<td>{_e(item["standard_error"])}</td><td>{_e(item["trials"])}</td>{cells}</tr>')
    curve = [("학습 정책", [(e["update"], e["mean_reward_per_step"]) for e in evaluations])]
    turns = [(name, [(i+1, reward) for i, reward in enumerate(comparisons[key]["mean_reward_by_turn"])])
             for key, name in conditions]
    history_keys = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "gradient_norm", "mean_reward_per_step")
    history_rows = "".join(
        f'<tr><th scope="row">{_e(row["update"])}</th>' + "".join(f'<td>{_e(row[k])}</td>' for k in history_keys) + '</tr>'
        for row in history
    )
    config_rows = "".join(f'<tr><th scope="row">{_e(key)}</th><td>{_e(value)}</td></tr>' for key, value in config.items())
    episode_family = ("평가와 학습에서 action_bonus의 규칙 구조가 다시 등장하며, 보상은 상대 행동에 의존하지 않습니다."
                      if config.get("family") == "action_bonus"
                      else "action_bonus 진단에서는 평가와 학습의 규칙 구조가 겹칩니다.")
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAMRL Recurrent PPO 학습 리포트</title><style>{_CSS}</style></head><body>
<header class="hero"><div class="shell"><div class="eyebrow">MAMRL · RECURRENT PPO</div><h1>학습 실행 리포트</h1>
<p>{_e(opponent_label)}와 유한 trial에서 실행한 단일 학습 seed의 엔지니어링 기준선입니다. 결과는 이 진단 조건 안에서 해석합니다.</p>
</div></header><div class="shell"><nav class="nav" aria-label="리포트 목차"><a href="#comparison">조건 비교</a><a href="#evaluation">평가 추이</a><a href="#turns">Trial 내 보상</a><a href="#training">학습 과정</a><a href="#config">실행 조건</a></nav>
<main><section class="card" id="comparison"><h2>평가 조건 비교</h2><p class="muted">평균과 episode별 값은 평가 행동당 P0 보상입니다. 초기·Random·학습·hidden 초기화 조건은 같은 평가 task와 환경 seed를 사용합니다.</p>
<dl class="stats"><div class="stat"><dt>학습 정책 평균 보상</dt><dd>{_e(comparisons["trained"]["mean_reward_per_step"])}</dd></div>
<div class="stat"><dt>평가 trial 표준오차</dt><dd>{_e(comparisons["trained"]["standard_error"])}</dd></div><div class="stat"><dt>파라미터 변화 L2</dt><dd>{_e(record["parameter_change_l2"])}</dd></div>
<div class="stat"><dt>평가 seed</dt><dd>{_e(record["evaluation_seed"])}</dd></div></dl>
<div class="table-wrap"><table><caption>조건별 평균·표준오차와 episode별 행동당 평균 보상</caption><thead><tr><th scope="col">조건</th><th scope="col">평균</th><th scope="col">표준오차</th><th scope="col">평가 trials</th>{headers}</tr></thead><tbody>{''.join(comparison_rows)}</tbody></table></div>
<p class="note">표준오차는 평가 trial 간 변동만 나타내며, 서로 다른 학습 seed 간 변동은 포함하지 않습니다. 표본 trial이 하나면 표준오차를 0으로 기록합니다.</p></section>
<section class="card" id="evaluation"><h2>학습 중 평가 추이</h2><p class="muted">각 update의 평가는 고정된 평가 task와 환경 seed를 사용하며 파라미터를 갱신하지 않습니다.</p>{_plot(curve,title="학습 update별 평가 행동당 평균 보상",x_label="학습 update")}</section>
<section class="card" id="turns"><h2>Trial 내 턴별 보상</h2><p class="muted">여러 episode를 이어 붙인 실제 행동 순서입니다. 경계의 관측 전용 이벤트는 턴 수에 포함하지 않습니다.</p>{_plot(turns,title="조건별 trial 내 턴별 평균 보상",x_label="trial 내 행동 순서")}</section>
<section class="card" id="training"><h2>PPO 학습 과정</h2><p class="muted">손실·엔트로피는 최적화 진단 수치이며 평가 보상과 구분합니다.</p><div class="table-wrap"><table class="history"><caption>학습 update별 최적화 진단</caption>
<thead><tr><th scope="col">Update</th><th scope="col">Loss</th><th scope="col">Policy loss</th><th scope="col">Value loss</th><th scope="col">Entropy</th><th scope="col">Approx. KL</th><th scope="col">Clip fraction</th><th scope="col">Gradient norm</th><th scope="col">수집 평균 보상</th></tr></thead><tbody>{history_rows}</tbody></table></div></section>
<section class="card" id="config"><h2>실행 조건과 해석 범위</h2><div class="warning"><strong>해석 주의:</strong> {_e(episode_family)} {_e(opponent_label)} 조건입니다. 단일 학습 seed의 결과는 미지 규칙 일반화나 탐색 성공의 증거가 아닙니다. n 간 학습 일반화와 다중 seed 결과는 확인하지 않았습니다.</div>
<p>런타임: Python {_e(runtime["python_version"])} · PyTorch {_e(runtime["torch_version"])} · {_e(runtime["device"])} · threads {_e(runtime["threads"])}</p>
<div class="table-wrap"><table><caption>학습 설정</caption><tbody>{config_rows}</tbody></table></div>
<p><a href="training.json">전체 학습 기록 JSON</a> · <a href="evaluation_trial.json">평가 trial JSON</a> · <a href="sample_episode/report.html">별도로 저장한 단일 episode 예제</a></p></section>
</main><footer><p>단일 실행 · 스키마 {_e(record["schema_version"])}</p></footer></div></body></html>'''
