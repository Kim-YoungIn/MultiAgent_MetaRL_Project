"""Render hydrated meta-exploration results as a standalone local HTML report."""

import html
import json
import statistics

def render_meta_report(record):
    assessment_steps=record['protocol'].get('assessment_steps',record['protocol']['steps'])
    esc=lambda x:html.escape(str(x),quote=True)
    num=lambda x:'—' if x is None else f'{x:.4f}'
    objective=record['protocol'].get('objective','retained_gain')
    target_description='후속 평균 보상' if objective=='assessment_return' else '평가−경험 제거 대조'
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>메타 탐색과 학습하는 상대</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:28px auto;padding:0 18px;background:#f5f7fb;color:#152538}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}td,th{border:1px solid #cbd4df;padding:8px;text-align:left}pre{white-space:pre-wrap}svg{background:white;width:100%}</style>',
           '<h1>메타 탐색 → 적응 학습 → 학습하는 상대</h1>',
           '<p>탐색자가 만든 경험을 같은 온라인 Q 학습기가 활용합니다. 상대도 자기 경험으로 Q와 전략을 갱신합니다.</p>',
           f'<p>탐색 {record["protocol"]["steps"]}회 + 실제 평가 {assessment_steps}회 + 경험 제거 대조 {assessment_steps}회. 탐색자의 학습 목표는 {target_description}입니다. G와 실제 성능을 함께 보고합니다. 탐색 중 보상을 더하지 않습니다.</p>',
           '<p><a href="experiment.json">전체 원자료</a> · <a href="protocol.json">프로토콜</a> · <a href="manifest.json">생성 규칙</a></p>']
    labels={'initial':'학습 전 탐색','trained':'메타 학습 탐색','random':'랜덤 탐색','epsilon_greedy':'ε-greedy 탐색','repeat':'반복 탐색'}
    if 'balanced' in record['baselines']: labels['balanced']='반복/변경 50:50'
    if 'change' in record['baselines']: labels['change']='항상 변경'
    if 'systematic' in record['baselines']:labels['systematic']='행동 순환 탐색'
    colors=('#8a50b0','#0068b5','#777','#d26900','#268344','#a17a00','#477b8b','#b14565')
    for split in record['comparisons']:
        parts.append('<h2>'+esc(split)+'</h2><table><tr><th>탐색 전략</th><th>경험 유지 이득 G</th><th>실제 후속 성능</th><th>경험 제거 성능</th><th>탐색 보상</th><th>실제 게임 전체</th><th>상대 분포 변화 누계</th><th>학습된 비균등 상대 행동 비율</th></tr>')
        curves={}
        for method in labels:
            if method in record['baselines']:
                stats=record['baselines'][method][split]['summary']
            else:
                samples=[r['evaluation'][method][split]['summary'] for r in record['runs']]
                stats={k:statistics.mean(s[k] for s in samples) for k in ('meta_reward','assessment_mean','erased_mean','exploration_mean','total_gameplay_mean','mean_opponent_policy_l1_change','opponent_nonuniform_action_fraction')}
                stats['assessment_curve']=[statistics.mean(s['assessment_curve'][i] for s in samples) for i in range(len(samples[0]['assessment_curve']))]
            curves[method]=stats['assessment_curve']
            parts.append('<tr><td>'+labels[method]+'</td>'+''.join('<td>'+num(stats[k])+'</td>' for k in ('meta_reward','assessment_mean','erased_mean','exploration_mean','total_gameplay_mean','mean_opponent_policy_l1_change','opponent_nonuniform_action_fraction'))+'</tr>')
        parts.append('</table><svg viewBox="0 0 700 160" role="img" aria-label="후속 학습 성능 곡선"><line x1="20" x2="680" y1="80" y2="80" stroke="#bbb"/>')
        for (method,curve),color in zip(curves.items(),colors):
            pts=' '.join(f'{20+660*i/max(1,len(curve)-1):.2f},{80-65*v:.2f}' for i,v in enumerate(curve))
            parts.append(f'<polyline points="{pts}" stroke="{color}" fill="none" stroke-width="2"><title>{labels[method]}</title></polyline>')
        parts.append('</svg><p>'+ ' · '.join(f'<span style="color:{c}">{labels[m]}</span>' for m,c in zip(curves,colors))+'</p>')
        parts.append('<details><summary>학습 seed별 결과</summary><pre>'+esc(json.dumps(record['comparisons'][split],ensure_ascii=False,indent=2))+'</pre></details>')
        parts.append('<details><summary>후속 구간의 경험 수별 성능과 곡선 기울기</summary><p>prefix AUC는 해당 구간 보상의 합을 구간 길이로 나눈 값입니다. 기울기는 상태 방문·상대 변화도 포함한 기술량이며 순수 적응 속도로 해석하지 않습니다.</p><table><tr><th>탐색</th><th>3턴 평균</th><th>6턴 평균</th><th>전체 평균</th><th>기울기</th></tr>')
        for method,label in labels.items():
            stats_list=[record['baselines'][method][split]['summary']] if method in record['baselines'] else [r['evaluation'][method][split]['summary'] for r in record['runs']]
            budgets=stats_list[0]['assessment_prefix_auc']
            def prefix(k):
                key=str(min(k,assessment_steps))
                return statistics.mean(s['assessment_prefix_auc'][key] for s in stats_list)
            slope=statistics.mean(s['assessment_slope'] for s in stats_list) if assessment_steps>1 else None
            parts.append('<tr><td>'+label+'</td>'+''.join('<td>'+num(v)+'</td>' for v in (prefix(3),prefix(6),prefix(assessment_steps),slope))+'</tr>')
        parts.append('</table></details>')
        families=record['runs'][0]['evaluation']['trained'][split].get('by_family',{})
        if families:
            parts.append('<details><summary>관계 가족별 후속 성능</summary><table><tr><th>가족</th><th>학습 탐색</th><th>랜덤</th><th>ε-greedy</th><th>반복/변경50:50</th></tr>')
            for family in families:
                neural=statistics.mean(r['evaluation']['trained'][split]['by_family'][family]['assessment_mean'] for r in record['runs'])
                vals=[neural]+[record['baselines'][method][split]['by_family'][family]['assessment_mean'] for method in ('random','epsilon_greedy','balanced')]
                parts.append('<tr><td>'+esc(family)+'</td>'+''.join('<td>'+num(v)+'</td>' for v in vals)+'</tr>')
            parts.append('</table></details>')
    if record.get('rule_catalog'):
        parts.append('<h2>생성된 의미 관계와 보상</h2><p>반복=C/변경=D이며,PD 태그는 최종 보상표의 독립 선택·영역별 동등성·T&gt;R&gt;P&gt;S를 검사한 결과입니다. 예외/숨은 보너스 뒤에도 자동 유지되지 않습니다.</p>')
        for entry in record['rule_catalog']:
            parts.append('<details><summary>'+esc(entry['id']+' / '+entry['family']+' / '+entry['split'])+'</summary><p>'+esc(entry['description'])+'</p><p>실제 PD 성립: '+esc(entry['audit']['pd']['is_prisoners_dilemma'])+' · 상대에 공통인 순수 최적 대응이 없는 상태 수: '+str(entry['audit']['states_without_universal_pure_best_response'])+'</p></details>')
    parts.append('<h2>구성과 측정의 한계</h2><ul><li>G는 경험을 보존한 Q와 지운 Q의 성능 차이입니다. 순수 정보량이 아니며 학습 능력·상태 방문·상대에 대한 영향도 포함합니다.</li></ul>')
    parts.append('<p>초기·최종 탐색 가중치는 평가 중 고정하고 학생과 상대의 Q는 갱신합니다. 최종 평가 규칙을 메타 학습에 쓰지 않습니다. 상대 초기 조건은 맞추되 탐색에 반응해 다르게 학습하는 것은 허용합니다.</p>')
    if record['protocol'].get('catalog') in ('relational','diverse_relational'):
        parts.append('<p>사회적 반복/변경·상성·상호 이력·숨은 보상·예외를 포함하는 제한된 관계 분포입니다. 현재 행동 일치를 모든 게임에 강제하지 않습니다. 상대는 Q-learning과 Expected SARSA이며 새 설정은 이미 학습한 Expected SARSA의 파라미터 변화입니다. 이전 catalog 결과와의 차이는 생성기 또는 상대 중 하나의 인과 효과가 아닙니다.</p>')
        if record['protocol'].get('catalog')=='diverse_relational':
            parts.append('<p>focused/broad는 생성 recipe의 폭을 바꿉니다. 규칙 개수와 전략 다양성은 별개이며,가족별·최선반응 audit를 함께 확인해야 합니다. 예외 평가는 기존 문법의 조건부 조합 전이이며 새 primitive 전이가 아닙니다.</p>')
    elif record['protocol'].get('catalog')=='probe_template':
        parts.append('<p>공통 safe/risk/marker Template의 미학습 파라미터와 반복·모방 보너스 재조합을 분리합니다. 역할 재명명은 독립 구조 다양성으로 세지 않습니다. 정보가치의 정확 계산은 고정 상대 대조이며 실제 학습 상대의 성공 보장이 아닙니다.</p>')
    else:
        parts.append('<p>기존 논리 catalog는 현재 행동 일치 관계를 공통으로 포함합니다. 상대 suite에 따라 사용 알고리즘을 protocol에서 확인하세요.</p>')
    n=record['protocol'].get('n',3)
    parts.append(f'<p>균등 행동은 n={n}에서 반복/변경을1:{n-1}로 만듭니다. 50:50 대조도 숨은 규칙은 알지 못합니다. PD의 즉시 우월성은 반복게임의 최적 전략이나 모든 변경 행동의 장기적 동등성을 보장하지 않습니다. 제한된 seed의 기술 통계이며 H1/H2 확증이 아닙니다.</p></html>')
    return ''.join(parts)


