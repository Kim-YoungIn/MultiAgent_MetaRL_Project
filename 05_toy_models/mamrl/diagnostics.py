"""Behavior, paired uncertainty, and supervised capacity diagnostics.

These measurements characterize a single action-bonus diagnostic family; they
do not establish adaptation to novel rules or multi-agent strategic behavior.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
from html import escape
import json
import math
import random
import statistics
from typing import Any

import torch

from .common import finite_reward, positive_integer, validate_action, validate_n, validate_seed
from .environment import Observation
from .recurrent import RecurrentActorCritic
from .rules import RuleSpec, RuleTerm
from .training import collect_trial


def balanced_specs(n: int = 3, repeats: int = 16, fixed_action: int | None = None) -> tuple[RuleSpec, ...]:
    """Interleave equal-sized target strata, or repeat a single fixed target."""
    n = validate_n(n)
    repeats = positive_integer(repeats, "repeats")
    if fixed_action is not None:
        fixed_action = validate_action(fixed_action, n)
    targets = (fixed_action,) * (n * repeats) if fixed_action is not None else tuple(range(n)) * repeats
    return tuple(RuleSpec(n, (RuleTerm("action_bonus", 1, target),)) for target in targets)


def _target(spec: RuleSpec) -> int:
    if not isinstance(spec, RuleSpec):
        raise TypeError("each evaluation task must be a RuleSpec")
    if len(spec.terms) != 1:
        raise ValueError("diagnostic tasks require one action_bonus term")
    term = spec.terms[0]
    if term.kind != "action_bonus" or term.coefficient != 1:
        raise ValueError("diagnostic tasks require action_bonus coefficient 1")
    return term.action


def _spec_fingerprint(spec: RuleSpec) -> str:
    """Hash all rule semantics, using the canonical UTF-8 JSON representation."""
    canonical = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _stratified(values: list[float], targets: list[int]) -> tuple[float | None, dict[str, dict[str, Any]]]:
    """Standard error of a fixed target-mixture mean, with unbiased strata variance."""
    if not values or len(values) != len(targets):
        raise ValueError("stratified values and targets must be nonempty and aligned")
    groups: dict[int, list[float]] = defaultdict(list)
    for target, value in zip(targets, values):
        groups[target].append(value)
    result: dict[str, dict[str, Any]] = {}
    variance = 0.0
    complete = True
    for target in sorted(groups):
        group = groups[target]
        se = statistics.stdev(group) / math.sqrt(len(group)) if len(group) >= 2 else None
        result[str(target)] = {"trials": len(group), "mean": statistics.mean(group), "standard_error": se}
        if se is None:
            complete = False
        else:
            variance += (len(group) / len(values)) ** 2 * se**2
    return (math.sqrt(variance) if complete else None), result


def behavior_metrics(model: RecurrentActorCritic | None, specs, *, episodes_per_trial: int,
                     max_steps: int, seed: int = 918273, reset_hidden: bool = False) -> dict[str, Any]:
    """Evaluate behavior using matched local trial seeds and record provenance."""
    specs = tuple(specs)
    if not specs:
        raise ValueError("evaluation requires at least one task")
    if model is not None and not isinstance(model, RecurrentActorCritic):
        raise TypeError("model must be a RecurrentActorCritic or None")
    positive_integer(episodes_per_trial, "episodes_per_trial")
    positive_integer(max_steps, "max_steps")
    if seed is None:
        raise TypeError("evaluation requires an explicit seed")
    validate_seed(seed)
    if type(reset_hidden) is not bool:
        raise TypeError("reset_hidden must be bool")
    targets = [_target(spec) for spec in specs]
    n = specs[0].n
    if any(spec.n != n for spec in specs):
        raise ValueError("all evaluation tasks must have the same action count")
    spec_fingerprints = [_spec_fingerprint(spec) for spec in specs]
    rng = random.Random(seed)
    trial_seeds = [rng.getrandbits(63) for _ in specs]
    trials = [collect_trial(model, spec, episodes_per_trial=episodes_per_trial, max_steps=max_steps,
                            seed=trial_seed, reset_hidden=reset_hidden)
              for spec, trial_seed in zip(specs, trial_seeds)]
    steps = episodes_per_trial * max_steps
    returns = [sum(trial.rewards) for trial in trials]
    means = [value / steps for value in returns]
    standard_error, groups = _stratified(means, targets)
    pooled = statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0
    opportunities = successes = 0
    for trial in trials:
        for episode in trial.episodes:
            for transition in episode.transitions:
                observation = transition.observations[0]
                if not observation.episode_start_flag and observation.previous_own_reward > 0:
                    opportunities += 1
                    successes += transition.actions[0] == observation.previous_own_action
    if episodes_per_trial > 1:
        second_first = statistics.mean(trial.episodes[1].transitions[0].rewards[0] for trial in trials)
        first_three = min(3, max_steps)
        second_first_three = statistics.mean(
            sum(t.rewards[0] for t in trial.episodes[1].transitions[:first_three]) / first_three
            for trial in trials
        )
    else:
        second_first = second_first_three = None
    return {
        "trials": len(trials), "mean_reward_per_step": statistics.mean(means),
        "mean_reward_by_episode": [statistics.mean(t.episodes[e].total_rewards[0] / max_steps for t in trials)
                                   for e in range(episodes_per_trial)],
        "mean_reward_by_turn": [statistics.mean(t.rewards[k] for t in trials) for k in range(steps)],
        "trial_returns": returns, "standard_error": standard_error,
        "pooled_standard_error": pooled, "standard_error_method": "stratified_by_target",
        "by_target": {target: {"trials": data["trials"], "mean_reward_per_step": data["mean"],
                               "standard_error": data["standard_error"]} for target, data in groups.items()},
        "repeat_after_success_rate": successes / opportunities if opportunities else None,
        "repeat_after_success_count": successes, "success_opportunities": opportunities,
        "second_episode_first_action_mean": second_first,
        "second_episode_first_three_actions_mean": second_first_three,
        "n": n, "episodes_per_trial": episodes_per_trial, "max_steps": max_steps,
        "steps_per_trial": steps, "trial_targets": targets, "trial_seeds": trial_seeds,
        "spec_fingerprints": spec_fingerprints,
    }


def paired_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Return left-minus-right paired differences; reject mismatched provenance."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise TypeError("paired evaluations must be dictionaries")
    for evaluation in (left, right):
        validate_n(evaluation["n"])
        episodes = positive_integer(evaluation["episodes_per_trial"], "episodes_per_trial")
        horizon = positive_integer(evaluation["max_steps"], "max_steps")
        steps = positive_integer(evaluation["steps_per_trial"], "steps_per_trial")
        if steps != episodes * horizon:
            raise ValueError("steps_per_trial must equal episodes_per_trial * max_steps")
    for key in ("n", "episodes_per_trial", "max_steps", "steps_per_trial",
                "trial_targets", "trial_seeds", "spec_fingerprints"):
        if left[key] != right[key]:
            raise ValueError(f"paired evaluations disagree on {key}")
    steps = positive_integer(left["steps_per_trial"], "steps_per_trial")
    targets, seeds = list(left["trial_targets"]), list(left["trial_seeds"])
    fingerprints = list(left["spec_fingerprints"])
    left_returns, right_returns = list(left["trial_returns"]), list(right["trial_returns"])
    if not targets or len({len(targets), len(seeds), len(fingerprints), len(left_returns), len(right_returns)}) != 1:
        raise ValueError("paired trial provenance and return lengths must match and be nonempty")
    if any(not isinstance(fingerprint, str) for fingerprint in fingerprints):
        raise TypeError("spec fingerprints must be strings")
    if left.get("trials") != len(targets) or right.get("trials") != len(targets):
        raise ValueError("paired trial count must match provenance")
    for target in targets:
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("trial target must be an integer")
        validate_action(target, left["n"])
    for seed in seeds:
        if seed is None:
            raise TypeError("trial seed must not be None")
        validate_seed(seed)
    differences = [(finite_reward(a) - finite_reward(b)) / steps for a, b in zip(left_returns, right_returns)]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("paired differences must be finite")
    standard_error, groups = _stratified(differences, targets)
    for target, group in groups.items():
        group["trial_differences"] = [difference for t, difference in zip(targets, differences) if str(t) == target]
    return {"mean": statistics.mean(differences), "standard_error": standard_error,
            "standard_error_method": "stratified_by_target", "trial_differences": differences,
            "by_target": groups}


def supervised_probe(*, seed: int, steps: int = 400, hidden_size: int = 32,
                     learning_rate: float = .003, n: int = 3) -> dict[str, Any]:
    """Fit all n*n*2 immediate observations with zero recurrent hidden state.

    Soft labels are training targets only. This is neither held-out evaluation
    nor evidence that the policy can remember information across timesteps.
    """
    n = validate_n(n)
    steps = positive_integer(steps, "steps")
    hidden_size = positive_integer(hidden_size, "hidden_size")
    if seed is None:
        raise TypeError("supervised probe requires an explicit seed")
    validate_seed(seed)
    learning_rate = finite_reward(learning_rate)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    model = RecurrentActorCritic(hidden_size=hidden_size, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    examples = [(Observation(own, other, float(reward), False), own, reward)
                for own in range(n) for other in range(n) for reward in (0, 1)]
    target_entropy = .5 * math.log(n - 1)

    def batch_metrics():
        losses, repeats, avoids = [], [], []
        for observation, own, reward in examples:
            logits, _, _ = model.forward_observation(observation, model.initial_hidden(), n=n)
            logs = torch.log_softmax(logits, dim=0)
            probs = logs.exp()
            if reward:
                losses.append(-logs[own]); repeats.append(probs[own])
            else:
                # Explicitly sum only non-own terms; no accidental -0 * -inf.
                losses.append(-torch.cat((logs[:own], logs[own + 1:])).mean())
                avoids.append(1 - probs[own])
        loss = torch.stack(losses).mean()
        repeat = torch.stack(repeats).mean()
        avoid = torch.stack(avoids).mean()
        return loss, repeat, avoid

    def snapshot() -> dict[str, float]:
        with torch.no_grad():
            loss, repeat, avoid = batch_metrics()
        return {"cross_entropy": float(loss), "excess_cross_entropy": float(loss) - target_entropy,
                "repeat_probability_after_success": float(repeat),
                "avoid_probability_after_failure": float(avoid)}

    initial = snapshot()
    for _ in range(steps):
        loss, _, _ = batch_metrics()
        if not torch.isfinite(loss).item():
            raise ValueError("supervised probe produced non-finite loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = snapshot()
    if not all(math.isfinite(v) for v in final.values()):
        raise ValueError("supervised probe produced non-finite metrics")
    return {"seed": seed, "steps": steps, "n": n, "hidden_size": hidden_size,
            "learning_rate": learning_rate, "examples": len(examples),
            "target_entropy": target_entropy, "initial": initial, "final": final}


def _html(value: object) -> str:
    return escape(str(value), quote=True)


def _number(value: Any) -> str:
    return "미정의" if value is None else _html(value)


def _run_rows(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return '<tr><td colspan="13">기록 없음</td></tr>'
    rows = []
    for run in runs:
        comparisons = run["comparisons"]
        initial = comparisons["initial"]
        random_policy = comparisons["random"]
        trained = comparisons["trained"]
        reset = comparisons["hidden_reset"]
        random_diff = run["paired_vs_random"]
        memory_diff = run["paired_memory_benefit"]
        target = "가변" if run["target"] is None else _html(run["target"])
        rows.append(f'''<tr><th scope="row">{_html(run["seed"])}</th><td>{target}</td><td>{_html(run["update"])}</td>
          <td>{_number(initial["mean_reward_per_step"])}</td><td>{_number(random_policy["mean_reward_per_step"])}</td>
          <td>{_number(trained["mean_reward_per_step"])}</td><td>{_number(trained["standard_error"])}</td>
          <td>{_number(reset["mean_reward_per_step"])}</td><td>{_number(random_diff["mean"])} ± {_number(random_diff["standard_error"])}</td>
          <td>{_number(memory_diff["mean"])} ± {_number(memory_diff["standard_error"])}</td>
          <td>{_number(trained["repeat_after_success_rate"])} ({_number(trained["repeat_after_success_count"])}/{_number(trained["success_opportunities"])})</td>
          <td>{_number(trained["second_episode_first_action_mean"])}</td>
          <td>{_number(trained["second_episode_first_three_actions_mean"])}</td></tr>''')
    return "".join(rows)


def _run_table(runs: list[dict[str, Any]], caption: str) -> str:
    return f'''<div class="scroll"><table><caption>{_html(caption)}</caption><thead><tr>
      <th scope="col">학습 seed</th><th scope="col">target</th><th scope="col">update</th>
      <th scope="col">초기 평균</th><th scope="col">Random 평균</th><th scope="col">학습 평균</th><th scope="col">학습 SE</th><th scope="col">hidden 초기화 평균</th>
      <th scope="col">학습 − Random ± SE</th><th scope="col">hidden 유지 − 초기화 ± SE</th>
      <th scope="col">성공 후 반복 (횟수/기회)</th><th scope="col">두 번째 episode 첫 행동 보상</th><th scope="col">두 번째 episode 첫 세 행동 평균 보상</th>
      </tr></thead><tbody>{_run_rows(runs)}</tbody></table></div>'''


def render_diagnostic_report(record: dict[str, Any]) -> str:
    """Create a local HTML report directly from a diagnostic result record."""
    fixed = record["fixed_runs"]
    variable = record["variable_runs"]
    supervised = record["supervised_runs"]
    supervised_rows = []
    for run in supervised:
        initial, final = run["initial"], run["final"]
        supervised_rows.append(f'''<tr><th scope="row">{_html(run["seed"])}</th><td>{_html(run["steps"])}</td><td>{_html(run["examples"])}</td>
          <td>{_number(run["target_entropy"])}</td><td>{_number(initial["cross_entropy"])}</td><td>{_number(final["cross_entropy"])}</td>
          <td>{_number(final["excess_cross_entropy"])}</td><td>{_number(final["repeat_probability_after_success"])}</td>
          <td>{_number(final["avoid_probability_after_failure"])}</td></tr>''')
    if not supervised_rows:
        supervised_rows.append('<tr><td colspan="9">기록 없음</td></tr>')
    protocol = _html(json.dumps(record["protocol"], ensure_ascii=False, indent=2, sort_keys=True))
    aggregates = record.get("aggregates") or {}
    aggregate_rows = []
    for family, label in (("fixed_by_target", "고정 target"), ("variable_by_update", "가변 update")):
        for key, data in (aggregates.get(family) or {}).items():
            aggregate_rows.append(
                f'<tr><th scope="row">{_html(label)} {_html(key)}</th>'
                f'<td>{_number(data["learning_seed_count"])}</td>'
                f'<td>{_number(data["mean_reward_across_learning_seeds"])}</td>'
                f'<td>{_number(data["sd_across_learning_seeds"])}</td>'
                f'<td>{_number(data["mean_paired_vs_random"])}</td>'
                f'<td>{_number(data["mean_paired_memory_benefit"])}</td></tr>'
            )
    overview = ""
    if aggregate_rows:
        overview = f'''<section class="card" id="overview"><h2>학습 seed별 결과 요약</h2>
<p class="muted">각 행은 해당 고정 target 또는 가변 학습 update의 독립 학습 seed 결과를 요약합니다.</p>
<div class="scroll"><table><caption>고정 target별 및 가변 update별 학습 결과</caption>
<thead><tr><th scope="col">조건</th><th scope="col">학습 seeds</th><th scope="col">평균 보상</th><th scope="col">학습 seed 간 SD</th><th scope="col">평균 짝지은 Random 차이</th><th scope="col">평균 짝지은 기억 차이</th></tr></thead>
<tbody>{''.join(aggregate_rows)}</tbody></table></div>
<p>여기의 SD는 독립 학습 seed 결과 간 표본 표준편차입니다. 아래 개별 실행 표의 SE는 해당 실행에서 평가 trial의 변동으로 계산하므로 서로 다른 불확실성입니다.</p></section>'''
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAMRL 학습 진단</title><style>
:root{{--ink:#142334;--muted:#5d6b78;--line:#dce4e8;--bg:#f2f5f3}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);font:15px/1.6 system-ui,-apple-system,sans-serif;color:var(--ink)}}
.shell{{width:min(1280px,calc(100% - 40px));margin:auto}}header{{background:#102d34;color:white;padding:48px 0 40px}}h1{{font-size:clamp(2rem,5vw,3rem);line-height:1.12;margin:10px 0}}h2{{margin:0 0 .4rem}}header p{{color:#d2e0dc;max-width:850px}}.eyebrow{{color:#a5d9c8;font-weight:700;letter-spacing:.13em;font-size:.78rem}}
a{{color:#096b64;text-underline-offset:3px}}a:focus-visible,summary:focus-visible{{outline:3px solid #e29b38;outline-offset:3px}}nav{{display:flex;gap:20px;flex-wrap:wrap;padding:17px 0;border-bottom:1px solid var(--line);font-weight:650}}
main{{padding:26px 0 60px}}.card{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin-bottom:18px}}.muted{{color:var(--muted)}}.warning{{background:#fff5e8;border-left:3px solid #da9a3b;padding:10px 14px;color:#684a1d}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:950px;font-variant-numeric:tabular-nums}}caption{{text-align:left;color:var(--muted);padding:8px 0}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:9px 11px;vertical-align:top}}th{{font-size:.85rem;color:#465967}}tbody th{{color:var(--ink)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f8f6;padding:16px;border-radius:8px}}summary{{cursor:pointer;color:#096b64;font-weight:650}}
@media(max-width:600px){{.shell{{width:calc(100% - 24px)}}header{{padding:34px 0}}.card{{padding:17px}}nav{{gap:12px}}}}
</style></head><body><header><div class="shell"><div class="eyebrow">MAMRL · LEARNING DIAGNOSTICS</div><h1>학습 진단</h1>
<p>고정 target, trial별 가변 target, 지도학습 표현력 점검을 비교합니다. 상대는 고정 Random 정책입니다.</p></div></header>
<div class="shell"><nav aria-label="리포트 목차">{'<a href="#overview">요약</a>' if aggregate_rows else ''}<a href="#fixed">고정 target</a><a href="#variable">가변 target</a><a href="#probe">지도학습 점검</a><a href="#protocol">조건과 한계</a></nav><main>{overview}
<section class="card" id="fixed"><h2>고정 target 학습</h2><p class="muted">실행별 실제 target·학습 seed·update를 표시합니다. 학습 평균은 행동당 평균 보상입니다.</p>
{_run_table(fixed,"고정 target별 진단 결과")}
<p class="warning">고정 target에서 높은 보상은 하나의 상수 행동을 학습하는 것으로도 얻을 수 있어, 보상 조건부 행동이나 적응을 증명하지 않습니다.</p></section>
<section class="card" id="variable"><h2>가변 target 학습</h2><p class="muted">target은 trial마다 바뀌며 평가에서는 target층을 균형 있게 사용합니다. update 값은 같은 경로의 checkpoint를 구별합니다.</p>
{_run_table(variable,"가변 target 진단 결과")}
<p>동일 예산 비교에서는 고정 조건과 가변 조건의 <strong>같은 update</strong>를 비교합니다. 기본 행렬에서는 고정 40과 가변 40이 같은 예산 비교이고, 가변 40과 가변 120은 동일 학습 경로에서의 추가 예산 비교입니다. 서로 다른 설정을 한 평균에 섞지 않습니다.</p></section>
<section class="card" id="probe"><h2>지도학습 표현력 점검</h2><p class="muted">각 관측은 zero hidden에서 처리합니다. 성공 후에는 이전 행동을, 실패 후에는 다른 행동들의 균등 분포를 목표로 둡니다.</p>
<div class="scroll"><table><caption>동일 seed 초기 모델의 전체 관측 집합 fit</caption><thead><tr><th scope="col">seed</th><th scope="col">steps</th><th scope="col">예제 수</th><th scope="col">목표 엔트로피</th><th scope="col">초기 교차 엔트로피</th><th scope="col">최종 교차 엔트로피</th><th scope="col">최종 초과 교차 엔트로피</th><th scope="col">성공 후 반복 확률</th><th scope="col">실패 후 회피 확률</th></tr></thead><tbody>{''.join(supervised_rows)}</tbody></table></div>
<p class="warning">이 점검은 {"18개" if supervised and all(x["examples"] == 18 for x in supervised) else "명시된 수의"} 고정 관측에 대한 학습 집합 fit입니다. 미관측 사례 일반화나 장기 기억 능력을 검증하지 않습니다.</p></section>
<section class="card" id="protocol"><h2>조건과 해석 범위</h2><p>SE는 target층별 표본비중을 반영한 평가 trial의 표준오차입니다. 어느 target층이든 trial이 2개 미만이면 SE는 미정의입니다. 학습 seed 간 변동과는 다릅니다. 짝지은 차이는 같은 target 순서와 외생 seed의 trial별 평균 보상 차이입니다.</p>
<p>진단 task는 action_bonus에 한정되어 있으며, 결과는 새로운 게임 규칙에 대한 일반화나 Multi-Agent 적응의 증거가 아닙니다. 각 행은 별도의 학습 seed 결과이며 평가 task는 재사용됩니다. 성공 임계값을 자동으로 가정하지 않습니다.</p>
<details><summary>전체 실행 조건 보기</summary><pre>{protocol}</pre></details><p><a href="diagnostic.json">원본 진단 기록 JSON</a></p></section>
</main></div></body></html>'''
