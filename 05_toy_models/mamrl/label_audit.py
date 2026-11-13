"""Action-label permutation and PPO gradient diagnostics.

The wrapper is an analysis intervention, not a new adopted baseline. The
gradient audits never change source parameters, stored gradients or RNG state.
"""

from __future__ import annotations

from copy import deepcopy
from html import escape
from itertools import product
import json
import random
from typing import Any

import torch

from .common import finite_reward, validate_action, validate_n
from .environment import MemoryOneEnv, Observation
from .ppo import compute_gae
from .recurrent import RecurrentActorCritic
from .rules import RuleSpec, RuleTerm, compile_rule
from .training import TrainConfig, collect_trial, replay_trial, sample_task


class PermutedActorCritic(RecurrentActorCritic):
    """Map physical labels to internal labels on input and back on output."""

    def __init__(self, hidden_size: int = 32, seed: int = 0, *, permutation=(0, 1, 2)):
        if not isinstance(permutation, tuple):
            raise TypeError("permutation must be a tuple")
        n = validate_n(len(permutation))
        if any(isinstance(x, bool) or not isinstance(x, int) for x in permutation):
            raise TypeError("permutation entries must be integers")
        if set(permutation) != set(range(n)):
            raise ValueError("permutation must be a bijection of 0..n-1")
        super().__init__(hidden_size=hidden_size, seed=seed)
        self.permutation = permutation

    def forward_observation(self, observation, hidden, *, n):
        n = validate_n(n)
        if n != len(self.permutation):
            raise ValueError("action count must equal the permutation length")
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        own = validate_action(observation.previous_own_action, n)
        other = validate_action(observation.previous_opponent_action, n)
        finite_reward(observation.previous_own_reward)
        if type(observation.episode_start_flag) is not bool:
            raise TypeError("episode_start_flag must be bool")
        mapped = Observation(self.permutation[own], self.permutation[other],
                             observation.previous_own_reward, observation.episode_start_flag)
        logits, value, next_hidden = super().forward_observation(mapped, hidden, n=n)
        return logits[list(self.permutation)], value, next_hidden


def reward_audit(n: int = 3) -> dict[str, Any]:
    """Exhaustively compare tensor and real environment reward to a predicate."""
    n = validate_n(n)
    # Seed each environment reset to cover every possible previous joint action.
    seeds: dict[tuple[int, int], int] = {}
    seed = 0
    while len(seeds) < n * n:
        rng = random.Random(seed)
        pair = (rng.randrange(n), rng.randrange(n))
        seeds.setdefault(pair, seed)
        seed += 1
    checked_entries = checked_steps = 0
    for target in range(n):
        tensor = compile_rule(RuleSpec(n, (RuleTerm("action_bonus", 1, target),)))
        env = MemoryOneEnv(tensor, max_steps=1)
        for p0, p1, a0, a1 in product(range(n), repeat=4):
            expected = (float(a0 == target), float(a1 == target))
            for player in (0, 1):
                actual = tensor[player, p0, p1, a0, a1]
                if actual != expected[player]:
                    raise AssertionError(f"tensor mismatch target={target}, previous={(p0,p1)}, actions={(a0,a1)}, player={player}")
                checked_entries += 1
            observations = env.reset(seed=seeds[(p0, p1)])
            if (observations[0].previous_own_action, observations[0].previous_opponent_action) != (p0, p1):
                raise AssertionError("environment reset failed to recreate the selected previous state")
            result = env.step(a0, a1)
            if result.rewards != expected:
                raise AssertionError(f"step reward mismatch target={target}, previous={(p0,p1)}, actions={(a0,a1)}")
            checked_steps += 1
    return {"passed": True, "tensor_entries_checked": checked_entries, "environment_steps_checked": checked_steps}


def actor_gradient_audit(model: RecurrentActorCritic, *, n: int = 3) -> dict[str, Any]:
    """Audit probability derivatives w.r.t. a float64 actor at detached states."""
    if type(model) is not RecurrentActorCritic:
        raise TypeError("actor audit expects an unpermuted base RecurrentActorCritic")
    n = validate_n(n)
    actor = deepcopy(model.actor).double()
    params = tuple(actor.parameters())
    names = tuple(name for name, _ in actor.named_parameters())
    inputs = []
    with torch.no_grad():
        for own, opponent in product(range(n), repeat=2):
            observation = Observation(own, opponent, 0.0, True)
            _, _, hidden = model.forward_observation(observation, model.initial_hidden(), n=n)
            actions = torch.arange(n, dtype=torch.float64)
            candidates = torch.stack((actions/(n-1), (actions == own).double(),
                                      (actions == opponent).double()), dim=1)
            inputs.append(torch.cat((hidden.detach().double().unsqueeze(0).expand(n, -1), candidates), dim=1))
    inputs = torch.stack(inputs)

    def probabilities():
        return torch.softmax(actor(inputs).squeeze(-1), dim=1)

    targets = []
    for target in range(n):
        # Each target needs a fresh graph after the previous finite-difference mutations.
        probabilities_tensor = probabilities()
        objective = probabilities_tensor[:, target].mean()
        gradients = torch.autograd.grad(objective, params, retain_graph=True)
        flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
        maximum = int(flat.abs().argmax())
        offset = 0
        chosen = 0
        for index, parameter in enumerate(params):
            if maximum < offset + parameter.numel():
                chosen = index
                break
            offset += parameter.numel()
        local_index = maximum - offset
        estimate = float(flat[maximum])
        per_observation = []
        for obs_idx in range(n*n):
            gs = torch.autograd.grad(probabilities_tensor[obs_idx, target], params, retain_graph=True)
            per_observation.append(float(torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in gs]))))
        # Mutation is confined to our copied actor and restored before next target.
        differences = []
        parameter = params[chosen]
        with torch.no_grad():
            original = float(parameter.reshape(-1)[local_index])
            for epsilon in (1e-4, 1e-5, 1e-6):
                parameter.reshape(-1)[local_index] = original + epsilon
                plus = float(probabilities()[:, target].mean())
                parameter.reshape(-1)[local_index] = original - epsilon
                minus = float(probabilities()[:, target].mean())
                parameter.reshape(-1)[local_index] = original
                central = (plus-minus)/(2*epsilon)
                differences.append({"epsilon": epsilon, "central_difference": central,
                                    "absolute_error": abs(central-estimate)})
        targets.append({"target": target, "mean_probability": float(objective.detach()),
                        "mean_gradient_norm": float(torch.linalg.vector_norm(flat)),
                        "mean_per_observation_gradient_norm": sum(per_observation)/len(per_observation),
                        "per_observation_gradient_norms": per_observation,
                        "largest_gradient_coordinate": {"parameter": names[chosen], "index": local_index,
                                                        "autograd": estimate, "finite_differences": differences}})
    # This deliberately simplified linear fixture is not the full actor model.
    w = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    scores = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    fixture_probs = torch.softmax(w*scores, dim=0)
    derivatives = [float(torch.autograd.grad(fixture_probs[i], w, retain_graph=True)[0]) for i in range(3)]
    return {"n": n, "observation_count": n*n, "targets": targets,
            "linear_ordinal_fixture": {"scores": [0.0, 0.5, 1.0], "derivatives": derivatives}}


def loss_gradient_audit(model: RecurrentActorCritic, config: TrainConfig) -> dict[str, Any]:
    """Reconstruct the first real PPO batch and decompose its gradients.

    Uses the same local task/rollout RNG stream and full-trial replay as train.
    No backward() or optimizer step is performed on the source model.
    """
    if not isinstance(model, RecurrentActorCritic):
        raise TypeError("model must be a RecurrentActorCritic")
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    rng = random.Random(config.seed)
    batch = []
    for _ in range(config.trials_per_update):
        spec = sample_task(config.n, config.family, rng.getrandbits(63), fixed_action=config.fixed_action)
        batch.append(collect_trial(model, spec, episodes_per_trial=config.episodes_per_trial,
                                   max_steps=config.max_steps, seed=rng.getrandbits(63)))
    advantages, returns = [], []
    for trial in batch:
        advantage, target = compute_gae(trial.rewards, trial.old_values,
                                        gamma=config.gamma, gae_lambda=config.gae_lambda, bootstrap_value=0.0)
        advantages.append(advantage)
        returns.append(target)
    advantage = torch.cat(advantages)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    target = torch.cat(returns)
    old_log_probs = torch.tensor([x for trial in batch for x in trial.old_log_probs])
    replay = [replay_trial(model, trial) for trial in batch]
    log_probs = torch.cat([x[0] for x in replay])
    values = torch.cat([x[1] for x in replay])
    entropies = torch.cat([x[2] for x in replay])
    ratio = torch.exp(log_probs - old_log_probs.detach())
    policy = -torch.minimum(ratio*advantage.detach(),
                            torch.clamp(ratio, 1-config.clip_ratio, 1+config.clip_ratio)*advantage.detach()).mean()
    value = .5*(values-target.detach()).square().mean()
    entropy = entropies.mean()
    named = tuple(model.named_parameters())
    params = tuple(parameter for _, parameter in named)
    group_indices = {group: [i for i, (name, _) in enumerate(named) if name.startswith(group + ".")]
                     for group in ("actor", "gru", "critic")}
    if sum(map(len, group_indices.values())) != len(params):
        raise ValueError("model has parameters outside actor/gru/critic groups")

    def gradients(objective):
        return torch.autograd.grad(objective, params, retain_graph=True, allow_unused=True)

    def group_vector(grads, group):
        return torch.cat([((grads[i] if grads[i] is not None else torch.zeros_like(params[i])).reshape(-1))
                          for i in group_indices[group]])

    def norms(grads):
        return {group: float(torch.linalg.vector_norm(group_vector(grads, group))) for group in group_indices}

    raw_gradients = {"policy": gradients(policy), "value": gradients(value), "entropy": gradients(entropy)}
    weights = {"policy": 1.0, "value": config.value_coefficient, "entropy": -config.entropy_coefficient}
    objectives = {"policy": policy, "value": value, "entropy": entropy}
    components = {}
    for name in ("policy", "value", "entropy"):
        raw = norms(raw_gradients[name])
        components[name] = {"raw_loss": float(objectives[name].detach()), "weight": weights[name],
                            "raw_group_norms": raw,
                            "weighted_group_norms": {group: abs(weights[name])*norm for group, norm in raw.items()}}
    total = policy + config.value_coefficient*value - config.entropy_coefficient*entropy
    total_grads = gradients(total)
    total_vector = torch.cat([((grad if grad is not None else torch.zeros_like(param)).reshape(-1))
                              for grad, param in zip(total_grads, params)])
    total_norm = float(torch.linalg.vector_norm(total_vector))
    policy_gru = group_vector(raw_gradients["policy"], "gru")
    value_gru = group_vector(raw_gradients["value"], "gru")
    policy_norm = float(torch.linalg.vector_norm(policy_gru))
    value_norm = float(torch.linalg.vector_norm(value_gru))
    cosine = float(torch.dot(policy_gru, value_gru) / (policy_norm*value_norm)) if policy_norm and value_norm else None
    return {"batch_trials": len(batch), "batch_steps": sum(len(trial.actions) for trial in batch),
            "components": components, "total_gradient_norm": total_norm,
            "global_clip_multiplier": min(1.0, config.max_grad_norm / (total_norm + 1e-6)),
            "shared_gru_policy_value_cosine": cosine}


def render_label_audit_report(record: dict[str, Any]) -> str:
    """Render measurements and crossed-label results without external assets."""
    def text(value):
        return escape(str(value), quote=True)

    def number(value):
        return '미정의' if value is None else text(format(value, '.6g'))

    # Each cell averages only the learning seeds for this physical/internal pair.
    cells = []
    for physical in range(3):
        row = [f'<th scope="row">물리 행동 {physical}</th>']
        for internal in range(3):
            runs = [r for r in record['runs'] if r['physical_target'] == physical and r['internal_target'] == internal]
            value = sum(r['comparisons']['trained']['mean_reward_per_step'] for r in runs)/len(runs) if runs else None
            row.append(f'<td>{number(value)}</td>')
        cells.append('<tr>'+''.join(row)+'</tr>')
    gradients = []
    for entry in record['initial_gradient_audits']:
        for target in entry['audit']['targets']:
            gradients.append(f'<tr><td>{text(entry["seed"])}</td><td>{text(target["target"])}</td>'
                             f'<td>{number(target["mean_probability"])}</td>'
                             f'<td>{number(target["mean_gradient_norm"])}</td>'
                             f'<td>{number(target["mean_per_observation_gradient_norm"])}</td></tr>')
    losses = []
    for entry in record['loss_gradient_audits']:
        audit = entry['audit']
        comp = audit['components']
        losses.append(f'<tr><td>{text(entry["seed"])}</td><td>{text(entry["physical_target"])}</td>'
                      f'<td>{number(comp["policy"]["weighted_group_norms"]["actor"])}</td>'
                      f'<td>{number(comp["policy"]["weighted_group_norms"]["gru"])}</td>'
                      f'<td>{number(comp["value"]["weighted_group_norms"]["gru"])}</td>'
                      f'<td>{number(audit["shared_gru_policy_value_cosine"])}</td>'
                      f'<td>{number(audit["global_clip_multiplier"])}</td></tr>')
    rows = []
    for run in record['runs']:
        comp = run['comparisons']
        rows.append(f'<tr><td>{text(run["seed"])}</td><td>{text(run["permutation"])}</td>'
                    f'<td>{text(run["physical_target"])}</td><td>{text(run["internal_target"])}</td>'
                    f'<td>{number(comp["initial"]["mean_reward_per_step"])}</td>'
                    f'<td>{number(comp["random"]["mean_reward_per_step"])}</td>'
                    f'<td>{number(comp["trained"]["mean_reward_per_step"])}</td>'
                    f'<td>{number(run["initial_to_trained"]["mean"])}</td>'
                    f'<td><a href="{text(run["raw_record"])}">JSON</a></td></tr>')
    reward = record['reward_audit']
    protocol = text(json.dumps(record['protocol'], ensure_ascii=False, indent=2))
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MAMRL 행동 번호 분석</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f3;color:#142334;font:15px/1.65 system-ui,sans-serif}}
header{{background:#102d34;color:white;padding:36px 24px}}main{{max-width:1200px;margin:auto;padding:24px}}
h1{{margin:4px 0}}h2{{margin:0 0 12px}}section{{background:white;border:1px solid #dce4e8;border-radius:12px;padding:22px;margin-bottom:18px}}
a{{color:#096b64}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #dce4e8}}th{{color:#465967}}caption{{text-align:left;color:#5d6b78;padding:8px 0}}
.note{{background:#eef7f4;padding:12px;border-left:3px solid #31a98f}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
summary{{cursor:pointer}}@media(max-width:600px){{main{{padding:12px}}section{{padding:15px}}}}
</style></head><body><header><h1>행동 번호와 학습 경사 분석</h1>
<p>물리 게임은 유지하고 모델 내부의 행동 번호만 바꾼 대조 실험입니다.</p></header><main>
<section><h2>핵심 비교: 보상 행동의 물리 번호와 내부 번호</h2>
<p>각 칸은 해당 물리·내부 번호 조합에서 학습 seed별 행동당 보상의 평균입니다. 실제 게임의 보상 규칙은 행마다 동일합니다.</p>
<div class="scroll"><table><caption>학습 후 평균 보상</caption><thead><tr><th>보상받는 물리 행동</th><th>내부 0</th><th>내부 1</th><th>내부 2</th></tr></thead><tbody>{''.join(cells)}</tbody></table></div>
<p class="note">이 개입은 GRU 관측 인코딩과 actor 후보 인코딩을 함께 바꿉니다. 결과를 actor 단독 원인이나 새 구조의 우월성으로 해석하지 않습니다.</p></section>
<section><h2>보상·인덱스 감사</h2><p>Passed: {text(reward['passed'])} · 텐서 항목 {text(reward['tensor_entries_checked'])}개 · 실제 환경 step {text(reward['environment_steps_checked'])}개를 독립 보상 조건과 대조했습니다.</p></section>
<section><h2>초기 actor의 정확한 경사</h2><p>9개 초기 관측, 고정 GRU 출력에서 float64 actor의 평균 p(target)를 미분했습니다. 평균 경사의 norm과 관측별 norm의 평균을 구분합니다. finite difference 원자료는 JSON에 있습니다.</p>
<div class="scroll"><table><thead><tr><th>seed</th><th>내부 target</th><th>평균 확률</th><th>norm(mean gradient)</th><th>mean(norm gradient)</th></tr></thead><tbody>{''.join(gradients)}</tbody></table></div>
<p>단순 선형 ordinal slope fixture의 미분은 (-1/6, 0, +1/6)입니다. 이 식은 실제 신경망 전체가 아닙니다.</p></section>
<section><h2>첫 PPO batch의 경사 분해</h2><p>실제 학습 seed와 GAE로 재구성했습니다. 가중치를 적용한 norm이며 GRU cosine은 공통 parameter에 대해서만 계산합니다. 크기 차이만으로 원인을 단정하지 않습니다.</p>
<div class="scroll"><table><thead><tr><th>seed</th><th>물리 target</th><th>Policy→actor</th><th>Policy→GRU</th><th>Value→GRU</th><th>GRU policy/value cosine</th><th>Global clip 배율</th></tr></thead><tbody>{''.join(losses)}</tbody></table></div></section>
<section><h2>모든 실행</h2><div class="scroll"><table><thead><tr><th>seed</th><th>물리→내부 매핑</th><th>물리 target</th><th>내부 target</th><th>초기</th><th>Random</th><th>학습 후</th><th>초기 대비</th><th>기록</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>같은 외생 환경·상대 seed를 사용하지만 learner의 확률과 선택 경로는 달라질 수 있습니다. 반복된 seed/매핑을 모두 독립 학습 seed로 세지 않습니다.</p></section>
<section><h2>재현과 해석 범위</h2><p>internal_weights.pt는 내부 좌표의 가중치입니다. 물리 정책 복원에는 기록된 permutation과 load_physical_model이 필요합니다. 기존 baseline 구조는 변경하지 않았습니다.</p>
<p>세 cyclic mapping과 고정 action_bonus 과제의 진단입니다. 미지 규칙 적응·장기 기억·행동 공간 일반화 검증은 아닙니다.</p>
<details><summary>실행 조건과 source hash</summary><pre>{protocol}</pre></details><p><a href="audit.json">전체 분석 JSON</a> · <a href="gradient_audits.json">보상·경사 원자료</a></p></section>
</main></body></html>'''
