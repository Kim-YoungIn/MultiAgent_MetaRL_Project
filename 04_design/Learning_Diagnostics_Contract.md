# Learning Diagnostics — v1

## 목적과 변경 경계

기존 recurrent PPO의 성능 정체를 fixed-target 학습 능력, 보상 조건부 행동,
trial마다 달라지는 target에 대한 적응으로 나누어 측정한다. GPU·모델 구조·PPO 기본값을 바꾸지 않는다.
진단 과제는 action_bonus 계수1만 사용하며 상대는 고정 Random이다. 연구용 Multi-Agent 적응 증거가 아니다.

## 고정 target 선택 확장

- `TrainConfig.fixed_action: int | None = None` 추가. None이면 기존 동작과 동일.
- 지정 시 family는 action_bonus여야 하며 action은 0..n-1 정수(bool 거부).
- `sample_task(n, family, seed, *, fixed_action=None)`와 training/evaluation_specs가 이를 적용한다.
- CLI `mamrl.train --fixed-action A`. target은 정책 입력에 전달하지 않는다.
- 기존 모델/checkpoint/관측/return/episode 경계는 유지한다.

## 사전에 정한 실험 행렬

- learning seeds: [7,17,27]. 각 seed의 같은 초기 weights를 고정/가변 조건에 재사용한다.
- 고정 조건: target=0,1,2 각각 40updates. 총9run. 평가 target은 같은 고정값.
- 가변 조건: 기존 trial마다 uniform target, 각 seed 120updates. 총3run.
  update40과120에서 checkpoint를 별도로 비교한다(조기결과를 숨기지 않음).
- n3,H32,8trials/update,2episodes×12steps,3epochs,Adam .003 등 나머지 기본값은 유지한다.
- 진단 평가48trials. 가변은 target0/1/2 각16개를 interleave, 고정은 같은target48개.
- 평가 외생 RNG root=918273, 모든 조건/학습seed에서 공유한다. weights frozen.
- 초기/Random/학습/episode마다 hidden reset을 같은 spec·외생draws로 평가한다.
- 평가 trials의 분산과 독립 학습 seeds의 분산을 구분한다. 같은 eval tasks를 반복 사용한 결과다.
- 다르게 설정한 실행은 metadata에 전부 기록하며 기본 행렬 결과와 혼동하지 않는다.

## API (mamrl.diagnostics)

- `balanced_specs(n=3, repeats=16, fixed_action=None)` → RuleSpec tuple.
  가변의 순서는 각 repeat마다 targets 0..n-1. 고정은 n*repeats개의 같은 spec.
- `behavior_metrics(model, specs, *, episodes_per_trial, max_steps, seed=918273, reset_hidden=False)` → dict.
  기존 evaluate 결과의 모든 필드에 다음 추가:
  `repeat_after_success_rate`(기회0이면None), `repeat_after_success_count`, `success_opportunities`,
  `second_episode_first_action_mean`(episode1개면None),
  `second_episode_first_three_actions_mean`(episode1개면None; min(3,max_steps) 사용).
  `steps_per_trial`, `trial_targets`, `trial_seeds`와 `standard_error_method`도 보존한다.
  `n`, `episodes_per_trial`, `max_steps`, `spec_fingerprints`도 보존한다.
  fingerprint는 각 RuleSpec의 asdict를 sort_keys=True,separators=(',',':'),allow_nan=False로
  JSON 직렬화한 UTF-8 SHA256이다. 평가 spec들의 n은 동일해야 한다.
  기존 evaluate의 pooled SE는 pooled_standard_error로 보존하고, standard_error는
  target층별 평균의 분산을 표본비중으로 가중한 SE로 표시한다. 층별 표본이 2개 미만이면 None.
  episode 시작 관측 및 observe-only event는 반복기회에서 제외한다.
  기회는 이전 자기보상>0, 성공 반복은 이번행동==이전자기행동.
- `paired_difference(left, right)` → {mean, standard_error, trial_differences}.
  left/right는 같은 spec/외생draw로 평가한 metrics. 각 trial_returns를 실제 행동 수로 나눈 차이.
  길이/실제 행동수/target순서/trialseed/n/episode 구성/spec fingerprint 불일치 거부.
  steps_per_trial=episodes_per_trial*max_steps도 검증한다.
  by_target에 층별 mean/standard_error/trials/trial_differences도 보존한다.
  전체 SE는 target층별 paired 차이의 분산을 표본비중으로 가중한다(층당 표본<2면None).
- `supervised_probe(*, seed, steps=400, hidden_size=32, learning_rate=.003, n=3)` → dict.
  같은 초기 architecture의 별도 모델, n*n*2개 고정 관측, 매 예제 zero hidden.
  reward1: 이전자기행동 one-hot, reward0: 다른 행동 균등 target. cross entropy soft targets.
  desired distribution은 loss label로만 제공. 기존 PPO weights/설정은 변경하지 않는다.
  출력 initial/final의 cross_entropy, repeat_probability_after_success,
  avoid_probability_after_failure; target_entropy=.5*log(n-1), excess_cross_entropy도 보존한다.
  seed/steps/n도 보존. 학습 데이터 fit이며 held-out 성능/장기 기억 능력 아님.
- `render_diagnostic_report(record)` → 외부 의존성 없는 HTML. 실제 조건·seed별 값·비교와 한계 표시.

## 실행/산출물 (mamrl.diagnose)

새 디렉터리에 diagnostic.json / report.html / 실행별 checkpoint와 raw JSON을 보존한다.
TrainConfig에 callback 추가: `train(config, *, checkpoint_callback=None)`.
callback(update,model)은 update0 및 매update의 optimizer/평가 종료 후 호출되며,
수정 없이 진단 checkpoint 복사/저장 용도로만 사용한다. 기본값None이면 기존 결과를 바꾸지 않는다.
가변120update 실행의40/120 snapshot은 동일 학습 경로에서 가져온다.

record: schema_version1, protocol(모든 설정), fixed_runs(list), variable_runs(list),
supervised_runs(list), aggregates(dict).
각 run: seed,target(None이면가변),update,config,comparisons(initial/random/trained/hidden_reset),
paired_vs_random, paired_memory_benefit, checkpoint(상대경로), history(해당update까지),
raw trace provenance. fixed/variable task의 raw reward를 단일평균으로 섞지 않는다.
원래 baseline은 유지하고 결과에 따라 원인 후보를 좁힌다. 성공 임계값을 자동 테스트로 강제하지 않는다.
fixed40↔variable40은 같은 예산 비교, variable40↔120은 같은 학습 경로의 예산 비교로 구별한다.
fixed 성공은 조건부 보상 사용의 증거가 아니며 상수 행동으로 풀 수 있다.
