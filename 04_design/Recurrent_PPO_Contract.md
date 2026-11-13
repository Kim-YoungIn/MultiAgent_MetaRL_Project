# Recurrent PPO Engineering Baseline — v1

## 범위와 채택한 작업 조건

- 단일 P0 learner vs P1 고정 RandomPolicy. Independent Self-Play는 아직 아님.
- 기본 n=3, 같은 RuleSpec의 episodes_per_trial개 episode를 하나의 유한 trial로 묶는다.
- 환경의 랜덤 초기화·비공개 fixed horizon·네 필드 관측은 바꾸지 않는다.
- learner hidden은 trial 시작 zero, episode 경계 유지. 평가 중 파라미터 업데이트 없음.
- 중간 episode의 마지막 관측(final action/reward)을 observe-only event로 GRU에 한 번 전달한 뒤
  새 reset 관측을 처리한다. observe-only에는 행동·RNG sampling·보상·loss·discount timestep이 없다.
- 최종 trial에서는 bootstrap=0. 중간 episode truncated는 return/GAE를 끊지 않는다.
  이 목적은 유한 trial의 discounted return이며 계속되는 무한 task를 임의로 자른 것과 다르다.
- PPO 매 epoch마다 zero hidden에서 전체 trial 관측 이벤트를 순서대로 다시 forward한다.
  episode 경계 detach나 저장 hidden 재사용, 시간 순서 shuffle은 하지 않는다.
- 진단 task family `action_bonus`는 target을 trial마다 균등 선택한 기존 action_bonus term(계수1).
  기존 generator의 zero_sum/coordination/memory_one/mixed도 선택 가능하다.
- action_bonus 평가 seed는 학습과 분리하지만 target 구조는 겹친다. 미지 규칙 일반화 주장은 금지한다.
- 이번 값은 실행 가능한 baseline이며 최종 연구 조건이 아니다. Exploration 별도 목적함수는 추가하지 않는다.

## 선택 의존성

코어 환경/리포트는 표준 라이브러리만 사용한다. 학습은 별도 .venv의 CPU PyTorch를 사용한다.
requirements-training.txt에 실제 실행 버전을 고정한다. torch가 없는 환경에서 학습 테스트만 skip한다.

## 모델과 정책 (mamrl.recurrent)

- `RecurrentActorCritic(hidden_size=32, seed=0)`은 CPU torch.nn.Module.
- `initial_hidden()` → float32 Tensor(H), zero.
- `forward_observation(observation, hidden, *, n)` → `(logits Tensor(n), value scalar Tensor, next_hidden Tensor(H))`.
- GRUCell 입력 6개: own/(n-1), opponent/(n-1), own==opponent,
  tanh(previous_own_reward/4), episode_start_flag, 1/n.
- 각 candidate 행동의 특징 3개: action/(n-1), action==previous_own_action,
  action==previous_opponent_action. shared scorer에 hidden과 결합하여 n개 logits 출력.
- value는 hidden으로 계산. 파라미터 shape는 n에 독립. 같은 weights로 n=3/4/5 실행 가능.
  정규화 index는 행동 라벨의 순서 구조를 가정하며 n 간 학습 일반화는 미검증이다.
- init은 전역 torch RNG를 보존한다. n·seed·관측·hidden shape/finite를 검증한다.
- `RecurrentPolicy(model)`은 기존 Policy API. initial_state는 플레이어별 hidden과 local torch.Generator를 생성.
  act는 no_grad inference + local multinomial sampling, 반환 PolicyOutput. 정책은 episode 길이를 전달받지 않음.
  명시 seed는 PyTorch RNG 범위에 맞춰 modulo 2**63로 매핑한다.
- `save_checkpoint(model, path)` / `load_checkpoint(path)`.
  format_version=1, architecture.hidden_size, state_dict만 저장. 새 파일만 생성.
  CPU weights_only=True 로드, shape/key/finite 검증. optimizer resume는 범위 밖.

## PPO 수치 계약 (mamrl.ppo)

- `compute_gae(rewards, values, *, gamma=.99, gae_lambda=.95, bootstrap_value=0.0)`
  → `(advantages Tensor(T), returns Tensor(T))` (float32 CPU).
  비어 있지 않은 동일 길이의 유한 실수 sequence, gamma/lambda in [0,1].
  delta[t]=reward[t]+gamma*value[t+1]-value[t], A[t]=delta[t]+gamma*lambda*A[t+1].
  returns=A+values, 마지막 next_value=bootstrap_value. episode별 mask 없음.
- `ppo_loss(new_log_probs, old_log_probs, new_values, returns, advantages, entropies,
  *, clip_ratio=.2, value_coefficient=.5, entropy_coefficient=.01)` → `(loss Tensor, metrics dict)`.
  모두 같은 비어 있지 않은 1D CPU float Tensor. old logprob/returns/advantages는 detach target.
  ratio=exp(new-old); policy_loss=-mean(min(ratio*A,clamp(ratio,1-clip,1+clip)*A)).
  value_loss=.5*mean((new_value-return)**2). loss=policy_loss+value_coeff*value_loss-entropy_coeff*mean(entropy).
  metrics: loss, policy_loss, value_loss, entropy, approx_kl(mean((ratio-1)-logratio)), clip_fraction.
  비유한 입력/ratio/loss 및 잘못된 shape/config는 명확히 거부한다.
- advantage normalization은 trainer가 전체 수집 batch에서 한번 수행(population std + 1e-8).
- full trial batch PPO epochs, Adam, gradient norm clipping. PPO minibatch/TBPTT는 아직 없음.

## 수집·평가·학습 (mamrl.training)

- `TrainConfig` frozen dataclass: n=3, family='action_bonus', seed=7, hidden_size=32,
  updates=40, trials_per_update=8, episodes_per_trial=2, max_steps=12, epochs=3,
  learning_rate=.003, gamma=.99, gae_lambda=.95, clip_ratio=.2,
  value_coefficient=.5, entropy_coefficient=.01, max_grad_norm=.5,
  evaluation_trials=16, evaluation_interval=10.
  선택적 fixed_action=None은 기존 task sampling을 유지한다. 정수 지정은 action_bonus 진단에서만
  사용하며 train/eval target을 고정한다. 보상 target은 정책 입력에 전달하지 않는다.
- `collect_trial(model, spec, *, episodes_per_trial, max_steps, seed, reset_hidden=False)` → TrialRollout.
  model=None이면 Random learner 비교군. 무경사 수집; rule/environment/opponent/action RNG 분리.
  TrialRollout: spec, events(Observation tuple), action_event_indices(int tuple),
  actions/rewards/old_log_probs/old_values(tuple), episodes(EpisodeResult tuple),
  environment_seeds/opponent_seeds/learner_seeds(tuple), reset_event_indices(tuple, 기본 (0,)).
  각 episode 기록은 기존 recording.build_record와 연결 가능. learner_seeds는 trial 첫 seed를 쓰며
  동일 trial 내 generator가 계속된다. 이후 episode 기록 seed는 재실행을 보장하는 독립 seed가 아님을 메타 설명한다.
- `replay_trial(model, trial)` → `(log_probs Tensor(T), values Tensor(T), entropies Tensor(T))`.
  events를 모두 forward하지만 action_event_indices에 있는 실제 행동에만 loss 값을 만든다.
  reset_event_indices는 해당 관측 처리 전에 hidden을 zero로 만드는 위치다(평가 reset ablation 재현용).
- `evaluate(model, specs, *, episodes_per_trial, max_steps, seed, reset_hidden=False)` → dict.
  모델 weights는 바꾸지 않는다. 동일 specs/seed는 모델·ablation 간 같은 환경 초기화/상대 행동을 제공한다.
  반환: trials, mean_reward_per_step, mean_reward_by_episode(list), mean_reward_by_turn(list),
  trial_returns(list), standard_error(독립 trial 평균reward의 표본SE, trial 1개면0).
  mean_reward_by_episode는 각 episode의 행동당 평균, mean_reward_by_turn은 trial 내 실제 행동
  순서의 평균(길이 episodes_per_trial * max_steps)이다. observe-only는 길이에 포함하지 않는다.
- `train(config, *, checkpoint_callback=None, model_factory=None)` → TrainingResult(initial_model, model, history, evaluations, config).
  history는 update/loss/entropy/value_loss/approx_kl/clip_fraction/mean_reward_per_step/gradient_norm 포함.
  evaluations는 update=0과 evaluation_interval 및 마지막 update에서 같은 평가 draws로 trained 측정.
  초기 모델/Random/hidden-reset ablation 비교 및 JSON/HTML/체크포인트는 CLI가 저장.
  수동 snapshot callback은 update0 및 각 update 완료 후 호출한다. 기본None은 기존 결과를 유지한다.
  model_factory는 분석용 선택 인자다. None은 원래 모델, callable은 hidden_size/seed를 받아
  RecurrentActorCritic 호환 모델을 만든다. 현재 cyclic label 교환 실험만 사용하며 기본 구조는 변경하지 않는다.

## CLI / 검증

`PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.train --output-dir runs/recurrent_demo`
새 디렉터리에 config·training.json·initial.pt·policy.pt·report.html과 진단 episode 기록을 저장한다.
보고서는 update별 평가 보상, episode/turn별 평가, Random/초기/학습/hidden-reset 비교를 표시한다.
수치는 동일 diagnostic 조건에서만 비교. 여러 독립 학습 seed와 일반화/Exploration 증거는 후속이다.

`training.json` schema_version=1, kind='recurrent_ppo_engineering_baseline':
config, runtime(python_version/torch_version/device/threads), history, evaluations,
comparisons(initial/random/trained/hidden_reset), parameter_change_l2,
evaluation_task_specs, evaluation_seed를 저장한다.
`evaluation_trial.json`은 첫 평가 trial의 trial_seed/model_checkpoint와 TrialRollout의
dataclass 직렬화(관측 전용 event 및 행동/reset index 포함)를 저장한다.
sample_episode는 별도 seed·새 hidden으로 수행한 단일 episode이며 전체 trial 평가와 구별한다.
일반 실행 CLI의 `--checkpoint PATH`는 P0에 저장된 RecurrentPolicy를 연결하고 P1은 RandomPolicy를 유지한다.

독립 검증: 손계산 GAE와 양/음 advantage PPO clipping, gradient 전파·target detach,
full-trial rollout/replay likelihood 일치, observe-only event/episode경계 유지와 trial초기화,
RNG 분리 및 weights frozen 평가, 파라미터 실제 변경, n=3/4/5 실행, 체크포인트 재현.
학습 smoke에서 향상을 강제하는 취약한 임계 테스트를 쓰지 않는다.
