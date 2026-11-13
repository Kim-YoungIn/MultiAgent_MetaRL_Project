# Implementation Status — 2026-11-13

## 독립 검토 후 현재 평가와 다음 작업

최초 Toy A/B의 최소 구현·통합 인수 조건은 완료했다. 아래의 추가 학습 진단과 구분한다.
소프트웨어 정확성/제한된 정보 활용 진단은 수행했지만, 다양성 효과·held-out 전이의 H1/H2 비교는 미실시다.
D-018에 따라 다음 중심 작업은 통제된 논리 게임 분포·MA 상대군·규칙 차이·held-out 평가 설계이며,
작은 진단의 학습 성공을 그 선행 관문으로 두지 않는다. 이후 공통 정책의 전이 파일럿을 수행한다.
독립 검토와 인수 기준은 `sessions/07_independent_research_review.md`, 전달용 초안은 `sessions/08_colleague_progress_note.md`에 있다.
이번 검토는 문서 조정만 수행했다. 아래 테스트·학습 수치는 직전 실제 실행 기록이며 새 실행을 의미하지 않는다.

## 완료한 마일스톤: 비서열적 정책 교정과 기능 벤치마크

`Nominal_Policy_Benchmark_Contract.md`를 채택했다. `action_set.py`의 공유 행동별 GRU/actor는
숫자형 action index를 입력하지 않는다. 임의 순열에서 logits/hidden은 함께 변환되고 value는 유지된다.
메인에서 학습 checkpoint18개를 n=3/4/5로 검사한 최대 오류는 logits/value 4.7684e-7, hidden0이었다.
old ordinal_v1 파라미터6114/hidden32, 새 모델 파라미터5922/hidden96(n3,H32).
이는 상태 구조/용량도 다른 모델 비교이며 인코딩만의 단일 요인 성능 개선으로 해석하지 않는다.
새 모델에는 trial 시작부터 고정된 행동 ID를 선호하는 prior가 없다.

일반 학습 CLI는 action_set_v1을 기본으로 선택한다. 라이브러리 TrainConfig 기본ordinal_v1과
구 checkpoint v1은 과거 재현용으로 유지했다. 새 recurrent/reactive 모델은 v2로 저장한다.
과거 v1 checkpoint의 평가 수치가 정확히 재현됨을 메인이 확인했다.

세 가지 명시적 논리 benchmark를 추가했다:

- hidden_action: 숨은 보상 행동 추적. 기존 action_bonus template.
- probe_memory: 숨은 repeat/copy mode, 정답+1/다른 reference−3/그 외 probe−.25 또는−.5.
  서로 다른 실제 두 history가 같은 관측(own0,opp1,reward1,startFalse)에 도달하지만
  다음 최적 행동은0/1로 달라짐을 검증했다. 복잡한 규칙 발견의 증거는 아니다.
- opponent_response: coordination rule 고정, copy_previous/avoid_previous 상대.
  상대별 기대 best-response 상한은1/.5이며 각각의 보상/regret를 따로 기록한다.

관측4필드/Memory-One tensor/state/비공개 horizon은 유지했다. TaskSpec/모드/상대 유형은
연구자 metadata이며 learner는 family prior/n/seed만 받는다.
Random, 수작업 family-informed reactive, 온라인Q, family-informed belief를 비교한다.
Q는 평가 중 표를 갱신(alpha.5,gamma.99,epsilon.1), 신경망 weights는 고정한다.
informed references의 정확한 family likelihood 지식을 학습 모델과 동등한 prior로 취급하지 않는다.

```bash
PYTHONPATH=05_toy_models .venv/bin/python -B -m mamrl.benchmark_run --output-dir runs/nominal_functional_pilot_v1
```

seed7/17/27 ×3가족× recurrent/매행동hidden0 =18회, 각40updates.
PPO720updates, learner행동138,240개. 평가48trials/가족, E2×H12, root918273.
가족마다 보상 척도가 다르므로 행 간 절대 순위를 만들지 않는다.

| 가족 | 학습 recurrent 평균 | 학습 매행동hidden0 평균 | family-informed belief |
|---|---:|---:|---:|
| hidden_action | 0.329572 | 0.332176 | 0.966146 |
| probe_memory | -0.440828 | -0.376447 | 0.942708 |
| opponent_response | 0.444734 | 0.481481 | 0.752604 |

일부 학습은 관찰됐다: C recurrent 초기평균0.34491→0.44473, 매행동hidden0 0.34462→0.48148.
B 매행동hidden0도 -0.44437→-0.37645로 상승했다. 다만 유용한 기억/적응의 일관된 증거는 부족하다.
C의 recurrent episode memory benefit은 seed별 +.04861/−.00347/0으로 한 seed에 집중됐다.
분석 후 추가한 '자기 직전 행동 항상 반복' 기준도 C에서 정확히0.5(copy1/avoid0)를 얻었다.
따라서 Random 대비 상승으로 상대 식별 성공을 주장하지 않는다. 추가 기준은 재학습 없이 따로 기록했다.

family-informed belief의 기억 유지−초기화 차이는 A+.04167/B+.05729/C+.01302였다.
이는 과제의 정보 활용 가능성 진단이며 neural policy 학습 성공이나 모든 reactive 정책 대비 최적 차이의 증명은 아니다.
소프트웨어의 label 대칭성 교정과 정보가 유용한 과제 성립을 확인했지만,
보상 경험을 조건부 행동·기억 활용으로 연결하는 학습 검증은 남았다. H1/H2는 아직 검증하지 않았다.

검증: 독립 validator와 메인이 shape/순열/gradient/checkpoint/정보경계/보상표/
memory witness/상대 상한/Q reset-bootstrap/동일 trajectory/CLI를 검사했다.
최종 메인 전체 테스트는 **79 tests, 22.250s, OK**였다.
Torch 없는 시스템 Python에서도 79개 중 선택 의존성 관련 43개를 건너뛰고 나머지는 통과했다
(0.452s, OK). `git diff --check`도 통과했다.

```bash
PYTHONPATH=05_toy_models .venv/bin/python -B -m unittest discover -s tests -q
```

18개 initial/trained/episode-reset 평가를 저장 checkpoint로 정확히 재현했고,
매행동hidden0의 episode-reset 차이는 전부0이었다. 실제 학습 weights의 n5·12턴 실행도
`runs/nominal_n5_demo_v1/`에 저장했다. n 간 학습 일반화 검증은 아니다.
`benchmark.json`, 실행별 weights/raw JSON, source snapshot 및 report.html을 저장했다.
학습 이후 추가한 C 진단 기준과 최종 renderer source는 별도 provenance로 보존했다.
브라우저 시각 검수는 미실행이며 HTML/로컬 링크/원자료는 확인했다.

## 이전 분석: 행동 번호 인코딩과 학습 경사

`Action_Label_Audit_Contract.md`에 따라 보상·경사 감사와 물리→내부 번호의 cyclic 교환을 실행했다.
기존 모델 구조/PPO 설정은 유지하고 선택적 model_factory 및 분석용 wrapper만 추가했다.
분석 코드는 `label_audit.py`/`audit_labels.py`다.

```bash
PYTHONPATH=05_toy_models .venv/bin/python -B -m mamrl.audit_labels --output-dir runs/action_label_audit_v1
```

seed7/17/27 × 물리 보상 행동0/1/2 × cyclic mapping3종 = **27개 새40-update 학습**.
과거 identity run을 재사용하지 않고 같은 현재 실행 경로로 다시 학습했다.
초기 내부 weights는 seed별 동일하며 환경/상대 외생 seed를 공유했다.
정책의 물리 행동 확률과 선택 경로는 mapping에 따라 달라질 수 있다.
총1080updates, learner 행동207,360개. 평가48trials, 2episodes×12steps, seed918273.

| 보상받는 물리 행동 | 내부0으로 표현 | 내부1로 표현 | 내부2로 표현 |
|---|---:|---:|---:|
| 0 | 0.817998 | 0.331887 | 0.839410 |
| 1 | 0.851852 | 0.330440 | 0.894676 |
| 2 | 0.844039 | 0.342303 | 0.877894 |

각 칸은 해당 조건의 학습 seed3개 평균이다. 내부 번호별 전체9조건 평균은
0:0.837963, 1:0.334877, 2:0.870660. 반복된 seed/mapping을 독립 seed9개로 세지 않는다.
정체는 특정 물리 보상 행동보다 **내부 번호1**을 따라갔다. 임의 행동 ID를
0/.5/1로 표현하는 현재 좌표 선택에 학습 난이도가 민감하다는 직접 대조 근거다.
GRU 관측과 actor 후보 인코딩을 함께 바꿨으므로 actor 단독 원인으로 확정하지 않는다.

초기 actor의 exact mean p(target) gradient norm은 내부1에서0.001619–0.005977,
끝 행동에서0.117219–0.137849였다. 가운데/끝 평균 비율은 seed별0.012–0.044.
개별 관측의 가운데 gradient norm 평균은0.162979–0.170646으로 존재하지만
관측 평균에서 상당히 상쇄된다. 분리한 float64 actor의 finite difference 오차 최대는
1.0842e-10이었다. 단순 선형 fixture의 slope 미분(-1/6,0,+1/6)은 설명용이며 전체 신경망이 아니다.
첫 PPO batch의 GRU gradient는 value 항이 policy 항보다 컸으나 이 크기만으로
최적화 원인을 단정하지 않는다. 수치·각 항 가중치·GRU cosine·clip 배율은 JSON에 보존했다.

보상 텐서486개 항목 및 실제 환경243step은 모두 독립 predicate와 일치했다.
저장된27개 initial/trained 물리 정책을 mapping과 함께 복원해 평가를 정확히 재현했다.
새 identity9개는 과거 분리 진단의 전체 history 및 최종 평가와 정확히 같았다.
실행 source snapshot/hash도 보존·대조했다.

검증: `PYTHONPATH=05_toy_models .venv/bin/python -B -m unittest discover -s tests -q`
→ **59 tests, 14.339s, OK**. torch 없는 시스템 Python은 core27개 통과/선택32개 skip.
worker의 분석 helper와 validator가 남긴 독립 테스트를 통합했다.
Subagent 사용 한도로 후반 HTML·추가 loss/CLI 테스트·전체 실행과 결과 검증은 메인이 수행했다.
브라우저 시각 검수는 미실행, HTML/로컬 링크/원자료 일치는 검사했다.

결론: 고정 과제의 행동 번호 비대칭은 현재 행동 인코딩 선택의 영향을 강하게 받는다.
보상 오류나 PPO 수치 오류는 이번 검사에서 발견하지 못했다. 이는 모든 구현 오류의 부재 증명은 아니다.
기억/미지 규칙 적응 정체의 전체 원인은 아직 확정하지 않았으며, 새 표현 구조는 아직 채택하지 않았다.
다음은 비서열적 행동 표현의 통제 비교다. internal_weights.pt는 반드시 기록된 permutation과
`load_physical_model`로 복원해야 하며 일반 물리 좌표 checkpoint로 취급하지 않는다.

## 이전 마일스톤: 학습 정체의 분리 진단

기존 모델/PPO/GPU 설정을 유지하고 `Learning_Diagnostics_Contract.md`의 사전 행렬을 실행했다.
기본 target sampling을 유지하는 선택적 fixed_action과 passive checkpoint callback,
`diagnostics.py`의 행동/층화 paired 평가·지도학습 probe,
`diagnose.py`의 재현 가능한 행렬 실행기 및 한국어 비교 리포트를 추가했다.

```bash
PYTHONPATH=05_toy_models .venv/bin/python -B -m mamrl.diagnose --output-dir runs/learning_diagnostics_v1
```

학습 seeds 7/17/27 × 고정 target 0/1/2 각각40updates(9회), 가변 target120updates(3회)의
40/120 snapshot(6개), 별도 지도학습400steps×3개를 빠짐없이 실행했다.
PPO 총720updates, learner 행동138,240개. 고정 조건과 가변 조건의 raw reward를 한 평균에 섞지 않았다.
평가는 조건마다48trials, 가변은 target별16개 균형 구성. 평가 seed918273과 외생draws를 공유한다.
과거 16-trial 평가와는 평가 구성이 다르므로 그 수치를 동일 표본의 변화로 비교하지 않는다.

| 조건 | 행동당 보상, 학습 seed 3개 평균 | 학습 seed 간 SD |
|---|---:|---:|
| 고정 target0, 40updates | 0.817998 | 0.072551 |
| 고정 target1, 40updates | 0.330440 | 0.001326 |
| 고정 target2, 40updates | 0.877894 | 0.048892 |
| 가변 target, 40updates | 0.330150 | 0.001326 |
| 가변 target, 120updates | 0.334491 | 0.004282 |

가변 조건의 공통 Random 평균은0.329861. 최종 학습−Random의 평균 paired 차이는+0.004630,
hidden 유지−초기화 차이는−0.001447이다. 작은 수치 상승은 있으나 뚜렷한 적응/기억 이득은 확인하지 못했다.
고정 target의 학습 비대칭은 세 seed 모두에서 관찰됐다. 상수 행동 학습 가능성과 조건부 행동 학습을 구별한다.
지도학습 18개 zero-hidden 관측 fit의 성공 후 반복 확률은 세 seed에서0.9987–0.9991,
초과 교차 엔트로피는0.00258–0.00469. 이는 즉시 관측에 대한 제한된 표현 가능성이며,
장기 기억이나 중앙 고정 행동 선택, PPO 적응을 검증한 결과가 아니다.
행동 표현/초기화/actor-critic 최적화/탐색 중 원인은 아직 분리되지 않았다.

연구 critic은 설계·결과를 독립 검토했다. Worker는 진단 helper·리포트,
validator는 독립 수치/경계/CLI 검증, 메인은 학습 선택 확장·행렬 실행·통합을 맡았다.
검증 중 서로 다른 n/episode 구성의 paired 비교를 수용하던 문제를 보완했다.
최종 paired 비교는 n/E/H/spec SHA256/target순서/trialseed와 T=E×H를 확인한다.
실제 행렬의 모든 조건은 원래부터 같은 n3/E2/H12였다. 저장한15개 snapshot을
재학습 없이 재평가하여 모든 기존 수치가 정확히 재현됨을 확인하고 추가 provenance를 저장했다.

실제 메인 전체 검증:

```bash
PYTHONPATH=05_toy_models .venv/bin/python -B -m unittest discover -s tests -q
```

**52 tests, 8.870s, OK**. 독립 validator 전체 실행도52개 통과했다.
torch 없는 시스템 Python은 코어27개 통과·선택 학습25개 skip(총52개)을 확인했다.
고정 target/default stream, callback의 학습 비변경, 행동별 기회 수,
층화/paired SE 손계산, 잘못된 paired 구성 거부, CE entropy floor,
checkpoint 재평가·CLI 산출물·기존 결과 보존을 확인했다.
`runs/learning_diagnostics_v1/report.html`과 diagnostic.json/protocol.json 및
실행별 raw JSON/checkpoint를 생성했다. 브라우저 시각 검수는 미실행이다.

## 이전 마일스톤: Recurrent PPO 첫 학습 실행

`recurrent.py`의 GRU actor-critic/shared candidate scorer와 checkpoint Policy 어댑터,
`ppo.py`의 유한 trial GAE/clipped PPO, `training.py`의 trial 수집·전체 재생·평가,
`train.py`의 CLI 및 `train_report.py`의 학습 리포트를 연결했다.
선택 의존성은 프로젝트 .venv의 CPU torch 2.15.0+cpu와 numpy 2.5.4이며
`requirements-training.txt`에 고정했다. 코어는 여전히 torch 없이 실행한다.

Worker는 모델/PPO 수치·HTML 렌더러, validator는 독립 테스트,
메인은 trial 학습/평가·CLI·통합을 담당했다. research_critic은 경계 설계와 실제 결과를 독립 검토했다.
현재 실행 조건과 미확정 연구 조건은 `../04_design/Recurrent_PPO_Contract.md`를 따른다.

실제 실행:

```bash
PYTHONPATH=05_toy_models .venv/bin/python -B -m mamrl.train --architecture ordinal_v1 --output-dir runs/recurrent_ppo_seed7
```

seed=7, n=3, 고정 Random 상대, action_bonus task, 2 episodes×12 steps/trial,
batch 8 trials, 40 updates×3 PPO epochs, 16 평가 trials. 각 trial의 보상 target은 다시 추출한다.
학습·평가에서 target 구조는 겹치며 보상은 상대 행동에 의존하지 않는다.

| 평가 조건 | 행동당 평균 P0 보상 | 평가 trial 표준오차 |
|---|---:|---:|
| 초기 정책 | 0.341146 | 0.028007 |
| 균등 Random | 0.343750 | 0.027560 |
| 학습 정책 | 0.338542 | 0.029677 |
| 학습 정책, episode마다 hidden 초기화 | 0.338542 | 0.029677 |

전체 모델 파라미터 변화 L2=8.763219. optimizer가 파라미터를 바꿨지만,
이 단일 실행에서 보상 향상이나 episode 간 기억 이득은 관찰하지 못했다.
성능 악화·알고리즘 실패·기억 불필요·탐색 성공으로 결론 내리지 않는다.
표준오차는 독립 평가 trial 변동이며 학습 seed 변동이 아니다.
update 0/10/20/30/40 평가 평균은 0.341146/0.338542/0.341146/0.343750/0.338542.

`runs/recurrent_ppo_seed7/report.html`, training/config JSON, initial.pt/policy.pt,
전체 첫 평가 trial의 evaluation_trial.json, 별도 sample_episode 리포트를 생성했다.
저장한 초기/학습 checkpoint로 네 조건 평가를 다시 실행해 모든 수치가 정확히 재현됐다.
같은 학습 weights의 n=5·12턴도 `runs/recurrent_checkpoint_n5/`에서 실행 확인했다.
이는 실행 확장성 검증이며 n 간 학습 일반화 결과는 아니다.

메인 전체 테스트:

```bash
PYTHONPATH=05_toy_models .venv/bin/python -B -m unittest discover -s tests -v
```

**43 tests, 5.675s, OK**. 손계산 PPO/GAE, target detach, 경계를 넘는 recurrent gradient,
rollout/replay 일치, observe-only의 행동/discount 제외, exogenous RNG 분리,
frozen 평가, 실제 파라미터 변경·재현, checkpoint 및 학습 CLI/리포트 산출물을 확인했다.
기존 코어 테스트도 포함한다. `pip check`: No broken requirements found.
torch 없는 시스템 Python도 43개를 실행해 코어 27개 통과·선택 학습 16개 skip을 확인했다.
브라우저 시각 검수는 미실행이며 HTML/SVG/수치/로컬 링크는 자동 검증했다.

### 후속 읽기 전용 진단

사용자 질문에 따라 학습 설정/가중치를 바꾸지 않고 checkpoint inference만 점검했다.
결과는 `runs/recurrent_ppo_seed7/inference_diagnostics.json`에 저장했다.
zero hidden에서 모든 n=3 own/opponent 조합 9개의 직전 보상만 0→1로 바꿀 때,
학습 정책의 최대 행동 확률 변화는 0.001174(약 0.12%p)였다.
이는 해당 통제 입력에서의 민감도이며 모든 도달 상태에 대한 결론은 아니다.
16 평가 trial의 동일 관측을 teacher forcing하고 episode 2 시작 hidden 유지/초기화를 비교하면,
192개 action event에서 최대 행동 확률 차이의 평균은 0.003665(약 0.37%p),
최대는 0.057730, hidden L2 차이의 평균은 0.325300이었다.
hidden이 달라져도 행동 선택에 대한 영향은 이 진단에서 평균적으로 작았다.
관측 경로/메모리 전달은 확인했지만, 보상 경험을 유용한 행동 차이로 연결하지 못한
근본 원인(학습량·정책 구조·최적화)은 아직 분리하지 못했다.
GPU 배치화는 모델/실험 확대 시의 후속 아이디어 I-011로만 보존하고 현재 설정은 바꾸지 않았다.

## 완료: 실행 기록·오프라인 리포트

사용자 요청에 따라 학습 정책 연결보다 실행 결과 확인을 앞당겼다(D-010).
`recording.py`는 버전 있는 JSON 저장/로드 및 규칙·텐서·전이·보상 근거·요약의 일치를 검증한다.
`report.py`는 외부 서버·JS·네트워크 없이 열리는 한국어 HTML을 만든다.
요약, 누적 보상 SVG, 턴별 보상 근거, 상태별 보상표 및 행동 빈도를 제공한다.
`--output-dir` 옵션이 없으면 기존 CLI stdout 동작을 유지하며, 기존 경로는 덮어쓰지 않는다.
정확한 스키마/API는 `../04_design/Run_Report_Contract.md`에 있다.

Worker는 HTML 렌더러, validator는 기록·HTML 독립 테스트, 메인은 기록 계층·CLI·통합을 담당했다.
독립 검증에서 발견한 거대 정수 오류 처리와 원본 JSON 텐서 해시 비교 문제를 수정하고 회귀 검증했다.

실제 메인 최종 실행:

```bash
PYTHONPATH=05_toy_models python3 -B -m unittest discover -s tests -v
```

**27 tests, 0.335s, OK**. 저장/로드, 손상 기록 거부, 덮어쓰기 방지,
손계산 항별 기여, 표의 글로벌 행/열과 값, 음수·0 보상의 SVG 좌표,
HTML escape, CLI 저장·재생성을 검증했다. 독립 validator도 27개 전체 통과를 확인했다.
추가로 n=3/4/5 × 4개 family의 CLI 12개 조합에서 기록 옵션 유무에 따른
원래 결과·transition의 동일성, 저장/로드/렌더 결과 일치와 n²개 보상표 생성을 확인했다.
브라우저 실행 도구가 없어 실제 시각적 표시와 반응형 레이아웃은 검증하지 않았다.

열어볼 수 있는 실제 산출물(ignored runs 디렉터리):

```bash
PYTHONPATH=05_toy_models python3 -B -m mamrl --n 3 --steps 32 --seed 7 --family mixed --output-dir runs/inspection_n3_seed7
```

`runs/inspection_n3_seed7/report.html` 및 `run.json`을 생성·재로드 확인했다.
32턴, 총 보상 (-80, -96). 이는 RandomPolicy 실행 확인이며 학습/적응 결과가 아니다.
이미 생성된 디렉터리에 같은 명령을 다시 실행하면 거부되므로 새 디렉터리를 지정한다.
여러 seed 집계, 학습/적응 곡선과 게임 간 정규화 비교는 아직 구현하지 않았다.

## 최초 Toy 마일스톤

에이전트 인터페이스·환경·실행기를 우선 구현하고, 작은 논리 규칙 생성/컴파일을 연결했다.
정확한 계약은 `../04_design/Interface_Contract.md`, 결정은 `Decisions.md`를 따른다.

- `05_toy_models/mamrl/environment.py`: Memory-One 동시 행동, 랜덤 초기 state,
  개별 관측·보상, 비공개 fixed horizon에서 truncation.
- `policy.py`, `rollout.py`: 명시적 플레이어별 정책 상태, RandomPolicy, episode 실행,
  frozen transition·총 보상 기록. 정책 상태는 episode마다 초기화한다.
- `tensor.py`, `rules.py`: 불변 글로벌 축 텐서, 불변 논리 명세, seed 기반 작은 합성 생성기,
  결정적 컴파일 및 정적 matrix 변환. 기본 n=3, n=4/5도 실행 검증했다.
- Python 표준 라이브러리만 사용. 패키지는 `PYTHONPATH=05_toy_models`로 실행한다.

Worker는 tensor/rules 생산 코드, validator는 tests만 담당했다.
메인은 계약·환경·정책·실행기·CLI·관리 문서를 담당하고 결과를 통합·직접 재실행했다.

## 실제 검증 증거

환경: WSL2 Ubuntu, Python 3.14.4.

```bash
PYTHONPATH=05_toy_models python3 -B -m unittest discover -s tests -v
```

메인 최종 실행: **18 tests, 0.137s, OK**. 독립 validator도 동일 18개를 실행해 통과했다.
손계산 fixture로 n=3/4/5의 전 보상 조합, 글로벌/로컬 순서, 정적 게임과 이력 의존성,
입력 불변성, 잘못된 입력, 초기화·전이·정확한 절단 시점, 정보 비노출,
전역 RNG 비변경과 프로세스 간 seed 재현성, 공유 정책 객체의 상태 분리,
실제 Generator→Compiler→Environment→RandomPolicy 경로를 확인했다.

추가 CLI 검증은 n=3/4/5 × zero_sum/coordination/memory_one/mixed 12개 조합에서
다음 명령을 각각 두 번 실행했다(n/family만 변경).

```bash
PYTHONPATH=05_toy_models python3 -B -m mamrl --n 3 --steps 32 --seed 7 --family mixed
```

모든 조합에서 32 step, 올바른 shape, terminated=False/truncated=True 및 동일 JSON 재현을 확인했다.
예시 조합 n=3/mixed의 total_rewards는 (-80, -96). 이는 실행 검증이며 학습 성능 결과가 아니다.
`--n 1`, `--steps 0`, `--seed -1`은 각각 종료 코드 2로 거부됐다.
`git diff --check`도 통과했다.

## 남은 작업과 결정

1. 통제된 논리 게임 분포·MA 상대군·거리 후보와 held-out 평가를 설계한 뒤 공통 정책의 전이 비교로 이동한다.
   조건부 보상 활용·policy/value 공유·학습 예산의 추가 진단은 사전 한도를 둔 보조 작업으로 남긴다.
   작은 benchmark 학습 성공을 Generator 연구의 필수 관문으로 두지 않는다. RandomPolicy는 학습하지 않는다.
2. 연구용 task/trial 길이·상대 구성·Self-Play·평가 조건은 아직 최종 결정이 아니다.
   현재는 trial 내 hidden 유지, 유한 trial bootstrap 0, 평가 중 가중치 갱신 없음으로 실행한다.
3. 환경/RandomPolicy 및 동일 recurrent weights의 n=3/4/5 실행은 지원한다. n 간 학습 일반화는 미검증이다.
4. 생성기는 가산 primitive 조합만 지원한다. 0 계수/상쇄/중복으로 퇴화 게임이 나올 수 있다.
   배율/조건부 대체, 규칙 난이도·식별 가능성·전략적 다양성 및 train/test 분리는 후속이다.
5. 최소 체류 길이 이후 확률적 규칙 변경은 보류했으며 episode 중 scheduler는 구현하지 않았다.

원본·외부 생각 노트·Codex 설정은 수정하지 않았다. Git commit은 만들지 않았다.
