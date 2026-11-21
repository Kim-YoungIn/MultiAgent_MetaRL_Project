# Multi-Agent Meta-RL — MAMRL

두 에이전트가 동시에 행동하고 각자 자기 보상으로 학습하는 반복 게임에서, 메타학습한 탐색 정책이 새 게임에 대한 적응을 빠르게 하는지 평가한다.

## D-031 실험 모델

탐색자가 먼저 경험을 수집하고 동일한 온라인 Q 학생에게 제어권을 넘긴다. 학생의 후속 성능을 학습 목표로 삼은 탐색자(M), 탐색 중 자기 보상을 최적화한 탐색자(P), 첫 관측 이후 상대 행동과 보상을 가린 탐색자(O)를 비교한다. 상대는 매 시행 Q-learning 또는 Expected SARSA로 학습한다. D-031 v2 생성기는 숨은 행동 보너스의 상대적 크기를 게임마다 표집해 학습 규칙과 미학습 규칙을 분리한다. 행동 이름을 바꾸거나 행동 수가 달라져도 같은 탐색자 가중치를 사용할 수 있다. [실험 설계와 판정 기준](04_design/Adaptation_Experiment.md)

WSL 또는 Linux의 저장소 루트에서 PyTorch와 NumPy가 설치된 Python을 사용한다. 아래 첫 줄은 훈련 수와 예산을 출력한다. 두 번째 줄은 별도 seed로 모든 블록의 훈련·평가·보고 경로를 짧게 실행한다.

```bash
.venv/bin/python -m experiment plan
.venv/bin/python -m experiment run --dev --block all --output runs/my_d031_dev
```

본실험 주 조건은 5개 훈련 규칙 집합 × 각 5개 초기화 × 세 방법으로 75회 학습한다. 한 모델당 800 updates × 32 trials, 탐색 24턴, 학생 평가 24턴이다. 제거 대조를 포함한 총 훈련 예산은 주 조건 138,240,000턴, 확장 전체 691,200,000턴이며 최종 평가 비용은 별도다.

```bash
.venv/bin/python -m experiment prepare --output runs/my_d031
.venv/bin/python -m experiment run --block primary --output runs/my_d031
```

- 확장은 같은 출력 경로에 `--block rules`, `opponents`, `families`, `actions` 순서로 실행한다. `--block all`로 전체를 실행할 수도 있다. `actions`는 n=3과 n=5 모델을 n=3/4/5 패널에서 평가한다.
- 규모를 나눠 실행할 때는 `--catalog 1 --replica 1`처럼 범위를 지정한다. 같은 출력 경로에서 서로 다른 조합을 병렬 실행할 수 있다. 각 작업에 `--defer-report`를 주고 끝에서 `python -m experiment report --output runs/my_d031`로 모으면 전체 자료를 한 번 읽는다.
- 중단 후 같은 명령을 실행하면 완료한 PPO 업데이트에서 재개한다. 평가 도중 중단된 방법은 해당 방법의 평가를 다시 실행한다. 코드·실험 명세·PyTorch 버전·장치가 바뀌면 새 출력 경로를 사용한다. 생성한 패널의 과거 중복 검사 범위는 `panels/panel.json`에 기록된다. 동료가 같은 평가 규칙으로 독립 실행하려면 준비 직후의 `study.json`, `panel_sha256.json`, `panels/`를 함께 복사한다. 새 체크아웃에서 과거 runs 없이 생성하면 제외 목록이 달라질 수 있다.
- 기본은 CPU다. CUDA가 보이는 WSL 또는 Colab에서 같은 명령에 `--device cuda`를 추가한다. 준비와 보고서 재생성에도 같은 장치를 지정한다. GPU는 신경망 연산을 수행하고 게임과 표 형식 학습자는 CPU에서 실행한다.
- 결과를 읽을 때는 출력 폴더의 **`summary.md`**에서 M−P 적응 곡선 면적, 탐색 길이별 절대 성능과 경험·상대 교차표를 먼저 본다. `report.json`에는 가족·상대별 결과, 계층 bootstrap 구간과 규칙 재표집 민감도가 있다. 개발 실행은 기능 검증용이며 주가설 판정에는 봉인된 본실험 패널과 25개 모델 쌍을 사용한다.
- 모델은 `cells/<조건>/c<catalog>/j<replica>/training/<방법>/`에 저장된다. Python에서 `from experiment.training import load_model`로 불러오고 초기 가중치는 `load_model(path, initial=True)`로 읽는다. 시행별 seed와 곡선은 `*.trials.jsonl.gz`, 훈련 노출·시간·메모리 기록은 `ledger.json`에 있다.

## 선행 모델과 공통 시뮬레이터

- 2인 동시 행동, 기본 행동 수 n=3. 환경과 규칙은 n=4/5도 지원한다.
- 보상은 직전·현재 joint action으로 결정한다. 현재 생성기는 양쪽이 자기 관점에서
  동일 함수를 적용하는 대칭 규칙이다. 공통 표는 n⁴개, 저장 텐서는 `(2,n,n,n,n)`이다.
- 논리 규칙을 생성한 후 텐서로 결정적으로 컴파일한다. 기본 생성기는 가산 조합이며,
  `zero_sum`, `coordination`, `memory_one`, `mixed` 가족을 제공한다.
- 별도 생성 실험은 행동 관계의 일치/불일치와 AND/OR/NOT, 조건부 보상·가산·배율의
  제한된 문법을 사용한다. 기존 가산 생성 CLI와 구분한다.
- 초기 joint action은 독립·균등 무작위다. 정책은 직전 자기/상대 행동, 직전 자기 보상,
  episode 시작 표시를 받고 규칙·상대 ID·종료 길이는 받지 않는다.
- 상대 정책은 Random, 직전 learner 행동을 복사하는 copy, 그 행동을 피하는 avoid,
  양의 자기 보상 후 반복하고 그 외에는 전환하는 win_stay가 있다.
- 메타 탐색 경로는 탐색 정책이 만든 경험으로 온라인 Q 학생을 학습시키고,
  상대도 자기 보상으로 Q 또는 Expected SARSA를 갱신한다. 탐색 정책은 후속 성능 차이로 PPO 학습한다.
- 관계형 생성기는 명시적 상성표, 승패·반복·모방 관계, 공동 조건 보너스, 숨은 행동 보상,
  조건부 예외를 지원한다. 최종 텐서에서 PD 여부와 최선반응을 검사하며 규칙 이름으로 판정하지 않는다.
- 기존 직접 플레이 경로는 RL² 스타일 recurrent PPO다. 두 경로의 목적함수와 상대 학습을 구분한다.
- 실행·학습·다중 seed 진단과 기능 벤치마크 결과를 JSON 및 오프라인 HTML로 저장한다.

## 설치

모든 명령은 프로젝트 루트의 WSL/Linux 터미널에서 실행한다.
환경·규칙·RandomPolicy 실행은 Python 3.10 이상과 표준 라이브러리만 사용한다.
학습·체크포인트 추론에는 선택 의존성이 필요하다.
실제 학습 환경은 WSL2, Python 3.14.4, CPU PyTorch 2.15.0+cpu, NumPy 2.5.4다.
다른 Python 버전에서 고정 패키지의 설치 가능성은 별도로 확인해야 한다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt
```

현재 모델과 설치 파일은 CPU 실행을 사용했다.

## 메타 탐색과 학습하는 상대 — 본 목적 실험

```bash
PYTHONPATH=05_toy_models OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m mamrl.meta_exploration --catalog relational --opponent-suite mixed_learning --output-dir runs/my_meta_exploration
```

| 역할 | 동작 |
|---|---|
| 탐색 정책 | 12턴의 탐색 행동을 선택. action_set_v1 recurrent 신경망 |
| 학생 에이전트 | 탐색 경험으로 온라인 Q 학습 후 12턴 플레이. 후속 구간에서도 계속 학습 |
| 상대 에이전트 | 두 구간 모두 자기 경험으로 Q 또는 Expected SARSA와 행동 전략 갱신 |

학생과 상대는 trial 시작에 새로 만들고 두 episode 사이에는 학습 상태를 유지한다.
양쪽은 같은 이전 관측으로 동시에 행동하고 상대의 이번 행동이나 규칙을 직접 받지 않는다.
후속 구간에서 학생은 greedy Q로 행동하며 상대는 설정된 ε-greedy 학습을 계속한다.

탐색자의 meta reward는 **후속 평균 보상 − 학생 Q만 지운 대조의 평균 보상**이다.
대조는 탐색이 끝난 같은 상대 Q·난수 상태와 같은 환경 reset에서 시작한다.
후속 상호작용 때문에 두 분기에서 상대가 다르게 학습하는 것은 허용한다.
탐색 중 보상을 이 학습 신호에 더하지 않는다. 별도 대조12턴도 비용에 포함하므로
trial당 실제 게임24턴 + 대조12턴 = 총36턴 simulation이다.

기본은 n=3,3seeds,40updates,update당8trials,H32다. 탐색자의 최종 가중치를 고정하고,
새 학생/상대의 학습은 허용한 상태로 초기·학습 탐색과 Random·ε-greedy·반복·balanced·변경을 비교한다.
위 명령은12개 학습 규칙,8개 새 규칙,8개 예외 구조 규칙을 사용한다.
학습 가족은 social/contest/reciprocity/hidden이며 exception은 구조 평가에만 사용한다.
사회적 유인층에는 PD·조정·반조정·협력 우세를 포함한다. 새 규칙은 계수만 바뀐 평가가 아니다.
Q(alpha=.3,epsilon=.15)와 Expected SARSA(.5,.2)로 훈련하고 Expected SARSA(.7,.1)를 별도 평가한다.
이는 학습에 포함한 알고리즘의 새 설정이며 미사용 알고리즘 전이가 아니다.

반복=C,변경=D로 두면 CC=R,CD=S,DC=T,DD=P를 표현할 수 있다. PD 태그는 최종 보상이
T>R>P>S이고 모든 이전 상태·같은 C/D 행동군에서 동일한 보상인지 검사한 뒤 붙인다.
보너스나 예외로 성질이 깨지면 태그가 사라진다. n3의 반복은1개/변경은2개 행동이므로
Random은 C:D=1:2다. balanced는 반복/변경에 각각1/2을 배정하고 변경 행동 중 균등 선택한다.
변경 대조군은 항상 이전 자기 행동과 다르게 고른다. 둘 다 숨은 규칙을 보지 않는다.

관계형 원식·상성표·최종 보상 명세·정규화·compiled audit가 manifest에 저장된다.
리포트의 규칙 설명은 원식이며 실제 플레이는 전체 보상을 양의 affine 변환으로[-1,1]에 맞춘다.
최선반응·보상/구조 거리는 기술적 지표이며 전략적 다양성이나 탐색 필요성의 보장이 아니다.

두 선택 인자를 생략하면 과거 logical 생성기(12/6/6규칙)와 Q-only 상대 설정을 사용한다.
과거 실행을 재현하기 위한 기본값이며 관계형 실험은 위 명령처럼 인자를 명시한다.

출력은 `protocol.json`, `manifest.json`, 관계형 `distances.json`, `baselines.json`, `experiment.json`,
seed별 초기/최종 체크포인트와 학습/평가 기록,source snapshot,`report.html`이다.
새 실행의 `experiment.json`(저장 형식v3)은 요약과 상세 파일의 경로/hash를 담는다.
상세 trial은 `baselines.json`과 seed별 `evaluation.json`에 한 번 저장한다.
분석할 때는 아래 loader로 읽는다. 기존 상세 내장 형식도 같은 함수로 읽을 수 있다.
실행 폴더를 옮길 때는 참조된 파일을 함께 유지해야 한다.

```python
from mamrl.experiment_artifacts import load_meta_experiment
from mamrl.meta_exploration_report import render_meta_report

record = load_meta_experiment("runs/my_meta_exploration/experiment.json")
# record에는 상세 trials까지 복원된다.
html = render_meta_report(record)
```

HTML에서 유지 경험 이득과 절대 후속 성능,탐색 비용,상대의 실제 학습된 행동 사용 비율을 함께 본다.
G가 커지는 것만으로 절대 성능이 좋아졌다고 해석하지 않는다. 후속1/3/6/전체턴 prefix 평균(AUC/길이),
턴별 곡선·기술적 기울기·가족별 성능도 제공한다. 기울기에는 방문 상태와 상대 학습 변화가 섞인다.

빠른 연결 확인:

```bash
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.meta_exploration --catalog relational --opponent-suite mixed_learning --output-dir runs/my_meta_smoke --seeds 7 --updates 1 --trials-per-update 1 --steps 2 --hidden-size 8 --reps 1
```

## 공통 Template와 탐색 경험 비교

```bash
PYTHONPATH=05_toy_models OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m mamrl.meta_exploration --catalog probe_template --catalog-seed 20261116 --train-rules 32 --fresh-rules 16 --structural-rules 16 --opponent-suite mixed_learning --seeds 7 17 27 --updates 200 --trials-per-update 32 --steps 12 --assessment-steps 48 --batch-size 32 --device cpu --reps 1 --output-dir runs/my_probe_template
```

safe/risk/상성/비용의 공통 구조와 숨은 방향θ를 사용한다. 학습·새 파라미터에서는 반복/모방 보너스를
각각 따로 쓰고,구조 평가는 두 요소의 새 조합이다. 역할 재명명은 독립 구조 다양성으로 세지 않는다.
규칙 수는 짝수이며 학습 규칙4개 이상에서 두 요소를 포함한다. source·정규화·seed·canonical 분할을 기록한다.
`information_audit.json`은 고정 상대/알려진 prior의 양성·음성 대조이며 실제 MA 상대의 성공 보장이 아니다.
실제 학습·평가에서는 상대도 Q/Expected SARSA를 계속 학습한다.

규칙 수를 비교할 때는 하나의 manifest를 공유하고 학습 규칙만 제한한다.

```bash
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.meta_exploration --catalog probe_template --manifest runs/my_probe_template/manifest.json --train-rule-limit 8 --opponent-suite mixed_learning --output-dir runs/my_probe_subset8
```

다른 제한값도 같은 manifest·평가 seed·학습 예산에서 실행한다. `seen_rule`과 실제 학습 ID를 기록한다.
`--train-rules`로 매번 catalog를 새로 만들면 평가 규칙도 바뀔 수 있으므로 다양성 효과 비교에 사용하지 않는다.

같은 경험을 다른 학생에 제공하거나,같은 학생에서 다른 탐색 경험을 비교한다.

```bash
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.history_study --manifest runs/my_probe_template/manifest.json --checkpoint runs/my_probe_template/seed7/policy.pt --initial-checkpoint runs/my_probe_template/seed7/initial.pt --steps 12 --assessment-steps 48 --reps 2 --output-dir runs/my_history_study
```

원래 full Q 학생의 자연 MA 성능과,같은 참조 상대 상태에서 재생한 조건부 성능을 구분한다.
세 학생은 정확한 reward/보상 제외/sign 보상 key의 Online Q이며 서로 다른 학습 알고리즘 비교는 아니다.
History 파일은한벌이며 학생별row는hash/id로 참조한다. full replay는 원학생 snapshot과 일치해야 한다.
짝 규칙 식별성은 알려진 prior의 연구자 진단이며 신경망 내부의 규칙 분류를 직접 측정하지 않는다.
`--threshold` 또는 `--reference-low/--reference-high/--threshold-fraction`으로 사전 기준을 지정한다.
reference high를 최대 달성 가능값으로 자동 가정하지 않는다. 미도달 시간은None/우측검열로 기록한다.
개별 연구결과의 독립 실행용 도구이며,이 명령 자체가H1의 다양성 효과를 검증하지는 않는다.

같은 조건에서 G 목표와 절대 후속보상 목표를 학습·비교하려면 다음을 실행한다.

```bash
PYTHONPATH=05_toy_models OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m mamrl.objective_study --manifest runs/my_probe_template/manifest.json --output-dir runs/my_objective_study
```

기본은 두 목표×3seed×200updates이며0/50/100/200의 고정 평가와 `report.html`을 남긴다.
같은 인자에 `--resume`을 붙여 update 경계에서 재개한다. 단일 메타 실행에서는
`--objective assessment_return`으로 선택하며 기본값은 `retained_gain`이다.
두 목표 모두 경험 제거 대조를 계산한다. 새 평가 trajectory 대신 task별 지표를 보존한다.

탐색 기록과 상대 상태를3×3으로 교차하고 학생 학습 유무를 비교하려면 다음을 실행한다.

```bash
PYTHONPATH=05_toy_models OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m mamrl.reference_study --manifest runs/my_probe_template/manifest.json --initial-checkpoint runs/my_probe_template/seed7/initial.pt --checkpoint runs/my_probe_template/seed7/policy.pt --output-dir runs/my_reference_study
```

기본 fresh 평가에서 각 조합의 online/frozen 점수와 원자료를 저장한다.
frozen은 학생만 고정하며 상대는 계속 학습한다. 대각선은 원래 자연 MA 조합이고,
교차 조합은 상대 상태 선택에 대한 민감도 분석이다. 순수 경험 품질의 인과 순위로 해석하지 않는다.

## 행동 공간과 규칙 생성 범위 확대

`diverse_relational`은 n=3/4/5와 가변 규칙 수를 지원한다. `focused`는 상성/시간 관계 recipe,
`broad`는 사회적 유인·상성·상호 이력·숨은 보상의 네 recipe를 섞는다.
두 조건은 같은 catalog seed와 평가 개수에서 같은 평가 규칙을 공유한다.
예외 분할은 기존 논리 요소의 조건부 결합이며 새 primitive 전이가 아니다.

```bash
PYTHONPATH=05_toy_models OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m mamrl.meta_exploration --catalog diverse_relational --n 5 --diversity broad --catalog-seed 20261118 --train-rules 128 --fresh-rules 16 --structural-rules 16 --opponent-suite mixed_learning --seeds 7 --updates 200 --trials-per-update 32 --steps 12 --assessment-steps 48 --batch-size 32 --reps 1 --output-dir runs/my_diverse_n5
```

같은 n·평가 개수·catalog seed에서 `--diversity focused` 또는 규칙 수8/32/128로 바꿔 비교한다.
하나의 manifest를 공유할 때는 `--manifest ... --train-rule-limit N`을 사용한다.
행동 수·diversity는 manifest와 일치시켜야 한다. 과거 `probe_template`/`logical` 메타 경로는 n3 전용이다.
가족/계수/보상표 차이가 전략 다양성이나 탐색 유용성을 보장하지 않으므로 manifest의 compiled audit를 함께 본다.

각 행동 수의 manifest를 준비한 뒤 같은 탐색 checkpoint를 재학습 없이 평가할 수 있다.

```bash
PYTHONPATH=05_toy_models OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m mamrl.action_transfer --checkpoint runs/my_diverse_n3/seed7/policy.pt --manifests runs/my_diverse_n3/manifest.json runs/my_diverse_n4/manifest.json runs/my_diverse_n5/manifest.json --batch-size 32 --output-dir runs/my_action_transfer
```

같은 n 안에서 랜덤 탐색과 task/난수를 맞추고 후속 성능·G·가족별 점수·반복률·행동 방문률을 저장한다.
탐색 가중치는 고정하지만 학생/상대는 학습한다. n 사이의 보상표는 달라 생보상 차이를 행동 수만의 효과로 보지 않는다.
훈련 행동 수는 원래 실행의 protocol로 확인한다. 작은 실행 성공은 학습된 행동 공간 일반화의 증거가 아니다.
같은 인자로 `--checkpoint .../initial.pt`와 별도 출력 경로를 지정해 초기 탐색도 평가한다.
두 실행의 task/seed를 맞춰 학습 전후를 비교한다. 이 평가는 메타 가중치를 고정한 학생 적응이며 학생 zero-shot이 아니다.

## GPU와 Colab 실행

[Colab 노트북](notebooks/MAMRL_GPU.ipynb)을 Colab에서 열고 GPU 런타임을 선택한다.
노트북은 저장소의 코드를 호출하며 별도 학습 구현을 포함하지 않는다.
아직 Git에 게시하지 않은 현재 작업본은 루트에서 아래 ZIP을 만들어 노트북에 업로드한다.

```bash
zip -r /tmp/mamrl-colab.zip 05_toy_models tests requirements-colab.txt -x '*__pycache__*'
```

GitHub로 가져올 때는 노트북의 `SOURCE_MODE="github"`와 GPU 구현이 포함된 `REVISION`을 명시한다.
Colab에 기존 CPU 전용 requirements를 설치하면 CUDA PyTorch가 교체될 수 있으므로 사용하지 않는다.
노트북의 초기 설정은 연결 검사용이며 확대 H1 실험 설정이 아니다.

먼저 **실제로 학습할 WSL 터미널/Colab 런타임**에서 진단한다.

```bash
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.diagnose_device --smoke --require-cuda
```

진단은 PyTorch 빌드/runtime,드라이버 조회,GPU 이름과 실제 CUDA 행렬 연산을 구분해 출력한다.
GPU가 가능한 기존 환경이면 PyTorch나 CUDA Toolkit을 다시 설치할 필요가 없다. CUDA 지원 PyTorch binary의
runtime을 사용하되 NVIDIA 드라이버/GPU 접근은 별도로 필요하다. 새 환경 설치는
[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)의 CUDA 명령을 따른다.
기존 `requirements-training.txt`는 CPU 전용 재현용이므로 정상 동작하는 CUDA 환경에 설치하지 않는다.
에이전트 샌드박스에서는 같은 경로의 Python도 GPU 장치가 차단되거나 다른 빌드로 보일 수 있다.
이 경우 진단의 `cuda_inaccessible`/`cpu_build`를 구분하고 일반 WSL 터미널에서 다시 확인한다.
`auto`는 현재 프로세스에서 GPU를 사용할 수 없으면 CPU로 실행하므로 GPU 검증 시 `cuda`를 명시한다.

CUDA PyTorch가 설치된 환경에서는 다음처럼 실행한다.

```bash
PYTHONPATH=05_toy_models OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m mamrl.meta_exploration --catalog relational --opponent-suite mixed_learning --device cuda --batch-size 32 --updates 40 --trials-per-update 32 --steps 24 --assessment-steps 96 --output-dir runs/my_gpu_run --checkpoint-every 10
```

`--device cpu`가 기본이며 `auto`는 CUDA 가능 시 GPU를 선택한다. `cuda`를 요청했는데 사용할 수 없으면
오류를 낸다. `--batch-size`는 병렬 신경망 추론·재생의 trial 수다. 학생/상대 Q-table과 환경은 CPU에서
독립 실행한다. 배치 크기1은 기존 순차 경로를 유지한다. 환경까지 모두 GPU로 옮긴 구현은 아니다.
학습·신경망 평가가 배치화되며 GPU 속도 향상은 장치와 batch 크기/환경 비용에 따라 달라진다.

중단 복구는 같은 명령에 `--resume`을 추가한다. `seed*/training_state.pt`는 모델·초기 모델·optimizer·
sampler/torch 난수·완료 update·이력을 원자적으로 저장한다. `--updates`는 최초부터 최종 목표로 고정한다.
`--stop-after-update 10`으로 의도적으로 멈췄다면 재개할 때 이 옵션을 제거한다.
마지막 저장 update 이후의 미저장 작업은 다시 수행한다. trial은 update마다 새로 시작하므로 내부 Q 상태를
중간에 복구하지 않는다. 설정·원본 source·Python/PyTorch·torch thread·device가 맞지 않으면 재개를 거부한다.
CPU/GPU 간 weight 추론 파일 이동은 지원하지만 서로 다른 backend의 학습 재개/비트 일치는 보장하지 않는다.

`--backup-dir`에 별도 영구 run 폴더를 주면 checkpoint 저장 경계와 주요 완료 시점에 파일을 복사한다.
Colab은 VM 로컬 디스크에서 실행하고 Drive에는 이 옵션으로 백업한다. 기존 run이 들어 있는 경로는 새 실행에
재사용하지 않는다. VM을 잃으면 백업 폴더 전체를 로컬로 복사하고 같은 source/설정으로 `--resume`한다.
노트북이 이 복원 경로를 제공한다. 학습 전 baseline 평가 중 중단되면 미완료 baseline은 다시 계산한다.

`seed*/timing-*.json`은 호출별 수집/최적화/전체 시간,실제 gameplay step/s와 제어 step 수를 기록한다.
같은 조건·장치에서 warm-up 후 여러 번 측정해 비교한다. 평가·catalog 생성 시간은 이 학습 처리량에 포함하지 않는다.
`--steps K --assessment-steps H`로 탐색/후속 길이를 분리한다. H를 생략하면 기존처럼 K와 같다.
실제 gameplay 비용은K+H,경험 제거 제어까지 K+2H다. 전체 실제 평균 보상은 길이로 가중한다.
기본 catalog/n3는 유지하며45회 다양성 matrix와 확대 catalog 구성은 후속이다.

## 기존 생성 게임의 전이 비교

```bash
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.generated_experiment --output-dir runs/my_generated_transfer
```

논리 규칙 생성→게임/상대 분할→공통 정책 학습→미학습 평가→HTML 분석을 한 명령으로 실행한다.
기본은12개 학습 규칙,6개 새 규칙,6개 중첩 구조 규칙과4종 상대다.
중복·상수·자기/상대 현재 행동에 의존하지 않는 후보는 이 MA 실험군에서 사전 제외하고 수를 기록한다.

규칙 수4/12 × 학습 상대 종류1/3의 네 조건을 각각3seeds,40updates,같은 행동 예산으로 학습한다.
각 모델은 자신의 조건에 속한 여러 게임/상대를 경험한다. 모든 조건을 같은56개 평가 과제에 비교한다.
평가는 제외한 조합·새 규칙·중첩 구조·새 상대의 네 집합이며, 조건마다 본 규칙/상대 여부도 기록한다.
`combination` 집합이 순수한 새 조합인 것은 high/high 조건이고 다른 조건에는 새 요소도 포함된다.

출력은 `manifest.json`, `protocol.json`, `distances.json`, `references.json`, `oracle.json`,
`information_witness.json`, `experiment.json`, 실행별 체크포인트/평가 기록, source snapshot, `report.html`이다.
HTML에서 조건별 점수·턴 곡선·기준 정책·생성 규칙을 확인하고 JSON에서 과제별 원자료를 볼 수 있다.

규모를 줄여 연결을 확인할 수도 있다.

```bash
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.generated_experiment --output-dir runs/my_generated_smoke --seeds 7 --updates 1 --trials-per-update 1 --max-steps 2 --hidden-size 8 --reps 1
```

실제 플레이 보상은 규칙 전체의 양의 affine 변환으로[-1,1]에 맞춘다.
이는 task 정의의 일부이며, 보상에 반응하는 win_stay의 원규칙 행동을 보존한다는 뜻은 아니다.
원시·정규화 텐서 차이, 공통 행동 재명명 후 차이, 구조 토큰 차이를 기록한다.
이 지표들이 전이 난이도나 전략적 거리를 보장하지는 않는다.

## 게임 실행과 리포트

```bash
PYTHONPATH=05_toy_models python3 -m mamrl --n 3 --steps 32 --seed 7 --output-dir runs/my_run
PYTHONPATH=05_toy_models python3 -m mamrl --n 5 --family memory_one --trace
```

첫 명령은 RandomPolicy 두 개로 실행하고 `run.json`과 `report.html`을 만든다.
HTML을 로컬 브라우저에서 열면 누적 보상, 턴별 보상 근거, 상태별 보상표와 행동 빈도를 볼 수 있다.
서버나 네트워크 연결은 필요 없다. `--output-dir` 없이 실행하면 JSON을 stdout으로 출력한다.

저장된 기록에서 HTML을 다시 만들 수도 있다.

```bash
PYTHONPATH=05_toy_models python3 -m mamrl.report runs/my_run/run.json --output runs/my_run/rebuilt.html
```

## 학습과 체크포인트 실행

```bash
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.train --output-dir runs/my_training
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.train --family probe_memory --output-dir runs/my_probe
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.train --family opponent_response --opponent-mode mixed --output-dir runs/my_opponents
```

기본은 n=3, 숨은 보상 행동을 찾는 `action_bonus`, 고정 Random 상대,
trial당 2 episodes×12 steps, 40 PPO updates다. 이 값은 실행용 기본값이며 연구 최적 조건이 아니다.
`opponent_response`에서 상대를 생략하면 copy/avoid 혼합을 사용한다.

`--architecture`로 정책을 선택한다.

| 값 | 구조 |
|---|---|
| `action_set_v1` | CLI 기본값. 행동별 공유 GRU와 scorer, 행동 수 n일 때 hidden 크기 nH |
| `action_set_ff_v1` | 매 행동 hidden을 0으로 초기화하며 학습하는 비교군 |
| `ordinal_v1` | 기존 숫자형 행동 입력 구조 |

라이브러리에서 `TrainConfig()`를 직접 쓰면 기본은 과거 재현용 `ordinal_v1`이다.
새 구조를 사용할 때는 `architecture`를 명시한다.

출력에는 `config.json`, `training.json`, 초기/학습 체크포인트,
평가 trial 기록, `report.html`, 단일 실행 예제 `sample_episode/`가 포함된다.
학습 리포트는 update별 평가와 초기·Random·학습·episode마다 기억 초기화 조건을 비교한다.

```bash
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl --checkpoint runs/my_training/policy.pt --n 5 --output-dir runs/checkpoint_n5
```

기존 v1 및 새 v2 체크포인트를 로드한다. optimizer 상태를 복원하는 학습 재개는 지원하지 않는다.

## 기능 벤치마크와 진단

```bash
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.benchmark_run --output-dir runs/my_functional_pilot
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.diagnose --output-dir runs/my_diagnostics
PYTHONPATH=05_toy_models .venv/bin/python -m mamrl.audit_labels --output-dir runs/my_label_audit
```

| 명령 | 실행 내용 |
|---|---|
| `benchmark_run` | 숨은 보상 행동·규칙 기억·상대 반응의 3가족. 기본 3 seeds×2구조×3가족=18회 학습, 각40updates |
| `diagnose` | 고정/가변 보상 행동과 다중 seed 학습, 별도 지도학습 표현력 검사 |
| `audit_labels` | 물리 게임을 유지한 행동 라벨 교환, 보상·gradient 감사. 기본27회 학습 |

각 명령은 조건·평가 원자료·HTML을 새 출력 디렉터리에 저장한다.
`benchmark_run`은 Random·수작업 반응형·온라인 Q·가족 지식 belief 기준을 함께 평가한다.
각 명령의 `--help`에서 seed·update·평가 규모 등 조절 옵션을 확인할 수 있다.

## 테스트

```bash
PYTHONPATH=05_toy_models python3 -m unittest discover -s tests -v
PYTHONPATH=05_toy_models .venv/bin/python -m unittest discover -s tests -v
```

첫 명령은 학습 의존성이 없으면 관련 테스트를 skip한다. 전체 검증에는 두 번째 환경을 사용한다.

## 사용 및 결과 해석 시 주의사항

- 출력 디렉터리와 리포트 파일은 기존 경로를 덮어쓰지 않는다. 재실행할 때 새 이름을 사용한다.
- `.venv/`, `runs/`, `checkpoints/` 등은 Git 제외 대상이다. 저장소를 받은 사람은 각자 설치하고 실행한다.
- 기존 실행 CLI의 생성기는 작은 가산 문법이고, 새 전이 실험은 별도의 제한된 논리 문법을 사용한다.
  중첩 평가는 조건과 보상 합성 방식이 함께 달라지므로 깊이만의 효과나 새 primitive 발견을 뜻하지 않는다.
- 기존 직접 플레이/생성 전이 경로의 Copy/Avoid/Win-stay는 정해진 반응 규칙이다.
  본 목적 메타 탐색 경로는 별도의 온라인 Q 학습 상대를 사용한다. 두 메타 정책의 self-play는 구현하지 않았다.
- `action_bonus`는 상대 행동과 무관한 진단이다. 기능 벤치마크의 평가 seed 분리도 미학습 규칙 구조 평가를 뜻하지 않는다.
- 기존 파일럿에서 일부 보상 상승은 있었지만 일관된 신경망 기억·적응 이점은 확인하지 못했다.
  새 전이 실험도 가설을 지지하는 결과를 보장하지 않는다. 같은 총예산에서는 과제당 노출과 혼합 비율이 다르다.
- 가족별 보상 척도가 다르므로 생보상으로 전체 순위를 만들지 않는다. Q는 평가 중 표를 갱신하고,
  신경망은 weights를 고정한다. 수작업 기준의 가족 사전지식도 별도로 고려해야 한다.
- 상대 반응 과제는 두 종류를 균등 평가할 때 자기 직전 행동만 반복해도 평균0.5를 얻는다.
  Random보다 높다는 이유만으로 상대 식별 성공을 주장할 수 없다.
- n=4/5에서 같은 weights를 실행할 수 있지만, 행동 공간 간 학습 일반화는 검증하지 않았다.
- 전체 trial 표준오차는 서로 다른 고정 과제의 변동도 섞인 기술적 지표다. 학습 seed SD와 구분한다.
  초기/후기 보상 차이는 상태 방문 변화도 포함하므로 그 자체를 적응량으로 해석하지 않는다.
- 생성 실험의 posterior 기준은 평가 과제까지 포함한 유한 catalog prior를 알고, DP 상한은 실제 규칙·상대·남은 길이를 안다.
  이들의 성적은 동등한 사전지식의 학습 비교나 Nash regret가 아니다.
- 메타 탐색은 한 학생 알고리즘과 제한된 생성 분포에서의 실험이다. G는 학생 Q 유지 효과이며
  순수 정보량이 아니다. 상태 방문·후속 학습·상대에 대한 영향도 성능에 섞인다.
- 메타 탐색의 ε-greedy 대조는 학생의 명시적 Q를 활용하고, 신경망 탐색자는 관측 이력을 사용한다.
  같은 경험으로 계산 가능하지만 표현 방식이 다른 대조군이다.
- `audit_labels`의 `internal_weights.pt`는 내부 행동 좌표다. 일반 `--checkpoint`에 바로 쓰지 말고
  `mamrl.audit_labels.load_physical_model(path, permutation)`으로 기록된 매핑과 함께 복원한다.
