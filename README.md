# Multi-Agent Meta-RL — MAMRL

두 에이전트가 논리적 보상 규칙 아래에서 반복 게임을 플레이하는 연구용 토이모델이다.
현재 구현은 Memory-One 환경, 작은 규칙 생성기와 Compiler, recurrent PPO 학습,
기능 벤치마크 및 로컬 HTML 리포트를 제공한다.

## 현재 구현

- 2인 동시 행동, 기본 행동 수 n=3. 환경과 규칙은 n=4/5도 지원한다.
- 보상은 직전·현재 joint action으로 결정한다. 현재 생성기는 양쪽이 자기 관점에서
  동일 함수를 적용하는 대칭 규칙이다. 공통 표는 n⁴개, 저장 텐서는 `(2,n,n,n,n)`이다.
- 논리 규칙을 생성한 후 텐서로 결정적으로 컴파일한다. 기본 생성기는 가산 조합이며,
  `zero_sum`, `coordination`, `memory_one`, `mixed` 가족을 제공한다.
- 초기 joint action은 독립·균등 무작위다. 정책은 직전 자기/상대 행동, 직전 자기 보상,
  episode 시작 표시를 받고 규칙·상대 ID·종료 길이는 받지 않는다.
- 상대 정책은 Random, 직전 learner 행동을 복사하는 copy, 그 행동을 피하는 avoid가 있다.
- 학습은 RL² 스타일 recurrent PPO다. 같은 task의 episode 사이에는 learner hidden을 유지하고
  trial 시작에 초기화한다. 평가는 가중치를 고정하고 hidden state를 갱신한다.
- 실행·학습·다중 seed 진단과 기능 벤치마크 결과를 JSON 및 오프라인 HTML로 저장한다.

## 설치

모든 명령은 프로젝트 루트의 WSL/Linux 터미널에서 실행한다.
환경·규칙·RandomPolicy 실행은 Python 3.10 이상과 표준 라이브러리만 사용한다.
학습·체크포인트 추론에는 선택 의존성이 필요하다.
실제 확인한 학습 환경은 WSL2, Python 3.14.4, CPU PyTorch 2.15.0+cpu, NumPy 2.5.4다.
다른 Python 버전에서 고정 패키지의 설치 가능성은 별도로 확인해야 한다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt
```

현재 모델과 설치 파일은 CPU 실행을 사용한다.

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
- 현재 생성기는 작은 가산 문법이다. 범용 조건부 규칙 생성, 규칙 거리, held-out 구조 전이 실험은 아직 구현하지 않았다.
- 현재 학습 상대는 고정 정책이며 self-play와 공동 학습은 구현하지 않았다.
- `action_bonus`는 상대 행동과 무관한 진단이다. 기능 벤치마크의 평가 seed 분리도 미학습 규칙 구조 평가를 뜻하지 않는다.
- 기존 파일럿에서 일부 보상 상승은 있었지만 일관된 신경망 기억·적응 이점은 확인하지 못했다.
  규칙·상대 다양성 및 재사용 가능한 구조가 전이에 주는 효과는 미검증이다.
- 가족별 보상 척도가 다르므로 생보상으로 전체 순위를 만들지 않는다. Q는 평가 중 표를 갱신하고,
  신경망은 weights를 고정한다. 수작업 기준의 가족 사전지식도 별도로 고려해야 한다.
- 상대 반응 과제는 두 종류를 균등 평가할 때 자기 직전 행동만 반복해도 평균0.5를 얻는다.
  Random보다 높다는 이유만으로 상대 식별 성공을 주장할 수 없다.
- n=4/5에서 같은 weights를 실행할 수 있지만, 행동 공간 간 학습 일반화는 검증하지 않았다.
- `audit_labels`의 `internal_weights.pt`는 내부 행동 좌표다. 일반 `--checkpoint`에 바로 쓰지 말고
  `mamrl.audit_labels.load_physical_model(path, permutation)`으로 기록된 매핑과 함께 복원한다.
