# Run Artifact / Local Report Contract — v1

## 우선순위와 범위

학습 정책 구현 전에 실행 결과의 저장·관찰 기능을 우선한다. 연구 조건/보상/정책 관측은 바꾸지 않는다.
이번 범위는 단일 episode의 연구자용 오프라인 리포트다. 학습 곡선, 적응 성능과 seed 집계는 후속이다.
실행 결과는 학습·적응 성능의 증거로 표시하지 않는다.

- `mamrl.recording.build_record(spec, tensor, episode, *, family, seed, policy_seeds, policy_names)` → JSON 호환 dict.
- `mamrl.recording.validate_record(record)` → None 또는 TypeError/ValueError.
- `mamrl.recording.load_record(path)` → 검증한 dict. schema 불일치·손상된 기록은 거부.
- `mamrl.recording.save_run(record, output_dir)` → `(run_json_path, report_html_path)`.
  output_dir는 새 디렉터리여야 한다. 기존 디렉터리/파일은 덮어쓰지 않는다.
- `mamrl.report.render_report(record)` → 외부 의존성 없는 단일 HTML 문자열.
- CLI: `python3 -m mamrl --output-dir runs/demo`는 기존 stdout 동작에 저장 경로를 추가한다.
  옵션이 없으면 기존 JSON stdout 동작을 유지한다.
- `python3 -m mamrl.report runs/demo/run.json --output /tmp/rebuilt.html`로 저장 기록에서 재생성.
  출력은 기존 파일을 덮어쓰지 않는다. report CLI는 load_record로 검증 후 렌더링한다.

## 저장 스키마 (schema_version=1)

- `schema_version`: 정수 1.
- `metadata`: `n`, `family`, `seed`, `policy_seeds` (2개),
  `policies` (policy_names에서 전달한 문자열 2개; CLI는 현재 RandomPolicy), `python_version`, `semantics_version`.
  seed는 rule/environment 공통 seed, policy_seeds는 별도 RNG seed.
- `rule`: `{n, terms:[{kind, coefficient, action}]}`.
- `rule_descriptions`: term 순서의 한국어 설명 문자열 목록.
- `tensor`: `{shape:[2,n,n,n,n], values:[...], sha256:...}`.
  sha256는 shape/values의 sort_keys=True,separators=(',',':'),allow_nan=False JSON UTF-8 해시.
- `summary`: `steps`, `total_rewards` (2개), `mean_rewards` (2개),
  `action_counts` (2×n), `terminated` (false), `truncated` (true).
- `transitions`: episode 순서 목록. 기본 Transition 필드
  `observations`, `actions`, `rewards`, `next_observations`, `terminated`, `truncated`와 함께
  `turn` (1부터), `previous_joint_action` (글로벌 순서), `cumulative_rewards` (2개),
  `reward_terms`를 저장.
- `reward_terms`: term 순서의 `{kind, coefficient, action, label, contributions:[r0,r1]}` 목록.
  각 term의 기여 합과 실제 보상은 일치해야 한다. 비활성 조건도 기여 0으로 보존한다.
- 관측은 기존 frozen Observation 네 필드를 dict로 저장.
- 기록 생성/로드 때 shape·rule·보상·관측·전이·경계·누적값·항별 기여·요약 일치를 확인한다.
  부동소수점 산술 비교는 rel_tol=abs_tol=1e-9. JSON 비유한 값은 거부한다.

## 리포트 화면

한국어 중심, 반응형, 로컬 파일 직접 열기. 외부 CDN/폰트/JS/서버가 필요 없어야 한다.
네이티브 details와 앵커로 턴별 보상 근거 및 상태별 행렬을 탐색한다.

1. 요약: 정책·family·n·세 seed·실행 길이·총/평균 보상·종료 사유. 정책 비공개 정보는 연구자에게만 보임을 명시.
2. 누적 보상 SVG 그래프: 두 플레이어, 턴 0의 보상 0 포함, 축/범례, 상수·음수 보상 처리.
3. 턴별 표: 이전/현재 joint action, 플레이어별 보상·누적 보상, 펼칠 수 있는 항별 보상 근거, 종료 표시.
4. 규칙: 자연어 term 설명과 coefficient, 이전 joint action별 n×n 보상표.
   행은 현재 player 0 행동, 열은 현재 player 1 행동, 셀은 (r0,r1).
5. 행동 빈도: 플레이어별 action counts. 적응 지표로 해석하지 않음.

문자열은 HTML escape한다. 값은 실제 기록에서 표시하고 임의의 성능/정규화 점수를 만들지 않는다.
원본 JSON에는 규칙·길이·계수가 있지만 정책 관측에는 추가하지 않는다.
