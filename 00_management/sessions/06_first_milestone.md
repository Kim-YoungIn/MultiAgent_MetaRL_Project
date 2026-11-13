# 세션: 최초 Toy Model 통합 구현

목표: `04_design/Work_Packages.md`의 Toy A/B 최소 요구를 구현하고, 하나의 규칙 텐서로 두 정책이 상호작용하는 실행 가능한 통합 테스트를 통과시켜라. 현재 단계에는 RL²/PPO 학습을 포함하지 않는다.

`Project_Baseline.md`와 `Interface_Contract.md`를 확인한다. 환경・명시적 상태 Policy 인터페이스・실행기를 우선하며, 정적 matrix game → tensor → environment → random agents 경로와 작은 논리 RuleSpec → Compiler 경로를 검증한다. 이후 Memory-One 규칙과 필요한 Variation 및 각각의 테스트를 완성한다. 기본 n=3, 환경·규칙·RandomPolicy의 n=4/5 실행도 확인한다.

Coordinator가 구현의 순서와 위임 여부를 판단한다. 의미적으로 연결된 모듈을 무조건 별도 worker에게 나눠 병렬 구현하지 않는다. 보상 인덱스, 초기 상태, 관측, 상태 전이의 예상값은 가능하면 `validator`가 독립적으로 정의하고 확인하게 하라. 작업 경계가 안정적일 때만 병렬 위임한다.

첫 턴은 균등 랜덤 joint action·초기 보상 0·시작 flag를 사용하고, episode는 비공개 고정 길이에서 절단한다. 정책 상태는 현재 episode마다 초기화한다. 그 외 구현에 필요한 새 선택은 가역적인 현재 작업 가정으로 명시하되 연구 설계상의 확정 결정을 가장하지 않는다. 구현 후 실제 명령과 테스트 결과, 변경 파일 및 남은 연구적 결정만 보고하라.
