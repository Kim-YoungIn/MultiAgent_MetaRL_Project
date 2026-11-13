# Open Questions

- 규칙의 변경은 task/trial 경계에서만 일어나는가, 아니면 하나의 trial 중에도 일어나는가?
- Meta-Policy의 적응을 순수 recurrent hidden state만으로 할 것인가, 테스트 시 추가 학습을 허용할 것인가?
- Independent Co-training을 필수 baseline으로 할 것인가, 고정 상대를 먼저 사용할 것인가?
- Exploration은 단일 policy에서 암묵적으로 학습하게 할 것인가, 목적함수 또는 정책을 분리할 것인가?
- 구조화된 게임 생성과 무구조 무작위 텐서를 어떻게 구분하고 난이도를 통제할 것인가?
- 어떤 성능 지표가 실제 적응과 초기 일반화를 구분하는가?
- 초기 `START` state와 episode 경계를 어떤 방식으로 처리할 것인가?

이 목록은 결정을 요구하는 검토 항목이며, 질문이 적혀 있다는 이유만으로 현재 Toy Model 구현을 막지 않는다.
