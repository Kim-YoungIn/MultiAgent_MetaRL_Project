# Fast Exploration · 96턴 배치 전이

v2의 36개 학습 블록과 세 master seed를 그대로 사용한다. 새 판별용 시드 스트림에서 블록당 128개 게임을 균등 계열로 표집하고 각 게임을 두 번 초기화한다. 연속 배치 96턴 중 처음 K턴만 Teacher가 선택하고 Student와 상대는 계속 갱신한다. v2에서 학습한 `pooled_fixed_total`과 `pooled_fixed_value`를 수정 없이 사용하고, pooled Student만 배치한 `K=0`을 대조로 둔다. Teacher는 K=8, 24에서 평가한다.

주 지표는 K=8에서 `pooled_fixed_total − pooled Student 단독`의 96턴 실현 누적 보상과 그 차이의 `96턴 − 48턴` 변화다. 48턴 점수는 같은 96턴 궤적의 첫 48턴이다. 두 Teacher 간 차이와 계열·K별 곡선도 보고한다. 블록 전체를 대응 재표집하고 등가성은 별도로 가정하지 않는다. 행동과 두 플레이어의 보상을 각 턴에 저장한다. 이 비교는 48턴을 위해 학습한 탐색의 장기 전이를 측정한다.
