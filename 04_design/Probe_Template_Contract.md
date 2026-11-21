# 공통 Template와 History 효과 비교 — D-028 (2026-11-16)

## 목적과 범위

사용자의 설계 메모를 채택해 구조가 유사한 게임에서 정보 식별·전략적 의미·학습 경험 활용을 분리한다.
학습 상대는 처음부터 유지한다. 상대의 학습/변화에 대한 적응 능력을 따로 조작하는 분석만 후속이다.
관측 네 필드·Memory-One·대칭 공통F·기본 n3·Q 학생/G 목표는 유지한다. 숨은 규칙을 정책에 제공하지 않는다.
고정 상대 계산은 분석용 witness이며 본 실험의 상대를 대체하지 않는다.

## 공통 게임

세 역할 safe/costly/free를 행동에 재명명한다. safe 보상q,나머지는 θ·3·C(a,b)에서 costly 역할만η 비용을 낸다.
C는 역할 순서의 cyclic 상성,θ∈{−1,+1}. 양쪽은 같은 자기 관점F를 사용한다.
짝의q/η/역할은 같고θ만 다르다. 비용 marker가 없으면θ반전도 재명명으로 같은 게임이므로 η>0을 유지한다.
역할 재명명을 독립 구조 다양성으로 세지 않는다. Generator→RelationalRule→Compiler를 사용한다.
q∈[.5,1.4],η∈[.15,.5]를 seed로 생성하고,전체 보상을 양의 affine 변환으로[-1,1]에 맞춘다.
θ짝의 정규화범위와 safe 관측이 같음을 검사하고,최종 tensor canonical 중복을 모든 분할에서 제거한다.

학습/동일-family fresh는 반복 보너스 또는 모방 보너스(.15)를 각각 단독 적용한다.
구조 holdout은 두 요소를 동시에 적용하는 조합이다. 새 관계를 추가한 전이로 부르지 않는다.
학습 규칙4개 이상이면 두 단독 요소가 모두 포함된다. 수·seed·파라미터·정규화·원식·hash·sampling 경로를 검증한다.
학습/평가 count는 양의 짝수이며32/16/16이 개발 실행 구성이다. runtime catalog는 `probe_template`이다.

## 정보 가치와 음성 대조

zero-temporal-bonus 원보상q1/r3/η.2,θ균등 prior,알려진 고정 상대b=safe,총2턴에서:
무정보 총보상2,최적 adaptive2.9,known-θ oracle5.8,순정보가치.9다.
첫 위험 행동의 즉시 기대0은 safe1보다 낮지만 후속 대응의 가치가 비용을 회수한다.
horizon1턴에서는 순정보가치0,safe보상4인 지배전략 대조에서도0이다.
`exact_information_audit`가 가능한 deterministic reward history와 posterior를 열거해 계산한다.
이 증거는 source/raw 보상·명시한 prior/상대/예산에만 해당하며 temporal 보너스나 실제 학습 상대에 자동 상속하지 않는다.

## History 평가

`mamrl.history_study`는 random/ε-greedy/balanced/repeat/systematic/initial/learned 탐색의 같은 과제·seed 경험을 비교한다.
자연스러운 원래Q학생+원래 상대의 후속 점수를 남기고,조건부 비교에서는 random 탐색이 만든 동일 상대 상태로 고정한다.
학생은 full reward key / reward 제외 key / sign-quantized key의 세 Online Q 표현이다. 서로 다른 알고리즘 전체 비교는 아니다.
같은 탐색 events를 각 학생에 같은 순서/초기화/학습률/업데이트 수로 재생한다.
마지막 탐색 update는 `terminal=False`,next는 `learning_next_observations`의 후속 reset이다.
full 재생 snapshot이 원래 학생과 같은지 검사한다. 학습된 상대의 실제 table/RNG를 clone해 평가하며 이후 학습은 허용한다.
조건부 재조합은 자연스럽게 생성된 조합과 다를 수 있어 순수 정보 인과 추정량이 아니다.

빈 경험 학생도 같은 상대/reset에서 평가한다. 이는 동일 K를 소비한 후 경험을 버리는 대조이지 무료 추가 예산이 아니다.
History는 한 번 저장하고 row는 hash/id로 참조한다. 학생 고유 state·재방문·후속 재사용·prefix AUC와 자연/조건부 성능을 보고한다.
paired-rule prior에 대한 후보 수/식별 시점은 연구자 진단이며 신경망 내부 규칙 분류의 증거가 아니다.
정책에 rule ID·θ·candidate set을 주지 않는다. 실제 규칙 분류를 주장하려면 별도 행동/표현 분석이 필요하다.

threshold는 사전에 선언한 절대값 또는 개발 reference low/high 사이의 비율이다. high를 true 최대값으로 가정하지 않는다.
창3/연속2창 이상 유지한 최초 종료step을 기록하며 미도달은None/우측검열과 도달비율로 보고한다.
G·후속 return·실제K+H 비용과,재생/빈 경험/paired 제어의 추가 simulation을 구분한다.

## 고정 평가 집합의 H1 실행 경로

`--manifest common.json --train-rule-limit N`은 공통 catalog에서 학습 prefix만 바꾼다.
평가 task/seed/규칙은 동일하고 실제 training_rule_ids/seen_rule을 기록한다. θ짝을 보존하도록 제한은 짝수다.
새 catalog를 매 조건 생성하며 train_rules만 바꾸면 sampler의 후속 평가 규칙도 바뀌므로 H1 비교에 사용하지 않는다.
서로 다른 catalog 반복과 고정 총예산/규칙당 노출량 비교는 별도 matrix로 실행해야 한다.
이 구현은 수를 바꾸는 경로이며 현재 생성 파라미터 수 증가가 전략 다양성 증가라는 뜻은 아니다.

## 확대와 검증

개발 실행:32train/16fresh/16조합,3seeds7/17/27,200updates×32trials,K12/H48,H32,batch32,reps1.
실제384,000+제어307,200=691,200 simulation steps/seed,3seed 총2,073,600(최종 평가 비용 별도).
이는 기존7,680 실제 steps/seed의50배다. 작은H32는 측정상CPU가 빨라 CPU1thread로 실행한다. CUDA 경로는 유지한다.
새 최종 평가를 열거나 이 개발 결과를 H1/H2의 확증으로 쓰지 않는다.

불변식: 역할 재명명 동등성/θ구별/교환대칭,정규화 누출 없음,seed/count/canonical provenance,
손계산VoI·음성대조,reset bootstrap 재생,학생 간 동일history/상대,threshold 검열,
CLI→checkpoint 재평가→History study와HTML을 독립 검증한다.
모든 결과·체크포인트·source snapshot은runs,현재 결론/다음 일은Implementation Status에 둔다.

## 개발 결과 해석의 제한

현재64규칙의 θ짝32/32에서는 단계별 최선반응 집합이 다르다. 반면 각θ/base에서 temporal 보너스만
제거해 비교하면 단계별 최선반응이 달라진 것은0/64다. 장기 전략이 같다는 증거도 아니지만,
재조합을 새로운 전략 구조로 부르지 않는다. 단독총보너스.15와조합최대.30의 규모/정규화 교란이 남는다.

조건부 reference가random인 기본평가에서 random history는자연쌍,learned history는교차쌍이다.
성능 차이를 순수history품질 인과순위로 단정하지 않는다. reference=random/initial/learned 교차는 아래 D-029로 구현했다.
세학습checkpoint의History비교는같은evaluation seed/과제/reps1을사용한paired비교다.
반복된random점수를세독립환경seed로세지않는다. θ후보진단은knownpaired reward-consistency일뿐
상대행동likelihood를활용한전체규칙추론/신경망belief검사가아니다.
threshold기능은있지만이번실행에서는None이므로적응도달시간개선을관찰했다고보고하지않는다.

## D-029 — 목표와 상대 상태를 분리하는 개발 비교

외부 비평의 G/절대 성능 불일치와 reference 비대칭을 별도 조작으로 검토한다.
`mamrl.objective_study`는 같은 manifest·초기 신경망 seed·task sampler·학생·학습 상대·예산에서
G(`retained_gain`)와 후속 평균(`assessment_return`)으로 각각 PPO를 학습한다.
어느 목표에도 탐색 중 즉시보상을 더하지 않는다. 절대 목표에서도 경험 제거 대조를 계산해 계산량을 맞춘다.
기본 CLI 목표는 G를 유지하며,목표 변경 시 optimizer 재개를 거부한다.

개발 실행은 기존32/16/16 manifest·seeds7/17/27·200updates×32trials·K12/H48·H32·batch32·CPU1thread다.
0/50/100/200update의 공통 평가를 남기고 최종200을 비교한다. 평가 최고점 선택/추가 튜닝은 하지 않는다.
6회 훈련 총4,147,200 simulation steps이며 중간/최종 평가 비용은 별도다.
후속 평균·G·전체 K+H 평균을 함께 보고,θ/상대/보상 부호/직전 행동 일치/초기·후기별 반복 횟수도 남긴다.
조건부 빈도에는 상태 방문 차이가 있으므로 같은 관측에서의 인과적인 정책 차이로 해석하지 않는다.
신규 평가 trajectory를 중복 저장하지 않고 task별 지표·가중치·훈련 이력·source hash를 보존한다.
`--resume`은 동일 protocol/source에서 update 경계 상태를 재개한다. 이전 평가 누락은 임의 재생성하지 않는다.

`mamrl.reference_study`는 각 과제에서 random/initial/learned가 만든 학생 History와
각 탐색이 만든 상대의 table/RNG 상태를3×3으로 교차한다. 모든 조건에서 같은 reset을 사용하고
학생 재생 snapshot을 검사하며 online 대각선은 원래 자연 MA 평가와 정확히 일치해야 한다.
각 조합은 학생 online/frozen 두 방식으로 평가한다. frozen은 학생 Q만 갱신하지 않으며
동률 행동 RNG는 진행하고 상대는 자기 보상으로 계속 학습한다. 순수 정보량/고정 상대 실험은 아니다.
K12/H48에서는 과제당3(K+2H)+18H=1,188 simulation steps다.
fresh32과제×기존3checkpoint는114,048steps이며 같은 평가 난수를 공유하는 paired 비교다.

확대는 이 개발 비교와 병행 가능한 다음 본실험으로 이어간다. 단순 update 증량보다
독립 catalog·학습 seed·평가 반복을 분리하고,규칙수와 전략 차이·규칙당 노출을 통제한다.
우선 고정된 학생/목표로 소수 조건의 다중catalog 실행을 수행하고 그 처리량/학습곡선으로
8/32/128×3catalog×5seed 계획의 예산을 확정한다. 개발 성공을 모든 후속 작업의 상시 관문으로 삼지 않는다.
Bayes 양성 대조와 실제 Q의 연결,총보너스/정규화 교란,전략적 다양성의 정의는 여전히 미해결이다.

후속 실행의 우선순위는 사용자 행동 공간·규칙 다양성 확대 요청에 따른
[D-030](Relational_Game_Contract.md#d-030--행동-공간과-생성-구조-확대-및-이번-주-설계-마감-2026-11-18)이 갱신한다.
기존 probe는 분석·회귀용으로 유지하고 가변 크기 관계형 분포와 n3/4/5 학습·전이 비교를 진행한다.
