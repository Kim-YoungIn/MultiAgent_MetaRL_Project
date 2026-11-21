# Relational games and broader learning opponents — v2 (2026-11-14)

## Research choices

Implement the user's named action relationships as a semantic generation layer, not a mandatory
list of completed games. Preserve four local observations, symmetric common F, n=3 pilot,
Memory-One tensor, meta explorer→same Q student→learning opponent→downstream/control scoring.
No mandatory current-action matching component. Existing v1 catalogs/checkpoints remain reproducible.
New catalog and opponent algorithm expansion form a NEW protocol, not a controlled performance
improvement over v1. Student, teacher architecture, meta G and K/H unchanged in the first v2 pilot.

Repeat=C/change=D is a semantic embedding. Prisoner's Dilemma requires T>R>P>S; 2R>T+S
is separately recorded, not automatically a necessary definition. For n>=3 multiple D actions
exist; action-uniform is not C/D-uniform. Retain uniform-action baseline and add a semantically
balanced repeat/change baseline (.5 repeat/.5 uniformly among changes). It uses only local previous
own action and never the hidden true rule, so not an oracle. Immediate PD action-class equivalence
does not imply repeated-game equivalence against label-sensitive/adaptive opponents.
Add always-change exploration as a rule-blind stage-defection control alongside always-repeat.

## Semantic rule grammar (worker owned relational_rules.py + rules.py dispatch)

- Relation(op,left=None,right=None,children=()): eq/ne/win/draw/lose on four local variables or
  integer action constants; and/or exactly2 children; not1. Same local variable names as old AST.
- Reward(op,value=0.0,left=None,right=None,condition=None,children=()): const; outcome(left,right)
  returns -1/0/1 from outcome table; if(condition,2 children),add(2 children),scale(value,1 child).
- RelationalRule(n,outcomes,root,scale=1.0,shift=0.0): immutable square -1/0/1 outcomes,
  diagonal0,antisymmetric; root generic Reward expression; positive scale and finite shift define
  actual played reward scale*raw+shift. Bound each root depth<=10,total nodes<=255.
- cyclic_outcomes(n) makes explicit antisymmetric cyclic table including nonidentical draws for
  even n. Action permutation may relabel table; do not infer ordering from label values.
- evaluate_relational_rule(rule,previous,actions),compile_relational_rule(rule)->existing tensor;
  symmetry by evaluating same expression in local player perspective.
- relational_to_dict/from_dict strict versioned schema with format='relational_rule',version=1,
  n,outcomes,root,scale,shift. No eval. Serialization preserves metadata needed for reconstruction.
- social_rule(n=3,T=5,R=3,P=1,S=0): repeat means own==prev_own, other repeat==other==prev_other;
  reward if own repeat: if other repeat R else S; else: if other repeat T else P.
  Arbitrary finite payoffs allowed; PD status is determined by independent compiled audit,not name.
- generate_relational_rule(n=3,seed=0,family='social'): local RNG; families social/contest/
  reciprocity/hidden/exception choose grammar expressions and coefficients.
  social samples PD/coordination/anti-coordination/harmony payoff orderings;
  contest uses sampled outcome table and optional relation cost; reciprocity uses joint/temporal
  conditions incl both losing to previous opponent; hidden adds action-specific payoff;
  exception uses explicit ordered conditional override (e.g symmetric outcome reversal).
  Never sample independent tensor cells. Every family need not use outcome or current matching.
  Generate enough normalized/permutation-distinct candidates for3train+2fresh per4base families,
  plus8exception holdouts. Keep complexity bounded; exact recipe recorded in source+rule.
- describe_relational_rule(rule)->human-readable Korean string, separates base outcome vs payoff.

## Main-owned catalog and integration

relational_catalog.py uses12 train rules(3 per social/contest/reciprocity/hidden),8fresh(2 per family),
8exception holdouts. Catalog local seed20261114. Exact/positive-affine/common-permutation tensor
duplicates rejected globally before model scores; constant rules rejected. Unlike old catalog,
do not require every rule's best response to depend on opponent; dominance is a valid control.
Record rejections,source and actual rule,positive normalization to[-1,1],all rule hashes,formula,
family,and full compiled payoff audit. Normalization defines actual task and may affect learning.

Audits per compiled table: per-axis dependence; for every previous state and current opponent
action the full best-response SET; any empty intersection across opponent actions indicates
no universal pure best response at that state (sufficient flag,not a diversity metric).
PD audit: classify action pairs by repeat/change for both players, verify both unilateral class
choices and all4 quadrants are reachable at every previous state,check all action variants yield
the same R/S/T/P and these payoffs are independent of previous labels. Only then T>R>P>S can
be marked exact embedded PD. Additive bonuses/exception can invalidate the tag. Save raw and
played audit and separately optional2R>T+S. Never label an arbitrary relation predicate as C/D PD.

Keep at least one explicit exact PD fixture in training by predeclared first social case;
the remaining cases use seeded generation. A hand fixture is a control,not a guarantee of
exploration benefit. Heldout rules/strata fixed before training,do not select for neural success.
Predeclare the5 social slots: train PD/coordination/anti-coordination; fresh cooperation-dominant/PD.
Rejection sample the seeded social recipe until its COMPILED payoff matches that incentive stratum,
record the rejection count and stratum. These are semantic coverage controls before any model score.
Distances reuse raw/normalized/common-permutation tensor and named-relation token Jaccard,
recorded as descriptive covariates,not strategic distance.

meta_exploration CLI adds --catalog relational (legacy default remains existing logical v1),
--opponent-suite mixed_learning (legacy suite default preserved). v2 uses both flags explicitly.
Protocol/schema saves catalog choice and suite. Generic decode supports legacy and relational
rule JSON without changing old serializer. Family-wise evaluation summaries,normalized prefix
assessment AUC after3/6/full steps and first/late means added; do not call slope pure adaptation.
Current full-horizon assessment mean is uniform discrete AUC/full steps already; name clearly.
Evaluation rules/classes/expr are researcher-only. Teacher neural forward unchanged.

## MA algorithm extension (main owned adaptive_agents.py/meta_exploration_rollout.py)

Add ExpectedSARSA opponent: same observation/action/RNG/table interface,but TD target is
reward+gamma*sum(policy_prob(next_obs)*Q(next_obs)),not max. epsilon-greedy behavior.
Algorithm-specific clone MUST retain subclass/algorithm,table,RNG and audits. Student remains OnlineQ.
v2 training profiles steady Q(.3,.15) and sarsa ExpectedSARSA(.5,.2); heldout profile ExpectedSARSA
(.7,.1). All gamma.95. v1 profiles unchanged. Protocol reports which algorithm,not only alpha/epsilon.
This is broader than Q parameter variation but not unseen-algorithm generalization in the heldout
setting because ExpectedSARSA already appears in v2 training. Learned strategy use audits retained.

## Execution, verification and reporting

v2 default experiment3seeds×40updates,8trials/update,K=H12,reps2,CPU1thread;
no tuning based on final heldout. Same inner budgets and paired Q-erased control across baselines.
Report G AND absolute downstream scores,perfamily and rules; positive-affine normalization and
repeat/change multiplicity explicit. Reuse full source snapshots/raw data/checkpoint reevaluation.
Tests: hand PD table/dominance/quadrants across n3/4/5 and allpreviousstates; false PD after hidden
bonus,draw/antisymmetry,mutual temporal bonus,exception order,codec corruption,RNG and normalization;
manifest disjointness and no forced matching; analytic ExpectedSARSA vs Q bootstrap difference,
clone/RNG/algorithm identity,paired control invariants,3/6/full prefix metrics,CLI/reevaluation/overwrite.
Independent scientific review and contract tests where available; main owns integration and final execution.

## 2026-11-16 연구 확대 계획 — 미구현, v2 실행 계약과 구별

근거: [전달된 중간 비판 보고서](<../03_ideas/Multi-Agent Meta-RL 프로젝트 설계 및 실험 방법론 비판 보고서>).
대상은 본 저장소인 첫번째 구현이다. 동료 구현에 대한 기술 주장은 전달 보고로만 취급하며 재감사하지 않았다.
D-021의 탐색→새 MA 과제 적응 목적을 유지하고,H1을 첫 본실험의 조작 축으로 삼는다.
이 절은 후속 구현/실험 계획이다. 위 v2의 n/K/H/학생/목표/분할·실행 증거를 소급 변경하지 않는다.

### 비판의 수용 범위

- 수용: 정보 식별성·전략적 유용성·탐색 비용 회수,학생의 경험 활용,목표/평가 차이,
  여러 catalog·학습 seed,총예산과 규칙당 노출량,개발/최종 평가 분리가 필요하다.
- 수정: 현 audit는 구문뿐 아니라 최종 보상/최선반응도 검사한다. 그러나 양의 정보 가치는 증명하지 않는다.
  action-set 대칭성이 상성 학습 불가능의 증거는 아니다. 후보별 이력 결합의 제약은 비교할 가설이다.
  reward key 제거는 상태 압축과 정보 손실이 함께 생기므로 자동 채택하지 않는다.
- 유지: 본실험은 처음부터 학습 상대를 사용한다. 상대 다양성 고정은 고정 정책 사용이 아니다.
  고정 반응형 상대는 분석용 대조군이다. 작은 neural 성공을 H1/규모 확대의 필수 관문으로 삼지 않는다.
  H2·두 저장소 코드 통합·dual-head·n4 전환은 자동 채택하지 않는다.

### 한정된 진단과 확장 기반을 병행

1. 기존 checkpoint3seeds의 초기/학습 탐색으로 학생φ와 상대ψ를 얻고2×2 교차 평가를 한 차례 수행한다.
   task별 동일 reset/난수,실제 공동 적응 평가도 병기한다. 이는 조건부 상태 재조합이며 순수 정보 인과효과가 아니다.
   평가136쌍/seed이면408쌍×(2K+4H)=29,376 steps(K=H12),G 대조 추가분은 별도 청구한다.
   저장된 RNG digest만으로 복원하지 말고 재수집 시 실제 RNG/학습 상태를 확보한다.
2. 동일 history에서 학생 exact-reward Q / 양자화 Q / reward 제외 Q를 개발 집합에서 비교한다.
   고유 상태·재방문·후속 재사용·정책 변화·후속 성능을 함께 본다. 주 실험에서는 한 학생을 고정한다.
3. 현재 action-set / index-aware one-hot GRU / memoryless action-set을 비교한다.
   기존 ordinal 실수 번호 인코딩을 일반 GRU의 대표로 삼지 않는다. 공통 관측/예산,
   파라미터 수와 실제 기억 용량을 보고하고 개발 집합에서 구조를 고정한다.
4. G 목표와 절대 후속 return 목표를 개발 집합에서 비교해 고정한다. G는 두 조건에서 진단으로 남긴다.
   학생·구조·목표 전체 조합을 교차 학습하지 않는다. 단계별 비교 예산을 실행 전에 고정하고
   미해결도 기록한 뒤 확장 학습 곡선을 진행한다. 바닥 성능에서 H1 무효과는 해석을 제한한다.

기존 trained 평가408trial의 기록 재분석: 탐색12턴의 평균 고유 Q-state8.0196,
후속 상태의 탐색 방문 key 일치율.443627,보상 제외 key로 계산한 일치율.607639.
이는 key fragmentation/학생 활용 가설의 근거이며 reward 제외의 성능 우위 증거가 아니다.

### 게임 분포와 평가

- 분석 가능한 template 하위집합과 폭넓은 관계형 문법 sampler를 함께 운용한다.
  식별 불가 / 식별 가능하나 대응 동일 / 정보가 있어도 비용 회수 불가 / 유용한 탐색의 네 대조를 분리한다.
  작은 하위집합만 prior·상대·예산을 고정해 정확 계산하고,큰 분포는 추정오차를 포함한 지표로 기술한다.
  Bayesian 기준의 실패만으로 정보 가치0을 단정하지 않는다. 모든 게임에 양의 정보 가치를 강제하지 않는다.
- no-exploration prior / random / ε-greedy / scheduled ε / systematic / learned history를 동일 학생에 제공한다.
  Bayesian·true-rule 기준은 특권 정보를 명시한다. 근사 reference를 수학적 상한/Nash regret로 부르지 않는다.
- 현재28규칙/heldout은 이미 반복 확인했으므로 개발·회귀 집합으로 사용한다.
  새 catalog는 train/dev/final의 생성 seed·규칙·행동 재명명 동등성 누수를 차단하고 최종 집합을 봉인한다.
- 첫 H1은 상대 학습 프로필 분포를 고정하고 규칙 수8/32/128의 nested 하위집합을 비교한다.
  가족·유인·정보 난도 층의 비율을 맞추고 여러 독립 catalog에서 반복한다.
  rule count·구문 거리·상대에 따른 최선반응 차이를 서로 다른 설명 변수로 기록한다.
- primary는 동일 예산의 후속 평균 return,보조는 전체 실제 상호작용 AUC/누적 return·G·탐색 비용이다.
  K 비교는 고정 총예산B에서 H=B−K로 정의한다. 고정 H의 K 비교는 별도 실험으로 표시한다.
  threshold는 개발 단계에서 reference/창길이/유지조건을 고정하고 미도달 비율과 우측 검열을 보고한다.
  reference gap<=0인 경우 정의 불가로 표시한다. 적응 곡선의 기울기만으로 적응 성공을 판단하지 않는다.

### 본실험 규모 목표와 계산량

아래는 실행 전 처리량/개발 검증으로 확정할 계획 목표다. 확정 전 최종 평가 점수를 열지 않는다.

| 항목 | 계획 목표 |
|---|---|
| 행동 공간 | 당시 계획은 n3 유지였으며,아래 D-030에서 n3/4/5의 학습·전이 비교로 확대 |
| 규칙 수 | 8/32/128,훈련 catalog3개,독립 training seeds5개 =45학습 실행 |
| 학습 | 2,000 updates×32trials,실제 학습 곡선/checkpoint 기록 |
| 적응 예산 | 기준K24/H96; 확장 단계에서K12/24/48를 같은B120으로 비교 |
| 평가 | 공통 미학습256규칙×2상대설정×10reps=5,120trial/모델,초기/최종 paired 평가 |
| 반복 확대 | 개발 분산으로5/10seeds 중 사전 확정;최종 점수 확인 후 유의해질 때까지 추가하지 않음 |

기준 실행당 실제 gameplay=2,000×32×120=7,680,000 steps(기존7,680의1,000배).
학생 제거H96을 전 trial에 계산하면 추가6,144,000 steps,합13,824,000.
45실행 합계는 실제345,600,000~제어 포함622,080,000 steps이며 개발/최종 평가 비용은 별도다.
개발에서 G 목표를 택하면 대조 비용을 모두 청구한다. absolute 목표에서는 사전 고정한 진단 표본에만
G를 계산할 수 있으나,그때도 방법별 실제 예산과 제어/평가 비용을 따로 보고한다.

Fixed-total-budget가 첫 본실험이다. 보완은 상위 두 규칙 수에서 규칙당 노출을 맞추고,
동일 추가 계산을 사용한 낮은 다양성 대조도 둬서 diversity×budget을 분석한다.
규칙당 노출 고정에서 총 계산이 증가한 결과를 순수 다양성 효과라고 해석하지 않는다.
catalog×learning seed 수준의 paired 차이와 신뢰구간을 보고하고 rollout들을 독립 학습 seed로 세지 않는다.
45실행을 독립 최종 데이터45개로 단순 pooling하지 않는다. 효과량과 catalog별 일관성을 보고한다.

### 실행 기반과 진행 기준

제한된 실행 범위는 GPU가 차단됐으나,호스트 확인으로 RTX5060Laptop8GB 및
기존torch2.15.0+cu132/runtime13.2가 동작함을 확인했다(2026-11-16). CPU16개/메모리 약11GiB다.
배치 rollout/추론,규칙 tensor 캐시,분리된K/H,n/catalog 주입,resume용 optimizer/RNG/sampler checkpoint,
설정 matrix·비용 집계·중단 복구를 먼저 구현한다. step/s·메모리·저장량을 측정해 병렬 수를 정한다.
환경 연산·추론·PPO를 따로 측정한다. GPU 실행/배치 전송은 구현·실측했으나 작은 설정의 GPU 우위는 가정하지 않는다.
모든 trial의 대형JSON 내장을 피하고 지표는 전수,상세 trace는 사전 고정 표본과 실패 재현분에 저장한다.

개발 단계는 기존K12/H12→K12/H96→K24/H96의 연결 대조로 경계/학습 곡선을 확인한다.
학습 곡선은200/800/2,000update에서 개발 지표를 기록하며 조건별 동일 규칙으로 실행한다.
새 학생/모델/목표를 채택한 최종 구성과 v2의 차이를 단일 요인 개선으로 해석하지 않는다.
진행 순서: 한정된 대조/계약 점검과 배치 기반 병행→개발 집합의 학생/모델/목표 선택·설정 봉인→
본격 학습 곡선과45실행 Rule Diversity 실험→노출량 보완→Opponent Diversity→H2 검토.
양성대조의 neural 실패는 결과로 남기며 무기한 재설계하지 않는다. 공학 오류·평가누수는 먼저 교정한다.

참고: [DREAM](https://proceedings.mlr.press/v139/liu21s.html)은 탐색·활용의 상호 의존 병목을,
[Xu2018](https://proceedings.mlr.press/v80/xu18d.html)은 학습된 탐색이 후속 학습기에 미치는 효과를 다룬다.
현재 방법과 동일한 실험/MA 보장은 아니다. [Agarwal2021](https://agarwl.github.io/rliable/)의 불확실성 논의를
참고해 여러 난수 층과 paired 효과를 분리한다. 독립 research_critic 검토를 반영했다.

## D-030 — 행동 공간과 생성 구조 확대 및 이번 주 설계 마감 (2026-11-18)

11/19 확정된 [D-031 본실험 명세](Adaptation_Experiment.md)가 아래 일정과 규모 초안을 대체한다.

사용자가 행동 공간·게임 규칙 다양성의 확대와 이번 주 실험 설계 마감을 명시했다.
기존 n3 probe는 분석·회귀용으로 유지하고,확대의 중심을 가변 크기 관계형 catalog로 옮긴다.
11월20일(금)을 설계 확정 목표,22일(일)을 실행 재현·인계 정리 여유로 둔다.
이는 실험 설계/설정 확정의 일정이며,그때까지 모든 장기 학습 결과가 나와야 한다는 뜻은 아니다.
이후 추가 아이디어는 다음 버전 후보에 두고 공학 오류·평가 누수 외에는 봉인된 조건을 바꾸지 않는다.

### 구현할 분포와 조작의 의미

- `diverse_relational`: n3/4/5,가변 train/fresh/structural 수를 지원한다.
  focused는 contest recipe, broad는 social/contest/reciprocity/hidden의 균형 혼합이다.
  규칙 내부 상대 보상 계수도 seed로 변화시키되 논리식→결정적 Compiler를 거친다.
  구조 평가는 exception recipe이며 기존 관계의 조건부 결합이다. 새 primitive 전이로 부르지 않는다.
- 같은 n/catalog seed에서 규칙 수·focused/broad를 바꿔도 평가 규칙을 먼저 예약해 동일하게 유지한다.
  훈련은 공통 생성 순서의 prefix로8/32/128을 비교한다. 평가와 재명명 동등한 훈련 규칙은 제외한다.
  focused에서도 공통 평가에는 미학습 가족이 포함되므로 contest 안의 전이와 새 가족 전이를 나눠 보고한다.
- 구문/계수/규칙 수/가족 수/최선반응 signature를 각각 기록한다. 가족 수를 전략 다양성의 증명으로 삼지 않는다.
  compiled 보상·상대별 최선반응·관계 축 의존성·사회적 유인 audit를 병기한다.
  focused/broad는 생성 분포 차이이며 난도·보상 모양까지 일치시킨 순수한 추상 다양성 조작은 아니다.
- 평가 비교는 공통 rule ID와 난수를 고정한다. across-n 게임은 서로 다른 보상표를 가지므로
  n3와n5 생보상의 차이를 행동 수 하나의 인과효과로 해석하지 않는다.
  n4 cyclic 상성에는 무승부/연결 구조 차이가 생길 수 있어 실제 상성표와 유인 층을 함께 기록한다.

### 단계별 실험과 계산량

전체 축을 한 번에 곱하지 않고 동일 checkpoint·조건을 공유한다.

| 실험 | 훈련 조건 | 비교/해석 |
|---|---|---|
| 학습 경로 인수 | n3/4/5의 작은 실제 학습·저장·재평가 | shape/관측/Compiler/학생·상대 업데이트 검증; 성능 증거 아님 |
| 개발 범위 확인 | n3/5×focused/broad×N32×3catalog×3seed =36회 | 분포·행동 수 효과와 처리량/저장량; 기존 n3 probe18회 계획을 우선 대체 |
| 규칙 수 본실험 | n3×broad×N8/32/128×3catalog×5seed =45회 | 고정 총예산에서 규칙 수 비교 |
| 생성 범위 비교 | n3×focused×N32×3catalog×5seed =15회 추가 | 대응 broad N32는 앞45회에서 공유 |
| 행동 수 비교 | n5×broad×N32×3catalog×5seed =15회 추가 | 대응 n3 조건은 앞45회에서 공유 |
| 행동 공간 전이 | 위 n3/n5 broad N32 checkpoint를 n3/4/5에서 평가 | weights 재학습 없는 실행과 실제 전이 성능을 구분; 평가만 추가 |

개발은 K12/H48·200updates×32trials,H32,batch32를 출발점으로 고정한다.
36회 훈련은 제어 포함24,883,200steps이며 평가 비용은 별도다. 아직 이36회는 실행하지 않았다.
개발 학습곡선을 사전 선언한200/800에서 확인하고 처리량·분산·예산으로 본실험 최종 update 수를 정한다.
heldout 최고 checkpoint로 종료 시점을 선택하지 않는다. 목표는 일단 G,학생·학습 상대는 현행으로 고정한다.
학생 표현 비교 등은 별도 한정 대조이며 주 matrix 전체를 학생/목표별로 다시 곱하지 않는다.

주 설계는 합75회이며 규칙 수×생성 범위의 상호작용까지 추정하지 않는다.
그 상호작용이 필수라는 연구 질문을 추가로 채택하면 focused N8/128의30회를 별도로 사전 선언한다.
장기 본실험의 K24/H96·2,000×32trials 상한을 그대로 쓰면75회 훈련은
실제576,000,000+제어460,800,000=1,036,800,000steps다. 목표 규모이지 이미 승인·실행된 계산량은 아니다.
고정 예산과 규칙당 노출량 보완은 별도로 해석한다. CPU/GPU는 해당 n/모델/batch 처리량으로 정한다.
n 증가만으로 GPU가 빨라진다고 가정하지 않는다.

### 이번 주 확정할 내용과 보고 기준

1. **11/18~19:** n3/4/5 학습 경로·가변 크기 catalog를 구현/검증하고 source/seed 재현·중복 제거를 확인한다.
   양성/음성 대조,보너스 총량/정규화,현재 Q의 정보 활용 한계는 인수 결과에 기록한다.
2. **11/20 설계 확정 목표:** 논리적 유인/상대 의존성을 갖는 분포 명세,개발/최종 seed 분리,
   sampler·학생·탐색자·목표·K/H·모델·예산·중단/재개·평가 반복/통계를 하나의 설정으로 확정한다.
   본실험의 최종 집합은 개발 결과와 분리하고 전체 훈련 catalog와 canonical 중복을 검사한다.
3. **11/21~22:** 확정 설정의 독립 재현·동료 인계용 README/결과표를 정리한다.
   남은 확장 후보는 미결 목록으로 이동하고 새 진단의 성공을 매번 선행 조건으로 추가하지 않는다.

primary는 같은 K/H와 학생에서의 후속 평균,paired learned−random/초기 탐색 차이다.
행동 공간 전이는 학습 run의 initial checkpoint도 동일 manifest/seed로 `action_transfer`에 따로 제공해
trained−initial을 계산한다. 도구의 trained 레이블은 전달받은 checkpoint를 뜻하며 학습 여부의 자동 판정이 아니다.
이는 메타 가중치 재학습 없는 전이와 학생의 과제 내 적응을 함께 측정하며 학생 zero-shot 성능은 아니다.
prefix AUC·전체 실제 K+H return·G·탐색 비용·실패/도달 비율은 보조다.
적응 속도 threshold는 유효한 개발 reference가 있을 때만 사전 지정하고,없으면 AUC를 보고한다.
정규화 entropy=entropy/log(n),반복률과 균등기준1/n 차이,행동 방문률·Q-state 재방문/재사용을 기록한다.
같은H32의 학습 파라미터 수는 n과 독립이지만 action-set hidden은 n×H라 기억량은 같지 않다.
행동 공간 일반화와 기억 용량 효과의 한계를 함께 명시한다.
catalog×훈련seed를 반복 단위로 보고하고 동일평가 난수/rollout을 독립 훈련seed로 세지 않는다.
H1의 수/분포 범위와 H2의 구조 재사용,행동 공간 전이는 별도 결론으로 낸다.
