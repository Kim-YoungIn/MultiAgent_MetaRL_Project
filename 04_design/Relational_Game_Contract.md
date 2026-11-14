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
