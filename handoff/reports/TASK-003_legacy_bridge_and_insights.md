# TASK-003 — Legacy Bridge and Results Insights

**Role:** `results_analyst` · **Status:** Proposal for Lead review · **Date:** 2026-11-27  
**Scope:** Required governance/brief files and the ten listed CSV tables only; no repository writes, training, or commits.

Auxiliary prediction improves reliable task inference and cooperative coordination, but its benefit reverses under some deployment conditions. V3 establishes conditional success; it does not establish general meta-exploration superiority.

## Evidence and interpretation limits

- **C:** `results/core_v3_sealed/tables/{methods,paired_contrasts,by_theta,behavior_repetition}.csv`
- **A:** `results/exp2a_online_opp/tables/{methods,paired_contrasts,restore_vs_continuous,behavior_repetition}.csv`
- **B:** `results/exp2b_symmetric_ma/tables/{matchups,paired_contrasts}.csv`
- Intervals and verdicts below are exported values; bootstrap uncertainty was not recomputed. Behavioral explanations are hypotheses unless supported by an intervention.
- These tables lack individual seed trajectories, direct early-adaptation AUC, and Exp 1 counterpart returns. Post-return and K4 adaptation scores cannot substitute for AUC. Oracle opening-necessity percentages are documented in `PRIOR_FINDINGS.md`, not independently reverified here.
- Paired improvement win-rates and head-to-head episode win-rates answer different questions.

## 1. Legacy-to-v3 coverage matrix

| Legacy failure mode | What v3 currently establishes | Remaining clean-v3 verification gap |
|---|---|---|
| **1. Mean–median skew** | C/R3 aux−vanilla: mean `+4.59`, paired median `+0.15`, wins `50%`. A/Q-learning matched−zero-shot aux: mean `+5.4296`, median `0`, wins `35.82%`, losses `49.10%`. Large average gains still coexist with uneven benefits. | No claim of general distributional improvement. Require seed/rule-stratified differences, lower-tail returns, and deployment-family reweighting before pooling broader task distributions. |
| **2. Baseline loss and weak feedback conditioning** | C/R3 aux−arm-UCB `+24.82`; aux−masked `+1.57` `[1.4426,1.6784]`; aux−structural `+0.02`, `RESOLVED_NULL`. A/SARSA aux−prior-only `−1.5062`, `SUPPORTED_NEG`; B/competitive aux loses to Q even in distribution. | Baseline loss is conditionally overcome. Separate auxiliary representation benefits from learning algorithm and training-budget effects; DRQN success does not establish recurrent PPO success. |
| **3. Passive information versus active exploration** | Legacy synthesis documents Oracle-required opening changes in active versus passive regimes. C behavior agrees: aux opening-probe rates `0/1/1` in R0/R1/R3. R3 K4 commitment gain over masked is `+10.40`; net-adaptive gain is `+0.84`. | Opening probes alone do not establish useful information gathering. Reverify Gate 0 for every changed distribution and add neutral information/value and mixed-prefix comparisons where absent from these tables. |
| **4. Opponent shaping versus dynamics shift d** | A/Q-learning aux retains gains over masked when restored: `+1.3602`. Continuous−restored aux post-return is `+0.492`, inconclusive; structural model gains `+1.4087`, supported. B/cooperative counterpart and welfare gains establish mutual-benefit outcomes. | No controlled reliability/dynamics sweep appears here. Restoration addresses opponent-state carryover, not every shaping pathway. B lacks a causal decomposition of information acquisition versus induced cooperation. |
| **5. D-031 diversity versus exposure; adapt_v1 reversal** | B documents cooperative transfer and competitive transfer failure; A documents opponent shift. These are deployment boundaries, not controlled diversity effects. | No rule-count × exposure intervention. No controlled concordance-q reversal. The documented anchor–orientation shortcut explanation remains a mechanistic hypothesis from aggregate CSV evidence alone. |

## 2. Hidden patterns and anomalies

### Exp 1: much of the auxiliary benefit is failure prevention

- C/R3 aux returns `17.52`, median `17.30`, seed SD `0.123`; vanilla returns `12.93`, median `16.90`, seed SD `4.3155`. The auxiliary method chiefly reduces severe failures rather than raising every successful outcome.
- Theta-level means localize the weakness: at θ=2, aux/vanilla/reward-only return `17.12/8.78/6.38`; at θ=3 they return `18.26/18.48/18.48`. The advantage is heterogeneous.
- Vanilla already probes at opening rate `1.0` and has branch-TV `1.0`, yet spends `35%` of exploitation on safe action S; aux spends `0%`. Thus successful probing/branching is insufficient: the policy must convert history into sustained exploitation.
- `aux_y` reaches `16.795`, while `aux_r` reaches `12.215`. However, aux-y versus vanilla remains inconclusive, and adding reward prediction to aux-y gives `+0.725`, also inconclusive at the practical threshold. The evidence supports combined auxiliaries over reward-only; it does not isolate a necessary reward head.
- Against masked, total-return gain `+1.57` accompanies commitment/adaptive/net-adaptive gains `+10.40/+2.02/+0.84`. Continued adaptation recovers much of a weak prefix’s disadvantage, so commitment gains should not be reported as deployment-return gains.

### Exp 2A: why vanilla beats auxiliary prediction on unseen SARSA

| Continuous SARSA | Matched aux-yr | Matched vanilla |
|---|---:|---:|
| Total mean / median | `3.2938 / 1.9` | `4.2353 / 4.2` |
| Total IQR | `15.8` | `2.1` |
| Opening-probe rate | `100%` | `60%` |
| Exploitation S / X-mismatch | `39.19% / 9.98%` | `72.87% / 3.66%` |
| Counterpart total return | `42.6357` | `42.9195` |

- Vanilla avoids more exposure to unreliable inference. Its advantage is consistent with conservative fallback, rather than superior information gathering: blind prior-only remains better at `4.8`.
- Aux−vanilla total return is `−0.9415` `[-1.4237,-0.4069]`, `SUPPORTED_NEG`; post-return difference is `−0.4698`, inconclusive. Subtracting post-return from total shows approximately `−0.4717` of the disadvantage occurs before the post phase.
- Early opponent honesty is `59.02%` with aux versus `63.85%` with vanilla. This is policy-dependent behavior, not a randomized measure of exogenous signal quality.
- Matching online training nevertheless sharply reduces aux mismatch versus zero-shot: `13.04%→1.96%` under Q-learning and `34.66%→9.98%` under SARSA. It learns caution, but insufficient caution for the unseen opponent.
- Restoring the opponent leaves aux−vanilla at `−0.8004`, although the exported verdict is inconclusive. The ordering therefore persists descriptively without proving an independent mechanism.
- The exported zero-shot `bayes_optimal` returns `−9.6441` on SARSA. It is a transferred diagnostic policy, not a valid optimal ceiling for the shifted online environment.

### Exp 2B: auxiliary prediction helps coordination, but can hurt competition

- Competitive in-distribution margins against Q are aux `−0.9781` versus vanilla `+0.8273`. Their paired margin difference is `−1.8055` `[-2.4898,-1.1906]`, supported negative.
- Against arm-UCB, both win: aux margin `+3.4117`, vanilla `+4.2172`. This opponent dependence is consistent with an auxiliary-induced policy that stateful opponents can exploit, but does not establish that mechanism.
- Late action-frequency entropy is almost identical against Q: aux `1.8159`, vanilla `1.8062`. It cannot establish stochastic policy randomization or explain predictability. Identical-seed competitive self-play ties every episode despite entropy `1.7356`, illustrating this limitation.
- On unseen competitive rules, aux-versus-UCB margin is **`−0.0836`**, CI `[-0.6844,0.6969]`, inconclusive. The commonly quoted `−0.04` is focal return, not margin.
- In unseen cooperative cross-seed self-play, aux versus vanilla yields welfare `31.6975` versus `20.6950`, cooperation `61.62%` versus `35.25%`, and trap rates `8.13%` versus `11.04%`. The paired welfare gain is `+11.0025`, median `+4.3`, wins `60%`, CI `[0.7532,24.6]`.
- Cooperation is therefore the clearest behavioral correlate of the auxiliary benefit. Its causal origin—better identification, compatible conventions, or altered learning dynamics—remains unresolved.
- Aux identical-seed welfare is `43.345`, versus cross-seed `31.6975`; cooperation falls `80.12%→61.62%`. The self/cross-seed welfare gap is approximately `11.65` in both splits. This suggests a substantial compatibility limitation alongside rule transfer, without identifying particular bad seeds.

### Negative individual margins coexist with positive welfare

All rows below concern unseen `coop_trap`; gains compare replacing player 1 in the corresponding non-meta self-play baseline.

| Partner | Meta / partner returns | Head-to-head margin | Meta return gain | Partner return gain¹ | Welfare gain |
|---|---:|---:|---:|---:|---:|
| Q-learner | `4.6284 / 6.4539` | `−1.8255` | `+2.1913` | `+4.0252` | `+6.2165`, supported |
| Arm-UCB | `9.6530 / 10.8018` | `−1.1488` | `−0.1039` | `+1.0479` | `+0.9440`, inconclusive |
| Contextual-UCB | `7.7872 / 8.4623` | `−0.6751` | `+7.6044` | `+8.3011` | `+15.9055`, supported |

¹ Derived as welfare gain minus focal gain; no separate uncertainty claim.

Both players can improve while the partner improves more. Unilateral exploration cost is a plausible explanation, but aggregate returns do not identify who paid which action costs. The `−1.15` margin and `+15.91` welfare gain concern **different UCB variants**.

## 3. Ranked next experiments

### 1. Reliability and dynamics stress test — evaluation first, no initial retraining

- **Design:** Evaluate existing checkpoints at prespecified `d∈{0,0.3,0.6}`. Separately manipulate R3 observation reliability and cooperative-success reliability in `coop_trap`; do not combine these into one perturbation. Pair random seeds and retain continuous/restored protocols where supported.
- **Measurements:** Total/median/paired wins, early AUC, counterpart return, mismatch/S rates, cooperation, and neutral information/value audit. Recompute the appropriate Oracle diagnostics; do not use transferred `zs_bayes_optimal` as the shifted ceiling.
- **Falsification:** “Cooperative gains depend on reliable cooperative outcomes” is contradicted if aux’s welfare advantage remains equivalent within ±`0.3` across the sweep. “Signal unreliability drives harmful commitment” is contradicted if degrading signals leaves mismatch and cost-inclusive aux−prior return equivalent while Oracle information value declines. Wide intervals remain inconclusive.
- **Leverage:** Directly tests legacy dynamics claims and the strongest observed OOD failure using existing policies. Gate 0 must precede any subsequent training.

### 2. D-031 diversity versus exposure — small factorial in `coop_trap`

- **Design:** Use nested, structurally balanced training sets of `N=4/8` rules from the existing training pool; preserve the eight unseen rules. Cross N with total transition budgets `B/2B`, comparing vanilla and aux-yr with three development seeds: **24 fits**. Match optimizer updates per transition.
- **Identification:** `(4,B)` versus `(8,B)` measures fixed-budget diversity; `(4,B)` versus `(8,2B)` matches nominal per-rule exposure. `(4,2B)` tests whether extra compute alone explains recovery. Counterbalance rule subsets; one subset cannot establish a diversity effect.
- **Falsification:** Exposure dilution predicts a fixed-budget penalty that shrinks with restored exposure. A persistent penalty at matched exposure contradicts dilution as a sufficient explanation. A diversity benefit requires a supported unseen-rule advantage at matched exposure; equivalence within ±`0.3` contradicts that benefit at the tested scale.
- **Leverage:** Resolves an entirely untested legacy confound while retaining the existing cooperative transfer setting.

### 3. Auxiliary Task Discovery with a controlled concordance diagnostic

- **Design:** Compare vanilla, fixed aux-yr, uniform candidate auxiliaries, and learned gates over public-observation prediction targets; use identical transition budgets and three development seeds. Choose gates through training/development return only.
- **Split control:** First establish disjoint competitive rule pools balanced for anchor×orientation, using additional payoff permutations if necessary. Otherwise label the exercise a trajectory-holdout diagnostic. Then vary a declared cue–orientation concordance from matched to reversed while keeping action-contingent evidence available.
- **Measurements:** Cost-inclusive return, early AUC, opponent-specific margins, prediction calibration, and history-conditional action probabilities. Evaluate competitive and cooperative families separately.
- **Falsification:** Adaptive auxiliary selection requires supported gains over both fixed aux-yr and uniform weighting; equivalence within ±`0.3` rejects a useful gating benefit at this budget. A concordance-reversal penalty surviving balanced training contradicts the narrow shortcut explanation as sufficient.
- **Leverage:** Tests the proposed extension against the observed negative transfer of auxiliaries and directly connects to `adapt_v1`.

For all proposals, preregister contrasts and practical thresholds, retain mandatory controls, keep development and final seeds separate, and treat negative or equivalent outcomes as informative.

## Validation

Five read-only, in-memory `unittest` checks passed: required-table coverage, welfare/margin identities, Exp 2A total-contrast arithmetic, win/tie/loss accounting, and behavior-share accounting. No source changes or new experimental results were produced.