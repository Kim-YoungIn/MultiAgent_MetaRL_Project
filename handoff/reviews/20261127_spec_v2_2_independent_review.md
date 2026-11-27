# Independent review — SPEC v2.2

- Reviewed snapshot: `refactor/v2-clean @ 16ba4e4`, context pack generated 2026-11-27.
- Status: **proposal, not adopted specification**. No repository changes or training performed. The supplied pack contains documents and tree, not implementation source. The R3 candidate below was independently evaluated with exact rational arithmetic DP.

## 1. Verdict and scope

Retain end-to-end RL² and the research question. Do not lock SPEC v2.2 unchanged. Existing legacy evidence supports studying active information acquisition, not assuming its benefit. In particular, fixed opponent and online-learning opponent experiments need different claims and Oracle scopes. The core question is not whether RL² wins, but whether and under what information structures meta-learning changes exploration and adaptation.

## 2. Required corrections

| Area | Correction |
|---|---|
| Baselines | Retain UCB/Greedy/Random; add one structural baseline trained on the same meta-training distribution (e.g. learned opponent/outcome model with greedy/TS), or at minimum contextual UCB. Report offline training and online interaction budgets separately. No test-task spec access. |
| Memory reset | Resetting GRU while retaining previous action/reward/opponent action is Memory-1, not zero-shot. Define separate prior-only policy without feedback. |
| Masking | Train masked policy under its own mask; do not merely mask at test. Explicit K and no retrospective revelation unless separately defined. Distinguish early masking from total absence of feedback. |
| R0–R3 | Match latent support/prior, action count, reward and horizon where possible. Manipulate joint observation likelihood (opponent action AND reward), not only reward. R2 Gaussian reward noise does not reduce rule ID if opponent action already identifies rule, and prevents simple finite belief enumeration. Use categorical observation noise if noise in information is the intended factor; otherwise rename R2 reward-noise optimization control. |
| Oracle | Distinguish Bayes-optimal prior-informed policy from clairvoyant task-aware policy. Define gate over priors or families, not per known latent type. Strict opening margin > tolerance; track actual realized types separately. Gaussian observations / online-learning opponent do not in general admit a cheap exact full-horizon Oracle. |
| Layer 1 | Posterior likelihood must condition on actions, observations and interventions. Report decision-relevant value, not merely entropy/ID accuracy. Add commitment value, continuation value and cumulative prefix cost separately. |
| Layer 2 | Balanced mixed prefixes mitigate, not eliminate, OOD. Include all evaluated prefix generators, K and reset protocols; split by training/test task and cross-fit; freeze exploiter. Shared exploiter can still have a capacity bottleneck. Secondary validation only. |
| Layer 3 | Oracle prefix into an on-policy RL² is OOD again. Do not infer exploitation efficiency from this contrast without training support. Compare actual-history policy continuation against Bayes continuation; linear probe on disjoint tasks is diagnostic, not proof of causal use. |
| Restore | Reset public physical state and opponent state to a common initial configuration while keeping θ; reset indicator must be visible and trained on. Define remaining horizon consistently. Restore removes state-mediated effects under the stated intervention, not every possible causal shaping effect. Counterpart return alone is insufficient for identifying shaping. |
| Diversity | Count distinct semantic tasks separately from relabelings; compare fixed total and fixed exposure only if resources permit. Match training/test distribution; record budget and per-task exposures. |
| Statistics | Separate information acquisition, net adaptation benefit, feedback-conditioned improvement and population coverage. Mean/median/win rate all reported, not all necessary to establish exploration improvement. Paired hierarchical bootstrap with training seed as independent unit; ties separate; practical effect threshold predeclared; distinguish null from inconclusive. |

## 3. Concrete R3 candidate: simultaneous delayed-response game

This is a **candidate diagnostic**, not a required replacement or a generic multi-agent benchmark. Focal action space A = {S,P,P0,P1,X00,X01,X10,X11}; opponent action y in {0,1}; hidden opponent type θ=(b,c) uniformly in {0,1}². Public state q = previous focal action, q0=S. Both players choose simultaneously; opponent observes q, not current focal action. Transition q'=a. Fixed opponent chooses fθ(q):

- fθ(S)=0; fθ(P)=b;
- fθ(P0)=1 iff b=0,c=1; fθ(P1)=1 iff b=1,c=1;
- fθ(Xij)=1 iff θ=(i,j).

Focal reward: S gives +0.2; any P action costs -0.1; Xij gives +1 if q=Xij and y=1, otherwise -2. Opponent utility can be 2·1[y=fθ(q)] + 0.5·1[a=q∈X and y=1]. Honest fixed response is strictly optimal, and the actual opponent action changes focal reward. Online opponent can learn Q(q,y) from its own rewards. This is a stateful simultaneous interaction, not a stateless arm-independent bandit, but a fixed opponent is still reducible to a POMDP; the online-learning condition is needed to test opponent adaptation.

One probe signal cannot identify four types in all cases. P then P_b identifies the type from two binary probe observations; with simultaneous delayed response, choose P at t=1, wait at t=2 to see b, choose P_b at t=3, observe c at t=4. This is one feasible identification policy, not necessarily the globally optimal policy. Any single probe yields at most one bit. An exploitation response can identify a lucky type earlier, but does not identify every type in one observation.

### Exact Gate-0 computation

Let B be the surviving subset of four equiprobable types, and let B_y={θ∈B:fθ(q)=y}. For h remaining turns:

V_h(q,B) = max_a Σ_y (|B_y|/|B|) [r(q,a,y)+V_{h-1}(a,B_y)], V_0=0.

At most 8×15×(T+1) belief-state-time tuples. Calculate with rational reward units (tenths). For T=24:

| Opening action | Exact expected total return, optimal continuation |
|---|---:|
| S | 16.775 |
| P | 17.450 |
| P0 or P1 | 17.575 |
| Any X | 15.675 |

Myopic opening S; Bayes-optimal opening P0/P1, margin 0.8 over S. Full blind policy best return 4.8 (all S), feedback value 12.775. Exact best K-turn open-loop prefix with all collected observations retrospectively disclosed at K, followed by optimal adaptive continuation: K=2:17.575; K=4:16.9; K=8:13.7. These are relaxed upper bounds for masked-prefix policies which cannot recover past observations. K=2 has no feedback-conditioning headroom due to observation delay; K=4 does. T=8 oracle return 1.75 vs all-safe 1.6; T=16 oracle 9.575 vs all-safe 3.2; T=32 oracle 25.575 vs all-safe 6.4. These are exact calculations of this candidate only, not empirical RL² results.

### R0–R3 and noise

To create matched controls, use common four-symbol opponent observation/action alphabet across regimes (adapt payoff only to the binary accept/reject semantics in exploitation states), R0 reveals θ in safe state, R1 reveals θ in a costly probe state, R3 binary partition as above. R2 can use discrete observation corruption with true payoff mechanism unchanged, or a reward-noise-only control. Re-run exact Gate-0 for each actual implementation; the numbers above verify only binary-response R3. Noise must be audited against the full joint observation. Full action-label permutation changes the prior and may destroy the strict opening margin; do not inherit gate results.

## 4. Audit definitions

- Opening necessity: Q*(prior,a*) - max_{a∈argmax E[r]}Q*(prior,a) > ε. Report fraction across task-family priors, with denominator and tolerance.
- Feedback value: V_adaptive - V_blind (or best K-open-loop, clearly specified). Full-blind and early-mask comparisons are different estimands.
- After common reset, commitment value C(b)=max_π∈Π_no-new-feedback E_b[R_{K+1:T}(π)]. For adaptive continuation V*(b), define opportunity gap V_clairvoyant(b)-V*(b). Use common post-prefix horizon; record prefix reward and net cost-inclusive return. Entropy is secondary.
- If finite hidden θ and all opponent Q updates are reconstructible from observed joint actions conditional on θ, exact posterior over θ may still be possible. Full-horizon optimization over continuous Q states is not guaranteed tractable. Keep exact gate on fixed reactive opponent (or explicitly finite-state learner); report bounded horizon approximate/limited diagnostics for online Q-learning.
- Restore vs continuous: pair tasks and seeds, train policies/exploiter with reset indicators; compare method-vs-baseline contrasts between protocols. This is intervention-specific state-mediated effect, not a universal additive causal decomposition. No claim based solely on counterpart reward.

## 5. Architecture and horizon recommendation

Standard GRU PPO+GAE for main study, fixed semantic action labels in core experiments. If testing unseen action permutations, use an equivariant architecture with tokenwise recurrent state/shared scoring and jointly transform actions, observations and hidden state. No positional embeddings leaking IDs; assert equivariance by permutation test. Compare under same budgets on dev; if permutation is hidden, Oracle must integrate it, not receive it. A full uniform relabeling can remove the distinct myopic safe action and increase hidden type support factorially. Do not make permutation generalization mandatory for core.

Primary objective undiscounted episode total reward, gamma=1 and lambda≈0.95 initially, with full episode sequences or correct recurrent burn-in/masking. T=24 primary, K=4 primary audit (K=2,8 diagnostic); T=16/32 sensitivity if warranted by Gate-0, not RL² advantage. Use pilot budget 2^18 transitions per trainable agent/seed; check wall time and learning curves on dev, lock final budget up to 2^20 if feasible. This is a planning proposal; actual throughput unmeasured.

## 6. Pre-registration and deadline

Primary question: at K=4 does RL² improve restored Bayes decision-relevant value over UCB and masked RL², with paired uncertainty and practical threshold? Net total return and early AUC answer the separate adaptation-benefit question. Report all distributions. Layer 2 corroborates model-free usability, Layer 3 diagnoses failures; do not require all metrics to win before admitting a measured information advantage. Avoid treating lack of significance as evidence of no effect.

Suggested core: R0/R1/R3 × fixed reactive opponents × RL²/Masked/Prior-Only (or Memory-1 clearly labeled)/UCB/Greedy/Random, plus Oracle diagnostics. Three trained agents × three regimes × five seeds = 45 fits; a separate mixed exploiter per regime adds 15 fits if retained. Use at least one structural baseline where feasible. R3 × online opponent as secondary (train matched policies if claiming learning-opponent generalization; otherwise clearly mark zero-shot transfer); R2 and H1 optional after core. Five training seeds are minimum, not five independent test episodes. Report cross-seed uncertainty. Do not make all R0–R3 × both opponent types × diversity controls × all architecture/audit ablations mandatory.

Milestones (proposal): Nov 27–Dec 1 lock definitions/implementation and exact gates; Dec 2–5 pilot and dev lock; Dec 6–11 sealed runs; Dec 12–15 analysis and video; Dec 16 buffer. Dec 17 video+quiz; Dec 20 report/code/slides. If throughput fails, remove H1/R2/Layer 2 before reducing essential independent seeds. No existing SPEC/PROGRESS mutation authorized by this review.

## References
- Duan et al., RL² (2016), https://arxiv.org/abs/1611.02779
- Russo & Van Roy, Learning to Optimize via Information-Directed Sampling, https://doi.org/10.1287/opre.2017.1663
- Planning to the Information Horizon of BAMDPs via Epistemic State Abstraction (2022), https://proceedings.neurips.cc/paper_files/paper/2022/hash/80b7bec60081f95d900973509744a306-Abstract-Conference.html
