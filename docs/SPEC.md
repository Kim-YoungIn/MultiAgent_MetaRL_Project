# Experimental Design Specification — Locked v3.0

> **Status:** Adopted & locked (incorporating independent review `handoff/reviews/20261127_spec_v2_2_independent_review.md`).
> **Guiding Principle:** Unbiased inquiry anchored strictly to the core research question. Positive, conditional, and null/inconclusive outcomes are treated with equal scientific value. No environment tuning solely to force Meta-RL to win.

---

## 1. Core Research Question & Two-Part Structure (Path A)

> **멀티에이전트 반복 게임 환경에서 메타 러닝이 탐색(exploration)을 개선하여 새로운 게임 규칙과 학습 상대에 대한 빠른 적응(fast adaptation)에 어떤 영향을 미치는가?**

- **Part 1 (Completed — `docs/PRIOR_FINDINGS.md`, git tag `archive/v0.9-legacy-studies`):**
  15 historical studies showed meta-learned exploration increased *mean* return (`+0.446 / +0.428`) while lowering the *median* (`−0.144 / −0.142`, win-rate 40%) and losing to UCB—because passive/greedy play already leaked task identity (only 6/96 pairs required changing the opening move), and long-horizon gains were confounded with opponent shaping (`+0.365` counterpart return).
- **Part 2 (Locked Core + Secondary Study):**
  Matched multi-agent information-structure study (`R0`, `R1`, `R3`) using unprivileged End-to-End `RL²` (Recurrent PPO+GAE) and rigorous decision-value decomposition to test *whether and under what information structures meta-learning changes exploration and adaptation*.

---

## 2. Benchmark Environment: Simultaneous Delayed-Response Game (`R0 / R1 / R3`)

Implemented in `src/mamrl/envs/delayed_response.py` (exact scalar + vectorized batch simulator).

### Shared Structure Across All Regimes
- **Focal Action Space ($n = 8$, fixed semantic labels in Core):**
  $A = \{S, P, P_0, P_1, X_{00}, X_{01}, X_{10}, X_{11}\}$.
- **Common Opponent Action/Observation Alphabet ($|\mathcal{Y}| = 4$):** $y \in \{0, 1, 2, 3\}$.
- **Hidden Opponent Type:** $\theta = (b, c) \in \{0, 1\}^2$ (4 equiprobable types, indexed $2b + c \in \{0, 1, 2, 3\}$).
- **Public State:** $q_t = a_{t-1}$ (previous focal action), initialized at $q_0 = S$.
- **Simultaneous Move Protocol:** At turn $t = 1 \dots T$, focal chooses $a_t$ and opponent chooses $y_t$ simultaneously. Opponent observes $q_{t-1}$ (not $a_t$). Next state $q_t = a_t$.
- **Focal Reward $r_F(q_{t-1}, a_t, y_t)$ (identical across R0/R1/R3):**
  - $a_t = S \Rightarrow +0.2$
  - $a_t \in \{P, P_0, P_1\} \Rightarrow -0.1$
  - $a_t \in \{X_{00}, X_{01}, X_{10}, X_{11}\} \Rightarrow +1.0$ if $(q_{t-1} = a_t \land y_t = 1)$ else $-2.0$.
- **Opponent Utility $r_O(q_{t-1}, a_t, y_t; \theta)$:**
  $2.0 \cdot \mathbb{1}[y_t = f_\theta(q_{t-1})] + 0.5 \cdot \mathbb{1}[a_t = q_{t-1} \in X \land y_t = 1]$.
  Honest fixed response $y_t = f_\theta(q_{t-1})$ is strictly incentive-compatible; online opponents learn $Q(q, y)$ from $r_O$.
- **Horizon & Audit Windows:** Primary $T = 24$, primary prefix audit $K = 4$ (diagnostics $K \in \{2, 8\}$; $T \in \{16, 32\}$ sensitivity if warranted by Gate 0).

### Matched Information Regimes (Only $f_\theta(q)$ Changes)

| Regime | Opponent Response Function $f_\theta(q)$ | Information Structure Tested |
|---|---|---|
| **R0: Passive Immediate** | $f_\theta(S) = \theta \in \{0,1,2,3\}$; $f_\theta(P)=f_\theta(P_0)=f_\theta(P_1)=0$; $f_\theta(X_{ij})=\mathbb{1}[\theta=(i,j)]$ | Safe opening $S$ passively reveals all 2 bits of $\theta$. Active probing unnecessary (`must_change = False`). |
| **R1: Active 1-Step** | $f_\theta(S)=0$; $f_\theta(P) = \theta \in \{0,1,2,3\}$; $f_\theta(P_0)=f_\theta(P_1)=0$; $f_\theta(X_{ij})=\mathbb{1}[\theta=(i,j)]$ | Costly opening $P$ required (`must_change = True`), but 1 open-loop probe reveals all 2 bits (no feedback branching needed). |
| **R3: Active 2-Step Adaptive** | $f_\theta(S)=0$; $f_\theta(P)=b$; $f_\theta(P_0)=\mathbb{1}[b=0,c=1]$; $f_\theta(P_1)=\mathbb{1}[b=1,c=1]$; $f_\theta(X_{ij})=\mathbb{1}[\theta=(i,j)]$ | Each probe yields $\le 1$ bit; strictly requires **both** active opening (`P0/P1` over `S`) **and** feedback-conditioned adaptive probing at $K=4$. |

*(Optional **R2: Categorical Observation Noise** applies discrete symbol-flip noise to $y$ while preserving finite belief DP, kept outside Core unless Core + Secondary finish early.)*

---

## 3. Agents, Baselines & Unprivileged Fairness Rules

**Zero test-task `spec` leakage:** No evaluated agent receives $\theta$ or $f_\theta$. Offline training transitions and online test interactions are reported separately.

### Core Evaluated Policies (5 Independent Sealed Training Seeds: `1001..1005`)
1. **`rl2_ppo` (On-Policy Pure-Reward `RL²`):** Standard GRU Actor-Critic trained with **PPO + GAE** ($\gamma = 1.0$, $\lambda \approx 0.95$, full episode sequences, undiscounted total return, zero auxiliary losses).
2. **`drqn_vanilla` (Off-Policy Pure-Reward `RL²`):** Identical GRU-64 trained with Double-DQN + Polyak target updates, zero auxiliary losses.
3. **Self-Supervised Auxiliary Signal Ablations (`drqn_aux_y`, `drqn_aux_r`, `drqn_aux_yr`):** Identical unprivileged GRU-64 + Double-DQN adding self-supervised prediction of public `(next_y)`, `(next_r)`, or both `(next_y, next_r)` — directly testing *which* auxiliary signal unlocks active exploration.
4. **`drqn_aux_yr_masked` / `rl2_masked` (Early-Masked Controls):** Identical architecture trained **under its own mask** for $t < K_{\text{audit}}$ (hiding opponent action & reward, **no retrospective revelation**, auxiliary loss also masked for $t < K_{\text{audit}}$).
5. **`drqn_prior_only` / `prior_only` (True Zero-Shot Blind Controls):** Identical training budget with **no feedback inputs at any turn** (and no auxiliary feedback loss).
6. **`structural_model` (Meta-Trained Structural Baseline):** Unprivileged empirical EM latent-cluster model $\hat{P}_m(y \mid q), \hat{r}(q,a,y)$ fitted on unlabeled offline rollouts (no `spec` access), paired with online belief lookahead.
7. **`contextual_ucb`, `arm_ucb`, `greedy`, `random`:** Strictly unprivileged online baselines.

### Diagnostic Ceilings (Privileged — Never Compared as Baselines)
- **`bayes_optimal`:** Exact rational DP policy starting from uniform prior over $\theta \in \{0,1,2,3\}$ (true achievable ceiling under fixed reactive opponent).
- **`clairvoyant`:** Knows true $\theta$ from turn 0 (measures total irreducible uncertainty cost).

---

## 4. Exact Gate-0 & Decision-Value Decomposition Audits

Implemented in `src/mamrl/eval/exact_dp.py` using exact rational arithmetic (`fractions.Fraction`):

1. **Gate-0 Exact Verification (Mandatory Before Training):**
   - **Opening Necessity:** $Q^*(\text{prior}, a^*) - \max_{a \in \arg\max E[r]} Q^*(\text{prior}, a) > \varepsilon$.
   - **Feedback Value:** $V_{\text{adaptive}} - V_{\text{blind}}$ and $V_{\text{adaptive}} - V_{\text{best } K\text{-open-loop}}$ at $K \in \{2, 4, 8\}$.
   - Exact rational verification runs on all actual regime implementations (`R0`, `R1`, `R3`) and is asserted in `tests/test_delayed_response.py`.
2. **Layer 1 — Primary Exploration Audit (Policy-Agnostic Exact Bayes Decision Value):**
   Given the realized first-$K$ trajectory $\tau_{1:K}$ collected by each policy (producing exact posterior $b(\tau_{1:K})$ over $\theta$):
   - **Restored Commitment Value:** $C(b) = \max_{\pi \in \Pi_{\text{blind}}} \mathbb{E}_b[R_{K+1:T}(\pi)]$ from common reset state $q = S$.
   - **Restored Adaptive Continuation Value:** $V^*(b)$ (Bayes-optimal remaining value from reset $q = S$, horizon $T - K$), plus opportunity gap $V_{\text{clairvoyant}}(b) - V^*(b)$.
   - **Prefix Cost & Net Return:** Prefix reward $R_{1:K}$ and net cost-inclusive return reported separately alongside posterior entropy $H(\theta \mid \tau_{1:K})$ (secondary).
3. **Layer 3 — Non-OOD Exploitation Efficiency Audit:**
   - **No OOD Oracle-prefix injection into on-policy `RL²`.** Instead, on each policy's *own realized histories*, compute the continuation gap $V^*(q_K, b(\tau_{1:K})) - R_{K+1:T}^{\text{actual}}$ (zero OOD shift!).
   - Post-hoc linear probe on `RL²` hidden states $h_t$ (trained/evaluated on disjoint rollouts) reported strictly as diagnostic, not causal proof.
4. **Layer 2 (Optional Secondary Corroboration):** Cross-fitted frozen Mixed-Prefix Neutral Exploiter trained on all evaluated prefix generators (including reset indicators), kept secondary to Layer 1.

---

## 5. Statistical Protocol & Pre-Registered Primary Endpoints

- **Independent Statistical Unit:** Training seed ($N \ge 5$ independent seeds per trained agent; paired hierarchical bootstrap over seeds/test episodes; ties reported separately).
- **Separate Estimands (Not Conflated):**
  1. **Information Acquisition (Primary Exploration Endpoint):** At $K = 4$, does `rl2` improve restored Bayes decision-relevant value ($C(b)$ and $V^*(b)$) over `contextual_ucb`, `arm_ucb`, and `rl2_masked` beyond a pre-declared practical threshold ($\delta_{\text{info}} = 0.25$ total return units over $T - K = 20$ turns)?
  2. **Net Adaptation Benefit:** End-to-end $T=24$ Total Return (Mean, **Median**, IQR, Win-Rate) and Early AUC.
  3. **Feedback Conditioning:** `rl2` vs. `rl2_masked` (and vs. `prior_only`).
  4. **Null vs. Inconclusive:** Distinguish statistically resolved null (95% CI wholly within $[-\delta, +\delta]$) from inconclusive (CI straddles threshold).

---

## 6. Scoped Execution Plan for 2026-12-17 / 12-20 Deadlines

| Tier | Matrix | Compute & Priority |
|---|---|---|
| **Core (Exp 1 — Mandatory)** | `R0, R1, R3` (`d=2` bits, 4 types) × Fixed Reactive Opponent × 5 seeds (`rl2`, `rl2_masked`, `prior_only`, `contextual_ucb`, `structural_model`, `arm_ucb`, `greedy`, `random`, `bayes_optimal`, `clairvoyant`) | 3 trained agents × 3 regimes × 5 seeds = **45 fits** (`2^18` dev pilot → up to `2^20` locked). Runs in minutes on local CPU multiprocessing. |
| **Secondary A (Exp 2A — Multi-Agent Non-Stationarity)** | `R3` × Online Q-Learning / Unseen Expected SARSA Opponents (zero-shot transfer + matched training; Restored vs. Continuous with visible reset indicator + Counterpart Return) | 15 fits (matched) + zero-shot eval. Bounded/approximate diagnostics (no false exact full-horizon Oracle claim). |
| **Secondary B (Exp 2B — Symmetric Multi-Agent Competition & Cooperation + Unseen Rules)** | Symmetric 2-player 4-action repeated games (hidden cyclic-dominance competition + hidden coordination/trap mixed-motive, strictly held-out unseen test rules) comparing **`Non-Meta vs. Non-Meta`**, **`Meta vs. Non-Meta`**, and **`Meta vs. Meta`** matchups | Directly validates true multi-agent competition/cooperation & unseen-game generalization without altering Exp 1. |
| **Priority Extension (Post-Core/Secondary — Auxiliary Task Discovery)** | **"보조 목표 탐색 (Auxiliary Task Discovery / Meta-Learning Auxiliary Objectives)"** (Veeriah et al. 2019; Zheng et al. 2018): meta-learning *which* auxiliary prediction objectives or learned loss gates are useful rather than hand-selecting `pred_y`/`pred_r` | Top-priority extension once Exp 1 (aux ablation baseline) and Exp 2A/2B are cleanly sealed and verified. |
| **Optional (Only If Time Permits)** | `R2` (categorical observation noise), `Layer 2` (cross-fitted mixed exploiter), `H1` exposure-vs-budget split | Drop before ever cutting independent training seeds below 5. |

### Milestones
- **Nov 27 – Dec 1:** Lock SPEC v3.0, implement R0/R1/R3 + exact rational DP Gate 0 + PPO-GAE + baselines + logging, pass unit tests.
- **Dec 2 – Dec 5:** Dev pilots (`2^18` transitions), verify learning curves & wall-clock throughput, lock hyperparameters.
- **Dec 6 – Dec 11:** Sealed multi-seed Core (Exp 1) + Secondary (Exp 2A/2B) runs & independent verification.
- **Dec 12 – Dec 15:** Figures, 5-page NeurIPS report draft, slides, 5-min video & 3 quiz questions.
- **Dec 16:** Buffer & `<5 MB` code zip audit (**Dec 17 23:59** Video+Quiz; **Dec 20 23:59** Report+Code+Slides).

---

## 7. Logging, Measurement & Statistical Verification Contract

### A. Artifact & Recording Layout (Local Raw `runs/` vs. Git-Tracked `results/`)
- **Local Raw (`runs/<study_id>/`, gitignored, resumable):**
  - `manifest.json`: full config, git commit, Python/Torch/NumPy versions, Gate-0 summary hash, seed list, UTC timestamp.
  - `fits/<regime>_<method>_seed<S>/`:
    - `receipt.json` (atomic `tmp -> rename`): offline train transitions, online eval turns, wall time, checkpoint & eval SHA-256.
    - `train_curve.csv`: periodic updates (`mean_return`, `median_return`, `policy_entropy`, `approx_kl`, `clip_frac`, `value_loss`, `explained_var`, opening action distribution, exploration/exploitation repetition rates).
    - `checkpoint.pt`: final model weights.
    - `eval_episodes.npz`: turn-level trajectories (`theta[E]`, `actions[E,T]`, `opp_actions[E,T]`, `focal_rewards[E,T]`, `opp_rewards[E,T]`) for zero-retrain independent audits (~150 KB/fit).
  - `summary.json` & `tables/*.csv`: aggregated metrics, paired contrasts, Layer-1 `K=2,4,8` audits, `by_theta` breakdowns.
- **Git-Tracked (`results/<study_id>/`, `< 5 MB` budget):** Only sealed `manifest_summary.json`, `summary.json`, compact CSV tables, and publication figures. Never commit checkpoints or raw `.npz` files.
- **Dev vs. Sealed Rule:** `runs/dev_*` is strictly for hyperparameter/throughput validation and never cited as final evidence.

### B. Mandatory 4-Axis Measurements
1. **Optimization Health:** Training return curves, policy entropy (collapse check), PPO KL/clip/explained variance, offline vs. online interaction budgets reported separately.
2. **End-to-End Distribution & Speed ($T=24$, balanced $\theta$, paired episodes):** Mean, Median, IQR, Bayes-Opt gap, **Win / Tie / Loss rates** (ties separate), turn-by-turn reward curve, Early/Mid/Late AUC, Time-to-Threshold (+ right-censored rate), and per-type (`by_theta`) breakdown.
3. **Behavioral & Repetition Diagnostics:** Phase-separated action distributions (`explore 1..K` vs `exploit K+1..T`: `S, P, P0, P1, X_match, X_mismatch`), **action repetition rates** (`repeat_rate_explore`, `repeat_rate_exploit`, `repeat_rate_all`), `unique_actions_explore`, `top_prefix_sequences`, `opening_probe_rate`, and **`k4_branching_tv`** (TV distance of turn-3 action distribution conditioned on $y_2=0$ vs $y_2=1$).
4. **Exploration vs. Exploitation Decomposition (`K ∈ {2, 4, 8}`, primary `K=4`) & 4-Way Statistical Verdict:**
   - **Layer 1:** `restored_commitment_value` $C(b)$, `restored_adaptive_value` $V^*(b)$, `restored_opportunity_gap`, `prefix_reward`, net cost-inclusive restored returns, posterior entropy & rule ID rate.
   - **Layer 3 (Non-OOD):** `live_continuation_gap` $= V^*(q_K, b(\tau_{1:K})) - R_{K+1:T}^{\text{actual}}$ + disjoint-split `hidden_state_linear_probe_acc` (diagnostic only).
   - **Hierarchical Paired Bootstrap (10,000 resamples, training seed as independent unit)** with pre-declared thresholds ($\delta_{\text{info}} = 0.25$, $\delta_{\text{total}} = 0.30$) classifying every primary contrast into `SUPPORTED_POS`, `SUPPORTED_NEG`, `RESOLVED_NULL`, or `INCONCLUSIVE`.


---

## 8. Post-Interim Extensions, Environment Design Principles & Active Roadmap (v3.2)

### Completed Diagnostics & Redesign Baseline
- **Exp 5A (`results/exp5a_gate0_and_surprise/`):** PBVI corrupted-channel Gate-0 (`d=0.6` achievable `12.00`, `VoI >= +7.20`); eval-only `pred_y` surprise switch (falsified escalation); Exp 2B/3B exact Gate-0 VoI (`100%` positive VoI on `coop_trap` & Exp 3B suites, `0` on `comp_rps`); 22-action Exp 4 `structural_model_mb` arbitration (`5/5` seeds unlock Tier 3, mean `84.84`, median `120.55`).
- **Exp 4B (`results/exp4b_compact_multibarrier/`):** `R3+Distractor` gate-validity check + compact 6-action 3-layer Stage-Lock (`src/mamrl/envs/stage_lock.py`, `T=48`, `|A|=6`, Gate-0 `9.6→30.9→60.1→90.1`). Recurrent Bellman target history bug fixed in `study_exp4b_compact.py` (`compute_recurrent_double_dqn_targets`), pending clean re-seal.

### Environment Design Principle (Adopted 2026-11-28)
- **No 1-to-1 Action-per-Attribute Enumeration:** Do not design future benchmarks by listing a separate dedicated action button for each hidden opponent attribute (e.g., `X00..X11`, `T3_000..T3_111`).
- **Mathematically Constructed Game Families + Pre-Built Exact Oracles:** Define reusable game-theoretic repeated-game families over compact shared action spaces, paired with exact mathematical Oracles / Bayes-Optimal DP verified *before* training.
- **Horizon & Scale Target:** Scale episode length (`T=48~64`) and multi-stage strategic depth without flat action-space explosion.

### Active Exploration Roadmap (7 Major Items — Pre-Freeze)
1. **`U1` Multi-Family × Auxiliary Task Discovery:** Combine Exp 3B's multi-family equivariant setup with auxiliary discovery (global gate vs context-conditioned dynamic gate $w(h_t)$) to test whether mixed families enable per-family tool discovery or cause interference.
2. **`U2` Multi-Barrier × Online Learning Opponents:** Combine multi-stage exploration barriers with non-stationary learning opponents where structure-informed fixed models break down.
3. **`U3` Auxiliary Discovery Mechanism Deepening:** Compare true bilevel meta-gradient vs local cosine alignment and richer public auxiliary candidates (multi-step predictions, reward deltas, counterpart rewards).
4. **`U4` Noise-Matched Training for OOD Calibration:** Train on $d \in \{0, 0.3\}$ and test held-out $d=0.6$ to separate support coverage from structural OOD overconfidence.
5. **`P1` Exp 3B Representation & Feedback Controls:** State-matched (`GRU-128`) × permutation-augmented `One-Hot` + `Equivariant` masked/prior controls (+ optional `D-031` `n2_b2`).
6. **`P2` Exp 4B Bug-Fixed Re-Seal & Probe Attribution:** Re-run fixed `study_exp4b_compact.py` (with Exp-1-matched `R3+distractor` hyperparams) + eval-only `P`-before-exploit probe attribution.
7. **`P3` Exp 1 `rl2_ppo` Budget Match (`691k`) & `Expected SARSA` Reconciliation:** Match `rl2_ppo` transition budget on `R3` and align Diag 1B vs Exp 2A SARSA configs.

### Adopted STOP Decisions
- **STOP** GRU-surprise retraining (Diag 1B failed escalation criterion vs `prior_only` and degraded $d=0.3$).
- **STOP** further 22-action Exp 4 interventions (Diag 1D + Exp 4B already isolated data sufficiency vs credit assignment).
- **STOP** further `comp_rps` expansion (frozen as exact `VoI=0` zero-sum negative control).
