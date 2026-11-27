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

### Core Evaluated Policies (5 Independent Training Seeds for All Learned Agents)
1. **`rl2` (Primary Meta-Learner):** Standard GRU Actor-Critic trained with **PPO + GAE** ($\gamma = 1.0$, $\lambda \approx 0.95$, full episode sequences, undiscounted total return). Input: `(one_hot(prev_a), one_hot(prev_y), prev_r, is_start, reset_flag, turn_frac)`.
2. **`rl2_masked` (Early-Masked Control):** Identical GRU + PPO+GAE trained **under its own mask** for $t < K_{\text{audit}}$ (hiding opponent action & reward for $t < K$, **no retrospective revelation**; begins observing live feedback at $t \ge K$).
3. **`prior_only` (True Zero-Shot Blind Control):** Identical training budget with **no feedback inputs at any turn** (time/self-action only). *(Distinct from Memory-1 reset, which retains 1-step feedback.)*
4. **`contextual_ucb` (State-Action UCB Baseline):** Unprivileged UCB over `(q, a)` pairs, hyperparameter tuned strictly on dev seeds.
5. **`structural_model` (Meta-Trained Structural Baseline):** Unprivileged empirical transition/reward model $\hat{P}(y \mid q), \hat{r}(q,a,y)$ fitted on the **same offline meta-training rollouts** (no `spec` access), paired with online posterior/Thompson or UCB action selection during test episodes.
6. **`arm_ucb`, `greedy`, `random`:** Standard stateless/myopic reference baselines.

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
| **Core (Mandatory)** | `R0, R1, R3` × Fixed Reactive Opponent × 5 seeds (`rl2`, `rl2_masked`, `prior_only`, `contextual_ucb`, `structural_model`, `arm_ucb`, `greedy`, `random`, `bayes_optimal`, `clairvoyant`) | 3 trained agents × 3 regimes × 5 seeds = **45 fits** (`2^18` pilot → up to `2^20` locked). Must complete & verify first. |
| **Secondary (Multi-Agent Non-Stationarity)** | `R3` × Online Q-Learning Opponent (both zero-shot transfer of fixed-trained `rl2` AND matched training on online Q-learning opponent; Restored vs. Continuous with visible reset indicator + Counterpart Return) | 15 fits (matched) + zero-shot eval. Bounded/approximate diagnostics (no false exact full-horizon Oracle claim). |
| **Optional (Only If Time Permits)** | `R2` (categorical observation noise), `Layer 2` (cross-fitted mixed exploiter), `H1` diversity (fixed budget vs. fixed exposure) | Drop before ever cutting independent training seeds below 5. |

### Milestones
- **Nov 27 – Dec 1:** Lock SPEC v3.0, implement R0/R1/R3 + exact rational DP Gate 0 + PPO-GAE + baselines, pass unit tests.
- **Dec 2 – Dec 5:** Dev pilots (`2^18` transitions), verify learning curves & wall-clock throughput, lock hyperparameters.
- **Dec 6 – Dec 11:** Sealed multi-seed Core + Secondary runs & independent verification.
- **Dec 12 – Dec 15:** Figures, 5-page NeurIPS report draft, slides, 5-min video & 3 quiz questions.
- **Dec 16:** Buffer & `<5 MB` code zip audit (**Dec 17 23:59** Video+Quiz; **Dec 20 23:59** Report+Code+Slides).
