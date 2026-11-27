"""Unprivileged Baselines & Privileged Diagnostic Ceilings for SPEC v3.0 (R0/R1/R3).

Unprivileged baselines (zero access to hidden theta or true f_theta):
- RandomPolicyV3
- GreedyPolicyV3 (empirical per-action greedy)
- ArmUCBPolicyV3 (stateless per-action UCB)
- ContextualUCBPolicyV3 (state-action (q, a) UCB)
- StructuralModelPolicyV3 (meta-trained on unlabeled offline rollouts via EM over latent
  response clusters + learned reward table; online Bayesian posterior + 2-step lookahead /
  Thompson or UCB action selection — zero hidden-type labels!)

Privileged diagnostic ceilings (never compared as baselines):
- BayesOptimalPolicyV3 (exact rational DP optimal action from belief mask)
- ClairvoyantPolicyV3 (exact rational DP optimal action knowing true theta)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import numpy as np
import torch
from torch.distributions import Categorical

from mamrl.agents.ppo_rl2 import RecurrentActorCritic, encode_obs_batch
from mamrl.envs.delayed_response import (
    ACT_S,
    N_ACTIONS,
    N_OPP_SYMBOLS,
    N_TYPES,
    VectorizedDelayedResponseEnv,
    focal_reward_rational,
)
from mamrl.eval.exact_dp import ExactDelayedDP, FULL_BELIEF_MASK


class BatchPolicyV3(ABC):
    """Unified batch-step interface for evaluating all policies identically."""

    name: str = "policy"
    is_privileged: bool = False
    offline_train_transitions: int = 0

    @abstractmethod
    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        ...

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        ...

    def last_hidden(self) -> np.ndarray | None:
        return None


class NeuralRL2PolicyV3(BatchPolicyV3):
    """Evaluation wrapper for trained RecurrentActorCritic (rl2 / rl2_masked / prior_only)."""

    def __init__(
        self,
        model: RecurrentActorCritic,
        *,
        variant: str,
        horizon: int = 24,
        mask_k: int = 4,
        deterministic: bool = False,
        offline_train_transitions: int = 0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.variant = variant
        self.name = variant
        self.horizon = horizon
        self.mask_k = mask_k
        self.deterministic = deterministic
        self.offline_train_transitions = int(offline_train_transitions)
        self.device = torch.device(device)
        self._h: torch.Tensor | None = None
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        self._h = torch.zeros((1, batch_size, self.model.hidden_dim), dtype=torch.float32, device=self.device)
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        assert self._h is not None
        x_np = encode_obs_batch(
            obs,
            horizon=self.horizon,
            step_t=step_t,
            variant=self.variant,
            mask_k=self.mask_k,
        )
        x_t = torch.from_numpy(x_np).to(self.device).unsqueeze(1)
        logits, _val, self._h = self.model.step(x_t, self._h)
        if self.deterministic:
            return torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Vectorized categorical draw using local numpy RNG for exact reproducibility
        cdf = np.cumsum(probs, axis=-1)
        u = self._rng.random((probs.shape[0], 1))
        acts = (u < cdf).argmax(axis=-1)
        return acts.astype(np.int64)

    def last_hidden(self) -> np.ndarray | None:
        if self._h is None:
            return None
        return self._h.squeeze(0).cpu().numpy().copy()


class RandomPolicyV3(BatchPolicyV3):
    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        self._batch = batch_size
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        return self._rng.integers(0, N_ACTIONS, size=self._batch, dtype=np.int64)


class GreedyPolicyV3(BatchPolicyV3):
    """Empirical per-action greedy baseline (zero prior knowledge, no exploration bonus)."""

    name = "greedy"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        self._batch = batch_size
        self._rng = np.random.default_rng(seed)
        self._sum = np.zeros((batch_size, N_ACTIONS), dtype=np.float64)
        self._cnt = np.zeros((batch_size, N_ACTIONS), dtype=np.float64)
        self._last_action = np.zeros(batch_size, dtype=np.int64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        if step_t > 0:
            prev_a = self._last_action
            prev_r = obs["prev_self_reward"]
            idx = np.arange(self._batch)
            self._sum[idx, prev_a] += prev_r
            self._cnt[idx, prev_a] += 1.0
        means = np.where(self._cnt > 0, self._sum / np.maximum(self._cnt, 1.0), 0.0)
        noise = self._rng.uniform(0.0, 1e-9, size=means.shape)
        acts = np.argmax(means + noise, axis=-1).astype(np.int64)
        self._last_action = acts.copy()
        return acts


class ArmUCBPolicyV3(BatchPolicyV3):
    """Stateless empirical arm-independent UCB baseline."""

    name = "arm_ucb"

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = float(beta)
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        self._batch = batch_size
        self._rng = np.random.default_rng(seed)
        self._sum = np.zeros((batch_size, N_ACTIONS), dtype=np.float64)
        self._cnt = np.zeros((batch_size, N_ACTIONS), dtype=np.float64)
        self._last_action = np.zeros(batch_size, dtype=np.int64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        if step_t > 0:
            prev_a = self._last_action
            prev_r = obs["prev_self_reward"]
            idx = np.arange(self._batch)
            self._sum[idx, prev_a] += prev_r
            self._cnt[idx, prev_a] += 1.0
        means = np.where(self._cnt > 0, self._sum / np.maximum(self._cnt, 1.0), 0.0)
        bonus = self.beta * np.sqrt(math.log(step_t + 2.0) / (self._cnt + 1.0))
        noise = self._rng.uniform(0.0, 1e-9, size=means.shape)
        acts = np.argmax(means + bonus + noise, axis=-1).astype(np.int64)
        self._last_action = acts.copy()
        return acts


class ContextualUCBPolicyV3(BatchPolicyV3):
    """Unprivileged state-action Contextual UCB over (q, a) pairs."""

    name = "contextual_ucb"

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = float(beta)
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        self._batch = batch_size
        self._rng = np.random.default_rng(seed)
        self._sum = np.zeros((batch_size, N_ACTIONS, N_ACTIONS), dtype=np.float64)
        self._cnt = np.zeros((batch_size, N_ACTIONS, N_ACTIONS), dtype=np.float64)
        self._prev_q = np.full(batch_size, ACT_S, dtype=np.int64)
        self._last_action = np.zeros(batch_size, dtype=np.int64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        idx = np.arange(self._batch)
        q_cur = obs["prev_self_action"]
        if step_t > 0:
            prev_r = obs["prev_self_reward"]
            self._sum[idx, self._prev_q, self._last_action] += prev_r
            self._cnt[idx, self._prev_q, self._last_action] += 1.0
        self._prev_q = q_cur.copy()

        s_q = self._sum[idx, q_cur]
        c_q = self._cnt[idx, q_cur]
        means = np.where(c_q > 0, s_q / np.maximum(c_q, 1.0), 0.0)
        bonus = self.beta * np.sqrt(math.log(step_t + 2.0) / (c_q + 1.0))
        noise = self._rng.uniform(0.0, 1e-9, size=means.shape)
        acts = np.argmax(means + bonus + noise, axis=-1).astype(np.int64)
        self._last_action = acts.copy()
        return acts


class StructuralModelPolicyV3(BatchPolicyV3):
    """Unprivileged meta-trained structural baseline (EM latent-cluster model + online lookahead).

    Fits ONLY on unlabeled offline trajectories (q, a, y, r) from the training regime
    (ZERO hidden theta labels):
    1. Learns empirical reward table r_hat(q, a, y).
    2. Fits an M=4 component discrete latent mixture P_m(y | q) + prior pi_m via EM.
    3. At test time, updates posterior b_t(m) from observed (q_{t-1}, y_t) and plans via
       short-horizon belief DP / Thompson-greedy over the LEARNED model (never the true spec).
    """

    name = "structural_model"

    def __init__(
        self,
        *,
        n_clusters: int = N_TYPES,
        horizon: int = 24,
        plan_depth: int = 4,
        em_iters: int = 25,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.horizon = int(horizon)
        self.plan_depth = int(plan_depth)
        self.em_iters = int(em_iters)
        self.pi_hat = np.full(self.n_clusters, 1.0 / self.n_clusters, dtype=np.float64)
        self.p_y_given_mq = np.full((self.n_clusters, N_ACTIONS, N_OPP_SYMBOLS), 1.0 / N_OPP_SYMBOLS, dtype=np.float64)
        self.r_hat = np.zeros((N_ACTIONS, N_ACTIONS, N_OPP_SYMBOLS), dtype=np.float64)
        self._rng = np.random.default_rng(0)

    def fit_offline(
        self,
        regime: str,
        *,
        n_episodes: int = 1024,
        seed: int = 0,
        opponent_factory=None,
    ) -> "StructuralModelPolicyV3":
        """Collect unlabeled random offline trajectories and fit empirical reward + EM response clusters."""
        rng = np.random.default_rng(seed)
        env = VectorizedDelayedResponseEnv(regime=regime, batch_size=n_episodes, horizon=self.horizon, seed=seed + 31)
        obs = env.reset()  # Notice: we NEVER read env.theta!
        opp = opponent_factory(n_episodes, seed + 53) if opponent_factory is not None else None
        if opp is not None:
            opp.reset(env.theta)

        q_hist = np.zeros((n_episodes, self.horizon), dtype=np.int64)
        a_hist = np.zeros((n_episodes, self.horizon), dtype=np.int64)
        y_hist = np.zeros((n_episodes, self.horizon), dtype=np.int64)
        r_hist = np.zeros((n_episodes, self.horizon), dtype=np.float64)

        for t in range(self.horizon):
            q_prev = obs["prev_self_action"].copy()
            q_hist[:, t] = q_prev
            acts = rng.integers(0, N_ACTIONS, size=n_episodes, dtype=np.int64)
            opp_a = opp.act(q_prev) if opp is not None else None
            obs, rf, ro, y = env.step(acts, opp_actions=opp_a)
            if opp is not None:
                opp.update(q_prev=q_prev, y_act=y, r_opp=ro, q_next=acts)
            a_hist[:, t] = acts
            y_hist[:, t] = y
            r_hist[:, t] = rf

        self.offline_train_transitions = int(n_episodes * self.horizon)

        # 1. Fit empirical reward table r_hat(q, a, y) from unlabeled transitions
        r_sum = np.zeros((N_ACTIONS, N_ACTIONS, N_OPP_SYMBOLS), dtype=np.float64)
        r_cnt = np.zeros((N_ACTIONS, N_ACTIONS, N_OPP_SYMBOLS), dtype=np.float64)
        np.add.at(r_sum, (q_hist, a_hist, y_hist), r_hist)
        np.add.at(r_cnt, (q_hist, a_hist, y_hist), 1.0)
        self.r_hat = np.where(r_cnt > 0, r_sum / np.maximum(r_cnt, 1.0), 0.0)

        # 2. Count (q, y) per unlabeled episode: counts[ep, q, y]
        ep_counts = np.zeros((n_episodes, N_ACTIONS, N_OPP_SYMBOLS), dtype=np.float64)
        ep_ids = np.repeat(np.arange(n_episodes), self.horizon)
        np.add.at(ep_counts, (ep_ids, q_hist.ravel(), y_hist.ravel()), 1.0)

        # 3. Multi-restart EM over M latent response models P_m(y | q) (no labels!)
        best_ll = -np.inf
        best_params = None
        for restart in range(5):
            r_rng = np.random.default_rng(seed + 1009 * (restart + 1))
            pi = np.full(self.n_clusters, 1.0 / self.n_clusters, dtype=np.float64)
            p_mqy = r_rng.dirichlet(np.ones(N_OPP_SYMBOLS), size=(self.n_clusters, N_ACTIONS))

            ll = -np.inf
            for _ in range(self.em_iters):
                log_p = np.log(np.clip(p_mqy, 1e-9, 1.0))  # (M, Q, Y)
                # E-step: log_lik[ep, m] = log(pi_m) + sum_{q,y} count[ep,q,y] * log P_m(y|q)
                log_lik = np.log(np.clip(pi, 1e-9, 1.0))[None, :] + np.einsum("eqy,mqy->em", ep_counts, log_p)
                max_ll = np.max(log_lik, axis=1, keepdims=True)
                lse = max_ll + np.log(np.sum(np.exp(log_lik - max_ll), axis=1, keepdims=True))
                ll = float(np.sum(lse))
                resp = np.exp(log_lik - lse)  # (E, M)

                # M-step with tiny Laplace smoothing (1e-3)
                pi = (np.sum(resp, axis=0) + 1e-2) / (n_episodes + self.n_clusters * 1e-2)
                weighted = np.einsum("em,eqy->mqy", resp, ep_counts) + 1e-3
                p_mqy = weighted / np.sum(weighted, axis=-1, keepdims=True)

            if ll > best_ll:
                best_ll = ll
                best_params = (pi.copy(), p_mqy.copy())

        assert best_params is not None
        self.pi_hat, self.p_y_given_mq = best_params

        # Precompute optimal per-cluster action values V_m(q) AND cross-cluster evaluation table
        self._v_cluster = np.zeros((self.n_clusters, self.horizon + 1, N_ACTIONS), dtype=np.float64)
        exp_r_all = np.einsum("mqy,qay->mqa", self.p_y_given_mq, self.r_hat)  # (M, Q, A)
        for m in range(self.n_clusters):
            for h in range(1, self.horizon + 1):
                q_vals = exp_r_all[m] + self._v_cluster[m, h - 1][None, :]
                self._v_cluster[m, h] = np.max(q_vals, axis=-1)

        # Precompute _cluster_policy_ret[target_m, rem, q_start, true_m] via DP (ultra-fast O(M^2*H*Q))
        self._cluster_policy_ret = np.zeros(
            (self.n_clusters, self.horizon + 1, N_ACTIONS, self.n_clusters), dtype=np.float64
        )
        for target_m in range(self.n_clusters):
            for h in range(1, self.horizon + 1):
                a_stars = np.argmax(exp_r_all[target_m] + self._v_cluster[target_m, h - 1][None, :], axis=-1)
                for q in range(N_ACTIONS):
                    a_s = int(a_stars[q])
                    self._cluster_policy_ret[target_m, h, q] = (
                        exp_r_all[:, q, a_s] + self._cluster_policy_ret[target_m, h - 1, a_s]
                    )

        return self

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        self._batch = batch_size
        self._rng = np.random.default_rng(seed)
        self._belief = np.tile(self.pi_hat[None, :], (batch_size, 1))
        self._prev_q = np.full(batch_size, ACT_S, dtype=np.int64)

    def _learned_voI_lookahead(self, q: int, belief: np.ndarray, rem: int) -> int:
        """2-step lookahead using ONLY the learned EM model + learned cluster continuation V_m."""
        # Expected immediate reward per action under current belief over learned clusters
        p_y_q = belief @ self.p_y_given_mq[:, q, :]  # (Y,)
        # Update belief for current observation y_now? Note y_now arrives AT THE END of this step!
        # Wait: when choosing a_t at state q, we also observe y_t ~ P_m(. | q) and transition to q' = a_t.
        # Compute exact expected value of action a_t with 1-step posterior update at q' = a_t,
        # plus 1 more adaptive probe step using learned cluster value continuation V_m(rem-2).
        best_a = ACT_S
        best_val = -1e18
        for a in range(N_ACTIONS):
            imm = float(np.sum(p_y_q * self.r_hat[q, a, :]))
            if rem <= 1:
                val = imm
            else:
                cont = 0.0
                for y in range(N_OPP_SYMBOLS):
                    py = float(p_y_q[y])
                    if py < 1e-9:
                        continue
                    b_next = belief * self.p_y_given_mq[:, q, y]
                    b_next /= float(np.sum(b_next))
                    # At next state q' = a, we observe y' ~ P_m(. | a) and then commit/continue
                    p_y2 = b_next @ self.p_y_given_mq[:, a, :]
                    best_a2 = -1e18
                    for a2 in range(N_ACTIONS):
                        imm2 = float(np.sum(p_y2 * self.r_hat[a, a2, :]))
                        if rem <= 2:
                            v2 = imm2
                        else:
                            # Expected value after observing y2 at state a:
                            # use max over a3 (commitment) or QMDP bound with entropy penalty
                            c2 = 0.0
                            for y2 in range(N_OPP_SYMBOLS):
                                py2 = float(p_y2[y2])
                                if py2 < 1e-9:
                                    continue
                                b2 = b_next * self.p_y_given_mq[:, a, y2]
                                b2 /= float(np.sum(b2))
                                # Conservative commitment value under learned clusters
                                # (best single-cluster policy evaluated under posterior b2)
                                v_commit = float(
                                    np.max(
                                        [
                                            float(np.sum(b2 * self._eval_cluster_policy(m_star, a2, rem - 2)))
                                            for m_star in range(self.n_clusters)
                                        ]
                                    )
                                )
                                c2 += py2 * v_commit
                            v2 = imm2 + c2
                        if v2 > best_a2:
                            best_a2 = v2
                    cont += py * best_a2
                val = imm + cont
            if val > best_val + 1e-12:
                best_val = val
                best_a = a
        return best_a

    def _eval_cluster_policy(self, target_m: int, q_start: int, rem: int) -> np.ndarray:
        """Expected return for each true cluster m if following target_m's greedy policy from q_start."""
        if rem <= 0:
            return np.zeros(self.n_clusters, dtype=np.float64)
        return self._cluster_policy_ret[target_m, rem, q_start]

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        q_cur = obs["prev_self_action"]
        if step_t > 0:
            prev_y = obs["prev_opp_action"]
            # Vectorized Bayesian update over learned clusters using observed (prev_q, prev_y)
            lik = self.p_y_given_mq[:, self._prev_q, prev_y].T  # (B, M)
            post = self._belief * lik
            denom = np.sum(post, axis=1, keepdims=True)
            self._belief = np.where(denom > 1e-12, post / np.maximum(denom, 1e-12), self._belief)
        self._prev_q = q_cur.copy()

        rem = self.horizon - step_t
        # Exact vectorized 2-step belief lookahead (exploiting degree-1 homogeneity of max expected return)
        exp_r_all = np.einsum("mqy,qay->mqa", self.p_y_given_mq, self.r_hat)  # (M, Q, A)
        exp_r_q = np.transpose(exp_r_all[:, q_cur, :], (1, 0, 2))  # (B, M, A)
        imm1 = np.einsum("bm,bma->ba", self._belief, exp_r_q)  # (B, A)
        if rem <= 1:
            return np.argmax(imm1, axis=-1).astype(np.int64)

        p_obs_q = np.transpose(self.p_y_given_mq[:, q_cur, :], (1, 2, 0))  # (B, Y, M)
        w1 = self._belief[:, None, :] * p_obs_q  # (B, Y, M)
        imm2 = np.einsum("bym,maq->byaq", w1, exp_r_all)  # (B, Y, A1, A2)
        if rem <= 2:
            v2 = imm2
        else:
            # w2[b, y, a1, y2, m] = w1[b, y, m] * P_m(y2 | a1)
            w2 = w1[:, :, None, None, :] * np.transpose(self.p_y_given_mq, (1, 2, 0))[None, None, :, :, :]
            # commit_ret[target_m, a2, true_m]
            cret = self._cluster_policy_ret[:, rem - 2, :, :]  # (M_t, A2, M)
            # val_commit[b, y, a1, y2, a2, target_m]
            val_c = np.einsum("byazm,tqm->byazqt", w2, cret)
            c2 = np.sum(np.max(val_c, axis=-1), axis=3)  # (B, Y, A1, A2)
            v2 = imm2 + c2

        cont1 = np.sum(np.max(v2, axis=-1), axis=1)  # (B, A1)
        return np.argmax(imm1 + cont1, axis=-1).astype(np.int64)


class BayesOptimalPolicyV3(BatchPolicyV3):
    """Privileged exact Bayes-optimal DP ceiling (starts from uniform prior over true theta)."""

    name = "bayes_optimal"
    is_privileged = True

    def __init__(self, regime: str, horizon: int = 24) -> None:
        self.regime = regime
        self.horizon = int(horizon)
        self.dp = ExactDelayedDP(regime=regime, horizon=horizon)
        self._f_table = self.dp._f_table
        self._policy_table: dict[tuple[int, int, int], int] = {}
        self._precompute_policy()

    def _precompute_policy(self) -> None:
        for h in range(1, self.horizon + 1):
            for q in range(N_ACTIONS):
                for mask in range(1, 1 << N_TYPES):
                    q_vals = self.dp.opening_q_values(horizon=h, q=q, belief_mask=mask)
                    best_a = max(range(N_ACTIONS), key=lambda a: (q_vals[a], -a))
                    self._policy_table[(h, q, mask)] = best_a

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        self._batch = batch_size
        self._masks = np.full(batch_size, FULL_BELIEF_MASK, dtype=np.int64)
        self._prev_q = np.full(batch_size, ACT_S, dtype=np.int64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        q_cur = obs["prev_self_action"]
        if step_t > 0:
            prev_y = obs["prev_opp_action"]
            for i in range(self._batch):
                m = int(self._masks[i])
                pq = int(self._prev_q[i])
                py = int(prev_y[i])
                new_m = 0
                for th in range(N_TYPES):
                    if ((m >> th) & 1) and int(self._f_table[th, pq]) == py:
                        new_m |= 1 << th
                if new_m != 0:
                    self._masks[i] = new_m
        self._prev_q = q_cur.copy()

        rem = self.horizon - step_t
        return np.array(
            [self._policy_table[(rem, int(q_cur[i]), int(self._masks[i]))] for i in range(self._batch)],
            dtype=np.int64,
        )


class ClairvoyantPolicyV3(BatchPolicyV3):
    """Privileged clairvoyant ceiling (knows true theta from turn 0)."""

    name = "clairvoyant"
    is_privileged = True

    def __init__(self, regime: str, horizon: int = 24) -> None:
        self.regime = regime
        self.horizon = int(horizon)
        self._bayes = BayesOptimalPolicyV3(regime=regime, horizon=horizon)

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        if true_theta is None:
            raise ValueError("ClairvoyantPolicyV3 requires true_theta.")
        self._batch = batch_size
        self._masks = (1 << np.asarray(true_theta, dtype=np.int64))

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        q_cur = obs["prev_self_action"]
        rem = self.horizon - step_t
        return np.array(
            [self._bayes._policy_table[(rem, int(q_cur[i]), int(self._masks[i]))] for i in range(self._batch)],
            dtype=np.int64,
        )
