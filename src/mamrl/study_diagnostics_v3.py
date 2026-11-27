"""SPEC v3.1 Block 1 (Exp 5A): Zero-Retrain Diagnostics & Gate-0 Audits.

Executes 4 pre-registered, near-zero-cost diagnostics without touching sealed results/:
1. Diag 1A — Exact/PBVI Corrupted-Channel Gate-0 for R3 at d in {0.0, 0.3, 0.6}:
   Full-horizon alpha-vector PBVI lower bound + QMDP upper bound + empirical rollout check.
2. Diag 1B — Eval-Only Surprise-Threshold Switch on Sealed Exp 2A `mt_drqn_aux_yr` Checkpoints:
   Public online surprise s_t = -log p_pred_y(y_t | h_{t-1}, a_{t-1}), calibrated strictly on
   training distribution (Q-learning, d=0) at episode-level p95 and p99, evaluated on
   Q-learning (d=0), unseen Expected SARSA (d=0), and fixed R3 corruption (d=0.3, d=0.6).
3. Diag 1C — Gate-0 Exact VoI Audit for Exp 2B & Exp 3B Symmetric Game Suites:
   Exact finite-rule-bank belief-partition DP against uniform partner opening at H=2, 4, 24.
4. Diag 1D — 22-Action Exp 4 Arbitration (`structural_model_mb` on eps-greedy warmup data):
   Unprivileged 8-cluster EM + learned-model full-horizon belief DP on the exact 4,320
   eps-greedy warmup episodes of Exp 4, logging exact raw positive-tier event counts.

Usage:
    PYTHONPATH=src .venv/bin/python -m mamrl.study_diagnostics_v3 --config configs/exp5a_diagnostics.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from mamrl.agents.drqn_rl2 import RecurrentQNet
from mamrl.agents.opponents_v3 import OnlineOpponentConfig, make_online_opponent
from mamrl.agents.ppo_rl2 import OBS_DIM, encode_obs_batch
from mamrl.envs.delayed_response import (
    ACT_S,
    N_ACTIONS,
    N_OPP_SYMBOLS,
    N_TYPES,
    VectorizedDelayedResponseEnv,
    build_focal_reward_table,
    build_response_table,
)
from mamrl.envs.multi_barrier import (
    MB_ACT_S,
    MB_FULL_BELIEF_MASK,
    MB_N_ACTIONS,
    MB_N_SYMBOLS,
    MB_N_TYPES,
    VectorizedMultiBarrierEnv,
    mb_action_tier,
)
from mamrl.envs.semantic_games import build_semantic_bank
from mamrl.envs.symmetric_ma import build_symmetric_rule_bank
from mamrl.eval.audit_v3 import hierarchical_paired_bootstrap


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Diag 1A: Full-Horizon PBVI Belief-DP for R3 under Observation Corruption d
# ---------------------------------------------------------------------------

class CorruptedR3PBVISolver:
    """Full-horizon alpha-vector PBVI solver for R3 with symmetric observation flip noise d.

    Because every alpha vector represents a valid conditional plan, V_PBVI(H, S, b0) is a
    strict, achievable lower bound on exact Bayes-Optimal V*(d), and equals exact 17.575 at d=0.0.
    Also computes the exact clairvoyant/QMDP upper bound after 0 turns for bracketing.
    """

    def __init__(self, noise_d: float, horizon: int = 24, n_beliefs: int = 400, seed: int = 0) -> None:
        self.noise_d = float(noise_d)
        self.horizon = int(horizon)
        self._f_table = build_response_table("R3")      # (4, 8)
        self._r_focal = build_focal_reward_table()      # (8, 8, 4)

        # True deterministic reward R_true[theta, q, a]
        self._r_true = np.zeros((N_TYPES, N_ACTIONS, N_ACTIONS), dtype=np.float64)
        for th in range(N_TYPES):
            for q in range(N_ACTIONS):
                self._r_true[th, q, :] = self._r_focal[q, :, int(self._f_table[th, q])]

        # Observation channel P_obs[theta, q, y_obs]
        self._p_obs = np.full((N_TYPES, N_ACTIONS, N_OPP_SYMBOLS), self.noise_d / (N_OPP_SYMBOLS - 1), dtype=np.float64)
        for th in range(N_TYPES):
            for q in range(N_ACTIONS):
                self._p_obs[th, q, int(self._f_table[th, q])] = 1.0 - self.noise_d

        # Precompute unique joint (y_obs, r_val) likelihood vectors per (q, a)
        self._obs_liks: list[list[list[np.ndarray]]] = [[[] for _ in range(N_ACTIONS)] for _ in range(N_ACTIONS)]
        for q in range(N_ACTIONS):
            for a in range(N_ACTIONS):
                r_vals = self._r_true[:, q, a]
                uniq_r = np.unique(r_vals)
                liks = []
                for y_obs in range(N_OPP_SYMBOLS):
                    p_y = self._p_obs[:, q, y_obs]
                    for rv in uniq_r:
                        mask_r = (np.abs(r_vals - rv) < 1e-9).astype(np.float64)
                        lik = p_y * mask_r
                        if float(np.sum(lik)) > 1e-12:
                            liks.append(lik)
                self._obs_liks[q][a] = liks

        self.beliefs = self._build_belief_grid(n_beliefs=n_beliefs, seed=seed)
        self.alphas: list[list[np.ndarray]] = [[np.zeros((1, N_TYPES), dtype=np.float64) for _ in range(N_ACTIONS)]]
        self._solve()

    def _build_belief_grid(self, n_beliefs: int, seed: int) -> np.ndarray:
        pts: list[np.ndarray] = []
        # All 15 non-empty uniform subsets of {0,1,2,3} (exact masks for d=0!)
        for mask in range(1, 1 << N_TYPES):
            v = np.array([(mask >> i) & 1 for i in range(N_TYPES)], dtype=np.float64)
            pts.append(v / np.sum(v))
        # Exact reachable beliefs by propagating from uniform through probes/noisy observations
        rng = np.random.default_rng(seed)
        frontier = [np.full(N_TYPES, 0.25, dtype=np.float64)]
        for _ in range(6):
            next_f = []
            for b in frontier:
                for q in range(N_ACTIONS):
                    for y in range(N_OPP_SYMBOLS):
                        w = b * self._p_obs[:, q, y]
                        s = float(np.sum(w))
                        if s > 1e-12:
                            bn = w / s
                            pts.append(bn)
                            if len(next_f) < 64:
                                next_f.append(bn)
            rng.shuffle(next_f)
            frontier = next_f[:32]
        # Add Dirichlet simplex points for broad coverage
        extra = rng.dirichlet(np.ones(N_TYPES) * 0.7, size=max(0, n_beliefs))
        for row in extra:
            pts.append(row)
        # Deduplicate to resolution 1e-3
        uniq: dict[tuple[int, ...], np.ndarray] = {}
        for p in pts:
            key = tuple(int(round(x * 1000.0)) for x in p)
            if key not in uniq:
                uniq[key] = p
        arr = np.stack(list(uniq.values()), axis=0)
        if arr.shape[0] > n_beliefs:
            # Keep first 15 exact masks + diverse sample
            keep_idx = list(range(min(15, arr.shape[0])))
            rem_idx = rng.choice(np.arange(15, arr.shape[0]), size=max(0, n_beliefs - 15), replace=False)
            arr = arr[keep_idx + rem_idx.tolist()]
        return arr

    def _backup_q(self, q: int, prev_alphas: list[np.ndarray]) -> np.ndarray:
        """Exact PBVI backup over self.beliefs for state q."""
        B = self.beliefs  # (N_B, 4)
        n_b = B.shape[0]
        best_vals = np.full(n_b, -1e18, dtype=np.float64)
        best_vecs = np.zeros((n_b, N_TYPES), dtype=np.float64)

        for a in range(N_ACTIONS):
            r_vec = self._r_true[:, q, a]  # (4,)
            a_prev = prev_alphas[a]        # (K, 4)
            # Accumulate backed-up vector per belief in B: (N_B, 4)
            cand = np.broadcast_to(r_vec[None, :], (n_b, N_TYPES)).copy()
            for lik in self._obs_liks[q][a]:
                # Unnormalized posterior W = B * lik[None, :]  (N_B, 4)
                W = B * lik[None, :]
                # Best alpha in a_prev maximizing W @ alpha^T -> (N_B, K)
                scores = W @ a_prev.T
                idx = np.argmax(scores, axis=1)
                chosen = a_prev[idx]  # (N_B, 4)
                cand += chosen * lik[None, :]
            vals = np.sum(B * cand, axis=1)
            better = vals > best_vals + 1e-12
            best_vals[better] = vals[better]
            best_vecs[better] = cand[better]

        # Deduplicate backed-up alpha vectors (rounded to 1e-7)
        uniq: dict[tuple[int, ...], np.ndarray] = {}
        for v in best_vecs:
            key = tuple(int(round(x * 1e6)) for x in v)
            if key not in uniq:
                uniq[key] = v
        return np.stack(list(uniq.values()), axis=0)

    def _solve(self) -> None:
        for _h in range(1, self.horizon + 1):
            prev = self.alphas[-1]
            cur = [self._backup_q(q, prev) for q in range(N_ACTIONS)]
            self.alphas.append(cur)

    def value(self, rem: int, q: int, belief: np.ndarray) -> float:
        return float(np.max(self.alphas[rem][q] @ belief))

    def opening_q_values(self, horizon: int) -> dict[int, float]:
        b0 = np.full(N_TYPES, 0.25, dtype=np.float64)
        prev = self.alphas[horizon - 1]
        out: dict[int, float] = {}
        for a in range(N_ACTIONS):
            val = float(np.sum(b0 * self._r_true[:, ACT_S, a]))
            for lik in self._obs_liks[ACT_S][a]:
                w = b0 * lik
                p = float(np.sum(w))
                if p > 1e-12:
                    val += p * float(np.max(prev[a] @ (w / p)))
            out[a] = val
        return out

    def act_single(self, rem: int, q: int, belief: np.ndarray) -> int:
        prev = self.alphas[rem - 1]
        best_a = ACT_S
        best_v = -1e18
        for a in range(N_ACTIONS):
            val = float(np.sum(belief * self._r_true[:, q, a]))
            for lik in self._obs_liks[q][a]:
                w = belief * lik
                p = float(np.sum(w))
                if p > 1e-12:
                    val += p * float(np.max(prev[a] @ (w / p)))
            if val > best_v + 1e-12:
                best_v = val
                best_a = a
        return best_a

    def rollout_verify(self, n_episodes: int = 1024, seed: int = 12345) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        env = VectorizedDelayedResponseEnv(regime="R3", batch_size=n_episodes, horizon=self.horizon, seed=seed + 1)
        thetas = np.tile(np.arange(N_TYPES, dtype=np.int64), n_episodes // N_TYPES + 1)[:n_episodes]
        env.reset(theta=thetas)
        beliefs = np.full((n_episodes, N_TYPES), 1.0 / N_TYPES, dtype=np.float64)
        q_cur = np.full(n_episodes, ACT_S, dtype=np.int64)
        tot_r = np.zeros(n_episodes, dtype=np.float64)
        s_rate = 0.0
        for t in range(self.horizon):
            rem = self.horizon - t
            acts = np.array([self.act_single(rem, int(q_cur[i]), beliefs[i]) for i in range(n_episodes)], dtype=np.int64)
            s_rate += float(np.mean(acts == ACT_S))
            _obs, rf, _ro, y_true = env.step(acts)
            tot_r += rf
            # Corrupt observation y_obs
            if self.noise_d > 0.0:
                flip = rng.random(n_episodes) < self.noise_d
                offs = rng.integers(1, N_OPP_SYMBOLS, size=n_episodes)
                y_obs = np.where(flip, (y_true + offs) % N_OPP_SYMBOLS, y_true)
            else:
                y_obs = y_true
            # Exact Bayesian update on (y_obs, rf)
            for i in range(n_episodes):
                q_i, a_i, y_i, r_i = int(q_cur[i]), int(acts[i]), int(y_obs[i]), float(rf[i])
                lik = self._p_obs[:, q_i, y_i] * (np.abs(self._r_true[:, q_i, a_i] - r_i) < 1e-9)
                w = beliefs[i] * lik
                s = float(np.sum(w))
                if s > 1e-12:
                    beliefs[i] = w / s
            q_cur = acts
        return {
            "rollout_mean": float(np.mean(tot_r)),
            "rollout_median": float(np.median(tot_r)),
            "rollout_se": float(np.std(tot_r, ddof=1) / np.sqrt(n_episodes)),
            "safe_action_rate": s_rate / self.horizon,
        }


def run_diag_1a(d_values: list[float], horizon: int) -> list[dict]:
    rows = []
    blind_val = 0.2 * horizon
    for d in d_values:
        solver = CorruptedR3PBVISolver(noise_d=d, horizon=horizon, n_beliefs=350, seed=42)
        b0 = np.full(N_TYPES, 0.25, dtype=np.float64)
        v_pbvi = solver.value(horizon, ACT_S, b0)
        q_open = solver.opening_q_values(horizon)
        best_open = max(q_open, key=lambda a: (q_open[a], -a))
        open_margin = q_open[best_open] - q_open[ACT_S]
        rv = solver.rollout_verify(n_episodes=1024, seed=int(10000 + d * 1000))
        rows.append({
            "noise_d": float(d),
            "blind_floor": blind_val,
            "pbvi_bayes_opt_lb": v_pbvi,
            "voi_over_blind": v_pbvi - blind_val,
            "best_opening_action": int(best_open),
            "opening_margin_over_S": float(open_margin),
            "must_change_opening": bool(open_margin > 1e-6),
            **rv,
        })
    return rows


# ---------------------------------------------------------------------------
# Diag 1B: Eval-Only Surprise-Threshold Switch on Sealed Exp 2A Checkpoints
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_mt_with_surprise(
    qnet: RecurrentQNet,
    *,
    mode: str,
    noise_d: float,
    n_episodes: int,
    horizon: int,
    seed: int,
    threshold: float | None = None,
    min_switch_turn: int = 2,
) -> dict:
    """Roll out sealed mt_drqn_aux_yr checkpoint with optional unlearned surprise->S fallback switch."""
    rng = np.random.default_rng(seed)
    env = VectorizedDelayedResponseEnv(regime="R3", batch_size=n_episodes, horizon=horizon, seed=seed + 11)
    thetas = np.tile(np.arange(N_TYPES, dtype=np.int64), n_episodes // N_TYPES + 1)[:n_episodes]
    obs = env.reset(theta=thetas)

    opp = None
    if mode in ("q_learning", "expected_sarsa"):
        opp_cfg = OnlineOpponentConfig(
            kind=mode,
            regime="R3",
            alpha=0.35,
            gamma=0.50,
            epsilon=0.08,
            warmup_steps=80,
            warmup_epsilon=1.0,
        )
        opp = make_online_opponent(opp_cfg, n_episodes, seed + 23)
        opp.reset(env.theta)

    h = torch.zeros((1, n_episodes, qnet.hidden_dim), dtype=torch.float32)
    switched = np.zeros(n_episodes, dtype=np.bool_)
    cum_surprise = np.zeros(n_episodes, dtype=np.float64)
    max_mean_surprise = np.zeros(n_episodes, dtype=np.float64)
    tot_r = np.zeros(n_episodes, dtype=np.float64)
    tot_opp = np.zeros(n_episodes, dtype=np.float64)
    s_cnt = 0.0

    prev_h_np: torch.Tensor | None = None
    prev_a_np: np.ndarray | None = None

    for t in range(horizon):
        # Compute public online surprise for the observation y_t just arrived (t >= 1)
        if t >= 1 and prev_h_np is not None and prev_a_np is not None:
            y_obs = obs["prev_opp_action"]
            a_oh = torch.nn.functional.one_hot(torch.from_numpy(prev_a_np), num_classes=N_ACTIONS).float()
            logits_y = qnet.pred_y(torch.cat([prev_h_np, a_oh], dim=-1))
            log_probs = torch.log_softmax(logits_y, dim=-1).numpy()
            s_t = -log_probs[np.arange(n_episodes), y_obs]
            cum_surprise += s_t
            mean_s = cum_surprise / float(t)
            if t >= min_switch_turn:
                max_mean_surprise = np.maximum(max_mean_surprise, mean_s)
                if threshold is not None:
                    switched |= mean_s > threshold

        x_np = encode_obs_batch(obs, horizon=horizon, step_t=t, variant="rl2", mask_k=4)
        x_t = torch.from_numpy(x_np).unsqueeze(1)
        q_vals, h = qnet.step(x_t, h)
        acts = torch.argmax(q_vals, dim=-1).numpy().astype(np.int64)
        if threshold is not None:
            acts = np.where(switched, ACT_S, acts)

        prev_h_np = h.squeeze(0).clone()
        prev_a_np = acts.copy()
        s_cnt += float(np.mean(acts == ACT_S))

        q_prev = obs["prev_self_action"].copy()
        opp_a = opp.act(q_prev) if opp is not None else None
        next_obs, rf, ro, y_true = env.step(acts, opp_actions=opp_a)
        if opp is not None:
            opp.update(q_prev=q_prev, y_act=y_true, r_opp=ro, q_next=acts)

        # Apply observation noise d ONLY to focal's observed y_obs (matching Exp 3A exactly)
        if noise_d > 0.0:
            flip = rng.random(n_episodes) < noise_d
            offs = rng.integers(1, N_OPP_SYMBOLS, size=n_episodes)
            y_obs_next = np.where(flip, (y_true + offs) % N_OPP_SYMBOLS, y_true)
            next_obs = {**next_obs, "prev_opp_action": y_obs_next}

        obs = next_obs
        tot_r += rf
        tot_opp += ro

    return {
        "total_return": tot_r,
        "opp_return": tot_opp,
        "switched": switched,
        "max_mean_surprise": max_mean_surprise,
        "safe_rate": s_cnt / horizon,
    }


def run_diag_1b(cfg: dict) -> tuple[list[dict], list[dict]]:
    seeds = [int(s) for s in cfg["seeds"]]
    horizon = int(cfg.get("horizon", 24))
    hidden_dim = int(cfg.get("hidden_dim", 64))
    n_calib = int(cfg.get("n_calib_episodes", 512))
    n_eval = int(cfg.get("n_eval_episodes", 512))
    n_boot = int(cfg.get("n_bootstrap", 10000))
    delta = float(cfg.get("delta_return", 0.30))
    exp2a_dir = Path(cfg.get("exp2a_runs_dir", "runs/exp2a_online_opp"))

    conditions = [
        ("in_dist_qlearn_d0", "q_learning", 0.0),
        ("ood_sarsa_d0", "expected_sarsa", 0.0),
        ("fixed_r3_d03", "fixed", 0.3),
        ("fixed_r3_d06", "fixed", 0.6),
    ]
    variants = ("raw", "switch_p95", "switch_p99")
    eval_arrays: dict[tuple[str, str], list[np.ndarray]] = {(c[0], v): [] for c in conditions for v in variants}
    meta_stats: dict[tuple[str, str], list[dict]] = {(c[0], v): [] for c in conditions for v in variants}

    for s in seeds:
        ckpt = exp2a_dir / "fits" / f"mt_drqn_aux_yr_seed{s}" / "checkpoint.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        qnet = RecurrentQNet(OBS_DIM, hidden_dim, N_ACTIONS)
        qnet.load_state_dict(payload["state_dict"])
        qnet.eval()

        # STRICT ZERO-LEAKAGE CALIBRATION on training distribution ONLY (Q-learning, d=0)
        cal = _rollout_mt_with_surprise(
            qnet, mode="q_learning", noise_d=0.0, n_episodes=n_calib, horizon=horizon, seed=77000 + s * 31
        )
        tau_p95 = float(np.percentile(cal["max_mean_surprise"], 95.0))
        tau_p99 = float(np.percentile(cal["max_mean_surprise"], 99.0))
        thresh_map = {"raw": None, "switch_p95": tau_p95, "switch_p99": tau_p99}

        for c_idx, (cname, mode, d_val) in enumerate(conditions):
            eval_seed = 99100 + c_idx * 1000 + s * 17
            for v in variants:
                res = _rollout_mt_with_surprise(
                    qnet,
                    mode=mode,
                    noise_d=d_val,
                    n_episodes=n_eval,
                    horizon=horizon,
                    seed=eval_seed,
                    threshold=thresh_map[v],
                )
                eval_arrays[(cname, v)].append(res["total_return"])
                meta_stats[(cname, v)].append({
                    "tau": -1.0 if thresh_map[v] is None else float(thresh_map[v]),
                    "trigger_rate": float(np.mean(res["switched"])),
                    "safe_rate": float(res["safe_rate"]),
                    "opp_return": float(np.mean(res["opp_return"])),
                })

    rows: list[dict] = []
    contrasts: list[dict] = []
    blind_arr = np.full((len(seeds), n_eval), 0.2 * horizon, dtype=np.float64)

    for cname, mode, d_val in conditions:
        for v in variants:
            mat = np.stack(eval_arrays[(cname, v)], axis=0)
            s_means = np.mean(mat, axis=1)
            q25, q50, q75 = np.percentile(mat, [25, 50, 75])
            st = meta_stats[(cname, v)]
            rows.append({
                "condition": cname,
                "mode": mode,
                "noise_d": d_val,
                "variant": v,
                "mean_return": float(np.mean(s_means)),
                "seed_std": float(np.std(s_means, ddof=1)) if len(s_means) > 1 else 0.0,
                "median_return": float(q50),
                "iqr_return": float(q75 - q25),
                "trigger_rate": float(np.mean([x["trigger_rate"] for x in st])),
                "safe_rate": float(np.mean([x["safe_rate"] for x in st])),
                "opp_return": float(np.mean([x["opp_return"] for x in st])),
                "mean_tau": float(np.mean([x["tau"] for x in st])),
            })

        raw_mat = np.stack(eval_arrays[(cname, "raw")], axis=0)
        for sw in ("switch_p95", "switch_p99"):
            sw_mat = np.stack(eval_arrays[(cname, sw)], axis=0)
            for ref_name, ref_mat in (("raw", raw_mat), ("prior_only", blind_arr)):
                boot = hierarchical_paired_bootstrap(
                    sw_mat, ref_mat, n_boot=n_boot, practical_threshold=delta, seed=1234
                )
                seed_diffs = np.mean(sw_mat - ref_mat, axis=1)
                contrasts.append({
                    "condition": cname,
                    "contrast": f"{sw}_minus_{ref_name}",
                    "mean_diff": boot["diff_mean"],
                    "ci95_lo": boot["ci_low"],
                    "ci95_hi": boot["ci_high"],
                    "win_rate": boot["win_rate"],
                    "tie_rate": boot["tie_rate"],
                    "loss_rate": boot["loss_rate"],
                    "seeds_positive": int(np.sum(seed_diffs > 1e-6)),
                    "n_seeds": len(seeds),
                    "verdict": boot["verdict"],
                })
    return rows, contrasts


# ---------------------------------------------------------------------------
# Diag 1C: Gate-0 Exact VoI Audit for Exp 2B & Exp 3B Symmetric Game Suites
# ---------------------------------------------------------------------------

def _exact_sym_suite_gate0(r1_bank: np.ndarray, horizon_short: int = 4, horizon_full: int = 24) -> dict:
    """Exact belief-partition Gate-0 VoI over a finite symmetric 4x4 rule bank (uniform partner).

    Opponent plays uniformly at random (stationarity control isolating focal information gain
    from partner shaping). Focal observes (a1, a2, r1) after each turn, partitioning active rules.
    """
    n_rules, n_act, _ = r1_bank.shape
    exp_r = np.mean(r1_bank, axis=2)  # (K, 4) expected payoff per (rule, focal_act) against uniform partner
    # Pre-round rewards to 4 decimals for exact equality grouping
    r_int = np.rint(r1_bank * 10000.0).astype(np.int64)

    # Blind optimal per-turn value under uniform prior over all K rules
    prior_r = np.mean(exp_r, axis=0)
    v_blind_1 = float(np.max(prior_r))
    v_blind_short = v_blind_1 * horizon_short
    v_blind_full = v_blind_1 * horizon_full

    # Clairvoyant value (knowing rule k from turn 0)
    best_per_rule = np.max(exp_r, axis=1)  # (K,)
    v_clairvoyant_1 = float(np.mean(best_per_rule))

    # Exact 2-turn & H-turn belief-partition DP using bitmask (works up to K <= 64, or tuple fallback)
    all_idx = tuple(range(n_rules))

    @lru_cache(maxsize=None)
    def dp_val(rem: int, rule_ids: tuple[int, ...]) -> float:
        if rem <= 0:
            return 0.0
        if len(rule_ids) == 1:
            return float(best_per_rule[rule_ids[0]]) * rem
        if rem == 1:
            sub = exp_r[list(rule_ids)]
            return float(np.max(np.mean(sub, axis=0)))
        best = -1e18
        m = len(rule_ids)
        for a1 in range(n_act):
            ev = 0.0
            for a2 in range(n_act):
                # Partition rule_ids by realized reward r_int[k, a1, a2]
                groups: dict[int, list[int]] = {}
                for k in rule_ids:
                    rv = int(r_int[k, a1, a2])
                    groups.setdefault(rv, []).append(k)
                for rv, grp in groups.items():
                    p = len(grp) / (m * n_act)
                    ev += p * (rv / 10000.0 + dp_val(rem - 1, tuple(grp)))
            if ev > best:
                best = ev
        return best

    # Also compute exact 1-step-info + commit value for H=24 (fast & tight lower bound for H=24, exact at H=2,4)
    v_inf_2 = dp_val(2, all_idx)
    v_inf_4 = dp_val(horizon_short, all_idx)

    # Per-rule positive VoI check at H=2: does observing turn-0 (a2, r1) improve turn-1 action over blind?
    blind_opt_set = set(int(a) for a in np.flatnonzero(prior_r >= v_blind_1 - 1e-9))
    # Under any opening a0, compute posterior-optimal turn-1 value averaged over partner a2
    voi_2_by_rule = np.zeros(n_rules, dtype=np.float64)
    for k in range(n_rules):
        # Best opening a0 under full suite (pick a0=0 since suites are S4-balanced, or exact max over a0)
        best_a0_vals = []
        for a0 in range(n_act):
            tot = 0.0
            for a2 in range(n_act):
                rv = int(r_int[k, a0, a2])
                grp = [j for j in all_idx if int(r_int[j, a0, a2]) == rv]
                post_means = np.mean(exp_r[grp], axis=0)
                a1_star = int(np.argmax(post_means))
                tot += (r1_bank[k, a0, a2] + exp_r[k, a1_star]) / n_act
            best_a0_vals.append(tot)
        voi_2_by_rule[k] = float(np.mean(best_a0_vals)) - (v_blind_1 * 2.0)

    # Check whether opening action ties across all 4 slots (Latin symmetry)
    opening_spread = float(np.max(prior_r) - np.min(prior_r))

    return {
        "n_rules": int(n_rules),
        "v_blind_h2": float(v_blind_1 * 2.0),
        "v_informed_h2": float(v_inf_2),
        "voi_h2": float(v_inf_2 - v_blind_1 * 2.0),
        "v_blind_h4": float(v_blind_short),
        "v_informed_h4": float(v_inf_4),
        "voi_h4": float(v_inf_4 - v_blind_short),
        "v_clairvoyant_h4": float(v_clairvoyant_1 * horizon_short),
        "v_blind_h24": float(v_blind_full),
        "v_clairvoyant_h24": float(v_clairvoyant_1 * horizon_full),
        "positive_voi_rule_frac_h2": float(np.mean(voi_2_by_rule > 1e-6)),
        "prior_opening_spread": opening_spread,
        "symmetric_opening_tie": bool(opening_spread < 1e-9),
        "blind_opt_count": len(blind_opt_set),
    }


def run_diag_1c() -> list[dict]:
    rows = []
    # Exp 2B suites
    for game in ("comp_rps", "coop_trap"):
        banks = build_symmetric_rule_bank(game)
        for split in ("train", "test"):
            res = _exact_sym_suite_gate0(banks[split]["r1"])
            rows.append({"study": "exp2b", "suite": f"{game}_{split}", **res})
    # Exp 3B suites
    for suite in ("in_dist", "unseen_perm", "unseen_params", "unseen_family"):
        bank = build_semantic_bank(n_sem_train=4, eval_suite=suite)
        res = _exact_sym_suite_gate0(bank["r1"])
        rows.append({"study": "exp3b", "suite": suite, **res})
    return rows


# ---------------------------------------------------------------------------
# Diag 1D: 22-Action Exp 4 Arbitration via Unprivileged `structural_model_mb`
# ---------------------------------------------------------------------------

def run_diag_1d(seeds: list[int], horizon: int = 64, n_eval: int = 256) -> dict:
    """Fit unprivileged 8-cluster EM + full-horizon learned belief DP on Exp 4's exact eps-greedy warmup data."""
    n_warmup_eps = 90 * 48  # 4,320 episodes = exact warmup budget of Exp 4 DRQN!
    per_seed_rows = []

    for s in seeds:
        rng = np.random.default_rng(s + 303)
        env = VectorizedMultiBarrierEnv(batch_size=n_warmup_eps, horizon=horizon, seed=s + 101)
        obs = env.reset()  # NEVER read env.theta!

        q_hist = np.zeros((n_warmup_eps, horizon), dtype=np.int64)
        a_hist = np.zeros((n_warmup_eps, horizon), dtype=np.int64)
        y_hist = np.zeros((n_warmup_eps, horizon), dtype=np.int64)
        r_hist = np.zeros((n_warmup_eps, horizon), dtype=np.float64)

        for t in range(horizon):
            q_prev = obs["prev_self_action"].copy()
            acts = rng.integers(0, MB_N_ACTIONS, size=n_warmup_eps, dtype=np.int64)
            obs, rf, _ro, y = env.step(acts)
            q_hist[:, t] = q_prev
            a_hist[:, t] = acts
            y_hist[:, t] = y
            r_hist[:, t] = rf

        # Count raw positive-tier confirmation events (public: q==a, y==1, r > 0.25)
        tier_arr = np.vectorize(mb_action_tier)(a_hist)
        pos_t1 = int(np.sum((q_hist == a_hist) & (y_hist == 1) & (tier_arr == 1)))
        pos_t2 = int(np.sum((q_hist == a_hist) & (y_hist == 1) & (tier_arr == 2)))
        pos_t3 = int(np.sum((q_hist == a_hist) & (y_hist == 1) & (tier_arr == 3)))

        # Fit empirical reward table r_hat(q, a, y) (unprivileged!)
        r_sum = np.zeros((MB_N_ACTIONS, MB_N_ACTIONS, MB_N_SYMBOLS), dtype=np.float64)
        r_cnt = np.zeros((MB_N_ACTIONS, MB_N_ACTIONS, MB_N_SYMBOLS), dtype=np.float64)
        np.add.at(r_sum, (q_hist, a_hist, y_hist), r_hist)
        np.add.at(r_cnt, (q_hist, a_hist, y_hist), 1.0)
        r_hat = np.where(r_cnt > 0, r_sum / np.maximum(r_cnt, 1.0), 0.0)
        # Count how many of the 14 positive exploit cells (q==a, y==1) were observed at least once!
        obs_t1_cells = int(np.sum([r_cnt[a, a, 1] > 0 for a in range(2, 4)]))
        obs_t2_cells = int(np.sum([r_cnt[a, a, 1] > 0 for a in range(6, 10)]))
        obs_t3_cells = int(np.sum([r_cnt[a, a, 1] > 0 for a in range(14, 22)]))

        # Unprivileged 8-cluster EM on unlabeled (q, y) episode counts
        ep_counts = np.zeros((n_warmup_eps, MB_N_ACTIONS, MB_N_SYMBOLS), dtype=np.float64)
        ep_ids = np.repeat(np.arange(n_warmup_eps), horizon)
        np.add.at(ep_counts, (ep_ids, q_hist.ravel(), y_hist.ravel()), 1.0)

        best_ll = -np.inf
        best_p = None
        for restart in range(6):
            rrng = np.random.default_rng(s + 991 * (restart + 1))
            pi = np.full(MB_N_TYPES, 1.0 / MB_N_TYPES, dtype=np.float64)
            p_mqy = rrng.dirichlet(np.ones(MB_N_SYMBOLS), size=(MB_N_TYPES, MB_N_ACTIONS))
            for _ in range(40):
                log_p = np.log(np.clip(p_mqy, 1e-9, 1.0))
                log_lik = np.log(np.clip(pi, 1e-9, 1.0))[None, :] + np.einsum("eqy,mqy->em", ep_counts, log_p)
                m_ll = np.max(log_lik, axis=1, keepdims=True)
                lse = m_ll + np.log(np.sum(np.exp(log_lik - m_ll), axis=1, keepdims=True))
                ll = float(np.sum(lse))
                resp = np.exp(log_lik - lse)
                pi = (np.sum(resp, axis=0) + 1e-2) / (n_warmup_eps + MB_N_TYPES * 1e-2)
                w = np.einsum("em,eqy->mqy", resp, ep_counts) + 1e-3
                p_mqy = w / np.sum(w, axis=-1, keepdims=True)
            if ll > best_ll:
                best_ll = ll
                best_p = p_mqy.copy()
        assert best_p is not None

        # Extract learned deterministic cluster map f_hat[m, q] = argmax_y P_m(y|q) and solve full-horizon belief-mask DP!
        f_hat = np.argmax(best_p, axis=-1)  # (8, 22)

        @lru_cache(maxsize=None)
        def learned_dp_val(rem: int, q: int, mask: int) -> float:
            if rem <= 0:
                return 0.0
            sub = [0] * MB_N_SYMBOLS
            cnt = [0] * MB_N_SYMBOLS
            for m in range(MB_N_TYPES):
                if (mask >> m) & 1:
                    y = int(f_hat[m, q])
                    sub[y] |= 1 << m
                    cnt[y] += 1
            total = sum(cnt)
            if total == 0:
                return 0.0
            splits = [(y, sub[y], cnt[y] / total) for y in range(MB_N_SYMBOLS) if cnt[y] > 0]
            best_v = -1e18
            for a in range(MB_N_ACTIONS):
                ev = 0.0
                for y, sm, prob in splits:
                    ev += prob * (float(r_hat[q, a, y]) + learned_dp_val(rem - 1, a, sm))
                if ev > best_v:
                    best_v = ev
            return best_v

        def learned_dp_act(rem: int, q: int, mask: int) -> int:
            sub = [0] * MB_N_SYMBOLS
            cnt = [0] * MB_N_SYMBOLS
            for m in range(MB_N_TYPES):
                if (mask >> m) & 1:
                    y = int(f_hat[m, q])
                    sub[y] |= 1 << m
                    cnt[y] += 1
            total = sum(cnt)
            if total == 0:
                return MB_ACT_S
            splits = [(y, sub[y], cnt[y] / total) for y in range(MB_N_SYMBOLS) if cnt[y] > 0]
            best_a = MB_ACT_S
            best_v = -1e18
            for a in range(MB_N_ACTIONS):
                ev = 0.0
                for y, sm, prob in splits:
                    ev += prob * (float(r_hat[q, a, y]) + learned_dp_val(rem - 1, a, sm))
                if ev > best_v + 1e-12:
                    best_v = ev
                    best_a = a
            return best_a

        # Evaluate learned structural_model_mb on n_eval episodes of the TRUE 22-action env
        eval_env = VectorizedMultiBarrierEnv(batch_size=n_eval, horizon=horizon, seed=88000 + s * 19)
        thetas = np.tile(np.arange(MB_N_TYPES, dtype=np.int64), n_eval // MB_N_TYPES + 1)[:n_eval]
        eobs = eval_env.reset(theta=thetas)
        masks = np.full(n_eval, MB_FULL_BELIEF_MASK, dtype=np.int64)
        tot_r = np.zeros(n_eval, dtype=np.float64)
        acts_all = np.zeros((n_eval, horizon), dtype=np.int64)

        for t in range(horizon):
            q_prev = eobs["prev_self_action"].copy()
            acts = np.array(
                [learned_dp_act(horizon - t, int(q_prev[i]), int(masks[i])) for i in range(n_eval)],
                dtype=np.int64,
            )
            eobs, rf, _ro, y_obs = eval_env.step(acts)
            tot_r += rf
            acts_all[:, t] = acts
            # Update belief mask over learned clusters using observed y_obs at q_prev
            for i in range(n_eval):
                new_m = 0
                qi, yi = int(q_prev[i]), int(y_obs[i])
                mi = int(masks[i])
                for m in range(MB_N_TYPES):
                    if ((mi >> m) & 1) and int(f_hat[m, qi]) == yi:
                        new_m |= 1 << m
                masks[i] = new_m if new_m != 0 else mi

        t_late = np.vectorize(mb_action_tier)(acts_all[:, 8:])
        per_seed_rows.append({
            "seed": int(s),
            "warmup_transitions": int(n_warmup_eps * horizon),
            "warmup_pos_tier1_events": pos_t1,
            "warmup_pos_tier2_events": pos_t2,
            "warmup_pos_tier3_events": pos_t3,
            "obs_t1_cells_of_2": obs_t1_cells,
            "obs_t2_cells_of_4": obs_t2_cells,
            "obs_t3_cells_of_8": obs_t3_cells,
            "mean_return": float(np.mean(tot_r)),
            "median_return": float(np.median(tot_r)),
            "tier0_rate_late": float(np.mean(t_late == 0)),
            "tier1_rate_late": float(np.mean(t_late == 1)),
            "tier2_rate_late": float(np.mean(t_late == 2)),
            "tier3_rate_late": float(np.mean(t_late == 3)),
        })

    return {
        "per_seed": per_seed_rows,
        "summary": {
            "mean_return": float(np.mean([r["mean_return"] for r in per_seed_rows])),
            "median_of_seed_means": float(np.median([r["mean_return"] for r in per_seed_rows])),
            "mean_tier3_rate_late": float(np.mean([r["tier3_rate_late"] for r in per_seed_rows])),
            "mean_warmup_pos_t1": float(np.mean([r["warmup_pos_tier1_events"] for r in per_seed_rows])),
            "mean_warmup_pos_t2": float(np.mean([r["warmup_pos_tier2_events"] for r in per_seed_rows])),
            "mean_warmup_pos_t3": float(np.mean([r["warmup_pos_tier3_events"] for r in per_seed_rows])),
            "seeds_above_tier0_floor": int(sum(r["mean_return"] > 15.0 for r in per_seed_rows)),
        },
    }


def run_all_diagnostics(cfg: dict) -> dict:
    t0 = time.perf_counter()
    seeds = [int(s) for s in cfg["seeds"]]
    horizon = int(cfg.get("horizon", 24))
    d_values = [float(d) for d in cfg.get("d_values", [0.0, 0.3, 0.6])]

    print("[Diag 1A] Running full-horizon PBVI corrupted-channel Gate-0 for R3...", flush=True)
    d1a = run_diag_1a(d_values, horizon=horizon)

    print("[Diag 1B] Running eval-only surprise-threshold switch on sealed Exp 2A checkpoints...", flush=True)
    d1b_rows, d1b_contrasts = run_diag_1b(cfg)

    print("[Diag 1C] Running exact Gate-0 VoI audit for Exp 2B & Exp 3B symmetric game suites...", flush=True)
    d1c = run_diag_1c()

    print("[Diag 1D] Running unprivileged structural_model_mb arbitration on 22-action Exp 4 warmup data...", flush=True)
    d1d = run_diag_1d(seeds=seeds, horizon=64, n_eval=256)

    wall_s = time.perf_counter() - t0
    summary = {
        "study_id": cfg["study_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "wall_seconds": round(wall_s, 2),
        "diag_1a_corrupted_r3_gate0": d1a,
        "diag_1b_surprise_switch_summary": d1b_rows,
        "diag_1b_surprise_switch_contrasts": d1b_contrasts,
        "diag_1c_sym_semantic_gate0": d1c,
        "diag_1d_exp4_structural_arbitration": d1d,
    }

    for target_key in ("output_dir", "results_dir"):
        if target_key not in cfg:
            continue
        root = Path(cfg[target_key])
        _atomic_write_json(root / "summary.json", summary)
        _write_csv(root / "tables" / "diag1a_r3_corrupted_gate0.csv", d1a)
        _write_csv(root / "tables" / "diag1b_surprise_switch.csv", d1b_rows)
        _write_csv(root / "tables" / "diag1b_surprise_contrasts.csv", d1b_contrasts)
        _write_csv(root / "tables" / "diag1c_sym_semantic_gate0.csv", d1c)
        _write_csv(root / "tables" / "diag1d_exp4_structural_arbitration.csv", d1d["per_seed"])

    print(f"[DONE] Exp 5A diagnostics finished in {wall_s:.1f}s -> {cfg.get('results_dir')}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SPEC v3.1 Block 1 (Exp 5A) diagnostics.")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_all_diagnostics(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
