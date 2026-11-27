"""SPEC v3.0 Exp 3B: Anti-Memorization (S4-Equivariant DRQN) + Semantic Game-Rule Diversity (D-031).

Compares:
- Exact S_4 Permutation-Equivariant DRQN ('equiv_aux_yr', 'equiv_vanilla') vs
  Standard One-Hot DRQN ('onehot_aux_yr', 'onehot_vanilla') vs Non-Meta ('q_learner', 'ucb')
- Integrated D-031 Semantic Diversity vs Exposure Factorial:
  - n2_b1 (2 semantic families, 1x budget)
  - n4_b1 (4 semantic families, 1x budget -> fixed-budget diversity)
  - n4_b2 (4 semantic families, 2x budget -> exposure-matched diversity)

Evaluated across 4 cleanly separated generalization suites:
1. in_dist        : Seen semantic families x Seen permutations
2. unseen_perm    : Seen semantic families x Held-Out permutations (anti-memorization)
3. unseen_params  : Held-Out payoff parameterizations (all 4 families) x Held-Out permutations
4. unseen_family  : Brand-new 5th strategic family ('dual_eq_battle') x Held-Out permutations

Usage:
    PYTHONPATH=src .venv/bin/python -m mamrl.study_exp3b --config configs/exp3b_equivariant_semantic.json --workers 6
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn

from mamrl.agents.equivariant_drqn import (
    TOKEN_FEAT_DIM,
    EquivariantDRQNPolicy,
    EquivariantRecurrentQNet,
    encode_equivariant_tokens_batch,
)
from mamrl.envs.semantic_games import SEM_N_ACTIONS, VectorizedSemanticGameEnv
from mamrl.eval.audit_v3 import hierarchical_paired_bootstrap
from mamrl.study_exp2b import (
    SYM_OBS_DIM,
    SymNeuralDRQNPolicy,
    SymPolicy,
    SymQLearningPolicy,
    SymUCBPolicy,
    SymmetricRecurrentQNet,
    encode_sym_obs_batch,
)


@dataclass
class SemanticTrainConfig:
    arch: str = "equivariant"    # "equivariant" or "onehot"
    aux_mode: str = "yr"         # "yr" or "none"
    n_sem_train: int = 4         # 2 or 4
    n_iters: int = 350
    warmup_iters: int = 50
    horizon: int = 24
    hidden_dim: int = 32         # 32 per token for equivariant; 64 flat for onehot
    batch_episodes: int = 64
    replay_capacity: int = 8192
    grad_steps_per_iter: int = 8
    minibatch_episodes: int = 128
    gamma: float = 1.0
    lr: float = 3e-3
    polyak_tau: float = 0.05
    aux_opp_weight: float = 1.5
    aux_r_weight: float = 1.0
    eps_start: float = 0.85
    eps_end: float = 0.05
    max_grad_norm: float = 5.0
    log_interval: int = 25
    seed: int = 0
    device: str = "cpu"


def train_semantic_meta_agent(cfg: SemanticTrainConfig) -> tuple[nn.Module, list[dict[str, float]], int]:
    """Train Equivariant or One-Hot DRQN on VectorizedSemanticGameEnv (eval_suite='in_dist' ONLY)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    is_eq = cfg.arch == "equivariant"

    if is_eq:
        qnet: nn.Module = EquivariantRecurrentQNet(TOKEN_FEAT_DIM, cfg.hidden_dim, SEM_N_ACTIONS).to(device)
        tnet: nn.Module = EquivariantRecurrentQNet(TOKEN_FEAT_DIM, cfg.hidden_dim, SEM_N_ACTIONS).to(device)
    else:
        qnet = SymmetricRecurrentQNet(SYM_OBS_DIM, cfg.hidden_dim, SEM_N_ACTIONS).to(device)
        tnet = SymmetricRecurrentQNet(SYM_OBS_DIM, cfg.hidden_dim, SEM_N_ACTIONS).to(device)
    tnet.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=cfg.lr)

    b = cfg.batch_episodes
    t_len = cfg.horizon
    n_self = b // 2
    n_ql = b // 4
    n_ucb = b - n_self - n_ql

    env = VectorizedSemanticGameEnv(
        n_sem_train=cfg.n_sem_train, eval_suite="in_dist", batch_size=b, horizon=t_len, seed=cfg.seed + 17
    )
    ql_opp = SymQLearningPolicy(alpha=0.35, gamma=0.50, epsilon=0.12)
    ucb_opp = SymUCBPolicy(beta=1.0)
    rng = np.random.default_rng(cfg.seed + 31)

    cap = cfg.replay_capacity
    obs_shape = (cap, t_len, SEM_N_ACTIONS, TOKEN_FEAT_DIM) if is_eq else (cap, t_len, SYM_OBS_DIM)
    buf_obs = np.zeros(obs_shape, dtype=np.float32)
    buf_act = np.zeros((cap, t_len), dtype=np.int64)
    buf_opp = np.zeros((cap, t_len), dtype=np.int64)
    buf_rew = np.zeros((cap, t_len), dtype=np.float32)
    ptr, size = 0, 0
    total_transitions = 0
    curve: list[dict[str, float]] = []

    for it in range(1, cfg.n_iters + 1):
        if it <= cfg.warmup_iters:
            eps = 1.0
        else:
            decay_span = max(1.0, 0.8 * float(cfg.n_iters - cfg.warmup_iters))
            eps = max(cfg.eps_end, cfg.eps_start * (1.0 - float(it - cfg.warmup_iters) / decay_span))

        obs1, obs2 = env.reset()
        ql_opp.reset(n_ql, seed=cfg.seed + it * 101 + 1)
        ucb_opp.reset(n_ucb, seed=cfg.seed + it * 101 + 2)

        if is_eq:
            h1 = qnet.init_hidden(b, device)  # type: ignore[operator]
            h2_self = tnet.init_hidden(n_self, device)  # type: ignore[operator]
            ep_o = np.zeros((b, t_len, SEM_N_ACTIONS, TOKEN_FEAT_DIM), dtype=np.float32)
        else:
            h1 = torch.zeros((1, b, cfg.hidden_dim), dtype=torch.float32, device=device)
            h2_self = torch.zeros((1, n_self, cfg.hidden_dim), dtype=torch.float32, device=device)
            ep_o = np.zeros((b, t_len, SYM_OBS_DIM), dtype=np.float32)

        ep_a = np.zeros((b, t_len), dtype=np.int64)
        ep_opp = np.zeros((b, t_len), dtype=np.int64)
        ep_r = np.zeros((b, t_len), dtype=np.float32)

        qnet.eval()
        tnet.eval()
        with torch.no_grad():
            for t in range(t_len):
                if is_eq:
                    x1 = encode_equivariant_tokens_batch(obs1, n_actions=SEM_N_ACTIONS, horizon=t_len, step_t=t)
                    ep_o[:, t] = x1
                    qv1, h1 = qnet.step(torch.from_numpy(x1).to(device), h1)  # type: ignore[operator]
                    q1_np = qv1.cpu().numpy()
                    g1 = np.argmax(q1_np + rng.uniform(0.0, 1e-7, size=q1_np.shape), axis=-1)
                else:
                    x1 = encode_sym_obs_batch(obs1, horizon=t_len, step_t=t, variant="rl2")
                    ep_o[:, t] = x1
                    qv1, h1 = qnet.step(torch.from_numpy(x1).to(device).unsqueeze(1), h1)  # type: ignore[operator]
                    g1 = qv1.argmax(dim=-1).cpu().numpy()
                rand1 = rng.integers(0, SEM_N_ACTIONS, size=b)
                a1 = np.where(rng.random(b) < eps, rand1, g1).astype(np.int64)

                a2 = np.zeros(b, dtype=np.int64)
                obs2_self = {k: v[:n_self] for k, v in obs2.items()}
                if is_eq:
                    x2s = encode_equivariant_tokens_batch(obs2_self, n_actions=SEM_N_ACTIONS, horizon=t_len, step_t=t)
                    qv2, h2_self = tnet.step(torch.from_numpy(x2s).to(device), h2_self)  # type: ignore[operator]
                    q2_np = qv2.cpu().numpy()
                    g2 = np.argmax(q2_np + rng.uniform(0.0, 1e-7, size=q2_np.shape), axis=-1)
                else:
                    x2s = encode_sym_obs_batch(obs2_self, horizon=t_len, step_t=t, variant="rl2")
                    qv2, h2_self = tnet.step(torch.from_numpy(x2s).to(device).unsqueeze(1), h2_self)  # type: ignore[operator]
                    g2 = qv2.argmax(dim=-1).cpu().numpy()
                r2_rand = rng.integers(0, SEM_N_ACTIONS, size=n_self)
                a2[:n_self] = np.where(rng.random(n_self) < 0.05, r2_rand, g2)

                obs2_ql = {k: v[n_self : n_self + n_ql] for k, v in obs2.items()}
                a2[n_self : n_self + n_ql] = ql_opp.act(obs2_ql, step_t=t)
                obs2_ucb = {k: v[n_self + n_ql :] for k, v in obs2.items()}
                a2[n_self + n_ql :] = ucb_opp.act(obs2_ucb, step_t=t)

                obs1, obs2, r1, _r2 = env.step(a1, a2)
                ep_a[:, t] = a1
                ep_opp[:, t] = a2
                ep_r[:, t] = r1.astype(np.float32)

        total_transitions += b * t_len
        for k in range(b):
            buf_obs[ptr] = ep_o[k]
            buf_act[ptr] = ep_a[k]
            buf_opp[ptr] = ep_opp[k]
            buf_rew[ptr] = ep_r[k]
            ptr = (ptr + 1) % cap
            size = min(cap, size + 1)

        qnet.train()
        td_vals, ao_vals, ar_vals = [], [], []
        mb_n = min(cfg.minibatch_episodes, size)
        for _ in range(cfg.grad_steps_per_iter):
            idx = rng.integers(0, size, size=mb_n)
            ox = torch.from_numpy(buf_obs[idx]).to(device)
            ax = torch.from_numpy(buf_act[idx]).to(device)
            oppx = torch.from_numpy(buf_opp[idx]).to(device)
            rx = torch.from_numpy(buf_rew[idx]).to(device)

            q_all, hs = qnet(ox)
            q_sa = q_all.gather(-1, ax.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_a = q_all[:, 1:, :].argmax(dim=-1, keepdim=True)
                tq, _ = tnet(ox)
                next_q = tq[:, 1:, :].gather(-1, next_a).squeeze(-1)
                tgt = rx.clone()
                tgt[:, :-1] += cfg.gamma * next_q

            td_loss = nn.functional.mse_loss(q_sa, tgt)
            if cfg.aux_mode == "yr":
                if is_eq:
                    opp_logits, r_pred = qnet.predict_aux(hs, ax)  # type: ignore[operator]
                else:
                    a_oh = nn.functional.one_hot(ax, num_classes=SEM_N_ACTIONS).float()
                    ha = torch.cat([hs, a_oh], dim=-1)
                    opp_logits = qnet.pred_opp(ha)  # type: ignore[operator]
                    r_pred = qnet.pred_r(ha).squeeze(-1)  # type: ignore[operator]
                opp_loss = nn.functional.cross_entropy(opp_logits.reshape(-1, SEM_N_ACTIONS), oppx.reshape(-1))
                r_loss = nn.functional.mse_loss(r_pred, rx)
            else:
                opp_loss = torch.tensor(0.0, device=device)
                r_loss = torch.tensor(0.0, device=device)

            loss = td_loss + cfg.aux_opp_weight * opp_loss + cfg.aux_r_weight * r_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                tau = cfg.polyak_tau
                for p, tp in zip(qnet.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * p.data)

            td_vals.append(float(td_loss.item()))
            ao_vals.append(float(opp_loss.item()))
            ar_vals.append(float(r_loss.item()))

        if it == 1 or it == cfg.n_iters or (it % cfg.log_interval == 0):
            curve.append(
                {
                    "update": float(it),
                    "transitions": float(total_transitions),
                    "eps": float(eps),
                    "mean_return": float(np.mean(ep_r.sum(axis=1))),
                    "td_loss": float(np.mean(td_vals)),
                    "aux_opp_loss": float(np.mean(ao_vals)),
                    "aux_r_loss": float(np.mean(ar_vals)),
                }
            )

    qnet.eval()
    return qnet, curve, total_transitions


def rollout_semantic_matchup(
    p1: SymPolicy,
    p2: SymPolicy,
    *,
    n_sem_train: int,
    eval_suite: str,
    n_episodes: int = 512,
    horizon: int = 24,
    seed: int = 0,
) -> tuple[dict, dict[str, np.ndarray]]:
    env = VectorizedSemanticGameEnv(
        n_sem_train=n_sem_train, eval_suite=eval_suite, batch_size=n_episodes, horizon=horizon, seed=seed
    )
    n_rules = env.n_rules
    if n_episodes % n_rules != 0:
        raise ValueError(f"n_episodes ({n_episodes}) must be divisible by n_rules ({n_rules}) for {eval_suite}.")
    rule_ids = np.tile(np.arange(n_rules, dtype=np.int64), n_episodes // n_rules)

    obs1, obs2 = env.reset(rule_ids=rule_ids)
    p1.reset(n_episodes, seed=seed + 13)
    p2.reset(n_episodes, seed=seed + 29)

    r1_hist = np.zeros((n_episodes, horizon), dtype=np.float64)
    r2_hist = np.zeros((n_episodes, horizon), dtype=np.float64)
    for t in range(horizon):
        a1 = p1.act(obs1, step_t=t)
        a2 = p2.act(obs2, step_t=t)
        obs1, obs2, r1, r2 = env.step(a1, a2)
        r1_hist[:, t] = r1
        r2_hist[:, t] = r2

    p1_tot = r1_hist.sum(axis=1)
    p2_tot = r2_hist.sum(axis=1)
    welfare = p1_tot + p2_tot
    margin = p1_tot - p2_tot
    # Normalized welfare efficiency relative to each rule's max possible 24-turn welfare (for positive-welfare rules)
    max_w = env.max_step_welfare[rule_ids] * horizon
    q25, q50, q75 = np.percentile(p1_tot, [25, 50, 75])

    metrics = {
        "p1_mean": float(np.mean(p1_tot)),
        "p1_median": float(q50),
        "p1_iqr": float(q75 - q25),
        "p2_mean": float(np.mean(p2_tot)),
        "welfare_mean": float(np.mean(welfare)),
        "welfare_median": float(np.median(welfare)),
        "margin_mean": float(np.mean(margin)),
        "p1_early_auc_1_4": float(np.mean(r1_hist[:, : min(4, horizon)])),
        "p1_late_auc_13_end": float(np.mean(r1_hist[:, 12:])) if horizon > 12 else 0.0,
        "welfare_ratio_to_max": float(np.mean(welfare / np.maximum(max_w, 1.0))),
    }
    return metrics, {"p1_return": p1_tot, "welfare": welfare, "margin": margin}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _train_or_load_exp3b_fit(out_dir_str: str, cond_name: str, cond_cfg: dict, seed: int, global_cfg: dict) -> tuple[str, int, dict]:
    torch.set_num_threads(1)
    out_dir = Path(out_dir_str)
    fit_dir = out_dir / "fits" / f"{cond_name}_seed{seed}"
    receipt_path = fit_dir / "receipt.json"
    ckpt_path = fit_dir / "checkpoint.pt"

    if receipt_path.exists() and ckpt_path.exists():
        rec = json.loads(receipt_path.read_text(encoding="utf-8"))
        if rec.get("checkpoint_sha256") == _sha256_file(ckpt_path):
            return cond_name, seed, rec

    fit_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    arch = str(cond_cfg["arch"])
    t_cfg = SemanticTrainConfig(
        arch=arch,
        aux_mode=str(cond_cfg["aux_mode"]),
        n_sem_train=int(cond_cfg["n_sem_train"]),
        n_iters=int(cond_cfg["n_iters"]),
        warmup_iters=int(cond_cfg.get("warmup_iters", 50)),
        horizon=int(global_cfg.get("horizon", 24)),
        hidden_dim=int(cond_cfg.get("hidden_dim", 32 if arch == "equivariant" else 64)),
        batch_episodes=int(global_cfg.get("drqn_batch_episodes", 64)),
        replay_capacity=int(global_cfg.get("drqn_replay_capacity", 8192)),
        grad_steps_per_iter=int(global_cfg.get("drqn_grad_steps", 8)),
        minibatch_episodes=int(global_cfg.get("drqn_minibatch", 128)),
        gamma=float(global_cfg.get("gamma", 1.0)),
        lr=float(global_cfg.get("drqn_lr", 3e-3)),
        polyak_tau=float(global_cfg.get("drqn_polyak_tau", 0.05)),
        aux_opp_weight=float(global_cfg.get("aux_opp_weight", 1.5)),
        aux_r_weight=float(global_cfg.get("aux_r_weight", 1.0)),
        log_interval=int(global_cfg.get("drqn_log_interval", 25)),
        seed=seed,
        device=str(global_cfg.get("device", "cpu")),
    )
    model, curve, off_trans = train_semantic_meta_agent(t_cfg)
    torch.save({"state_dict": model.state_dict(), "config": t_cfg.__dict__}, ckpt_path)
    _write_csv(fit_dir / "train_curve.csv", curve)
    rec = {
        "condition": cond_name,
        "seed": int(seed),
        "arch": arch,
        "aux_mode": t_cfg.aux_mode,
        "n_sem_train": t_cfg.n_sem_train,
        "n_iters": t_cfg.n_iters,
        "offline_train_transitions": int(off_trans),
        "wall_time_sec": round(time.perf_counter() - t0, 3),
        "checkpoint_sha256": _sha256_file(ckpt_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(receipt_path, rec)
    return cond_name, seed, rec


def _load_exp3b_policy(out_dir: Path, name: str, seed: int, cfg: dict) -> SymPolicy:
    if name == "q_learner":
        return SymQLearningPolicy(alpha=0.35, gamma=0.50, epsilon=0.12)
    if name == "ucb":
        return SymUCBPolicy(beta=1.0)
    cond_cfg = cfg["conditions"][name]
    ckpt_path = out_dir / "fits" / f"{name}_seed{seed}" / "checkpoint.pt"
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = str(cond_cfg["arch"])
    horizon = int(cfg.get("horizon", 24))
    hdim = int(cond_cfg.get("hidden_dim", 32 if arch == "equivariant" else 64))
    if arch == "equivariant":
        m = EquivariantRecurrentQNet(TOKEN_FEAT_DIM, hdim, SEM_N_ACTIONS)
        m.load_state_dict(payload["state_dict"])
        m.eval()
        return EquivariantDRQNPolicy(m, name=name, horizon=horizon, eps_eval=0.0, device="cpu")
    m2 = SymmetricRecurrentQNet(SYM_OBS_DIM, hdim, SEM_N_ACTIONS)
    m2.load_state_dict(payload["state_dict"])
    m2.eval()
    return SymNeuralDRQNPolicy(m2, name=name, variant="rl2", horizon=horizon, eps_eval=0.0, device="cpu")


def run_study_exp3b(cfg: dict, workers: int = 1) -> dict:
    study_id = str(cfg["study_id"])
    out_dir = Path(cfg.get("output_dir", f"runs/{study_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = cfg["conditions"]
    seeds = [int(s) for s in cfg["seeds"]]
    horizon = int(cfg.get("horizon", 24))
    n_eval = int(cfg.get("n_eval_episodes", 512))
    n_boot = int(cfg.get("n_bootstrap", 10000))
    delta = float(cfg.get("delta_return", 0.30))
    eval_seed_base = int(cfg.get("eval_seed_base", 99000))

    manifest = {
        "study_id": study_id,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "config": cfg,
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)

    # Phase 1: Train all (condition, seed) fits in parallel
    jobs = [(cname, ccfg, s) for cname, ccfg in conditions.items() for s in seeds]
    if workers <= 1:
        for cname, ccfg, s in jobs:
            _train_or_load_exp3b_fit(str(out_dir), cname, ccfg, s, cfg)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_train_or_load_exp3b_fit, str(out_dir), cname, ccfg, s, cfg) for cname, ccfg, s in jobs]
            done = 0
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    print(f"  [{study_id}] training progress: {done}/{len(jobs)} fits done", flush=True)

    # Phase 2: Evaluate across 4 suites (in_dist, unseen_perm, unseen_params, unseen_family)
    # Matchups: for each trained condition, evaluate cross-seed self-play (Meta x Meta) and Meta vs UCB + Non-Meta baselines
    eval_suites = ["in_dist", "unseen_perm", "unseen_params", "unseen_family"]
    matchup_defs: list[tuple[str, str, str, bool]] = [
        ("ucb_vs_ucb", "ucb", "ucb", False),
        ("q_learner_vs_q_learner", "q_learner", "q_learner", False),
    ]
    for cname in conditions:
        matchup_defs.append((f"{cname}__xseed", cname, cname, True))
        matchup_defs.append((f"{cname}__vs_ucb", cname, "ucb", False))

    results_by_suite: dict[str, dict] = {}
    matchup_rows: list[dict] = []
    contrast_rows: list[dict] = []

    for s_idx, suite in enumerate(eval_suites):
        agg_m: dict[str, dict] = {}
        arrs_m: dict[str, dict[str, np.ndarray]] = {}

        for lbl, p1_n, p2_n, xseed in matchup_defs:
            per_seed_m = []
            w_list, p1_list = [], []
            for i_s, seed in enumerate(seeds):
                s2 = seeds[(i_s + 1) % len(seeds)] if (xseed and len(seeds) > 1) else seed
                p1 = _load_exp3b_policy(out_dir, p1_n, seed, cfg)
                p2 = _load_exp3b_policy(out_dir, p2_n, s2, cfg)
                md, ad = rollout_semantic_matchup(
                    p1,
                    p2,
                    n_sem_train=4,  # Canonical 4-family reference for in_dist / unseen_perm evaluation
                    eval_suite=suite,
                    n_episodes=n_eval,
                    horizon=horizon,
                    seed=eval_seed_base + s_idx * 1000 + seed * 17,
                )
                per_seed_m.append(md)
                w_list.append(ad["welfare"])
                p1_list.append(ad["p1_return"])

            w_mat = np.stack(w_list, axis=0)
            p1_mat = np.stack(p1_list, axis=0)
            arrs_m[lbl] = {"welfare": w_mat, "p1_return": p1_mat}
            seed_w = np.mean(w_mat, axis=1)
            seed_p1 = np.mean(p1_mat, axis=1)
            agg = {
                "p1_mean": float(np.mean(seed_p1)),
                "p1_seed_sd": float(np.std(seed_p1, ddof=1)) if len(seeds) > 1 else 0.0,
                "p1_median": float(np.median(p1_mat)),
                "p1_iqr": float(np.percentile(p1_mat, 75) - np.percentile(p1_mat, 25)),
                "welfare_mean": float(np.mean(seed_w)),
                "welfare_seed_sd": float(np.std(seed_w, ddof=1)) if len(seeds) > 1 else 0.0,
                "welfare_median": float(np.median(w_mat)),
                "p1_early_auc_1_4": float(np.mean([m["p1_early_auc_1_4"] for m in per_seed_m])),
                "p1_late_auc_13_end": float(np.mean([m["p1_late_auc_13_end"] for m in per_seed_m])),
            }
            agg_m[lbl] = agg
            matchup_rows.append(
                {
                    "suite": suite,
                    "matchup": lbl,
                    "p1_mean": round(agg["p1_mean"], 4),
                    "p1_seed_sd": round(agg["p1_seed_sd"], 4),
                    "p1_median": round(agg["p1_median"], 4),
                    "p1_iqr": round(agg["p1_iqr"], 4),
                    "welfare_mean": round(agg["welfare_mean"], 4),
                    "welfare_seed_sd": round(agg["welfare_seed_sd"], 4),
                    "welfare_median": round(agg["welfare_median"], 4),
                    "early_auc_1_4": round(agg["p1_early_auc_1_4"], 4),
                    "late_auc_13_end": round(agg["p1_late_auc_13_end"], 4),
                }
            )

        # Key scientific contrasts (Anti-memorization + Aux + D-031 diversity vs exposure)
        contrast_pairs = [
            # Equivariant vs One-Hot (same N=4, same budget 1x)
            ("equiv_aux_yr_n4_b1__xseed", "onehot_aux_yr_n4_b1__xseed"),
            ("equiv_aux_yr_n4_b1__vs_ucb", "onehot_aux_yr_n4_b1__vs_ucb"),
            ("equiv_vanilla_n4_b1__xseed", "onehot_vanilla_n4_b1__xseed"),
            # Aux vs Vanilla inside Equivariant & One-Hot
            ("equiv_aux_yr_n4_b1__xseed", "equiv_vanilla_n4_b1__xseed"),
            ("onehot_aux_yr_n4_b1__xseed", "onehot_vanilla_n4_b1__xseed"),
            # Equivariant vs Non-Meta baselines
            ("equiv_aux_yr_n4_b1__xseed", "ucb_vs_ucb"),
            ("equiv_aux_yr_n4_b1__xseed", "q_learner_vs_q_learner"),
            ("equiv_aux_yr_n4_b1__vs_ucb", "ucb_vs_ucb"),
            # D-031 Semantic Diversity: Fixed-budget (n4_b1 vs n2_b1) & Exposure-matched (n4_b2 vs n2_b1)
            ("equiv_aux_yr_n4_b1__xseed", "equiv_aux_yr_n2_b1__xseed"),
            ("equiv_aux_yr_n4_b2__xseed", "equiv_aux_yr_n2_b1__xseed"),
            ("equiv_aux_yr_n4_b2__xseed", "equiv_aux_yr_n4_b1__xseed"),
            ("onehot_aux_yr_n4_b1__xseed", "onehot_aux_yr_n2_b1__xseed"),
            ("onehot_aux_yr_n4_b2__xseed", "onehot_aux_yr_n2_b1__xseed"),
        ]
        contrasts = {}
        for a, b in contrast_pairs:
            if a not in arrs_m or b not in arrs_m:
                continue
            c_key = f"{a}_minus_{b}"
            contrasts[c_key] = {}
            for m_key in ("welfare", "p1_return"):
                res = hierarchical_paired_bootstrap(arrs_m[a][m_key], arrs_m[b][m_key], practical_threshold=delta, n_boot=n_boot, seed=88001)
                contrasts[c_key][m_key] = res
                contrast_rows.append(
                    {
                        "suite": suite,
                        "contrast": c_key,
                        "metric": m_key,
                        "diff_mean": round(float(res["diff_mean"]), 4),
                        "diff_median": round(float(res["diff_median"]), 4),
                        "ci_low": round(float(res["ci_low"]), 4),
                        "ci_high": round(float(res["ci_high"]), 4),
                        "win_rate": round(float(res["win_rate"]), 4),
                        "tie_rate": round(float(res["tie_rate"]), 4),
                        "loss_rate": round(float(res["loss_rate"]), 4),
                        "threshold": delta,
                        "verdict": res["verdict"],
                    }
                )
        results_by_suite[suite] = {"matchups": agg_m, "paired_contrasts": contrasts}

    summary = {
        "study_id": study_id,
        "git_commit": manifest["git_commit"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "suites": results_by_suite,
    }
    _atomic_write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "tables" / "matchups.csv", matchup_rows)
    _write_csv(out_dir / "tables" / "paired_contrasts.csv", contrast_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp 3B Equivariant DRQN + Semantic Rule Diversity (D-031).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 1))
    t0 = time.perf_counter()
    summary = run_study_exp3b(cfg, workers=workers)
    print(f"[{cfg['study_id']}] Complete in {time.perf_counter() - t0:.1f}s")
    for suite, sdata in summary["suites"].items():
        print(f"\n=== Suite: {suite} ===")
        for lbl, md in sdata["matchups"].items():
            print(
                f"  {lbl:34s} | P1={md['p1_mean']:+6.2f} (sd={md['p1_seed_sd']:4.2f}, med={md['p1_median']:+6.2f}) "
                f"| W={md['welfare_mean']:+6.2f} (med={md['welfare_median']:+6.2f}) "
                f"| eAUC={md['p1_early_auc_1_4']:+5.2f} lAUC={md['p1_late_auc_13_end']:+5.2f}"
            )


if __name__ == "__main__":
    main()
