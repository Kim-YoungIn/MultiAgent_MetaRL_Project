"""SPEC v3.0 Secondary Study A (Exp 2A): R3 x Online Learning Opponents.

Evaluates:
- Condition A (Zero-Shot Transfer): Exp 1 fixed-opponent R3 policies dropped directly
  against online Q-learning (in-dist) and unseen Expected SARSA (out-of-dist) opponents.
- Condition B (Matched Training): Policies trained directly against online Q-learning
  opponents with mid-episode restore indicator (reset_prob=0.5 at K=4) so Restored
  evaluation is strictly in-distribution, tested on both Q-learning and unseen Expected SARSA.
- Online Baselines: contextual_ucb, arm_ucb, greedy, random.
- Protocols: Paired Continuous vs. Restored (K=4, q->S, opponent Q->Q_0, reset_flag=True)
  with counterpart return, both_up/both_down rates, opponent Q L1 change, opponent honesty
  trajectory, and hierarchical paired bootstrap (10k resamples over independent training seeds).

Note: Strictly no false exact full-horizon Oracle claim for online learning opponents.
zs_bayes_optimal is flagged privileged=True strictly as a fixed-opponent diagnostic reference.

Usage:
    PYTHONPATH=src .venv/bin/python -m mamrl.study_exp2a --config configs/exp2a_online_opp.json --workers 6
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
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

from mamrl.agents.baselines_v3 import (
    ArmUCBPolicyV3,
    BayesOptimalPolicyV3,
    BatchPolicyV3,
    ContextualUCBPolicyV3,
    GreedyPolicyV3,
    NeuralRL2PolicyV3,
    RandomPolicyV3,
    StructuralModelPolicyV3,
)
from mamrl.agents.drqn_rl2 import DRQNConfig, NeuralDRQNPolicyV3, RecurrentQNet, train_drqn_rl2
from mamrl.agents.opponents_v3 import OnlineOpponentConfig, VectorizedOnlineOpponent, make_online_opponent
from mamrl.agents.ppo_rl2 import OBS_DIM, RecurrentActorCritic
from mamrl.envs.delayed_response import (
    ACTION_NAMES,
    ACT_P,
    ACT_P1,
    ACT_S,
    ACT_X00,
    N_ACTIONS,
    N_TYPES,
    VectorizedDelayedResponseEnv,
)
from mamrl.eval.audit_v3 import _action_category_dist, _time_to_threshold, hierarchical_paired_bootstrap

MT_DRQN_METHODS = {
    "mt_drqn_aux_yr": ("rl2", "yr"),
    "mt_drqn_aux_y": ("rl2", "y"),
    "mt_drqn_vanilla": ("rl2", "none"),
    "mt_drqn_aux_yr_masked": ("rl2_masked", "yr"),
    "mt_drqn_prior_only": ("prior_only", "none"),
}

ZS_DRQN_METHODS = {
    "zs_drqn_aux_yr": ("drqn_aux_yr", "rl2"),
    "zs_drqn_aux_y": ("drqn_aux_y", "rl2"),
    "zs_drqn_vanilla": ("drqn_vanilla", "rl2"),
    "zs_drqn_aux_yr_masked": ("drqn_aux_yr_masked", "rl2_masked"),
    "zs_drqn_prior_only": ("drqn_prior_only", "prior_only"),
}


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


def _opp_cfg_from_dict(d: dict, regime: str) -> OnlineOpponentConfig:
    return OnlineOpponentConfig(
        kind=str(d["kind"]),
        regime=regime,
        alpha=float(d["alpha"]),
        gamma=float(d["gamma"]),
        epsilon=float(d.get("epsilon", 0.08)),
        temperature=float(d.get("temperature", 0.45)),
        warmup_steps=int(d.get("warmup_steps", 80)),
        warmup_epsilon=float(d.get("warmup_epsilon", 1.0)),
        warmup_temperature=float(d.get("warmup_temperature", 0.80)),
    )


def rollout_online_protocol(
    policy: BatchPolicyV3,
    *,
    regime: str,
    opp_cfg: OnlineOpponentConfig,
    protocol: str,
    n_episodes: int = 512,
    horizon: int = 24,
    restore_k: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run vectorized evaluation against an online learning opponent under continuous or restored protocol.

    Both protocols share identical theta, identical initial opponent Q_0 (via seed+301),
    and identical prefix RNG so turns 1..K match identically for deterministic policies.
    Under protocol='restored', after turn K (t == restore_k - 1), public state q -> S,
    opponent Q -> Q_0, and obs['reset_flag'] -> True (preserving turn-K (y, r)).
    """
    if protocol not in ("continuous", "restored"):
        raise ValueError(f"Unsupported protocol: {protocol}")
    if n_episodes % N_TYPES != 0:
        raise ValueError(f"n_episodes ({n_episodes}) must be divisible by {N_TYPES}.")

    thetas = np.tile(np.arange(N_TYPES, dtype=np.int64), n_episodes // N_TYPES)
    env = VectorizedDelayedResponseEnv(regime=regime, batch_size=n_episodes, horizon=horizon, seed=seed)
    obs = env.reset(theta=thetas)

    opp = make_online_opponent(opp_cfg, batch_size=n_episodes, seed=seed + 301)
    opp.reset(thetas, seed=seed + 301)
    init_greedy_honest = opp.greedy_honest_rate()

    policy.reset(batch_size=n_episodes, seed=seed + 13, true_theta=thetas if policy.is_privileged else None)

    actions = np.zeros((n_episodes, horizon), dtype=np.int64)
    opp_actions = np.zeros((n_episodes, horizon), dtype=np.int64)
    focal_rewards = np.zeros((n_episodes, horizon), dtype=np.float64)
    opp_rewards = np.zeros((n_episodes, horizon), dtype=np.float64)
    honest_flags = np.zeros((n_episodes, horizon), dtype=np.float64)

    for t in range(horizon):
        q_prev = env.q.copy()
        a = policy.act(obs, step_t=t)
        y_opp = opp.act(q_prev)
        honest_y = env.honest_opp_actions()
        honest_flags[:, t] = (y_opp == honest_y).astype(np.float64)

        obs, rf, ro, y = env.step(a, opp_actions=y_opp)
        opp.update(q_prev=q_prev, y_act=y, r_opp=ro, q_next=a)

        actions[:, t] = a
        opp_actions[:, t] = y
        focal_rewards[:, t] = rf
        opp_rewards[:, t] = ro

        if protocol == "restored" and (t + 1) == restore_k and (t + 1) < horizon:
            obs = env.restore_public_state(obs, mask=None)
            opp.restore(snap=None, mask=None)
            # Re-seed opponent continuation deterministically so all methods face matched post-reset randomness
            opp._rng = np.random.default_rng(seed + 7919)

    return {
        "theta": thetas,
        "actions": actions,
        "opp_actions": opp_actions,
        "focal_rewards": focal_rewards,
        "opp_rewards": opp_rewards,
        "honest_flags": honest_flags,
        "opp_q_l1_change": opp.q_l1_distance_from_init(),
        "opp_init_greedy_honest": init_greedy_honest,
        "opp_final_greedy_honest": opp.greedy_honest_rate(),
    }


def summarize_online_rollout(data: dict[str, np.ndarray], *, primary_k: int = 4) -> tuple[dict, dict[str, np.ndarray]]:
    """Compute distribution, speed, opponent dynamics, and behavioral metrics for Exp 2A."""
    theta = data["theta"]
    actions = data["actions"]
    opp_actions = data["opp_actions"]
    fr = data["focal_rewards"]
    orr = data["opp_rewards"]
    hf = data["honest_flags"]
    n_ep, horizon = actions.shape
    k_p = min(primary_k, horizon)

    ep_total = fr.sum(axis=1)
    ep_prefix = fr[:, :k_p].sum(axis=1)
    ep_post = fr[:, k_p:].sum(axis=1) if horizon > k_p else np.zeros(n_ep, dtype=np.float64)
    ep_opp_total = orr.sum(axis=1)
    ep_opp_post = orr[:, k_p:].sum(axis=1) if horizon > k_p else np.zeros(n_ep, dtype=np.float64)

    q25, q50, q75 = np.percentile(ep_total, [25, 50, 75])
    ttt_mean, ttt_cens = _time_to_threshold(fr, threshold=0.8, window=3)

    act_exp = actions[:, :k_p]
    act_expt = actions[:, k_p:]
    rep_exp = float(np.mean(act_exp[:, 1:] == act_exp[:, :-1])) if k_p > 1 else 0.0
    rep_expt = float(np.mean(actions[:, k_p:] == actions[:, k_p - 1 : -1])) if horizon > k_p else 0.0
    rep_all = float(np.mean(actions[:, 1:] == actions[:, :-1])) if horizon > 1 else 0.0
    open_probe_rate = float(np.mean((actions[:, 0] >= ACT_P) & (actions[:, 0] <= ACT_P1)))

    y2 = opp_actions[:, 1] if horizon >= 3 else np.zeros(n_ep, dtype=np.int64)
    a3 = actions[:, 2] if horizon >= 3 else np.zeros(n_ep, dtype=np.int64)
    m0, m1 = (y2 == 0), (y2 == 1)
    if int(np.sum(m0)) > 0 and int(np.sum(m1)) > 0:
        p0 = np.bincount(a3[m0], minlength=N_ACTIONS) / float(np.sum(m0))
        p1 = np.bincount(a3[m1], minlength=N_ACTIONS) / float(np.sum(m1))
        k4_branch_tv = float(0.5 * np.sum(np.abs(p0 - p1)))
    else:
        k4_branch_tv = 0.0

    by_theta = {}
    for th in range(N_TYPES):
        vals = ep_total[theta == th]
        by_theta[str(th)] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
        }

    metrics = {
        "total_mean": float(np.mean(ep_total)),
        "total_median": float(q50),
        "total_iqr": float(q75 - q25),
        "prefix_mean": float(np.mean(ep_prefix)),
        "post_mean": float(np.mean(ep_post)),
        "post_median": float(np.median(ep_post)),
        "opp_total_mean": float(np.mean(ep_opp_total)),
        "opp_post_mean": float(np.mean(ep_opp_post)),
        "opp_q_l1_change": float(np.mean(data["opp_q_l1_change"])),
        "opp_init_greedy_honest": float(np.mean(data["opp_init_greedy_honest"])),
        "opp_final_greedy_honest": float(np.mean(data["opp_final_greedy_honest"])),
        "opp_realized_honest_early": float(np.mean(hf[:, :k_p])),
        "opp_realized_honest_late": float(np.mean(hf[:, k_p:])) if horizon > k_p else 0.0,
        "early_auc_1_4": float(np.mean(fr[:, : min(4, horizon)])),
        "mid_auc_5_12": float(np.mean(fr[:, 4 : min(12, horizon)])) if horizon > 4 else 0.0,
        "late_auc_13_end": float(np.mean(fr[:, 12:])) if horizon > 12 else 0.0,
        "ttt_mean": ttt_mean,
        "ttt_censored_rate": ttt_cens,
        "open_probe_rate": open_probe_rate,
        "k4_branching_tv": k4_branch_tv,
        "repeat_rate_explore": rep_exp,
        "repeat_rate_exploit": rep_expt,
        "repeat_rate_all": rep_all,
        "action_dist_explore": _action_category_dist(act_exp, theta),
        "action_dist_exploit": _action_category_dist(act_expt, theta),
        "by_theta": by_theta,
    }
    arrays = {
        "total_return": ep_total,
        "post_return": ep_post,
        "opp_total_return": ep_opp_total,
        "opp_post_return": ep_opp_post,
        "opp_q_l1_change": data["opp_q_l1_change"],
    }
    return metrics, arrays


def _build_policy_for_exp2a(
    *,
    method: str,
    seed: int,
    cfg: dict,
    fit_dir: Path,
    train_opp_cfg: OnlineOpponentConfig,
) -> tuple[BatchPolicyV3, int, str | None]:
    """Load zero-shot policy, train matched policy, or construct online baseline."""
    regime = str(cfg.get("regime", "R3"))
    horizon = int(cfg.get("horizon", 24))
    primary_k = int(cfg.get("primary_k", 4))
    hidden_dim = int(cfg.get("hidden_dim", 64))
    device = str(cfg.get("device", "cpu"))
    zs_root = Path(cfg.get("zero_shot_runs_dir", "runs/core_v3_sealed"))

    if method in ZS_DRQN_METHODS:
        src_method, variant = ZS_DRQN_METHODS[method]
        ckpt_path = zs_root / "fits" / f"{regime}_{src_method}_seed{seed}" / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Zero-shot checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        qnet = RecurrentQNet(OBS_DIM, hidden_dim, N_ACTIONS).to(device)
        qnet.load_state_dict(payload["state_dict"])
        qnet.eval()
        src_receipt = ckpt_path.parent / "receipt.json"
        off_trans = int(json.loads(src_receipt.read_text(encoding="utf-8"))["offline_train_transitions"]) if src_receipt.exists() else 0
        pol: BatchPolicyV3 = NeuralDRQNPolicyV3(
            qnet,
            name=method,
            variant=variant,
            horizon=horizon,
            mask_k=primary_k,
            offline_train_transitions=off_trans,
            device=device,
        )
        return pol, off_trans, _sha256_file(ckpt_path)

    if method == "zs_rl2_ppo":
        ckpt_path = zs_root / "fits" / f"{regime}_rl2_ppo_seed{seed}" / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Zero-shot checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = RecurrentActorCritic(OBS_DIM, hidden_dim, N_ACTIONS).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        src_receipt = ckpt_path.parent / "receipt.json"
        off_trans = int(json.loads(src_receipt.read_text(encoding="utf-8"))["offline_train_transitions"]) if src_receipt.exists() else 0
        pol = NeuralRL2PolicyV3(
            model,
            variant="rl2",
            horizon=horizon,
            mask_k=primary_k,
            deterministic=False,
            offline_train_transitions=off_trans,
            device=device,
        )
        pol.name = method
        return pol, off_trans, _sha256_file(ckpt_path)

    if method == "zs_structural_model":
        struct_eps = int(cfg.get("structural_offline_episodes", 1024))
        pol = StructuralModelPolicyV3(horizon=horizon).fit_offline(
            regime=regime,
            n_episodes=struct_eps,
            seed=seed + 777,
            opponent_factory=None,  # Fixed reactive opponent (matches Exp 1 exactly)
        )
        pol.name = method
        return pol, pol.offline_train_transitions, None

    if method == "zs_bayes_optimal":
        pol = BayesOptimalPolicyV3(regime=regime, horizon=horizon)
        pol.name = method
        return pol, 0, None

    if method in MT_DRQN_METHODS:
        variant, aux_mode = MT_DRQN_METHODS[method]
        drqn_cfg = DRQNConfig(
            regime=regime,
            variant=variant,
            aux_mode=aux_mode,
            mask_k=primary_k,
            horizon=horizon,
            hidden_dim=hidden_dim,
            batch_episodes=int(cfg.get("drqn_batch_episodes", 64)),
            n_iters=int(cfg.get("drqn_iters", 450)),
            warmup_iters=int(cfg.get("drqn_warmup_iters", 100)),
            replay_capacity=int(cfg.get("drqn_replay_capacity", 8192)),
            grad_steps_per_iter=int(cfg.get("drqn_grad_steps", 8)),
            minibatch_episodes=int(cfg.get("drqn_minibatch", 128)),
            gamma=float(cfg.get("gamma", 1.0)),
            lr=float(cfg.get("drqn_lr", 3e-3)),
            polyak_tau=float(cfg.get("drqn_polyak_tau", 0.05)),
            aux_y_weight=float(cfg.get("drqn_aux_y_weight", 2.0)),
            aux_r_weight=float(cfg.get("drqn_aux_r_weight", 1.0)),
            log_interval=int(cfg.get("drqn_log_interval", 15)),
            seed=seed,
            device=device,
            reset_prob=float(cfg.get("train_reset_prob", 0.5)),
            reset_step=primary_k,
        )
        opp_factory = lambda b_sz, s_val: make_online_opponent(train_opp_cfg, batch_size=b_sz, seed=s_val)
        qnet, train_curve, off_trans = train_drqn_rl2(drqn_cfg, opponent_factory=opp_factory)
        ckpt_path = fit_dir / "checkpoint.pt"
        torch.save({"state_dict": qnet.state_dict(), "drqn_config": drqn_cfg.__dict__}, ckpt_path)
        _write_csv(fit_dir / "train_curve.csv", train_curve)
        pol = NeuralDRQNPolicyV3(
            qnet,
            name=method,
            variant=variant,
            horizon=horizon,
            mask_k=primary_k,
            offline_train_transitions=off_trans,
            device=device,
        )
        return pol, off_trans, _sha256_file(ckpt_path)

    if method == "mt_structural_model":
        struct_eps = int(cfg.get("structural_offline_episodes", 1024))
        opp_factory = lambda b_sz, s_val: make_online_opponent(train_opp_cfg, batch_size=b_sz, seed=s_val)
        pol = StructuralModelPolicyV3(horizon=horizon).fit_offline(
            regime=regime,
            n_episodes=struct_eps,
            seed=seed + 777,
            opponent_factory=opp_factory,
        )
        pol.name = method
        return pol, pol.offline_train_transitions, None

    if method == "contextual_ucb":
        return ContextualUCBPolicyV3(beta=float(cfg.get("contextual_ucb_beta", 1.0))), 0, None
    if method == "arm_ucb":
        return ArmUCBPolicyV3(beta=float(cfg.get("arm_ucb_beta", 1.0))), 0, None
    if method == "greedy":
        return GreedyPolicyV3(), 0, None
    if method == "random":
        return RandomPolicyV3(), 0, None

    raise ValueError(f"Unknown Exp 2A method: {method}")


def _run_or_load_exp2a_unit(
    *,
    out_dir_str: str,
    method: str,
    seed: int,
    cfg: dict,
) -> tuple[str, int, dict, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Run one (method, seed) fit and evaluate on all configured online opponents & protocols."""
    torch.set_num_threads(1)
    out_dir = Path(out_dir_str)
    regime = str(cfg.get("regime", "R3"))
    horizon = int(cfg.get("horizon", 24))
    primary_k = int(cfg.get("primary_k", 4))
    n_eval = int(cfg.get("n_eval_episodes", 512))
    eval_seed_base = int(cfg.get("eval_seed_base", 95000))
    eval_opponents = cfg["eval_opponents"]
    train_opp_cfg = _opp_cfg_from_dict(cfg["train_opponent"], regime=regime)

    fit_dir = out_dir / "fits" / f"{method}_seed{seed}"
    receipt_path = fit_dir / "receipt.json"
    eval_json_path = fit_dir / "eval_metrics.json"
    arrays_path = fit_dir / "eval_arrays.npz"

    if receipt_path.exists() and eval_json_path.exists() and arrays_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("arrays_sha256") == _sha256_file(arrays_path):
            saved_metrics = json.loads(eval_json_path.read_text(encoding="utf-8"))
            npz = np.load(arrays_path)
            ep_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
            for opp_name in eval_opponents:
                ep_arrays[opp_name] = {}
                for proto in ("continuous", "restored"):
                    ep_arrays[opp_name][proto] = {
                        k: npz[f"{opp_name}__{proto}__{k}"].astype(np.float64)
                        for k in ("total_return", "post_return", "opp_total_return", "opp_post_return", "opp_q_l1_change")
                    }
            return method, seed, {"receipt": receipt, "eval": saved_metrics}, ep_arrays

    fit_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    policy, off_trans, ckpt_sha = _build_policy_for_exp2a(
        method=method,
        seed=seed,
        cfg=cfg,
        fit_dir=fit_dir,
        train_opp_cfg=train_opp_cfg,
    )

    eval_metrics: dict[str, dict[str, dict]] = {}
    ep_arrays = {}
    flat_npz: dict[str, np.ndarray] = {}

    for opp_idx, (opp_name, opp_dict) in enumerate(eval_opponents.items()):
        opp_cfg = _opp_cfg_from_dict(opp_dict, regime=regime)
        eval_metrics[opp_name] = {}
        ep_arrays[opp_name] = {}
        opp_eval_seed = eval_seed_base + seed * 1009 + opp_idx * 313
        for proto in ("continuous", "restored"):
            raw = rollout_online_protocol(
                policy,
                regime=regime,
                opp_cfg=opp_cfg,
                protocol=proto,
                n_episodes=n_eval,
                horizon=horizon,
                restore_k=primary_k,
                seed=opp_eval_seed,
            )
            m_dict, arr_dict = summarize_online_rollout(raw, primary_k=primary_k)
            eval_metrics[opp_name][proto] = m_dict
            ep_arrays[opp_name][proto] = arr_dict
            for k, arr in arr_dict.items():
                flat_npz[f"{opp_name}__{proto}__{k}"] = arr.astype(np.float64)

    np.savez_compressed(arrays_path, **flat_npz)
    _atomic_write_json(eval_json_path, eval_metrics)
    wall_sec = time.perf_counter() - t0
    receipt = {
        "regime": regime,
        "method": method,
        "seed": int(seed),
        "is_privileged": bool(policy.is_privileged),
        "offline_train_transitions": int(off_trans),
        "online_eval_turns_per_protocol": int(n_eval * horizon),
        "wall_time_sec": round(wall_sec, 3),
        "checkpoint_sha256": ckpt_sha,
        "arrays_sha256": _sha256_file(arrays_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(receipt_path, receipt)
    return method, seed, {"receipt": receipt, "eval": eval_metrics}, ep_arrays


def run_study_exp2a(cfg: dict, workers: int = 1) -> dict:
    """Execute Exp 2A study across all methods, seeds, opponents, and protocols."""
    study_id = str(cfg["study_id"])
    out_dir = Path(cfg.get("output_dir", f"runs/{study_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [str(m) for m in cfg["methods"]]
    seeds = [int(s) for s in cfg["seeds"]]
    eval_opponents = list(cfg["eval_opponents"].keys())
    n_boot = int(cfg.get("n_bootstrap", 10000))
    delta_total = float(cfg.get("delta_total", 0.30))
    delta_post = float(cfg.get("delta_post", 0.25))

    manifest = {
        "study_id": study_id,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "config": cfg,
        "scientific_note": (
            "No exact full-horizon Oracle is claimed for online learning opponents. "
            "Restored vs Continuous contrasts measure intervention-specific state/opponent-mediated "
            "effects under the stated K=4 reset protocol."
        ),
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)

    jobs = [(m, s) for m in methods for s in seeds]
    collected: dict[tuple[str, int], tuple[dict, dict]] = {}

    if workers <= 1:
        for m, s in jobs:
            m_, s_, rec, arrs = _run_or_load_exp2a_unit(out_dir_str=str(out_dir), method=m, seed=s, cfg=cfg)
            collected[(m_, s_)] = (rec, arrs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_run_or_load_exp2a_unit, out_dir_str=str(out_dir), method=m, seed=s, cfg=cfg)
                for m, s in jobs
            ]
            done = 0
            for fut in as_completed(futs):
                m_, s_, rec, arrs = fut.result()
                collected[(m_, s_)] = (rec, arrs)
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    print(f"  [{study_id}] progress: {done}/{len(jobs)} units done", flush=True)

    contrast_pairs = [
        ("mt_drqn_aux_yr", "zs_drqn_aux_yr"),
        ("mt_drqn_aux_yr", "mt_drqn_aux_yr_masked"),
        ("mt_drqn_aux_yr", "mt_drqn_prior_only"),
        ("mt_drqn_aux_yr", "mt_drqn_vanilla"),
        ("mt_drqn_aux_yr", "mt_drqn_aux_y"),
        ("mt_drqn_aux_yr", "mt_structural_model"),
        ("mt_drqn_aux_yr", "zs_structural_model"),
        ("mt_drqn_aux_yr", "contextual_ucb"),
        ("mt_drqn_aux_yr", "arm_ucb"),
        ("mt_drqn_aux_yr", "greedy"),
        ("mt_drqn_aux_yr", "random"),
        ("mt_structural_model", "zs_structural_model"),
        ("zs_drqn_aux_yr", "zs_structural_model"),
        ("zs_drqn_aux_yr", "zs_drqn_aux_yr_masked"),
        ("zs_drqn_aux_yr", "contextual_ucb"),
        ("zs_drqn_aux_yr", "greedy"),
    ]

    results_by_opp: dict[str, dict] = {}
    method_rows: list[dict] = []
    contrast_rows: list[dict] = []
    restore_rows: list[dict] = []
    behavior_rows: list[dict] = []

    for opp_name in eval_opponents:
        results_by_opp[opp_name] = {"protocols": {}, "restore_vs_continuous_did": {}}

        # Aggregate per protocol
        for proto in ("continuous", "restored"):
            agg_methods: dict[str, dict] = {}
            arrays_by_method: dict[str, dict[str, np.ndarray]] = {}

            for m in methods:
                recs = [collected[(m, s)][0] for s in seeds]
                evals = [r["eval"][opp_name][proto] for r in recs]
                arrs = {
                    k: np.stack([collected[(m, s)][1][opp_name][proto][k] for s in seeds], axis=0)
                    for k in ("total_return", "post_return", "opp_total_return", "opp_post_return", "opp_q_l1_change")
                }
                arrays_by_method[m] = arrs
                seed_tot_means = np.mean(arrs["total_return"], axis=1)
                seed_post_means = np.mean(arrs["post_return"], axis=1)

                agg_methods[m] = {
                    "is_privileged": recs[0]["receipt"]["is_privileged"],
                    "offline_train_transitions": recs[0]["receipt"]["offline_train_transitions"],
                    "total_mean": float(np.mean(seed_tot_means)),
                    "total_seed_sd": float(np.std(seed_tot_means, ddof=1)) if len(seeds) > 1 else 0.0,
                    "total_median": float(np.median(arrs["total_return"])),
                    "total_iqr": float(np.percentile(arrs["total_return"], 75) - np.percentile(arrs["total_return"], 25)),
                    "post_mean": float(np.mean(seed_post_means)),
                    "post_seed_sd": float(np.std(seed_post_means, ddof=1)) if len(seeds) > 1 else 0.0,
                    "post_median": float(np.median(arrs["post_return"])),
                    "opp_total_mean": float(np.mean(arrs["opp_total_return"])),
                    "opp_post_mean": float(np.mean(arrs["opp_post_return"])),
                    "opp_q_l1_change": float(np.mean(arrs["opp_q_l1_change"])),
                    "opp_honest_early": float(np.mean([e["opp_realized_honest_early"] for e in evals])),
                    "opp_honest_late": float(np.mean([e["opp_realized_honest_late"] for e in evals])),
                    "early_auc_1_4": float(np.mean([e["early_auc_1_4"] for e in evals])),
                    "mid_auc_5_12": float(np.mean([e["mid_auc_5_12"] for e in evals])),
                    "late_auc_13_end": float(np.mean([e["late_auc_13_end"] for e in evals])),
                    "open_probe_rate": float(np.mean([e["open_probe_rate"] for e in evals])),
                    "k4_branching_tv": float(np.mean([e["k4_branching_tv"] for e in evals])),
                    "repeat_rate_explore": float(np.mean([e["repeat_rate_explore"] for e in evals])),
                    "repeat_rate_exploit": float(np.mean([e["repeat_rate_exploit"] for e in evals])),
                    "action_dist_explore": {
                        k: float(np.mean([e["action_dist_explore"][k] for e in evals]))
                        for k in ("S", "P", "P0", "P1", "X_match", "X_mismatch")
                    },
                    "action_dist_exploit": {
                        k: float(np.mean([e["action_dist_exploit"][k] for e in evals]))
                        for k in ("S", "P", "P0", "P1", "X_match", "X_mismatch")
                    },
                }

                method_rows.append(
                    {
                        "opponent": opp_name,
                        "protocol": proto,
                        "method": m,
                        "privileged": agg_methods[m]["is_privileged"],
                        "offline_transitions": agg_methods[m]["offline_train_transitions"],
                        "total_mean": round(agg_methods[m]["total_mean"], 4),
                        "total_seed_sd": round(agg_methods[m]["total_seed_sd"], 4),
                        "total_median": round(agg_methods[m]["total_median"], 4),
                        "total_iqr": round(agg_methods[m]["total_iqr"], 4),
                        "post_mean": round(agg_methods[m]["post_mean"], 4),
                        "opp_total_mean": round(agg_methods[m]["opp_total_mean"], 4),
                        "opp_q_l1": round(agg_methods[m]["opp_q_l1_change"], 4),
                        "opp_honest_early": round(agg_methods[m]["opp_honest_early"], 4),
                        "opp_honest_late": round(agg_methods[m]["opp_honest_late"], 4),
                        "open_probe_rate": round(agg_methods[m]["open_probe_rate"], 4),
                        "k4_branch_tv": round(agg_methods[m]["k4_branching_tv"], 4),
                        "rep_explore": round(agg_methods[m]["repeat_rate_explore"], 4),
                        "rep_exploit": round(agg_methods[m]["repeat_rate_exploit"], 4),
                    }
                )
                behavior_rows.append(
                    {
                        "opponent": opp_name,
                        "protocol": proto,
                        "method": m,
                        "rep_explore": round(agg_methods[m]["repeat_rate_explore"], 4),
                        "rep_exploit": round(agg_methods[m]["repeat_rate_exploit"], 4),
                        "open_probe_rate": round(agg_methods[m]["open_probe_rate"], 4),
                        "k4_branch_tv": round(agg_methods[m]["k4_branching_tv"], 4),
                        **{f"exp_{k}": round(v, 4) for k, v in agg_methods[m]["action_dist_explore"].items()},
                        **{f"expt_{k}": round(v, 4) for k, v in agg_methods[m]["action_dist_exploit"].items()},
                    }
                )

            contrasts: dict[str, dict] = {}
            for a_name, b_name in contrast_pairs:
                if a_name not in methods or b_name not in methods:
                    continue
                c_key = f"{a_name}_minus_{b_name}"
                contrasts[c_key] = {}

                # Joint focal+counterpart movement (both_up / both_down)
                df_tot = arrays_by_method[a_name]["total_return"] - arrays_by_method[b_name]["total_return"]
                do_tot = arrays_by_method[a_name]["opp_total_return"] - arrays_by_method[b_name]["opp_total_return"]
                both_up = float(np.mean((df_tot > 1e-9) & (do_tot > 1e-9)))
                both_down = float(np.mean((df_tot < -1e-9) & (do_tot < -1e-9)))

                for m_key, thresh in (
                    ("total_return", delta_total),
                    ("post_return", delta_post),
                    ("opp_total_return", delta_total),
                ):
                    res_b = hierarchical_paired_bootstrap(
                        arrays_by_method[a_name][m_key],
                        arrays_by_method[b_name][m_key],
                        practical_threshold=thresh,
                        n_boot=n_boot,
                        seed=23456,
                    )
                    if m_key == "total_return":
                        res_b["both_up_rate"] = both_up
                        res_b["both_down_rate"] = both_down
                    contrasts[c_key][m_key] = res_b
                    contrast_rows.append(
                        {
                            "opponent": opp_name,
                            "protocol": proto,
                            "contrast": c_key,
                            "metric": m_key,
                            "diff_mean": round(float(res_b["diff_mean"]), 4),
                            "diff_median": round(float(res_b["diff_median"]), 4),
                            "ci_low": round(float(res_b["ci_low"]), 4),
                            "ci_high": round(float(res_b["ci_high"]), 4),
                            "win_rate": round(float(res_b["win_rate"]), 4),
                            "tie_rate": round(float(res_b["tie_rate"]), 4),
                            "loss_rate": round(float(res_b["loss_rate"]), 4),
                            "both_up_rate": round(both_up, 4) if m_key == "total_return" else "",
                            "both_down_rate": round(both_down, 4) if m_key == "total_return" else "",
                            "threshold": thresh,
                            "verdict": res_b["verdict"],
                        }
                    )

            results_by_opp[opp_name]["protocols"][proto] = {
                "methods": agg_methods,
                "paired_contrasts": contrasts,
            }

        # Restored vs Continuous per-method state-mediated delta & DiD contrasts
        per_m_delta: dict[str, np.ndarray] = {}
        for m in methods:
            cont_post = np.stack([collected[(m, s)][1][opp_name]["continuous"]["post_return"] for s in seeds], axis=0)
            rest_post = np.stack([collected[(m, s)][1][opp_name]["restored"]["post_return"] for s in seeds], axis=0)
            per_m_delta[m] = cont_post - rest_post
            boot_delta = hierarchical_paired_bootstrap(
                cont_post, rest_post, practical_threshold=delta_post, n_boot=n_boot, seed=34567
            )
            results_by_opp[opp_name]["restore_vs_continuous_did"][f"delta_{m}"] = boot_delta
            restore_rows.append(
                {
                    "opponent": opp_name,
                    "comparison": f"continuous_minus_restored_post::{m}",
                    "diff_mean": round(float(boot_delta["diff_mean"]), 4),
                    "diff_median": round(float(boot_delta["diff_median"]), 4),
                    "ci_low": round(float(boot_delta["ci_low"]), 4),
                    "ci_high": round(float(boot_delta["ci_high"]), 4),
                    "win_rate": round(float(boot_delta["win_rate"]), 4),
                    "tie_rate": round(float(boot_delta["tie_rate"]), 4),
                    "loss_rate": round(float(boot_delta["loss_rate"]), 4),
                    "verdict": boot_delta["verdict"],
                }
            )

        for a_name, b_name in (
            ("mt_drqn_aux_yr", "mt_drqn_aux_yr_masked"),
            ("mt_drqn_aux_yr", "mt_structural_model"),
            ("mt_drqn_aux_yr", "contextual_ucb"),
            ("zs_drqn_aux_yr", "zs_structural_model"),
        ):
            if a_name not in methods or b_name not in methods:
                continue
            did = hierarchical_paired_bootstrap(
                per_m_delta[a_name], per_m_delta[b_name], practical_threshold=delta_post, n_boot=n_boot, seed=45678
            )
            key = f"did_{a_name}_minus_{b_name}"
            results_by_opp[opp_name]["restore_vs_continuous_did"][key] = did
            restore_rows.append(
                {
                    "opponent": opp_name,
                    "comparison": f"DiD_cont_minus_rest::{a_name}_minus_{b_name}",
                    "diff_mean": round(float(did["diff_mean"]), 4),
                    "diff_median": round(float(did["diff_median"]), 4),
                    "ci_low": round(float(did["ci_low"]), 4),
                    "ci_high": round(float(did["ci_high"]), 4),
                    "win_rate": round(float(did["win_rate"]), 4),
                    "tie_rate": round(float(did["tie_rate"]), 4),
                    "loss_rate": round(float(did["loss_rate"]), 4),
                    "verdict": did["verdict"],
                }
            )

    summary = {
        "study_id": study_id,
        "git_commit": manifest["git_commit"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "opponents": results_by_opp,
    }
    _atomic_write_json(out_dir / "summary.json", summary)

    tables_dir = out_dir / "tables"
    _write_csv(tables_dir / "methods.csv", method_rows)
    _write_csv(tables_dir / "paired_contrasts.csv", contrast_rows)
    _write_csv(tables_dir / "restore_vs_continuous.csv", restore_rows)
    _write_csv(tables_dir / "behavior_repetition.csv", behavior_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPEC v3.0 Exp 2A (R3 x Online Learning Opponents).")
    parser.add_argument("--config", required=True, help="Path to Exp 2A JSON config.")
    parser.add_argument("--workers", type=int, default=None, help="Override worker count.")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 1))
    t0 = time.perf_counter()
    summary = run_study_exp2a(cfg, workers=workers)
    out_dir = cfg.get("output_dir", f"runs/{cfg['study_id']}")
    print(f"[{cfg['study_id']}] Complete in {time.perf_counter() - t0:.1f}s -> {out_dir}/summary.json")
    for opp_name, odata in summary["opponents"].items():
        for proto in ("continuous", "restored"):
            print(f"\n=== {opp_name} | {proto} ===")
            for m, md in odata["protocols"][proto]["methods"].items():
                print(
                    f"  {m:22s} | Ret={md['total_mean']:+6.3f} (sd={md['total_seed_sd']:4.2f}) med={md['total_median']:+6.3f} "
                    f"| post={md['post_mean']:+6.3f} | opp={md['opp_total_mean']:5.2f} qL1={md['opp_q_l1_change']:.3f} "
                    f"| hon(E/L)={md['opp_honest_early']:.2f}/{md['opp_honest_late']:.2f} "
                    f"| prb={md['open_probe_rate']:.2f} tv={md['k4_branching_tv']:.2f}"
                )


if __name__ == "__main__":
    main()
