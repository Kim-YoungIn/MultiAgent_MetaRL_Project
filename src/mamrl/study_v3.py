"""Config-driven SPEC v3.0 Core Study Runner with Atomic Receipts, Parallelism, Resume, and CSV/JSON Exports.

Usage:
    PYTHONPATH=src .venv/bin/python -m mamrl.study_v3 --config configs/core_v3_sealed.json --workers 6
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
    ClairvoyantPolicyV3,
    ContextualUCBPolicyV3,
    GreedyPolicyV3,
    NeuralRL2PolicyV3,
    RandomPolicyV3,
    StructuralModelPolicyV3,
)
from mamrl.agents.drqn_rl2 import DRQNConfig, NeuralDRQNPolicyV3, train_drqn_rl2
from mamrl.agents.ppo_rl2 import PPOConfig, train_ppo_rl2
from mamrl.eval.audit_v3 import (
    TrajectoryBundle,
    compute_episode_audit,
    hierarchical_paired_bootstrap,
    rollout_policy_v3,
)
from mamrl.eval.exact_dp import run_gate0_audit

PPO_METHODS = {
    "rl2": "rl2",
    "rl2_ppo": "rl2",
    "rl2_masked": "rl2_masked",
    "prior_only": "prior_only",
}
DRQN_METHODS = {
    "drqn_vanilla": ("rl2", "none"),
    "drqn_aux_y": ("rl2", "y"),
    "drqn_aux_r": ("rl2", "r"),
    "drqn_aux_yr": ("rl2", "yr"),
    "drqn_aux_yr_masked": ("rl2_masked", "yr"),
    "drqn_prior_only": ("prior_only", "none"),
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


def _save_eval_npz(path: Path, bundle: TrajectoryBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        theta=bundle.theta.astype(np.int8),
        actions=bundle.actions.astype(np.int8),
        opp_actions=bundle.opp_actions.astype(np.int8),
        focal_rewards=bundle.focal_rewards.astype(np.float64),
        opp_rewards=bundle.opp_rewards.astype(np.float64),
    )


def _load_eval_npz(path: Path, regime: str, method: str) -> TrajectoryBundle:
    d = np.load(path)
    return TrajectoryBundle(
        regime=regime,
        policy_name=method,
        theta=d["theta"].astype(np.int64),
        actions=d["actions"].astype(np.int64),
        opp_actions=d["opp_actions"].astype(np.int64),
        focal_rewards=d["focal_rewards"].astype(np.float64),
        opp_rewards=d["opp_rewards"].astype(np.float64),
        hidden_states=None,
    )


def _run_or_load_fit(
    *,
    out_dir_str: str,
    regime: str,
    method: str,
    seed: int,
    cfg: dict,
) -> tuple[str, str, int, dict, dict[str, np.ndarray]]:
    """Run one (regime, method, seed) unit or resume cleanly from verified receipt."""
    torch.set_num_threads(1)
    out_dir = Path(out_dir_str)
    fit_dir = out_dir / "fits" / f"{regime}_{method}_seed{seed}"
    receipt_path = fit_dir / "receipt.json"
    eval_path = fit_dir / "eval_episodes.npz"
    audit_path = fit_dir / "audit_summary.json"

    horizon = int(cfg.get("horizon", 24))
    primary_k = int(cfg.get("primary_k", 4))
    audit_ks = tuple(int(k) for k in cfg.get("audit_ks", [2, 4, 8]))
    n_eval = int(cfg.get("n_eval_episodes", 256))
    eval_seed = int(cfg.get("eval_seed_base", 50000)) + seed * 1009

    if receipt_path.exists() and eval_path.exists() and audit_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("eval_sha256") == _sha256_file(eval_path):
            bundle = _load_eval_npz(eval_path, regime, method)
            audit = compute_episode_audit(bundle, audit_ks=audit_ks, primary_k=primary_k)
            saved_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if saved_audit.get("layer3", {}).get("hidden_state_linear_probe_acc"):
                audit["layer3"]["hidden_state_linear_probe_acc"] = saved_audit["layer3"]["hidden_state_linear_probe_acc"]
            return (
                regime,
                method,
                seed,
                {
                    "receipt": receipt,
                    "end_to_end": audit["end_to_end"],
                    "behavior": audit["behavior"],
                    "layer1": audit["layer1"],
                    "layer3": audit["layer3"],
                },
                audit["episode_arrays"],
            )

    fit_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ckpt_sha = None
    offline_transitions = 0
    record_hidden = False

    if method in PPO_METHODS:
        ppo_variant = PPO_METHODS[method]
        ppo_cfg = PPOConfig(
            regime=regime,
            variant=ppo_variant,
            mask_k=primary_k,
            horizon=horizon,
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            batch_episodes=int(cfg.get("batch_episodes", 256)),
            total_transitions=int(cfg.get("total_transitions", 262144)),
            ppo_epochs=int(cfg.get("ppo_epochs", 4)),
            minibatches=int(cfg.get("minibatches", 4)),
            gamma=float(cfg.get("gamma", 1.0)),
            gae_lambda=float(cfg.get("gae_lambda", 0.95)),
            clip_eps=float(cfg.get("clip_eps", 0.2)),
            lr=float(cfg.get("lr", 1e-3)),
            entropy_coef=float(cfg.get("entropy_coef", 0.015)),
            entropy_coef_end=(
                float(cfg["entropy_coef_end"]) if cfg.get("entropy_coef_end") is not None else None
            ),
            value_coef=float(cfg.get("value_coef", 0.5)),
            max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
            log_interval=int(cfg.get("log_interval", 2)),
            seed=seed,
            device=str(cfg.get("device", "cpu")),
        )
        model, train_curve, offline_transitions = train_ppo_rl2(ppo_cfg)
        ckpt_path = fit_dir / "checkpoint.pt"
        torch.save({"state_dict": model.state_dict(), "ppo_config": ppo_cfg.__dict__}, ckpt_path)
        ckpt_sha = _sha256_file(ckpt_path)
        _write_csv(fit_dir / "train_curve.csv", train_curve)
        policy: BatchPolicyV3 = NeuralRL2PolicyV3(
            model,
            variant=ppo_variant,
            horizon=horizon,
            mask_k=primary_k,
            deterministic=bool(cfg.get("eval_deterministic", False)),
            offline_train_transitions=offline_transitions,
            device=str(cfg.get("device", "cpu")),
        )
        policy.name = method
        record_hidden = True
    elif method in DRQN_METHODS:
        drqn_variant, aux_mode = DRQN_METHODS[method]
        drqn_cfg = DRQNConfig(
            regime=regime,
            variant=drqn_variant,
            aux_mode=aux_mode,
            mask_k=primary_k,
            horizon=horizon,
            hidden_dim=int(cfg.get("hidden_dim", 64)),
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
            device=str(cfg.get("device", "cpu")),
        )
        qnet, train_curve, offline_transitions = train_drqn_rl2(drqn_cfg)
        ckpt_path = fit_dir / "checkpoint.pt"
        torch.save({"state_dict": qnet.state_dict(), "drqn_config": drqn_cfg.__dict__}, ckpt_path)
        ckpt_sha = _sha256_file(ckpt_path)
        _write_csv(fit_dir / "train_curve.csv", train_curve)
        policy = NeuralDRQNPolicyV3(
            qnet,
            name=method,
            variant=drqn_variant,
            horizon=horizon,
            mask_k=primary_k,
            offline_train_transitions=offline_transitions,
            device=str(cfg.get("device", "cpu")),
        )
        record_hidden = True
    elif method == "structural_model":
        struct_eps = int(cfg.get("structural_offline_episodes", 1024))
        policy = StructuralModelPolicyV3(horizon=horizon).fit_offline(
            regime=regime,
            n_episodes=struct_eps,
            seed=seed + 777,
        )
        offline_transitions = policy.offline_train_transitions
    elif method == "contextual_ucb":
        policy = ContextualUCBPolicyV3(beta=float(cfg.get("contextual_ucb_beta", 1.0)))
    elif method == "arm_ucb":
        policy = ArmUCBPolicyV3(beta=float(cfg.get("arm_ucb_beta", 1.0)))
    elif method == "greedy":
        policy = GreedyPolicyV3()
    elif method == "random":
        policy = RandomPolicyV3()
    elif method == "bayes_optimal":
        policy = BayesOptimalPolicyV3(regime=regime, horizon=horizon)
    elif method == "clairvoyant":
        policy = ClairvoyantPolicyV3(regime=regime, horizon=horizon)
    else:
        raise ValueError(f"Unknown method: {method}")

    bundle = rollout_policy_v3(
        policy,
        regime=regime,
        n_episodes=n_eval,
        horizon=horizon,
        seed=eval_seed,
        record_hidden=record_hidden,
    )
    _save_eval_npz(eval_path, bundle)
    audit = compute_episode_audit(bundle, audit_ks=audit_ks, primary_k=primary_k)
    wall_sec = time.perf_counter() - t0

    audit_clean = {
        "end_to_end": audit["end_to_end"],
        "behavior": audit["behavior"],
        "layer1": audit["layer1"],
        "layer3": audit["layer3"],
    }
    _atomic_write_json(audit_path, audit_clean)

    receipt = {
        "regime": regime,
        "method": method,
        "seed": int(seed),
        "is_privileged": bool(policy.is_privileged),
        "offline_train_transitions": int(offline_transitions),
        "online_eval_turns": int(n_eval * horizon),
        "wall_time_sec": round(wall_sec, 3),
        "checkpoint_sha256": ckpt_sha,
        "eval_sha256": _sha256_file(eval_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(receipt_path, receipt)
    return regime, method, seed, {"receipt": receipt, **audit_clean}, audit["episode_arrays"]


def run_study_v3(cfg: dict, workers: int = 1) -> dict:
    """Execute full SPEC v3.0 study and export summary JSON + CSV tables."""
    study_id = str(cfg["study_id"])
    out_dir = Path(cfg.get("output_dir", f"runs/{study_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    regimes = [str(r) for r in cfg.get("regimes", ["R0", "R1", "R3"])]
    methods = [str(m) for m in cfg["methods"]]
    seeds = [int(s) for s in cfg.get("seeds", [101, 102])]
    horizon = int(cfg.get("horizon", 24))
    audit_ks = tuple(int(k) for k in cfg.get("audit_ks", [2, 4, 8]))
    n_boot = int(cfg.get("n_bootstrap", 10000))
    delta_info = float(cfg.get("delta_info", 0.25))
    delta_total = float(cfg.get("delta_total", 0.30))

    gate0 = {reg: run_gate0_audit(reg, horizon=horizon, audit_ks=(1, 2, 4, 8)).to_summary_dict() for reg in regimes}

    manifest = {
        "study_id": study_id,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "config": cfg,
        "gate0": gate0,
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)

    # Execute all (regime, method, seed) jobs (parallel if workers > 1)
    jobs = [(reg, m, s) for reg in regimes for m in methods for s in seeds]
    collected: dict[tuple[str, str, int], tuple[dict, dict[str, np.ndarray]]] = {}

    if workers <= 1:
        for reg, m, s in jobs:
            r_, m_, s_, rec, arrs = _run_or_load_fit(
                out_dir_str=str(out_dir), regime=reg, method=m, seed=s, cfg=cfg
            )
            collected[(r_, m_, s_)] = (rec, arrs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_run_or_load_fit, out_dir_str=str(out_dir), regime=reg, method=m, seed=s, cfg=cfg)
                for reg, m, s in jobs
            ]
            done = 0
            for fut in as_completed(futs):
                r_, m_, s_, rec, arrs = fut.result()
                collected[(r_, m_, s_)] = (rec, arrs)
                done += 1
                if done % 15 == 0 or done == len(jobs):
                    print(f"  [{study_id}] progress: {done}/{len(jobs)} fits done", flush=True)

    results_by_regime: dict[str, dict] = {}
    method_rows: list[dict] = []
    contrast_rows: list[dict] = []
    layer1_rows: list[dict] = []
    theta_rows: list[dict] = []
    behavior_rows: list[dict] = []

    for reg in regimes:
        per_method_records: dict[str, list[dict]] = {m: [] for m in methods}
        per_method_arrays: dict[str, dict[str, list[np.ndarray]]] = {
            m: {
                "total_return": [],
                "layer1_commitment_k4": [],
                "layer1_adaptive_k4": [],
                "layer1_net_adaptive_k4": [],
                "layer3_live_gap_k4": [],
            }
            for m in methods
        }

        for m in methods:
            for s in seeds:
                rec, arrs = collected[(reg, m, s)]
                per_method_records[m].append(rec)
                for k_arr in per_method_arrays[m]:
                    per_method_arrays[m][k_arr].append(arrs[k_arr])

        agg_methods: dict[str, dict] = {}
        for m in methods:
            recs = per_method_records[m]
            tot_mat = np.stack(per_method_arrays[m]["total_return"], axis=0)
            c4_mat = np.stack(per_method_arrays[m]["layer1_commitment_k4"], axis=0)
            v4_mat = np.stack(per_method_arrays[m]["layer1_adaptive_k4"], axis=0)
            nv4_mat = np.stack(per_method_arrays[m]["layer1_net_adaptive_k4"], axis=0)
            lg4_mat = np.stack(per_method_arrays[m]["layer3_live_gap_k4"], axis=0)

            seed_tot_means = np.mean(tot_mat, axis=1)
            beh_agg = {
                "repeat_rate_explore": float(np.mean([r["behavior"]["repeat_rate_explore"] for r in recs])),
                "repeat_rate_exploit": float(np.mean([r["behavior"]["repeat_rate_exploit"] for r in recs])),
                "repeat_rate_all": float(np.mean([r["behavior"]["repeat_rate_all"] for r in recs])),
                "unique_actions_explore": float(np.mean([r["behavior"]["unique_actions_explore"] for r in recs])),
                "opening_probe_rate": float(np.mean([r["behavior"]["opening_probe_rate"] for r in recs])),
                "k4_branching_tv": float(np.mean([r["behavior"]["k4_branching_tv"] for r in recs])),
                "action_dist_explore": {
                    k: float(np.mean([r["behavior"]["action_dist_explore"][k] for r in recs]))
                    for k in ("S", "P", "P0", "P1", "X_match", "X_mismatch")
                },
                "action_dist_exploit": {
                    k: float(np.mean([r["behavior"]["action_dist_exploit"][k] for r in recs]))
                    for k in ("S", "P", "P0", "P1", "X_match", "X_mismatch")
                },
                "top_prefix_seed0": recs[0]["behavior"]["top_prefix_sequences"][:3],
            }
            l1_summary = {
                f"K={k}": {
                    mk: float(np.mean([r["layer1"][f"K={k}"][mk] for r in recs]))
                    for mk in recs[0]["layer1"][f"K={k}"]
                }
                for k in audit_ks
                if f"K={k}" in recs[0]["layer1"]
            }
            probe_keys = list(recs[0]["layer3"]["hidden_state_linear_probe_acc"].keys())
            probe_summary = {
                pk: float(np.mean([r["layer3"]["hidden_state_linear_probe_acc"][pk] for r in recs]))
                for pk in probe_keys
            }

            agg_methods[m] = {
                "is_privileged": recs[0]["receipt"]["is_privileged"],
                "offline_train_transitions": recs[0]["receipt"]["offline_train_transitions"],
                "online_eval_turns_per_seed": recs[0]["receipt"]["online_eval_turns"],
                "total_mean": float(np.mean(seed_tot_means)),
                "total_seed_sd": float(np.std(seed_tot_means, ddof=1)) if len(seeds) > 1 else 0.0,
                "total_median": float(np.median(tot_mat)),
                "total_iqr": float(np.percentile(tot_mat, 75) - np.percentile(tot_mat, 25)),
                "opp_total_mean": float(np.mean([r["end_to_end"]["opp_total_mean"] for r in recs])),
                "bayes_opt_gap": float(gate0[reg]["bayes_optimal_value"]) - float(np.mean(seed_tot_means)),
                "early_auc_1_4": float(np.mean([r["end_to_end"]["early_auc_1_4"] for r in recs])),
                "mid_auc_5_12": float(np.mean([r["end_to_end"]["mid_auc_5_12"] for r in recs])),
                "late_auc_13_end": float(np.mean([r["end_to_end"]["late_auc_13_end"] for r in recs])),
                "ttt_mean": float(np.mean([r["end_to_end"]["ttt_mean"] for r in recs])),
                "ttt_censored_rate": float(np.mean([r["end_to_end"]["ttt_censored_rate"] for r in recs])),
                "layer1_k4_commitment_mean": float(np.mean(c4_mat)),
                "layer1_k4_adaptive_mean": float(np.mean(v4_mat)),
                "layer1_k4_net_adaptive_mean": float(np.mean(nv4_mat)),
                "layer3_k4_live_gap_mean": float(np.mean(lg4_mat)),
                "layer1_by_k": l1_summary,
                "layer3_linear_probe_acc": probe_summary,
                "behavior": beh_agg,
            }

            method_rows.append(
                {
                    "regime": reg,
                    "method": m,
                    "privileged": agg_methods[m]["is_privileged"],
                    "offline_transitions": agg_methods[m]["offline_train_transitions"],
                    "online_eval_turns": agg_methods[m]["online_eval_turns_per_seed"],
                    "total_mean": round(agg_methods[m]["total_mean"], 4),
                    "total_seed_sd": round(agg_methods[m]["total_seed_sd"], 4),
                    "total_median": round(agg_methods[m]["total_median"], 4),
                    "total_iqr": round(agg_methods[m]["total_iqr"], 4),
                    "opp_total_mean": round(float(np.mean([r["end_to_end"]["opp_total_mean"] for r in recs])), 4),
                    "early_auc_1_4": round(agg_methods[m]["early_auc_1_4"], 4),
                    "bayes_opt_gap": round(agg_methods[m]["bayes_opt_gap"], 4),
                    "l1_k4_commit": round(agg_methods[m]["layer1_k4_commitment_mean"], 4),
                    "l1_k4_adapt": round(agg_methods[m]["layer1_k4_adaptive_mean"], 4),
                    "l1_k4_net_adapt": round(agg_methods[m]["layer1_k4_net_adaptive_mean"], 4),
                    "l3_k4_live_gap": round(agg_methods[m]["layer3_k4_live_gap_mean"], 4),
                    "open_probe_rate": round(beh_agg["opening_probe_rate"], 4),
                    "k4_branch_tv": round(beh_agg["k4_branching_tv"], 4),
                    "rep_explore": round(beh_agg["repeat_rate_explore"], 4),
                    "rep_exploit": round(beh_agg["repeat_rate_exploit"], 4),
                }
            )

            behavior_rows.append(
                {
                    "regime": reg,
                    "method": m,
                    "rep_explore": round(beh_agg["repeat_rate_explore"], 4),
                    "rep_exploit": round(beh_agg["repeat_rate_exploit"], 4),
                    "rep_all": round(beh_agg["repeat_rate_all"], 4),
                    "uniq_explore": round(beh_agg["unique_actions_explore"], 4),
                    "open_probe_rate": round(beh_agg["opening_probe_rate"], 4),
                    "k4_branch_tv": round(beh_agg["k4_branching_tv"], 4),
                    **{f"exp_{k}": round(v, 4) for k, v in beh_agg["action_dist_explore"].items()},
                    **{f"expt_{k}": round(v, 4) for k, v in beh_agg["action_dist_exploit"].items()},
                }
            )

            for k_str, l1_d in l1_summary.items():
                layer1_rows.append({"regime": reg, "method": m, "K": k_str, **{k: round(v, 4) for k, v in l1_d.items()}})

            for th_str in ("0", "1", "2", "3"):
                theta_rows.append(
                    {
                        "regime": reg,
                        "method": m,
                        "theta": th_str,
                        "mean": round(float(np.mean([r["end_to_end"]["by_theta"][th_str]["mean"] for r in recs])), 4),
                        "median": round(float(np.mean([r["end_to_end"]["by_theta"][th_str]["median"] for r in recs])), 4),
                    }
                )

        contrasts: dict[str, dict] = {}
        contrast_pairs = [
            ("drqn_aux_yr", "drqn_aux_yr_masked"),
            ("drqn_aux_yr", "drqn_prior_only"),
            ("drqn_aux_yr", "drqn_vanilla"),
            ("drqn_aux_yr", "drqn_aux_y"),
            ("drqn_aux_yr", "drqn_aux_r"),
            ("drqn_aux_y", "drqn_vanilla"),
            ("drqn_aux_r", "drqn_vanilla"),
            ("drqn_aux_yr", "rl2_ppo"),
            ("drqn_aux_yr", "structural_model"),
            ("drqn_aux_yr", "contextual_ucb"),
            ("drqn_aux_yr", "arm_ucb"),
            ("drqn_aux_yr", "greedy"),
            ("drqn_aux_yr", "random"),
            ("rl2", "rl2_masked"),
            ("rl2", "prior_only"),
            ("rl2", "contextual_ucb"),
            ("structural_model", "contextual_ucb"),
        ]
        metric_specs = [
            ("total_return", delta_total),
            ("layer1_commitment_k4", delta_info),
            ("layer1_adaptive_k4", delta_info),
            ("layer1_net_adaptive_k4", delta_info),
        ]
        for a_name, b_name in contrast_pairs:
            if a_name not in methods or b_name not in methods:
                continue
            c_key = f"{a_name}_minus_{b_name}"
            contrasts[c_key] = {}
            for m_key, thresh in metric_specs:
                a_mat = np.stack(per_method_arrays[a_name][m_key], axis=0)
                b_mat = np.stack(per_method_arrays[b_name][m_key], axis=0)
                res_b = hierarchical_paired_bootstrap(
                    a_mat,
                    b_mat,
                    practical_threshold=thresh,
                    n_boot=n_boot,
                    seed=12345,
                )
                contrasts[c_key][m_key] = res_b
                contrast_rows.append(
                    {
                        "regime": reg,
                        "contrast": c_key,
                        "metric": m_key,
                        "diff_mean": round(float(res_b["diff_mean"]), 4),
                        "diff_median": round(float(res_b["diff_median"]), 4),
                        "ci_low": round(float(res_b["ci_low"]), 4),
                        "ci_high": round(float(res_b["ci_high"]), 4),
                        "win_rate": round(float(res_b["win_rate"]), 4),
                        "tie_rate": round(float(res_b["tie_rate"]), 4),
                        "loss_rate": round(float(res_b["loss_rate"]), 4),
                        "threshold": thresh,
                        "verdict": res_b["verdict"],
                    }
                )

        results_by_regime[reg] = {
            "gate0": gate0[reg],
            "methods": agg_methods,
            "paired_contrasts": contrasts,
        }

    summary = {
        "study_id": study_id,
        "git_commit": manifest["git_commit"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "regimes": results_by_regime,
    }
    _atomic_write_json(out_dir / "summary.json", summary)

    tables_dir = out_dir / "tables"
    _write_csv(tables_dir / "methods.csv", method_rows)
    _write_csv(tables_dir / "paired_contrasts.csv", contrast_rows)
    _write_csv(tables_dir / "layer1_k248.csv", layer1_rows)
    _write_csv(tables_dir / "by_theta.csv", theta_rows)
    _write_csv(tables_dir / "behavior_repetition.csv", behavior_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPEC v3.0 Core/Dev Study.")
    parser.add_argument("--config", required=True, help="Path to JSON study config.")
    parser.add_argument("--workers", type=int, default=None, help="Override parallel worker count.")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 1))
    t0 = time.perf_counter()
    summary = run_study_v3(cfg, workers=workers)
    out_dir = cfg.get("output_dir", f"runs/{cfg['study_id']}")
    print(f"[{cfg['study_id']}] Complete in {time.perf_counter() - t0:.1f}s -> {out_dir}/summary.json")
    for reg, data in summary["regimes"].items():
        print(f"\n=== {reg} (Bayes-Opt={data['gate0']['bayes_optimal_value']:.3f}, Blind={data['gate0']['blind_value']:.3f}) ===")
        for m, md in data["methods"].items():
            print(
                f"  {m:19s} | Ret mean={md['total_mean']:+6.3f} (sd={md['total_seed_sd']:4.2f}) med={md['total_median']:+6.3f} "
                f"| L1_K4 adapt={md['layer1_k4_adaptive_mean']:+6.3f} commit={md['layer1_k4_commitment_mean']:+6.3f} "
                f"| L3_gap={md['layer3_k4_live_gap_mean']:+5.3f} "
                f"| probe={md['behavior']['opening_probe_rate']:.2f} tv={md['behavior']['k4_branching_tv']:.2f} "
                f"rep(E/X)={md['behavior']['repeat_rate_explore']:.2f}/{md['behavior']['repeat_rate_exploit']:.2f}"
            )


if __name__ == "__main__":
    main()
