"""Single config-driven CLI entrypoint for Path-A experiments.

Usage:
    PYTHONPATH=src .venv/bin/python -m mamrl.runner --config configs/pilot_active_explore.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from mamrl.agents.explorers import (
    GreedyExplorer,
    OracleExplorer,
    RandomExplorer,
    UCBExplorer,
    train_meta_explorer,
)
from mamrl.envs.active_probe_game import generate_task_suite
from mamrl.eval.oracle import audit_task_suite
from mamrl.eval.protocol import evaluate_suite


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def run_experiment(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    mode = str(config["mode"])
    diversity = str(config.get("diversity", "low"))
    n_train = int(config.get("n_train_tasks", 32))
    n_test = int(config.get("n_test_tasks", 64))
    opponent_kind = str(config.get("opponent_kind", "q_learning"))
    student_kind = str(config.get("student_kind", "bayesian"))
    explore_steps = int(config.get("explore_steps", 4))
    total_horizon = int(config.get("total_horizon", 32))
    updates = int(config.get("updates", 60))
    batch_size = int(config.get("batch_size", 16))
    ucb_beta = float(config.get("ucb_beta", 0.5))
    objective = str(config.get("objective", "total_return"))

    train_tasks = generate_task_suite(n_train, mode=mode, diversity=diversity, seed=seed)
    test_tasks = generate_task_suite(n_test, mode=mode, diversity=diversity, seed=seed + 10000)

    # Gate 0: Always audit exact 2-turn Oracle VoI & active-opening necessity before training
    oracle_audit = audit_task_suite(test_tasks)

    meta_full, full_trace = train_meta_explorer(
        train_tasks,
        opponent_kind=opponent_kind,
        student_kind=student_kind,
        explore_steps=explore_steps,
        total_horizon=total_horizon,
        updates=updates,
        batch_size=batch_size,
        mask_feedback=False,
        objective=objective,
        seed=seed,
    )
    meta_masked, masked_trace = train_meta_explorer(
        train_tasks,
        opponent_kind=opponent_kind,
        student_kind=student_kind,
        explore_steps=explore_steps,
        total_horizon=total_horizon,
        updates=updates,
        batch_size=batch_size,
        mask_feedback=True,
        objective=objective,
        seed=seed,
    )

    factories = {
        "greedy": lambda _spec: GreedyExplorer(),
        "random": lambda _spec: RandomExplorer(n_actions=4),
        "ucb": lambda _spec: UCBExplorer(beta=ucb_beta),
        "meta_full": lambda _spec: meta_full,
        "meta_masked": lambda _spec: meta_masked,
        "oracle": lambda spec: OracleExplorer(spec),
    }

    continuous_eval = evaluate_suite(
        test_tasks,
        factories,
        opponent_kind=opponent_kind,
        student_kind=student_kind,
        explore_steps=explore_steps,
        total_horizon=total_horizon,
        restore_opponent=False,
        seed=seed + 20000,
    )
    controlled_eval = evaluate_suite(
        test_tasks,
        factories,
        opponent_kind=opponent_kind,
        student_kind=student_kind,
        explore_steps=explore_steps,
        total_horizon=total_horizon,
        restore_opponent=True,
        seed=seed + 20000,
    )

    return {
        "experiment_name": config.get("name", "unnamed"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": config,
        "gate0_oracle_audit": oracle_audit,
        "training_summary": {
            "meta_full_start": full_trace[0] if full_trace else None,
            "meta_full_end": full_trace[-1] if full_trace else None,
            "meta_masked_start": masked_trace[0] if masked_trace else None,
            "meta_masked_end": masked_trace[-1] if masked_trace else None,
        },
        "continuous_deployment": continuous_eval,
        "controlled_info_only": controlled_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a declarative Path-A MAMRL experiment.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    result = run_experiment(config)

    out_path = Path(args.output) if args.output else Path(config.get("output", "runs/latest_result.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    audit = result["gate0_oracle_audit"]
    print(f"[{result['experiment_name']}] Saved -> {out_path}")
    print(
        f"  Gate-0 Oracle: VoI>0={audit['positive_voi_rate']:.1%}, "
        f"must_change_opening={audit['must_change_opening_rate']:.1%}, "
        f"mean_VoI={audit['mean_voi']:.4f}"
    )
    for name, m in result["continuous_deployment"]["methods"].items():
        print(
            f"  {name:12s} | total mean={m['total_mean']:+.3f} "
            f"median={m['total_median']:+.3f} | opp={m['counterpart_mean']:+.3f}"
        )


if __name__ == "__main__":
    main()
