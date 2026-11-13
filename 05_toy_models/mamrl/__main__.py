"""Run a reproducible interaction: PYTHONPATH=05_toy_models python3 -m mamrl."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .environment import MemoryOneEnv
from .policy import RandomPolicy
from .rollout import run_episode
from .rules import compile_rule, generate_rule
from .recording import build_record, save_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--family", choices=("zero_sum", "coordination", "memory_one", "mixed"),
        default="mixed",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="save run.json and a local report.html in a new directory")
    parser.add_argument("--checkpoint", type=Path, help="optional recurrent learner checkpoint for player 0")
    args = parser.parse_args()
    try:
        spec = generate_rule(n=args.n, seed=args.seed, family=args.family)
        tensor = compile_rule(spec)
        env = MemoryOneEnv(tensor, max_steps=args.steps)
        if args.checkpoint is None:
            policies = (RandomPolicy(), RandomPolicy())
        else:
            # Keep the standard-library core usable without training dependencies.
            from .recurrent import RecurrentPolicy, load_checkpoint
            policies = (RecurrentPolicy(load_checkpoint(args.checkpoint)), RandomPolicy())
        policy_seeds = (args.seed + 1, args.seed + 2)
        result = run_episode(env, policies, seed=args.seed, policy_seeds=policy_seeds)
        artifact_paths = None
        if args.output_dir is not None:
            record = build_record(
                spec, tensor, result, family=args.family, seed=args.seed,
                policy_seeds=policy_seeds, policy_names=tuple(type(p).__name__ for p in policies),
            )
            artifact_paths = save_run(record, args.output_dir)
    except (TypeError, ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    output = {
        "rule": asdict(spec),
        "tensor_shape": tensor.shape,
        "steps": len(result.transitions),
        "total_rewards": result.total_rewards,
        "terminated": result.transitions[-1].terminated,
        "truncated": result.transitions[-1].truncated,
    }
    if args.trace:
        output["transitions"] = [asdict(item) for item in result.transitions]
    if artifact_paths is not None:
        output["artifacts"] = {
            "run_json": str(artifact_paths[0].resolve()),
            "report_html": str(artifact_paths[1].resolve()),
        }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
