"""Analyze paired block results from the adaptation experiment.

Run with ``python -m research.adapt_v1.analyze OUTPUT``. Each block supplies
metrics.npz and may supply shift.npz. Outputs are written into OUTPUT/analysis.
Inference resamples independent training blocks, preserving every pairing inside
a block. Curves and subgroup summaries are descriptive.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TARGETS = (0.75, 0.5, 0.25)
CURVE_FIELDS = ("value", "brier")
ONLINE_FIELDS = (
    "online_reward", "online_expected", "opponent_reward", "opponent_move",
    "opponent_prob", "field_abs",
)
PRIMARY = (
    ("adaptation_vs_independent", "g128_p75_mix", "independent", 0.75),
    ("diversity_on_balanced", "g128_p50_mix", "g128_p100_mix", 0.5),
)


def load_block(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {"signs", "rate", "source", "methods", "budgets", *CURVE_FIELDS, *ONLINE_FIELDS}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")
    signs = data["signs"]
    if signs.ndim != 2 or signs.shape[1] != 2 or not np.isin(signs, (-1, 1)).all():
        raise ValueError(f"{path}: signs must be N × 2 and contain only -1 and 1")
    n = len(signs)
    methods = data["methods"]
    budgets = data["budgets"]
    if methods.ndim != 1 or len(set(methods.tolist())) != len(methods):
        raise ValueError(f"{path}: methods must be a unique one-dimensional array")
    if budgets.ndim != 1 or budgets.tolist() != [0, 1, 2, 4, 8, 16]:
        raise ValueError(f"{path}: unexpected budgets {budgets}")
    for name in ("rate", "source"):
        if data[name].shape != (n,):
            raise ValueError(f"{path}: {name} must have shape (N,)")
    if not np.isin(data["source"], (0, 1)).all():
        raise ValueError(f"{path}: source must contain 0 or 1")
    if not np.isclose(data["rate"][:, None], [0.12, 0.65]).any(axis=1).all():
        raise ValueError(f"{path}: unexpected opponent learning rate")
    m, k = len(methods), len(budgets)
    for name in CURVE_FIELDS:
        if data[name].shape != (m, n, k, 2):
            raise ValueError(f"{path}: {name} has wrong shape {data[name].shape}")
    horizon = data["online_reward"].shape[-1]
    if horizon < 4:
        raise ValueError(f"{path}: horizon must be at least four")
    for name in ONLINE_FIELDS:
        if data[name].shape != (m, n, horizon):
            raise ValueError(f"{path}: {name} has wrong shape {data[name].shape}")
    for name in ("rate", *CURVE_FIELDS, *ONLINE_FIELDS):
        if not np.isfinite(data[name]).all():
            raise ValueError(f"{path}: {name} contains non-finite values")
    return data


def standardize(values, same, target, mask=None):
    """Average conditional means with a specified probability of matching signs."""
    if mask is None:
        mask = np.ones(len(same), dtype=bool)
    if not (mask & same).any() or not (mask & ~same).any():
        raise ValueError("each analysis stratum must contain both sign relationships")
    return target * values[mask & same].mean(axis=0) + (1 - target) * values[mask & ~same].mean(axis=0)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_blocks(output):
    manifest_path = output / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected = manifest["config"]["blocks"]
    if not isinstance(expected, int) or expected <= 0:
        raise ValueError("manifest config.blocks must be a positive integer")
    dirs = sorted(path for path in output.glob("block_[0-9][0-9][0-9]") if path.is_dir())
    if len(dirs) != expected:
        raise ValueError(f"expected {expected} block directories, found {len(dirs)}")
    blocks = []
    for folder in dirs:
        files = {"matched": folder / "metrics.npz"}
        if (folder / "shift.npz").exists():
            files["shift"] = folder / "shift.npz"
        blocks.append((folder.name, {key: load_block(path) for key, path in files.items()}))
    if len({tuple(sorted(sets)) for _, sets in blocks}) != 1:
        raise ValueError("shift.npz must be present for every block or none")
    reference = blocks[0][1]["matched"]
    for block, sets in blocks:
        for name, data in sets.items():
            if not np.array_equal(data["methods"], reference["methods"]) or not np.array_equal(data["budgets"], reference["budgets"]):
                raise ValueError(f"{block}/{name}: method or budget ordering differs")
    names = set(reference["methods"].tolist())
    for _, left, right, _ in PRIMARY:
        if left not in names or right not in names:
            raise ValueError(f"missing primary method {left} or {right}")
    return blocks


def summarize(blocks):
    curves, online, effects = [], [], []
    for block, populations in blocks:
        for population, data in populations.items():
            same = data["signs"][:, 0] == data["signs"][:, 1]
            for target in TARGETS:
                for index, method in enumerate(data["methods"].tolist()):
                    for rate in ("all", "slow", "fast"):
                        mask = np.ones(len(same), dtype=bool) if rate == "all" else np.isclose(data["rate"], 0.12 if rate == "slow" else 0.65)
                        for metric in CURVE_FIELDS:
                            values = standardize(data[metric][index], same, target, mask)
                            for ki, budget in enumerate(data["budgets"].tolist()):
                                for ai, arena in enumerate(("source", "other")):
                                    curves.append(dict(block=block, population=population, target_same=target,
                                                       rate=rate, method=method, metric=metric, budget=budget, arena=arena,
                                                       mean=float(values[ki, ai]), gain=float(values[ki, ai] - values[0, ai])))
                    for rate in ("all", "slow", "fast"):
                        mask = np.ones(len(same), dtype=bool) if rate == "all" else np.isclose(data["rate"], 0.12 if rate == "slow" else 0.65)
                        for metric in ONLINE_FIELDS:
                            series = standardize(data[metric][index], same, target, mask)
                            for period, value in (("all", series.mean()), ("early4", series[:4].mean()), ("late4", series[-4:].mean())):
                                online.append(dict(block=block, population=population, target_same=target,
                                                   rate=rate, method=method, metric=metric, period=period,
                                                   mean=float(value)))
                if population == "matched":
                    lookup = {name: i for i, name in enumerate(data["methods"].tolist())}
                    for hypothesis, left, right, goal in PRIMARY:
                        if target == goal:
                            def gain(name):
                                arr = data["value"][lookup[name], :, :, 1]
                                return float(standardize(arr[:, 3] - arr[:, 0], same, target))
                            effects.append(dict(block=block, hypothesis=hypothesis, left=left, right=right,
                                                target_same=target, left_gain=gain(left), right_gain=gain(right),
                                                difference=gain(left) - gain(right)))
    return curves, online, effects


def bootstrap(effects, count=10000, seed=79113):
    rng = np.random.default_rng(seed)
    result = []
    for hypothesis, left, right, target in PRIMARY:
        values = np.array([row["difference"] for row in effects if row["hypothesis"] == hypothesis])
        if len(values) < 2:
            raise ValueError("at least two completed blocks are required for inference")
        draws = rng.integers(0, len(values), size=(count, len(values)))
        low, high = np.quantile(values[draws].mean(axis=1), [0.0125, 0.9875])
        result.append(dict(hypothesis=hypothesis, left=left, right=right, target_same=target,
                           blocks=len(values), estimate=float(values.mean()), lower_97_5=float(low),
                           upper_97_5=float(high), resamples=count, seed=seed))
    return result


def plot_curves(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    populations = sorted({row["population"] for row in curves}, key=lambda x: x != "matched")
    methods = ("g128_p75_mix", "g128_p50_mix", "g128_p100_mix", "independent")
    labels = ("Meta p=.75", "Meta p=.50", "Meta p=1.00", "Independent")
    colors = ("#2455a4", "#16846b", "#a85b28", "#555555")
    fig, axes = plt.subplots(len(populations), 2, figsize=(10, 3.7 * len(populations)), squeeze=False, sharex=True, sharey=True)
    for row_index, population in enumerate(populations):
        for column, target in enumerate((0.75, 0.25)):
            ax = axes[row_index, column]
            for method, label, color in zip(methods, labels, colors):
                subset = [r for r in curves if r["population"] == population and r["target_same"] == target
                          and r["method"] == method and r["metric"] == "value" and r["arena"] == "other" and r["rate"] == "all"]
                budgets = sorted({r["budget"] for r in subset})
                means = [np.mean([r["gain"] for r in subset if r["budget"] == k]) for k in budgets]
                ax.plot(budgets, means, marker="o", markersize=3, label=label, color=color)
            ax.axhline(0, color="#999999", linewidth=0.7)
            ax.set_title(f"{population.capitalize()} tasks, P(same)={target:.2f}")
            ax.set_xlabel("Interactions before query")
            ax.set_ylabel("Other-arena expected payoff gain")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Adaptation curves by task distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    blocks = read_blocks(args.output)
    curves, online, effects = summarize(blocks)
    inference = bootstrap(effects)
    output = args.output / "analysis"
    output.mkdir(exist_ok=True)
    write_csv(output / "curves_by_block.csv", curves)
    write_csv(output / "online_by_block.csv", online)
    write_csv(output / "primary_by_block.csv", effects)
    write_csv(output / "primary_inference.csv", inference)
    plot_curves(curves, output / "adaptation_curves.png")
    print(json.dumps(inference, indent=2))


if __name__ == "__main__":
    main()
