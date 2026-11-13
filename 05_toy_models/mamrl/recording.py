"""Versioned researcher-only run artifacts, validated before saving or viewing."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random

from .common import finite_reward, validate_action, validate_n, validate_seed
from .environment import Observation
from .rollout import EpisodeResult, Transition
from .rules import RuleSpec, RuleTerm, compile_rule, evaluate_rule
from .tensor import RewardTensor

SCHEMA_VERSION = 1
SEMANTICS_VERSION = "memory-one-additive-v1"


def _pair(values, name):
    values = tuple(values)
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    return values


def _text(value, name):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _equal(expected, actual):
    """Strict schema types, with tolerance only for floating-point values."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(_equal(value, actual[key]) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(_equal(a, b) for a, b in zip(expected, actual)))
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        try:
            actual = finite_reward(actual)
        except (TypeError, ValueError):
            return False
        return math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9)
    return type(expected) is type(actual) and expected == actual


def _require_equal(expected, actual, name):
    if not _equal(expected, actual):
        raise ValueError(f"inconsistent {name}")


def _validate_observation(observation, n):
    if not isinstance(observation, Observation):
        raise TypeError("observations must be Observation values")
    validate_action(observation.previous_own_action, n)
    validate_action(observation.previous_opponent_action, n)
    finite_reward(observation.previous_own_reward)
    if type(observation.episode_start_flag) is not bool:
        raise TypeError("episode_start_flag must be bool")


def describe_term(term: RuleTerm) -> str:
    coefficient = f"{term.coefficient:g}"
    conditions = {
        "coordination": "현재 두 플레이어의 행동이 같으면",
        "repeat_own": "현재 자기 행동이 직전 자기 행동과 같으면",
        "copy_opponent": "현재 자기 행동이 직전 상대 행동과 같으면",
        "action_bonus": f"현재 자기 행동이 {term.action}이면",
        "probe_repeat": "직전 자기 행동을 반복하면 +1, 직전 상대 행동이면 -3, 그 외 정보 행동은 -0.25에 계수",
        "probe_copy": "직전 상대 행동을 복사하면 +1, 직전 자기 행동이면 -3, 그 외 정보 행동은 -0.5에 계수",
    }
    if term.kind == "cyclic_zero_sum":
        return f"순환 비교 결과(+1/0/-1)에 계수 {coefficient}를 곱함 (짝수 n의 반대편은 무승부)"
    if term.kind in ("probe_repeat", "probe_copy"):
        return f"{conditions[term.kind]} {coefficient}를 곱함 (두 기준이 같으면 정답 우선)"
    return f"{conditions[term.kind]} {coefficient}를 더함"


def _build_record(spec, tensor, episode, *, family, seed, policy_seeds,
                  policy_names, python_version):
    if not isinstance(spec, RuleSpec) or not isinstance(tensor, RewardTensor):
        raise TypeError("spec and tensor must be RuleSpec and RewardTensor")
    if not isinstance(episode, EpisodeResult):
        raise TypeError("episode must be an EpisodeResult")
    if compile_rule(spec) != tensor:
        raise ValueError("tensor does not match the supplied rule specification")
    n = spec.n
    family = _text(family, "family")
    python_version = _text(python_version, "python_version")
    seed = validate_seed(seed)
    policy_seeds = tuple(validate_seed(s) for s in _pair(policy_seeds, "policy_seeds"))
    policy_names = tuple(_text(p, "policy name") for p in _pair(policy_names, "policies"))
    if not episode.transitions:
        raise ValueError("episode must contain at least one transition")
    descriptions = [describe_term(term) for term in spec.terms]
    counts = [[0] * n for _ in range(2)]
    totals = [0.0, 0.0]
    transitions = []
    expected_observations = None
    for index, transition in enumerate(episode.transitions):
        if not isinstance(transition, Transition):
            raise TypeError("episode transitions must be Transition values")
        observations = _pair(transition.observations, "observations")
        next_observations = _pair(transition.next_observations, "next_observations")
        for obs in (*observations, *next_observations):
            _validate_observation(obs, n)
        previous = (observations[0].previous_own_action, observations[1].previous_own_action)
        for player in (0, 1):
            if observations[player].previous_opponent_action != previous[1-player]:
                raise ValueError("observation has inconsistent global/local action ordering")
            if observations[player].episode_start_flag != (index == 0):
                raise ValueError("invalid episode start boundary")
        if index == 0:
            for obs in observations:
                _require_equal(0.0, obs.previous_own_reward, "initial reward")
            if seed is not None:
                rng = random.Random(seed)
                if previous != (rng.randrange(n), rng.randrange(n)):
                    raise ValueError("initial actions do not match environment seed")
        else:
            _require_equal(expected_observations, [asdict(o) for o in observations], "transition history")
        actions = tuple(validate_action(a, n) for a in _pair(transition.actions, "actions"))
        rewards = tuple(finite_reward(r) for r in _pair(transition.rewards, "rewards"))
        expected_rewards = [tensor[p, *previous, *actions] for p in (0, 1)]
        _require_equal(expected_rewards, list(rewards), "tensor reward")
        _require_equal(False, transition.terminated, "terminated flag")
        _require_equal(index == len(episode.transitions)-1, transition.truncated, "truncated flag")
        expected_next = [
            asdict(Observation(actions[p], actions[1-p], rewards[p], False)) for p in (0, 1)
        ]
        _require_equal(expected_next, [asdict(o) for o in next_observations], "next observations")
        expected_observations = expected_next
        breakdown = []
        for term, label in zip(spec.terms, descriptions):
            contributions = evaluate_rule(RuleSpec(n, (term,)), previous, actions)
            breakdown.append({**asdict(term), "label": label, "contributions": list(contributions)})
        for p in (0, 1):
            _require_equal(rewards[p], sum(item["contributions"][p] for item in breakdown), "term contributions")
            counts[p][actions[p]] += 1
            totals[p] = finite_reward(totals[p] + rewards[p])
        transitions.append({
            "observations": [asdict(o) for o in observations],
            "actions": list(actions), "rewards": list(rewards),
            "next_observations": expected_next,
            "terminated": False, "truncated": transition.truncated,
            "turn": index+1, "previous_joint_action": list(previous),
            "cumulative_rewards": list(totals), "reward_terms": breakdown,
        })
    actual_totals = [finite_reward(r) for r in _pair(episode.total_rewards, "total_rewards")]
    _require_equal(totals, actual_totals, "total rewards")
    payload = {"shape": list(tensor.shape), "values": list(tensor.values)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode("utf-8")).hexdigest()
    steps = len(transitions)
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {"n": n, "family": family, "seed": seed,
                     "policy_seeds": list(policy_seeds), "policies": list(policy_names),
                     "python_version": python_version, "semantics_version": SEMANTICS_VERSION},
        "rule": {"n": n, "terms": [asdict(t) for t in spec.terms]},
        "rule_descriptions": descriptions,
        "tensor": {**payload, "sha256": digest},
        "summary": {"steps": steps, "total_rewards": totals,
                    "mean_rewards": [r/steps for r in totals], "action_counts": counts,
                    "terminated": False, "truncated": True},
        "transitions": transitions,
    }


def build_record(spec, tensor, episode, *, family, seed, policy_seeds, policy_names):
    """Snapshot a completed episode without adding any information to policies."""
    return _build_record(spec, tensor, episode, family=family, seed=seed,
                         policy_seeds=policy_seeds, policy_names=policy_names,
                         python_version=platform.python_version())


def validate_record(record) -> None:
    """Reconstruct canonical values and reject inconsistent or unknown schemas."""
    if not isinstance(record, dict):
        raise TypeError("record must be an object")
    # Reject unsupported Python objects and all non-finite JSON numbers up front.
    json.dumps(record, allow_nan=False)
    try:
        if type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        # Hash the saved representation before numeric normalization (0 != 0.0 in JSON).
        payload = {"shape": record["tensor"]["shape"], "values": record["tensor"]["values"]}
        saved_digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        _require_equal(saved_digest, record["tensor"]["sha256"], "saved tensor digest")
        metadata = record["metadata"]
        n = validate_n(metadata["n"])
        spec = RuleSpec(record["rule"]["n"], (RuleTerm(**t) for t in record["rule"]["terms"]))
        tensor = RewardTensor(n, record["tensor"]["values"])
        transitions = tuple(Transition(
            tuple(Observation(**o) for o in t["observations"]), tuple(t["actions"]), tuple(t["rewards"]),
            tuple(Observation(**o) for o in t["next_observations"]), t["terminated"], t["truncated"],
        ) for t in record["transitions"])
        episode = EpisodeResult(transitions, tuple(record["summary"]["total_rewards"]))
        canonical = _build_record(spec, tensor, episode, family=metadata["family"],
                                  seed=metadata["seed"], policy_seeds=metadata["policy_seeds"],
                                  policy_names=metadata["policies"], python_version=metadata["python_version"])
        canonical["tensor"]["sha256"] = saved_digest
        _require_equal(canonical, record, "record schema or derived values")
    except (KeyError, IndexError, AttributeError) as exc:
        raise ValueError("incomplete or malformed run record") from exc


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def load_record(path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"),
                        object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    validate_record(record)
    return record


def save_run(record, output_dir) -> tuple[Path, Path]:
    """Create a new artifact directory, leaving all existing paths untouched."""
    validate_record(record)
    from .report import render_report

    contents = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    report = render_report(record)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    json_path, report_path = directory / "run.json", directory / "report.html"
    try:
        with json_path.open("x", encoding="utf-8") as stream:
            stream.write(contents)
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(report)
    except OSError:
        # Both names are exclusively owned by this newly created directory.
        json_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        directory.rmdir()
        raise
    return json_path, report_path
