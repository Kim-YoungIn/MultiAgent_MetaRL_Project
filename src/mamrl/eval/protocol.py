"""Paired evaluation protocol separating Controlled Info Gain vs Continuous Deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from mamrl.agents.base import Explorer
from mamrl.agents.opponents import build_opponent
from mamrl.agents.students import build_student
from mamrl.envs.active_probe_game import ActiveProbeGame, ActiveProbeSpec
from .metrics import summarize_method, summarize_paired_contrast


@dataclass
class EpisodeOutcome:
    task_id: str
    explore_return: float
    assess_return: float
    total_return: float
    counterpart_return: float
    assess_curve: np.ndarray


def _run_paired_episode(
    spec: ActiveProbeSpec,
    explorer: Explorer,
    *,
    opponent_kind: str,
    student_kind: str,
    explore_steps: int,
    total_horizon: int,
    restore_opponent: bool,
    ep_seed: int,
) -> EpisodeOutcome:
    n_actions = spec.n_actions
    env = ActiveProbeGame(spec, seed=ep_seed)
    opp = build_opponent(opponent_kind, n_actions=n_actions, seed=ep_seed + 11)
    student = build_student(student_kind, n_actions=n_actions, spec=spec, seed=ep_seed + 23)

    obs_f, obs_o = env.reset(seed=ep_seed)
    opp.reset(seed=ep_seed + 11)
    student.reset(seed=ep_seed + 23)
    explorer.reset(seed=ep_seed + 37)

    # Snapshot pre-exploration opponent state for pure-information controlled evaluation
    opp_init_snapshot = opp.snapshot()

    explore_ret = 0.0
    opp_explore_ret = 0.0
    for t in range(explore_steps):
        a_f = explorer.act(obs_f, student, t)
        a_o = opp.act(obs_o)
        res = env.step(a_f, a_o)
        student.update(obs_f, a_f, res.focal_reward, res.focal_obs)
        opp.update(obs_o, a_o, res.opp_reward, res.opp_obs)
        obs_f, obs_o = res.focal_obs, res.opp_obs
        explore_ret += res.focal_reward
        opp_explore_ret += res.opp_reward

    if restore_opponent:
        # Pure information evaluation: keep Student posterior, restore opponent & env start
        opp.restore(opp_init_snapshot)
        obs_f, obs_o = env.reset(seed=ep_seed + 99)

    assess_steps = total_horizon - explore_steps
    assess_curve = np.zeros(assess_steps, dtype=np.float64)
    opp_assess_ret = 0.0

    for k in range(assess_steps):
        a_f = student.act(obs_f)
        a_o = opp.act(obs_o)
        res = env.step(a_f, a_o)
        if not restore_opponent:
            student.update(obs_f, a_f, res.focal_reward, res.focal_obs)
            opp.update(obs_o, a_o, res.opp_reward, res.opp_obs)
        obs_f, obs_o = res.focal_obs, res.opp_obs
        assess_curve[k] = res.focal_reward
        opp_assess_ret += res.opp_reward

    assess_ret = float(np.sum(assess_curve))
    return EpisodeOutcome(
        task_id=spec.task_id,
        explore_return=explore_ret,
        assess_return=assess_ret,
        total_return=explore_ret + assess_ret,
        counterpart_return=opp_explore_ret + opp_assess_ret,
        assess_curve=assess_curve,
    )


def evaluate_suite(
    tasks: list[ActiveProbeSpec],
    explorer_factories: dict[str, Callable[[ActiveProbeSpec], Explorer]],
    *,
    opponent_kind: str = "q_learning",
    student_kind: str = "bayesian",
    explore_steps: int = 4,
    total_horizon: int = 32,
    restore_opponent: bool = False,
    seed: int = 0,
) -> dict[str, dict]:
    """Evaluate all explorers on the exact same paired task seeds."""
    raw: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, dict] = {}

    for name, factory in explorer_factories.items():
        outcomes = [
            _run_paired_episode(
                spec,
                factory(spec),
                opponent_kind=opponent_kind,
                student_kind=student_kind,
                explore_steps=explore_steps,
                total_horizon=total_horizon,
                restore_opponent=restore_opponent,
                ep_seed=seed + i * 101,
            )
            for i, spec in enumerate(tasks)
        ]
        tot = np.array([o.total_return for o in outcomes], dtype=np.float64)
        exp = np.array([o.explore_return for o in outcomes], dtype=np.float64)
        ass = np.array([o.assess_return for o in outcomes], dtype=np.float64)
        opp = np.array([o.counterpart_return for o in outcomes], dtype=np.float64)
        crv = np.stack([o.assess_curve for o in outcomes], axis=0)
        raw[name] = {"total": tot, "assess": ass, "counterpart": opp}
        summary[name] = summarize_method(tot, exp, ass, opp, crv)

    contrasts: dict[str, dict[str, float]] = {}
    metric_key = "assess" if restore_opponent else "total"
    for name in explorer_factories:
        for base in ("greedy", "ucb", "meta_masked"):
            if name != base and base in raw:
                contrasts[f"{name}_minus_{base}"] = summarize_paired_contrast(
                    raw[name][metric_key],
                    raw[base][metric_key],
                    raw[name]["counterpart"],
                    raw[base]["counterpart"],
                )

    return {"methods": summary, "paired_contrasts": contrasts}
