"""Modular agents: opponents, exploitation students, and exploration policies."""

from .base import Explorer, Opponent, Student
from .explorers import (
    GreedyExplorer,
    OracleExplorer,
    RandomExplorer,
    RecurrentMetaExplorer,
    UCBExplorer,
    train_meta_explorer,
)
from .opponents import QLearningOpponent, ReactiveOpponent, build_opponent
from .students import BayesianRewardStudent, TabularQStudent, build_student

__all__ = [
    "BayesianRewardStudent",
    "Explorer",
    "GreedyExplorer",
    "Opponent",
    "OracleExplorer",
    "QLearningOpponent",
    "RandomExplorer",
    "ReactiveOpponent",
    "RecurrentMetaExplorer",
    "Student",
    "TabularQStudent",
    "UCBExplorer",
    "build_opponent",
    "build_student",
    "train_meta_explorer",
]
