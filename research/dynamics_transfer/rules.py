"""Paired perturbations of one coordination transition in v2 structured games."""
from dataclasses import replace
from math import isfinite

from mamrl.structured_games import Predicate


def investment_clause(rule):
    """Find the first accumulation clause and validate the supported family grammar."""
    index = next(i for i, clause in enumerate(rule.transitions) if clause.delta > 0)
    clause = rule.transitions[index]
    expected = (Predicate('own_is', action=rule.roles[0]),)
    if rule.family == 'sequence':
        expected += (Predicate('repeat', lag=2),)
    if clause.scope != 'both' or clause.delta != 1 or clause.when != expected:
        raise ValueError('unsupported investment clause')
    return index, clause


def variant(base, distance, extra_lag):
    index, clause = investment_clause(base)
    if not isfinite(distance) or not 0 <= distance < clause.probability:
        raise ValueError('distance must leave a positive gate probability')
    if extra_lag not in (0, 1):
        raise ValueError('extra_lag must be zero or one')
    if distance == 0 and not extra_lag:
        return base
    predicates = clause.when + ((Predicate('repeat', lag=1),) if extra_lag else ())
    modified = replace(clause, probability=clause.probability - distance, when=predicates)
    return replace(base, transitions=base.transitions[:index] + (modified,) + base.transitions[index + 1:])
