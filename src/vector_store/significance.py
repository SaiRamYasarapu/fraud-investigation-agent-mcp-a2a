"""
Is one search setup really better, or did it win by luck on these 63 queries?

Two checks, both on per-query scores (one score per query, in the same query
order for every setup):
- bootstrap_ci: resample the queries with replacement many times and keep the
  middle 95% of the averages. A wide range means the average is not settled.
- paired_test: compare two setups on the same queries. It gives the average
  difference, a 95% range for that difference, and a p-value from a sign-flip
  test. A p-value below 0.05 means the difference is unlikely to be luck.

Both use a fixed seed, so the same scores always give the same result.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

RESAMPLES = 10_000
LEVEL = 0.95


@dataclass
class Interval:
    mean: float
    low: float
    high: float


@dataclass
class PairedResult:
    mean_diff: float
    low: float
    high: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 1 - LEVEL


def _middle_range(values: list[float], level: float) -> tuple[float, float]:
    values = sorted(values)
    tail = (1 - level) / 2
    last = len(values) - 1
    return values[round(tail * last)], values[round((1 - tail) * last)]


def bootstrap_ci(scores: list[float], resamples: int = RESAMPLES, level: float = LEVEL,
                 seed: int = 0) -> Interval:
    """Average of the scores, with a bootstrap range around it."""
    if not scores:
        raise ValueError("need at least one score")
    rng = random.Random(seed)
    n = len(scores)
    means = [sum(rng.choices(scores, k=n)) / n for _ in range(resamples)]
    low, high = _middle_range(means, level)
    return Interval(mean=sum(scores) / n, low=low, high=high)


def paired_test(a: list[float], b: list[float], resamples: int = RESAMPLES, level: float = LEVEL,
                seed: int = 0) -> PairedResult:
    """Compare setup a with setup b on the same queries (positive = a is better)."""
    if len(a) != len(b):
        raise ValueError("both setups need a score for every query")
    diffs = [x - y for x, y in zip(a, b)]
    interval = bootstrap_ci(diffs, resamples, level, seed)

    # Sign-flip test: if a and b were equally good, each difference would be
    # just as likely to be positive as negative. Count how often random flips
    # give an average at least as far from zero as the real one.
    rng = random.Random(seed)
    observed = abs(interval.mean)
    n = len(diffs)
    as_extreme = 0
    for _ in range(resamples):
        flipped = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(flipped) >= observed - 1e-12:
            as_extreme += 1
    p_value = (as_extreme + 1) / (resamples + 1)
    return PairedResult(mean_diff=interval.mean, low=interval.low, high=interval.high,
                        p_value=p_value)
