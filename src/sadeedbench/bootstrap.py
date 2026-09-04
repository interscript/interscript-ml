"""Sentence-level paired bootstrap over per-item metric deltas: a
point delta without a CI is not evidence. Deterministic under the
fixed default seed (42, 1,000 resamples, percentile CIs)."""

from __future__ import annotations

import random
import statistics
from collections.abc import Iterable


def bootstrap_means(deltas: list[float], seed: int = 42, n: int = 1000) -> dict:
    """Bootstrap the mean of a delta list (percentile CIs, fixed seed)."""
    if not deltas:
        raise ValueError("empty deltas")
    rng = random.Random(seed)
    m = len(deltas)
    means = [
        statistics.fsum(rng.choice(deltas) for _ in range(m)) / m for _ in range(n)
    ]
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[min(n - 1, int(0.975 * n))]
    delta = statistics.fsum(deltas) / m
    p_leq0 = sum(1 for u in means if u <= 0) / n
    return {
        "delta": round(delta, 4),
        "ci95": (round(lo, 4), round(hi, 4)),
        "p_leq0": round(p_leq0, 4),
        "n": m,
    }


def bootstrap_delta(
    a: Iterable[float | None],
    b: Iterable[float | None],
    seed: int = 42,
    n: int = 1000,
) -> dict:
    """Bootstrap the mean of (a - b) over pairwise-aligned items; pairs
    with a None on either side are dropped (the aggregate DER call
    skips paragraphs with no scorable positions). (candidate,
    reference): positive delta = candidate worse."""
    deltas = [x - y for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not deltas:
        raise ValueError("no scorable pairs")
    return bootstrap_means(deltas, seed=seed, n=n)
