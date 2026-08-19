"""Direct standardisation: what a rate would have been had every group faced the same mix of situations.

A raw rate over a player's moves confounds how it reasons with what it faced. A weak player spends most
of its moves in already-lost or forced positions -- in this corpus the forced share runs from 23% to 65%
across cells -- so comparing raw rates partly compares position mixes. Reweighting each group's
within-stratum rates to one shared reference mix takes that term out."""

from __future__ import annotations

import math
from collections.abc import Mapping

from plybench.analysis.statistics.bundle import CIBundle
from plybench.analysis.statistics.intervals import ConfidenceInterval, z_score

type StratumRates[K] = Mapping[K, tuple[int, int]]  # stratum -> (hits, n)
type Reference[K] = Mapping[K, float]  # stratum -> share of the reference population


def reference_mix[K](counts: Mapping[K, int]) -> Reference[K]:
    total = sum(counts.values())
    return {stratum: count / total for stratum, count in counts.items()} if total else {}


def standardized_rate[K](rates: StratumRates[K], reference: Reference[K], confidence: float = 0.95) -> CIBundle | None:
    """The reference-weighted rate, its interval and the sample it rests on.

    None when the group covers none of the reference -- a group with no moves in any weighted stratum has
    no standardised rate, which is different from having one of zero. Strata the group never entered are
    dropped and the remaining weights renormalised, so the figure is honest about its own coverage rather
    than imputing a rate the data never saw."""
    usable = {stratum: rates[stratum] for stratum in reference if rates.get(stratum, (0, 0))[1]}
    covered = sum(reference[stratum] for stratum in usable)
    if not usable or covered <= 0:
        return None

    value, variance, n = 0.0, 0.0, 0
    for stratum, (hits, size) in usable.items():
        weight = reference[stratum] / covered  # renormalised over what this group actually covers
        proportion = hits / size
        value += weight * proportion
        # the strata are independent samples, so their weighted variances add; a stratum of one move
        # contributes no variance estimate of its own and is carried at zero rather than dropped
        variance += (weight**2) * proportion * (1 - proportion) / size
        n += size

    # a normal interval on a weighted sum of independent proportions, so it lands in the sem slot: it is
    # a standard-error interval, and calling it Wilson would claim a construction it does not have
    half = z_score(confidence) * math.sqrt(variance)
    interval = ConfidenceInterval(value, max(0.0, value - half), min(1.0, value + half))
    return CIBundle(value, n, sem=interval)
