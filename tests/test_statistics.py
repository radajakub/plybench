"""The general statistical primitives: family dispatch and the between-group tests. Nothing here knows
about games or moves."""

from __future__ import annotations

import pytest

from plybench.analysis.statistics.bundle import bundle_for_family
from plybench.analysis.statistics.comparison import compare_for_family, mean_difference_test, two_proportion_test
from plybench.analysis.statistics.distribution import Distribution
from plybench.common.enums import CIFamily


# --- between-group tests ---------------------------------------------------------------------
def test_two_proportion_test_separates_and_signs_the_difference():
    # 80% vs 20% over 10 each: a real, significant gap, and A - B is positive
    high, low = Distribution([1] * 8 + [0] * 2), Distribution([1] * 2 + [0] * 8)
    comp = two_proportion_test(high, low)
    assert comp.test == "two_proportion_z"
    assert comp.difference == pytest.approx(0.6)
    assert comp.p_value is not None and comp.significant
    assert comp.interval is not None and comp.difference is not None
    assert comp.interval.lower < comp.difference < comp.interval.upper


def test_two_proportion_test_identical_groups_are_not_significant():
    same = two_proportion_test(Distribution([1, 0, 1, 0]), Distribution([1, 0, 1, 0]))
    assert same.difference == 0.0 and not same.significant
    assert same.p_value is not None and same.p_value > 0.99


def test_two_proportion_test_gracefully_handles_empty_and_degenerate_groups():
    empty = two_proportion_test(Distribution([1, 0]), Distribution())
    assert empty.difference is None and empty.p_value is None and empty.se is None and not empty.significant
    assert empty.n_a == 2 and empty.n_b == 0

    # no variance in either group -> pooled SE is zero -> undefined test statistic, but a CI still exists
    flat = two_proportion_test(Distribution([1, 1]), Distribution([1, 1]))
    assert flat.difference == 0.0 and flat.interval is not None and flat.p_value is None


def test_mean_difference_test_separates_and_needs_two_per_group():
    comp = mean_difference_test(Distribution([10.0, 11.0, 12.0, 13.0]), Distribution([1.0, 2.0, 3.0, 4.0]))
    assert comp.test == "welch_t"
    assert comp.difference is not None and comp.difference > 0 and comp.significant

    tiny = mean_difference_test(Distribution([1.0]), Distribution([2.0, 3.0]))
    assert tiny.difference is None and tiny.p_value is None and tiny.se is None and not tiny.significant


def test_mean_difference_test_zero_variance_is_a_point_with_no_p_value():
    comp = mean_difference_test(Distribution([5.0, 5.0, 5.0]), Distribution([5.0, 5.0, 5.0]))
    assert comp.difference == 0.0 and comp.p_value is None and not comp.significant
    assert comp.interval is not None and comp.interval.unwrap() == (0.0, 0.0, 0.0)


def test_comparison_carries_the_unpooled_se_behind_its_interval():
    # the SE the CI was built from, so callers combining differences never recompute it
    a, b = Distribution([10.0, 11.0, 12.0, 13.0]), Distribution([1.0, 2.0, 3.0, 4.0])
    comp = mean_difference_test(a, b)
    assert comp.se == pytest.approx((a.std(ddof=1) ** 2 / a.n + b.std(ddof=1) ** 2 / b.n) ** 0.5)
    # the interval is the one that SE built: centred on the difference, t_crit wide on either side
    assert comp.interval is not None and comp.interval.value == pytest.approx(comp.difference)
    half_width = (comp.interval.upper - comp.interval.lower) / 2
    assert comp.se is not None
    assert half_width == pytest.approx(2.4469 * comp.se, rel=1e-3)  # df ~= 6 at n=4 per group

    ratio = two_proportion_test(Distribution([1] * 8 + [0] * 2), Distribution([1] * 2 + [0] * 8))
    assert ratio.se == pytest.approx(((0.8 * 0.2 / 10) + (0.2 * 0.8 / 10)) ** 0.5)


def test_family_dispatch_selects_the_right_intervals_and_test():
    ratio = bundle_for_family(CIFamily.RATIO, Distribution([1, 0, 1, 1]))
    assert ratio.wilson is not None and ratio.sem is None
    mean = bundle_for_family(CIFamily.MEAN, Distribution([1.0, 2.0, 3.0]))
    assert mean.sem is not None and mean.t is not None and mean.wilson is None

    assert compare_for_family(CIFamily.RATIO, Distribution([1, 1, 0]), Distribution([0, 0, 1])).test == "two_proportion_z"
    assert compare_for_family(CIFamily.MEAN, Distribution([1.0, 2.0, 3.0]), Distribution([4.0, 5.0, 6.0])).test == "welch_t"
