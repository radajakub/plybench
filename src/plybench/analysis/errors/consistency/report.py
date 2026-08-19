from __future__ import annotations

from collections.abc import Sequence

from plybench.analysis.errors.consistency.stats import ConsistencyReport
from plybench.analysis.errors.format import Scope, header


def report_consistency(scope: Scope, n_moves: int, reports: Sequence[ConsistencyReport]) -> None:
    """Did the model play what its own trace concluded? Independent of optimality -- a trace can conclude
    a losing move and be perfectly consistent -- so `slips` is what says whether a disagreement mattered."""
    print(header(scope, n_moves))
    if not reports:
        print("  no stored verdicts for these moves -- run: --do consistency")
        return

    for report in reports:
        counts = ", ".join(f"{verdict.value}={count}" for verdict, count in report.counts.items() if count)
        print(f"  judge {report.annotator}")
        print(f"    scored {report.n_scored}/{report.n_moves} analysable moves ({report.coverage:.1%}): {counts}")
        print(f"    inconsistency {report.inconsistency.fmt(8, interval=True)} over {report.inconsistency.n} decided traces")
        for outcome, verdicts in report.by_outcome.items():
            graded = ", ".join(f"{verdict.value}={count}" for verdict, count in verdicts.items() if count)
            print(f"    {outcome.value:14s} {sum(verdicts.values()):6d}  {graded}")
        slips = ", ".join(f"{kind.value}={count}" for kind, count in report.slips.items() if count)
        if slips:
            print(f"    slips among inconsistent moves: {slips}")
