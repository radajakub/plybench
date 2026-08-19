from __future__ import annotations

from plybench.analysis.errors.format import header
from plybench.analysis.errors.procedural.stats import LabelReport
from plybench.common.enums import StateClass

POSITION_COLUMNS: tuple[tuple[StateClass, str, int], ...] = (
    (StateClass.DECISION, "decision", 9),
    (StateClass.DONT_CARE, "forced", 7),
    (StateClass.LOST, "lost", 6),
)


def _histogram(counts: dict[int, int]) -> str:
    return "  ".join(f"{depth}:{count}" for depth, count in sorted(counts.items()))


def report_labels(report: LabelReport) -> None:
    print(header(report.scope, report.n_moves))
    if not report.n_moves:
        print("  no move put an action on the board -- nothing to grade")
        return

    print(f"  {report.n_moves} graded move(s), {report.n_suboptimal} suboptimal; diagnostic classes explain {report.explained.fmt(8, interval=True)} of those")
    print(f"  {'label':24s} {'n':>6s} {'rate (95% CI)':>27s}" + "".join(f" {name:>{width}s}" for _, name, width in POSITION_COLUMNS) + f" {'depth':>6s} {'kind':>11s}")
    for row in report.prevalence:
        counts = "".join(f" {row.by_class[state]:{width}d}" for state, _, width in POSITION_COLUMNS)
        # blank rather than 0.0 where the label only ever fired on moves the solver agreed with: those
        # carry no refutation to be deep or shallow, and a zero would read as "seen immediately"
        depth = f"{row.depth.value:6.2f}" if row.depth.n else f"{'':6s}"
        print(f"  {row.label.value:24s} {row.n:6d} {row.rate.fmt(8, interval=True):>27s}{counts} {depth} {row.kind:>11s}")

    if not report.depth.n:
        return
    print(f"  refutation depth {report.depth.fmt(6, interval=True)} plies, width {report.width.fmt(6, interval=True)} of replies")
    print(f"  blunders by depth  {_histogram(report.depths)}")
    if report.residual_depths:
        # a residual at the shallow end of that spread is a detector nobody has written yet, not an error
        # that needed deep search -- which is the one thing the count of deep_error alone cannot tell you
        print(f"  residual by depth  {_histogram(report.residual_depths)}")
