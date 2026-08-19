from __future__ import annotations

from plybench.analysis.errors.format import Scope, header
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.funnel.stats import FunnelReport, metrics
from plybench.analysis.errors.moves import FunnelStage, group_by
from plybench.common.enums import MetricName

FUNNEL_COLUMNS: tuple[tuple[MetricName, str, int], ...] = (
    (MetricName.OPTIMALITY_RATE, "optimality", 12),
    (MetricName.OPTIMALITY_RATE_NON_TRIVIAL, "opt@decision", 14),
    (MetricName.REGRET, "regret", 10),
    (MetricName.OUTPUT_TOKENS_PER_MOVE, "out_tokens", 12),
)


def report_funnel(report: FunnelReport) -> None:
    print(header(report.scope, report.n_moves))
    print(f"  {'bucket':16s} {'n':>6s} {'share':>7s}" + "".join(f" {label:>{width}s}" for _, label, width in FUNNEL_COLUMNS))
    for stage, summary in report.stages.items():
        row = f"  {stage.value:16s} {summary.n_moves:6d} {summary.share:7.3f}"
        row += "".join(f" {summary.metrics[name].fmt(width - 2):>{width}s}" for name, _, width in FUNNEL_COLUMNS)
        print(row)

    if report.lost_share is not None:
        print(f"  non_decision is {report.lost_share:.1%} already-lost positions (rest genuinely forced)")
    failures = ", ".join(f"{kind.value}={count}" for kind, count in report.output_failures.items() if count)
    if failures:
        print(f"  output failures: {failures}")


def report_traces(funnel: FunnelResult) -> None:
    print(header(Scope.of(funnel), len(funnel.moves)))
    traced = [move for move in funnel.moves if move.has_trace]
    untraced = funnel.stage(FunnelStage.NON_REASONING)
    print(f"  {'group':16s} {'n':>6s} {'opt@decision':>14s}  95% CI")
    for label, moves in (("with trace", traced), ("no trace", untraced)):
        bundle = metrics(moves)[MetricName.OPTIMALITY_RATE_NON_TRIVIAL]
        print(f"  {label:16s} {len(moves):6d} {bundle.fmt(12, interval=True)}")


def report_subgroups(funnel: FunnelResult) -> None:
    print(header(Scope.of(funnel), len(funnel.moves)))
    print(f"  {'bucket':16s} {'position':14s} {'n':>6s} {'of bucket':>10s} {'out_tokens':>12s}")
    for stage in FunnelStage:
        moves = funnel.stage(stage)
        for label, group in sorted(group_by(moves, lambda move: move.decision.value).items()):
            tokens = metrics(group)[MetricName.OUTPUT_TOKENS_PER_MOVE]
            print(f"  {stage.value:16s} {label:14s} {len(group):6d} {len(group) / len(moves):10.3f} {tokens.fmt(10):>12s}")


def report_opponents(funnel: FunnelResult) -> None:
    print(header(Scope.of(funnel), len(funnel.moves)))
    print(f"  {'opponent':52s} {'n':>6s} {'optimality':>12s} {'regret':>10s}")
    for label, moves in group_by(funnel.analyzable, lambda move: move.matchup.opponent).items():
        bundles = metrics(moves)
        print(f"  {label:52s} {len(moves):6d} {bundles[MetricName.OPTIMALITY_RATE].fmt(10):>12s} {bundles[MetricName.REGRET].fmt(8):>10s}")
