from __future__ import annotations

from collections.abc import Sequence

from plybench.analysis.errors.format import Scope, header
from plybench.analysis.errors.moves import FunnelStage
from plybench.analysis.errors.reasoning.agreement import ReliabilityReport
from plybench.analysis.errors.reasoning.stats import FREEZE_THRESHOLD, PrevalenceReport


def report_mistakes(scope: Scope, n_moves: int, reports: Sequence[PrevalenceReport]) -> None:
    """Per-code prevalence over annotated moves. `by_outcome` is the column that matters: a code whose
    rate is the same on optimal and suboptimal moves describes how the model talks, not how it fails.
    `std` is that rate reweighted to the corpus position mix, so two models are compared facing the same
    board even though one of them reached forced positions three times as often."""
    print(header(scope, n_moves))
    if not reports:
        print("  no stored annotations for these moves -- run: --do induce then --do annotate")
        return

    for report in reports:
        print(f"  judge {report.annotator}  (codebook {report.codebook_version})")
        print(f"    {report.n_annotated}/{report.n_moves} moves annotated ({report.coverage:.1%}), {report.n_clean} carrying no error")
        print(f"    {'any uncorrected error':30s} {report.any_error.fmt(8, interval=True)}")
        # errors the annotator found but no code covered. The descriptive rate is over every annotated
        # move; the figure that measures completeness is the one below it, over games induction never read
        print(f"    {'uncovered by the codebook':30s} {report.uncovered.fmt(8, interval=True)}  ({report.n_uncovered} move(s))")
        basis = "games" if report.game_level_holdout else "moves (no game provenance)"
        print(f"    {'  holding back induced ' + basis:30s} {report.uncovered_naive.fmt(8, interval=True)}  ({report.n_uncovered_naive}/{report.n_naive} move(s))")
        if report.n_naive and report.uncovered_naive.value > FREEZE_THRESHOLD:
            print(f"    ! above the {FREEZE_THRESHOLD:.0%} freeze threshold -- feed the uncovered descriptions back into induction and run another wave")
        print(f"    {'code':30s} {'rate':>8s} {'95% CI':>18s} {'std':>8s} {'optimal':>9s} {'subopt':>9s} {'recovery':>9s}")
        for code in report.codes:
            optimal = code.by_outcome.get(FunnelStage.OPTIMAL)
            suboptimal = code.by_outcome.get(FunnelStage.SUBOPTIMAL)
            interval = code.rate.wilson
            bounds = f"[{interval.lower:.3f}, {interval.upper:.3f}]" if interval is not None else ""
            recovery = f"{code.recovery:.2f}" if code.recovery is not None else "-"
            standardized = f"{code.standardized.value:8.3f}" if code.standardized is not None else f"{'':8s}"
            print(
                f"    {code.code_id:30s} {code.rate.value:8.3f} {bounds:>18s} {standardized} "
                f"{optimal.value if optimal else float('nan'):9.3f} {suboptimal.value if suboptimal else float('nan'):9.3f} {recovery:>9s}"
            )
        for description in report.uncovered_descriptions[:5]:
            print(f"      uncovered: {description[:110]}")
        if report.accounts:
            print("    why each suboptimal move was suboptimal:")
            for account, count in report.accounts.items():
                if count:
                    print(f"      {account.value:16s} {count:5d}")


def report_reliability(scope: Scope, n_moves: int, report: ReliabilityReport | None) -> None:
    """Cohen's kappa per code between two annotators, over the moves both of them judged. This is
    inter-*judge* agreement, not validity: two models can agree and both be wrong, which only a
    human-coded gold subsample can rule out."""
    print(header(scope, n_moves))
    if report is None:
        print("  needs two annotators -- re-run --do annotate with a second --model")
        return

    print(f"  {report.first}  vs  {report.second}  ({report.n_shared} move(s) judged by both)")
    if not report.same_protocol:
        # the only pair on file runs two different prompts, so the disagreement below mixes the judges
        # with the protocol. That is a useful experiment and a meaningless reliability figure
        print("  ! different prompt revisions -- this measures the protocol change, not inter-judge reliability")
    print(f"  {'code':30s} {'n':>6s} {'observed':>9s} {'kappa':>8s} {'either':>7s}")
    for code_id, kappa in report.codes.items():
        value = f"{kappa.kappa:8.3f}" if kappa.kappa is not None else f"{'n/a':>8s}"
        print(f"  {code_id:30s} {kappa.n:6d} {kappa.observed:9.3f} {value} {kappa.n_positive:7d}")
