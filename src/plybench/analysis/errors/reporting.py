"""The report registry and the run's cost table. Each stage owns its own reporters; this module only
names them, so adding a stage means adding a package and one line here."""

from __future__ import annotations

from collections.abc import Callable

from plybench.analysis.errors.analysis import Analysis
from plybench.analysis.errors.consistency.report import report_consistency
from plybench.analysis.errors.funnel.report import report_funnel, report_opponents, report_subgroups, report_traces
from plybench.analysis.errors.judge.cost import CostLedger, StepCost
from plybench.analysis.errors.procedural.report import report_labels
from plybench.analysis.errors.reasoning.report import report_correlations, report_mistakes, report_reliability
from plybench.utils.const import MILLION

Reporter = Callable[[Analysis], None]

# the stage reporters take their own report object or the funnel, never Analysis, so the same functions
# print a pooled row as print a cell -- the adapters here are what keeps them that way
REPORTERS: dict[str, Reporter] = {
    "funnel": lambda analysis: report_funnel(analysis.funnel_report),
    "traces": lambda analysis: report_traces(analysis.funnel),
    "subgroups": lambda analysis: report_subgroups(analysis.funnel),
    "labels": lambda analysis: report_labels(analysis.label_report),
    "opponents": lambda analysis: report_opponents(analysis.funnel),
    "consistency": lambda analysis: report_consistency(analysis.scope, len(analysis.funnel.moves), analysis.consistency_reports),
    "mistakes": lambda analysis: report_mistakes(analysis.scope, len(analysis.funnel.moves), analysis.prevalence_reports),
    "reliability": lambda analysis: report_reliability(analysis.scope, len(analysis.funnel.moves), analysis.reliability_report),
    "correlations": lambda analysis: report_correlations(analysis.scope, len(analysis.funnel.moves), analysis.correlation_reports),
}


def _cost_row(row: StepCost) -> str:
    tokens = row.tokens
    cost = f"{row.cost:8.4f}" if row.cost is not None else f"{'n/a':>8s}"
    return (
        f"  {row.step:14s} {row.n_calls:6d} {row.n_cached:7d} {row.n_failed:7d} "
        f"{tokens.input_tokens / MILLION:8.3f}M {tokens.cached_input_tokens / MILLION:8.3f}M {tokens.output_tokens / MILLION:8.3f}M {cost}"
    )


def report_cost(ledger: CostLedger) -> None:
    if not ledger:
        return
    total = ledger.total()
    print(f"\ncost  judge {total.model or 'several models'}")
    print(f"  {'step':14s} {'calls':>6s} {'cached':>7s} {'failed':>7s} {'in':>9s} {'cached_in':>9s} {'out':>9s} {'USD':>8s}")
    for row in ledger.rows():
        print(_cost_row(row))
    print(_cost_row(total))
    if total.cost is None:
        print("  the judge carries no pricing in the model registry, so no dollar figure is available")
        return
    per_move = total.cost_per_unit
    if per_move is not None:
        print(f"  ${per_move:.5f} per move covered ({total.n_units} move(s)) -- multiply by a --dry-run count to price a larger run")
    if total.n_cached:
        print(f"  {total.n_cached} call(s) served from cache and cost nothing this run")
