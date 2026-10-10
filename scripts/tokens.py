"""Sum the tokens an experiment spent from its recorded results, with the USD each group costs.

Every recorded call is counted, for both sides of every matchup (the opponent bills too when it is a
model). Groups are printed in the order asked for; the grand total is always printed last.

Prices come from the model catalogue of the providers this process has credentials for -- an entry
whose provider is not configured is counted in tokens but not in USD, and the affected totals are
marked with `*`. Costs are upper bounds: the recorded steps carry no prompt-cache split, so all input
is charged at the uncached rate.

Examples:
    uv run python scripts/tokens.py --experiment my_experiment
    uv run python scripts/tokens.py --experiment my_experiment --by model game matchup
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shared import add_source_args, benchmark_from_args, build_op  # noqa: E402
from plyllm import LLM  # noqa: E402

from plybench.analysis.usage import MatchupUsage, TokenUsage, benchmark_usage, group_by, model_label, total_cost, total_usage  # noqa: E402

GROUPS: dict[str, Callable[[MatchupUsage], str]] = {
    "model": lambda usage: model_label(usage.model),
    "game": lambda usage: usage.game.to_string(),
    "opponent": lambda usage: usage.opponent.to_string(),
    "matchup": lambda usage: f"{usage.game.to_string()} | {model_label(usage.model)} vs {usage.opponent.to_string()}",
}
HEADER = "group"


def _cost_cell(cost: float, unpriced: int, priced: int) -> str:
    if priced == 0:
        return "n/a"
    return f"${cost:,.2f}" + ("*" if unpriced else "")


def _rows(llm: LLM, groups: dict[str, list[MatchupUsage]]) -> list[tuple[str, TokenUsage, str]]:
    rows = []
    for label, entries in groups.items():
        cost, unpriced = total_cost(llm, entries)
        rows.append((label, total_usage(entries), _cost_cell(cost, unpriced, len(entries) - unpriced)))
    return sorted(rows, key=lambda row: row[1].total, reverse=True)


def _print_table(title: str, rows: list[tuple[str, TokenUsage, str]]) -> None:
    width = max(len(label) for label, _, _ in rows + [(HEADER, TokenUsage(), "")])
    print(f"\n=== {title} ===")
    print(f"  {HEADER:<{width}} {'input':>14} {'output':>14} {'reasoning':>14} {'billed total':>15} {'calls':>8} {'USD':>12}")
    for label, usage, cost in rows:
        tokens = usage.tokens
        print(f"  {label:<{width}} {tokens.input_tokens:>14,} {tokens.output_tokens:>14,} {tokens.reasoning_tokens:>14,} {usage.total:>15,} {usage.calls:>8,} {cost:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument("--by", nargs="+", choices=list(GROUPS), default=["model"], metavar="NAME", help=f"breakdowns to print: {', '.join(GROUPS)} (default: model)")
    args = parser.parse_args()

    op = build_op()
    results = benchmark_from_args(op, args).get_results()
    usages = benchmark_usage(results)
    if not usages:
        raise SystemExit("no recorded calls found (bot-only matchups record no tokens)")

    for name in args.by:
        _print_table(f"by {name}", _rows(op.llm, group_by(usages, GROUPS[name])))

    _print_table("total", _rows(op.llm, {"all matchups": usages}))
    unpriced = total_cost(op.llm, usages)[1]
    if unpriced:
        print(f"  * {unpriced} of {len(usages)} matchup sides have no price (provider not configured, or not a model): USD is a lower bound")


if __name__ == "__main__":
    main()
