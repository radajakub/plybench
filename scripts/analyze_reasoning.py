"""Reasoning-trace analysis: the funnel and its reports (free), plus the judge passes that build and
apply the mistake codebook (these call an LLM and cost money).

Stage one is always the funnel: every recorded move of a (game, player) cell, pooled over opponents and
routed to exactly one bucket -- no trace, failed, no decision to make, optimal, suboptimal. Everything
else consumes those buckets, so this script grows by adding reporters and passes rather than by changing
its own control flow.

Reports (--report, read-only):
    funnel       bucket sizes, shares and the usual move-metrics per bucket
    traces       what dropping the trace-less moves does to the sample (selection bias check)
    subgroups    every bucket cut by the position it was played in (no decision / optimal / suboptimal)
    labels       procedural move labels read off the solved tree: what could have been blocked, won
                 outright or forced, plus the residual no shallow class explains (free, no judge)
    opponents    the analysable moves broken down by the opponent they were played against
    consistency  whether the move played is the one the trace concluded, read off the stored verdicts
    mistakes     per-code prevalence with Wilson intervals, self-correction rates, how often an error fell
                 outside the codebook entirely, and the decomposition of every suboptimal move into
                 slip / reasoning error / both / unexplained
    reliability  per-code Cohen's kappa between two annotators over the moves both judged

Passes (--do, these spend money):
    consistency  did the model play the move its own trace concluded? Blind: the judge sees the position,
                 the legal moves and the trace, never the move that was played, and the comparison itself
                 is done in code. Runs on every successful reasoning move (forced positions included,
                 which are the no-stakes baseline).
    induce       grow the mistake codebook by open coding to saturation. Informed: the judge sees the
                 chosen move and the solver's optimal set, because discovery needs to see what went wrong.
                 Sequential by necessity, pooled over every cell, since the codebook is global.
    etalons      pick the clearest real instance of each code, quoted verbatim, to print as its reference
                 example. One call per code, not per move. Provenance only: it never moves the codebook
                 version, so electing one does not invalidate annotations already made.
    annotate     apply the frozen codebook to every analysable move and store the labels. The judge is shown the
                 move that was played, because three of the coding rules are defined against it, but not
                 the solver's verdict. Reports separate held-out evaluation, full-corpus descriptive,
                 and discovery populations.
    informed     the same pass with the solver's optimal set revealed, written to its own annotator column.
                 Which variant measures the reasoning better is untested: blind risks a judge that cannot
                 verify anything calling sound reasoning wrong, informed risks one reasoning backwards
                 from the outcome. Run both over the same moves and compare with --report reliability.

Passes always run in the order above, whatever order they are named in, because a codebook must exist
before it is applied and the suboptimal decomposition needs consistency verdicts to join against.
Verdicts are keyed by (move, annotator), where the annotator carries the model *and* the prompt revision,
so a re-run resumes rather than re-pays, and a changed prompt writes a new column instead of mixing two
protocols. Annotation additionally resumes only over its own codebook version: extending the codebook
re-annotates, because a label chosen from a different set of codes is not a label under the current one --
check `--dry-run` before re-running it. Raw responses are cached under `cache/reasoning/`.

Examples:
    uv run python scripts/analyze_reasoning.py --experiment ttt
    uv run python scripts/analyze_reasoning.py --experiment ttt --report funnel traces --json out.json
    uv run python scripts/analyze_reasoning.py --experiment ttt --do all --dry-run
    uv run python scripts/analyze_reasoning.py --experiment ttt --do consistency --model openai:gpt-5-mini
    uv run python scripts/analyze_reasoning.py --experiment ttt --do induce --per-stratum 6
    uv run python scripts/analyze_reasoning.py --experiment ttt --do annotate --per-stratum 40 --seed s1
    uv run python scripts/analyze_reasoning.py --experiment ttt --report mistakes reliability
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shared import add_source_args, benchmark_from_args, build_op  # noqa: E402

from plybench.analysis.errors.aggregate import pooled_reports, position_mix, report_pooled  # noqa: E402
from plybench.analysis.errors.analysis import Analysis, analyse  # noqa: E402
from plybench.analysis.errors.export import write_moves  # noqa: E402
from plybench.analysis.errors.facets import FACETS, parse_axis  # noqa: E402
from plybench.analysis.errors.format import describe  # noqa: E402
from plybench.analysis.errors.funnel.result import FunnelResult  # noqa: E402
from plybench.analysis.errors.funnel.run import build_funnels, unsupported_games  # noqa: E402
from plybench.analysis.errors.judge.cost import CostLedger  # noqa: E402
from plybench.analysis.errors.judge.runner import ResponseCache  # noqa: E402
from plybench.analysis.errors.pipeline import ETALONS, INDUCE, PASS_ORDER, PassOptions, Pipeline  # noqa: E402
from plybench.analysis.errors.reporting import REPORTERS, report_cost  # noqa: E402
from plybench.analysis.errors.split import DEFAULT_SPLIT_SEED, SplitConfig  # noqa: E402
from plybench.analysis.errors.stores import AnalysisStores  # noqa: E402
from plybench.analysis.replay import ReplayerCache  # noqa: E402
from plybench.llm import DEFAULT_CONCURRENCY, ModelConfig  # noqa: E402
from plybench.observability.notifications import NotificationClient  # noqa: E402
from plybench.utils.const import MILLION  # noqa: E402

ALL = "all"


def _model_config(args: argparse.Namespace) -> ModelConfig:
    try:
        return ModelConfig.from_string(args.model)
    except ValueError as error:
        raise SystemExit(f"--model: {error}") from error


AUTO_CODEBOOK = "auto"


def _codebook_label(args: argparse.Namespace, model: ModelConfig) -> str:
    """A fork is opt-in and must look like one. `auto` names it after the judge, which keeps two judges'
    taxonomies apart without having to invent a label; anything else is taken literally. An unusable label
    would otherwise resolve to a path that does not exist, and a missing codebook reads as "nothing
    induced yet" rather than as a mistake."""
    label = (args.codebook or "").strip()
    if label == AUTO_CODEBOOK:
        return model.slug
    if label and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
        raise SystemExit(f"--codebook must be a plain name (letters, digits, dot, dash, underscore) or 'auto', got {label!r}")
    return label


def _passes(args: argparse.Namespace) -> list[str]:
    requested = set(PASS_ORDER) if args.do and ALL in args.do else set(args.do or ())
    return [name for name in PASS_ORDER if name in requested]


def _reports(args: argparse.Namespace) -> list[str]:
    # a run that only spends money reports through its passes; asking for reports is what adds them back
    return args.report if args.report is not None else ([] if args.do else ["funnel"])


def _calls(pipeline: Pipeline, name: str, total: int, funnels: list[FunnelResult]) -> int:
    """What a pass would actually cost. Only the per-move passes make one call per move: induction batches
    them, and the etalon pass asks once per code that still needs one -- printing `total` for either would
    overstate the bill by two orders of magnitude, which is the number people budget against."""
    if name == INDUCE:
        return -(-total // pipeline.options.batch_size)
    if name == ETALONS:
        experiment = next((funnel.experiment for funnel in funnels if funnel.experiment), "")
        return sum(code.etalon is None for code in pipeline.stores.codebook(experiment).active())
    return total


def _dry_run(pipeline: Pipeline, passes: list[str], funnels: list[FunnelResult]) -> None:
    for name in passes:
        selection = pipeline.select(name, funnels)
        if name == INDUCE:
            planned = pipeline.plan_discovery(selection)
            planned_uids = {move.uid for move in planned.moves}
            selection = [(funnel, [move for move in moves if move.uid in planned_uids]) for funnel, moves in selection]
            total = len(planned.moves)
        else:
            total = sum(len(moves) for _, moves in selection)
        calls = _calls(pipeline, name, total, funnels)
        print(f"\n[{name}] {total} move(s) selected -> at most {calls} judge call(s) before caching and resume")
        for funnel, moves in selection:
            print(f"  {funnel.game.to_string()}  |  {funnel.player.to_string()}  |  {len(moves)} move(s)")


def _summary(passes: list[str], ledger: CostLedger, cells: int, elapsed: float) -> str:
    total = ledger.total()
    spent = f"${total.cost:.2f}" if total.cost is not None else "unpriced"
    ran = ", ".join(passes) if passes else "reports only"
    return f"reasoning analysis finished: {ran} over {cells} cell(s), {total.n_calls} judge call(s), {spent}, {elapsed / 60:.1f} min"


async def _run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    op = build_op(notif_enabled=args.notify, concurrency=args.concurrency, limit_scale=args.limit_scale)
    if args.notify and not op.notif.configured:
        print("warning: --notify set but NTFY_URL is not configured; notifications will be skipped")
    results = benchmark_from_args(op, args).get_results()
    skipped = unsupported_games(results, op.registry)
    replayers = ReplayerCache(op.registry)  # one solved tree per game, shared by the funnel and the labelling pass
    split_config = SplitConfig(args.discovery_fraction, args.split_seed)
    print(f"analysis split: {split_config.discovery_fraction:.0%} discovery / {split_config.evaluation_fraction:.0%} evaluation, seed={split_config.seed!r}")
    funnels = list(build_funnels(results, op.registry, replayers, split_config=split_config))
    model = _model_config(args)
    stores, ledger = AnalysisStores(_codebook_label(args, model)), CostLedger(op.llm)
    if stores.codebook_label:
        print(f"using the {stores.codebook_label!r} codebook, forked from the experiment's own")
    passes = _passes(args)

    if passes or args.dry_run:
        options = PassOptions(
            args.per_stratum,
            args.seed,
            args.batch_size,
            args.patience,
            args.min_instances,
            not args.no_consolidate,
            args.discovery_suboptimal_cap,
            args.discovery_optimal_cap,
            args.discovery_non_decision_cap,
            args.discovery_coverage_per_stratum,
        )
        pipeline = Pipeline(op.llm, model, stores, ResponseCache(enabled=not args.no_cache), ledger, options)
        populated = [funnel for funnel in funnels if funnel.moves]
        if not populated:
            raise SystemExit("nothing recorded for any configured cell")
        if args.dry_run:
            _dry_run(pipeline, passes or list(PASS_ORDER), populated)
            return
        for index, name in enumerate(passes, start=1):
            started_pass = time.monotonic()
            await pipeline.run(name, populated)
            # a sweep runs for hours across several passes, so "2 of 4 done" is the message worth having;
            # the end-of-run one arrives far too late to act on
            op.notif.notify(f"[{index}/{len(passes)}] {name} done in {(time.monotonic() - started_pass) / 60:.1f} min")

    analyses: list[Analysis] = []
    empty = 0
    for funnel in funnels:
        if not funnel.moves:  # a configured cell with nothing on disk is a gap, not a result
            empty += 1
            continue
        # every report object is built once here and shared, so the printed tables and the JSON row below
        # are the same numbers rather than two independent computations of them
        analysis = analyse(funnel, stores, replayers)
        analyses.append(analysis)
        for name in _reports(args):
            REPORTERS[name](analysis)

    # one reference mix for the whole corpus, so every standardised rate in the run is on the same scale
    reference = position_mix(analyses) if args.standardize else None
    pools = {axis: pooled_reports(analyses, parse_axis(axis), reference=reference) for axis in args.pool or ()}
    for axis, reports in pools.items():
        print(f"\n\n===== split by {describe(parse_axis(axis))} ({len(reports)} group(s)) =====")
        for report in reports:
            report_pooled(report)

    if args.dump_moves:
        written, index = write_moves(args.dump_moves, analyses)
        print(f"\n{written} graded move(s) -> {args.dump_moves} ({args.dump_moves.stat().st_size / MILLION:.1f} MB), cell table -> {index}")

    cells = len(analyses)

    print(f"\n{cells} (game, player) cell(s) funnelled" + (f", {empty} configured cell(s) had nothing recorded." if empty else "."))
    if skipped:
        # the buckets are defined by the solver's optimal set, so an unsolvable game is skipped outright --
        # said out loud, because otherwise it is indistinguishable from having recorded nothing
        print(f"{len(skipped)} configured game(s) are not minimax-solvable and were skipped: {', '.join(skipped)}")
    if cells == 0 and empty:
        print("nothing on disk for any configured cell -- the experiment file and the recorded results disagree (check the observation type / model list)")
    report_cost(ledger)
    if args.json:
        summary = {
            "split": {"discovery_fraction": split_config.discovery_fraction, "evaluation_fraction": split_config.evaluation_fraction, "seed": split_config.seed},
            "cells": [analysis.to_dict() for analysis in analyses],
            "pools": {axis: [report.to_dict() for report in reports] for axis, reports in pools.items()},
            "cost": ledger.to_dict(),
        }
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"summaries written to {args.json}")

    op.notif.notify(_summary(passes, ledger, cells, time.monotonic() - started))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument("--report", nargs="+", choices=list(REPORTERS), help=f"reports to print: {', '.join(REPORTERS)} (default: funnel, or none when --do is given)")
    parser.add_argument("--do", nargs="+", choices=[*PASS_ORDER, ALL], help=f"judge passes to run -- these cost money: {', '.join(PASS_ORDER)}, or all")
    parser.add_argument("--json", type=Path, help="also write the funnel summaries and the cost breakdown as JSON to this path")
    parser.add_argument(
        "--pool",
        nargs="+",
        metavar="FACET[,FACET]",
        help=f"also report cells grouped by a facet or a comma-separated combination of them ({', '.join(FACETS)}). "
        "'model' is one row per model over every game it played; 'model,presentation' is that model under each obfuscation. "
        "Grouping pools over moves, never over the cells' own rates",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="add a position-mix-standardised rate to every mistake code. A weak model spends most of its moves in forced or "
        "already-lost positions, so raw rates partly compare what each model faced; this reweights every group to the corpus mix",
    )
    parser.add_argument("--dump-moves", type=Path, help="write one JSONL row per graded move (uid, labels, refutation), joinable with the judge stores by move uid")
    parser.add_argument(
        "--discovery-fraction",
        type=float,
        default=0.2,
        metavar="FRACTION",
        help="fraction of complete games reserved for taxonomy discovery (default 0.2; the remainder is held-out evaluation)",
    )
    parser.add_argument(
        "--split-seed",
        default=DEFAULT_SPLIT_SEED,
        metavar="SEED",
        help=f"stable game-level discovery/evaluation split seed (default {DEFAULT_SPLIT_SEED!r})",
    )

    judge = parser.add_argument_group("judge (only used with --do)")
    judge.add_argument(
        "--model",
        default="openai:gpt-5-mini",
        metavar="CONFIG",
        help="judge model in the same syntax as a player's, minus the observation and strategy: "
        "<provider>:<model>[:<options>], e.g. metacentrum:gemma-4:thinking_enabled=True,reasoning_effort=medium "
        "(options: thinking_enabled, reasoning_effort, temperature, max_tokens; default openai:gpt-5-mini)",
    )
    judge.add_argument("--per-stratum", type=int, help="optional per-pass judge-call cap per (game, player, outcome); not used by induction; omit for complete annotation")
    judge.add_argument("--seed", default="", help="deterministic discovery ordering and optional sampling seed")
    judge.add_argument("--batch-size", type=int, default=8, help="moves per induction batch (induce only; default 8)")
    judge.add_argument("--patience", type=int, default=4, help="stop induction after this many coding batches that needed no new code (default 4)")
    judge.add_argument(
        "--min-instances",
        type=int,
        default=30,
        help="refuse to call induction saturated until this many error instances have been coded, however quiet the batches were (default 30)",
    )
    judge.add_argument(
        "--discovery-suboptimal-cap",
        type=int,
        default=32,
        metavar="N",
        help="maximum suboptimal moves per (game, player, opponent, outcome) discovery stratum; small strata are kept whole (default 32)",
    )
    judge.add_argument(
        "--discovery-optimal-cap",
        type=int,
        default=4,
        metavar="N",
        help="optimal control moves per discovery stratum (default 4)",
    )
    judge.add_argument(
        "--discovery-non-decision-cap",
        type=int,
        default=2,
        metavar="N",
        help="forced/non-decision control moves per discovery stratum (default 2)",
    )
    judge.add_argument(
        "--discovery-coverage-per-stratum",
        type=int,
        default=2,
        metavar="N",
        help="moves from every represented discovery stratum that must be read before induction may declare saturation (default 2)",
    )
    judge.add_argument("--no-consolidate", action="store_true", help="skip the merge/parent pass after induction")
    judge.add_argument(
        "--codebook",
        metavar="LABEL",
        help="induce into (and annotate against) a codebook forked beside the experiment's own, as codebook.<LABEL>.json. "
        "Pass 'auto' to name it after the judge. Use it to try a second judge's taxonomy without overwriting the first: "
        "annotations still share one file, since they record the codebook version they were made under",
    )
    judge.add_argument("--no-cache", action="store_true", help="ignore the raw-response cache and re-ask the judge")
    judge.add_argument("--dry-run", action="store_true", help="print how many calls each pass would make and exit")
    judge.add_argument("--notify", action="store_true", help="push an ntfy notification after each pass and when the run ends or fails (needs NTFY_URL)")
    judge.add_argument(
        "--concurrency",
        type=int,
        metavar="N",
        help=f"max in-flight judge requests per provider (default {DEFAULT_CONCURRENCY}) -- the binding constraint on a full sweep, where the default turns hours into days",
    )
    judge.add_argument("--limit-scale", type=float, default=1.0, help="scale the per-model quotas, to leave headroom when another run shares the account (default 1.0)")
    args = parser.parse_args()

    op_notif = NotificationClient.from_env(enabled=args.notify)
    # wrap rather than notify in a finally: a run that died halfway has still spent money, and the
    # difference between "finished" and "failed after 40 minutes" is the whole reason to be told
    op_notif.wrap("reasoning analysis", lambda: asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
