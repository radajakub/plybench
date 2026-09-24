"""The judge passes, in the one order they can legally run: a codebook has to exist before it can be
applied, and the suboptimal decomposition needs consistency verdicts to join against. Naming the passes
in any order runs them in this one."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

from plybench.analysis.errors.consistency.prompt import CONSISTENCY_REVISION
from plybench.analysis.errors.consistency.run import run_consistency
from plybench.analysis.errors.consistency.stats import consistency_report
from plybench.analysis.errors.format import Scope, cell
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.judge.cost import CostLedger
from plybench.analysis.errors.judge.runner import Generator, Judge, ResponseCache
from plybench.analysis.errors.judge.sampling import DiscoveryPlan, by_matchup_outcome, discovery_plan, sample, strata
from plybench.analysis.errors.moves import FunnelStage, TracedMove
from plybench.analysis.errors.reasoning.annotation import ANNOTATION_REVISION, INFORMED_ANNOTATION_REVISION, run_annotation
from plybench.analysis.errors.reasoning.annotations import AnnotationStore
from plybench.analysis.errors.reasoning.codebook import Codebook
from plybench.analysis.errors.reasoning.coverage import Coverage
from plybench.analysis.errors.reasoning.etalons import ETALON_REVISION, elect_etalons
from plybench.analysis.errors.reasoning.induction import INDUCTION_REVISION, consolidate, induce
from plybench.analysis.errors.reasoning.stats import FREEZE_THRESHOLD, prevalence_report
from plybench.analysis.errors.stores import AnalysisStores, consistency_join
from plybench.llm import ModelConfig

CONSISTENCY, INDUCE, ETALONS, ANNOTATE, INFORMED = "consistency", "induce", "etalons", "annotate", "informed"
# etalons sit after induce and before annotate: the taxonomy has to have settled, and picking one is a
# read of the codebook that nothing downstream depends on, so a run may skip it entirely
# `informed` is the same pass shown the solver's optimal set: a second opinion written to its own
# column, so the cost of blinding can be measured rather than assumed
PASS_ORDER: tuple[str, ...] = (CONSISTENCY, INDUCE, ETALONS, ANNOTATE, INFORMED)
REVISIONS: dict[str, str] = {
    CONSISTENCY: CONSISTENCY_REVISION,
    INDUCE: INDUCTION_REVISION,
    ETALONS: ETALON_REVISION,
    ANNOTATE: ANNOTATION_REVISION,
    INFORMED: INFORMED_ANNOTATION_REVISION,
}

Selection = list[tuple[FunnelResult, list[TracedMove]]]


@dataclass(frozen=True)
class PassOptions:
    per_stratum: int | None = None
    seed: str = ""
    batch_size: int = 8
    patience: int = 4
    # saturation is refused until this many errors have been coded. A 20-30 code taxonomy over a skewed
    # distribution needs several hundred instances before its tail is represented at all, so a low guard
    # lets a run declare completeness having seen almost nothing
    min_instances: int = 300
    consolidate: bool = True
    # hold the games induction read out of the annotation pass, so the uncovered rate is measured on
    # traces the codebook was not built from. This is the coverage test; the census leaves it off
    coverage_test: bool = False
    discovery_suboptimal_cap: int = 32
    discovery_optimal_cap: int = 4
    discovery_non_decision_cap: int = 2
    discovery_coverage_per_stratum: int = 2
    discovery_state_cap: int = 2  # moves drawn per (cell, position)


class Pipeline:
    def __init__(self, generator: Generator, model: ModelConfig, stores: AnalysisStores, cache: ResponseCache, ledger: CostLedger, options: PassOptions) -> None:
        self.generator = generator
        self.model = model
        self.stores = stores
        self.cache = cache
        self.ledger = ledger
        self.options = options

    def judge(self, pass_name: str) -> Judge:
        # the revision is part of the annotator, so each pass writes its own column even under one model
        return Judge(self.generator, self.model, REVISIONS[pass_name])

    def plan_discovery(self, selection: Selection) -> DiscoveryPlan:
        return discovery_plan(
            [move for _, moves in selection for move in moves],
            suboptimal_cap=self.options.discovery_suboptimal_cap,
            optimal_cap=self.options.discovery_optimal_cap,
            non_decision_cap=self.options.discovery_non_decision_cap,
            coverage_per_stratum=self.options.discovery_coverage_per_stratum,
            state_cap=self.options.discovery_state_cap,
            seed=self.options.seed,
        )

    def _excluded(self, pass_name: str, funnels: list[FunnelResult]) -> set[str]:
        """Move uids the pass may not look at.

        The census holds nothing back: it is meant to cover every analysable move, and
        `prevalence_report` measures the uncovered rate twice anyway, once on the induction-naive games.
        The coverage test is the opposite case -- it exists only to measure completeness, so every call it
        spends on a game induction has already read is a call spent measuring nothing."""
        if not (self.options.coverage_test and pass_name in (ANNOTATE, INFORMED)):
            return set()
        codebook = self.stores.codebook(funnels[0].experiment) if funnels else None
        if codebook is None or not codebook.induced:
            return set()
        if not codebook.induced_games_known:
            print("  ! this codebook predates game-level provenance, so only the induced moves themselves can be held back")
        return {move.uid for funnel in funnels for move in funnel.analyzable if not codebook.naive(move)}

    def select(self, pass_name: str, funnels: list[FunnelResult]) -> Selection:
        """Chosen at pass time rather than up front: induction adds to the held-back set, so annotation
        run in the same invocation must see what induction just consumed."""
        excluded = self._excluded(pass_name, funnels)
        selection = []
        for funnel in funnels:
            moves = funnel.analyzable
            # dropped before sampling, so the sample fills up from held-out moves rather than shrinking
            moves = [move for move in moves if move.uid not in excluded]
            if self.options.per_stratum is None or pass_name == INDUCE:
                chosen = moves
            else:
                chosen = sample(moves, self.options.per_stratum, self.options.seed)
            if chosen:
                selection.append((funnel, chosen))
        return selection

    async def run(self, pass_name: str, funnels: list[FunnelResult]) -> None:
        selection = self.select(pass_name, funnels)
        if not selection:
            print(f"\n[{pass_name}] nothing left to judge for any cell")
            return
        judge = self.judge(pass_name)
        print(f"\n=== {pass_name} === annotator {judge.annotator}")
        if pass_name == CONSISTENCY:
            await self._consistency(judge, selection)
        elif pass_name == INDUCE:
            await self._induce(judge, selection, funnels)
        elif pass_name == ETALONS:
            await self._etalons(judge, selection)
        else:
            await self._annotate(judge, selection, informed=pass_name == INFORMED)

    # --- consistency ------------------------------------------------------------------------------
    async def _consistency(self, judge: Judge, selection: Selection) -> None:
        for funnel, moves in selection:
            store = self.stores.consistency(funnel.experiment)
            stats = await run_consistency(judge, moves, store, self.cache)
            self.ledger.record(CONSISTENCY, self.model, stats)
            report = consistency_report(Scope.of(funnel), funnel.analyzable, store.by_move(judge.annotator), judge.annotator)
            print(f"\n{cell(funnel)}")
            print(f"  judged {stats.n} ({stats.n_cached} cached, {stats.n_failed} without a verdict)")
            for error in stats.errors:
                print(f"  ! {error}")
            counts = ", ".join(f"{verdict.value}={count}" for verdict, count in report.counts.items() if count)
            print(f"  scored {report.n_scored}/{report.n_moves} moves: {counts or 'nothing'}")
            if report.inconsistency.n:
                print(f"  inconsistency {report.inconsistency.fmt(8)} over {report.inconsistency.n} decided traces")
            slips = ", ".join(f"{kind.value}={count}" for kind, count in report.slips.items() if count)
            if slips:
                print(f"  slips: {slips}")

    # --- induction --------------------------------------------------------------------------------
    async def _induce(self, judge: Judge, selection: Selection, funnels: list[FunnelResult]) -> None:
        experiment = selection[0][0].experiment
        codebook = self.stores.codebook(experiment)
        design: dict[str, int | str] = {
            "strata": "game,player,opponent,outcome",
            "suboptimal_cap": self.options.discovery_suboptimal_cap,
            "optimal_cap": self.options.discovery_optimal_cap,
            "non_decision_cap": self.options.discovery_non_decision_cap,
            "coverage_per_stratum": self.options.discovery_coverage_per_stratum,
            "state_cap": self.options.discovery_state_cap,
            "sampling_seed": self.options.seed,
        }
        if codebook.discovery_design and codebook.discovery_design != design:
            raise SystemExit(
                f"codebook was induced with discovery design {codebook.discovery_design}, but this run requested {design}; use the original parameters or a new --codebook"
            )
        codebook.discovery_design = design
        candidates = [move for _, moves in selection for move in moves]
        plan = self.plan_discovery(selection)
        pooled = list(plan.moves)
        sampled_counts = strata(pooled, key=by_matchup_outcome)
        outcomes = {
            outcome.value: sum(count for stratum, count in sampled_counts.items() if stratum[-1] == outcome.value)
            for outcome in (FunnelStage.SUBOPTIMAL, FunnelStage.OPTIMAL, FunnelStage.NON_DECISION)
        }
        all_strata = {by_matchup_outcome(move) for funnel in funnels for move in funnel.analyzable if move.decision is FunnelStage.SUBOPTIMAL}
        discovery_strata = {stratum for stratum in sampled_counts if stratum[-1] == FunnelStage.SUBOPTIMAL.value}
        missing = all_strata - discovery_strata
        print(
            f"inducing over {len(pooled)}/{len(candidates)} discovery moves from {len(selection)} cell(s), "
            f"{len(sampled_counts)}/{plan.candidate_strata} represented matchup/outcome strata; "
            f"outcomes={outcomes}; codebook starts with {len(codebook.active())} active code(s)"
        )
        print(f"  saturation disabled until the {plan.coverage_moves}-move stratum-coverage prefix has been read")
        dropped = plan.n_candidates - plan.n_after_position_cap
        if dropped:
            print(f"  {dropped} move(s) dropped by the {self.options.discovery_state_cap}-per-position cap before stratification")
        if missing:
            print(f"  ! {len(missing)} suboptimal matchup stratum/strata have no move in discovery; raise the per-stratum caps or the per-position cap")

        run = await induce(
            judge,
            pooled,
            codebook,
            self.options.batch_size,
            self.options.patience,
            self.options.min_instances,
            plan.coverage_moves,
            self.cache,
        )
        if run.stats:
            self.ledger.record(INDUCE, self.model, reduce(lambda left, right: left + right, run.stats), run.moves_seen)
        print(f"  {run.batches} batch(es), {run.moves_seen} moves: {len(run.new_codes)} new code(s), {run.assignments} assignment(s) to existing codes")
        stop = "saturated" if run.saturated else "ran out of moves while still finding codes"
        print(f"  new codes per batch: {run.new_per_batch}  -> {stop}")
        print(f"  {run.instances} error instance(s) coded in total -- what the saturation claim rests on")
        print(f"  coverage: {Coverage.of(run.per_code).summary()}")
        if run.unmixed_batches:
            print(f"  {run.unmixed_batches} batch(es) held no suboptimal move -- coded, but they cannot end the loop, since there was little in them to find")
        if run.silent_batches:
            print(f"  {run.silent_batches} batch(es) had a suboptimal move but the judge attributed no error -- also unable to end the loop")
        if run.failed_batches:
            print(f"  ! {run.failed_batches} batch(es) produced no answer -- they do not count toward saturation, but they are unsampled traces")
        if run.unknown_codes:
            print(f"  ! judge referenced {len(run.unknown_codes)} code id(s) that do not exist: {sorted(set(run.unknown_codes))[:5]}")
        if run.rejected_evidence:
            print(f"  ! rejected {len(run.rejected_evidence)} induction label(s) without a trace quote")
        if run.rejected_scope:
            print(f"  ! rejected {len(run.rejected_scope)} induction assignment(s) outside the code scope")
        for stats in run.stats:
            for error in stats.errors:
                print(f"  ! {error}")

        if self.options.consolidate and codebook.active():
            consolidation = await consolidate(judge, codebook, self.cache)
            self.ledger.record("consolidate", self.model, consolidation.stats, 0)
            print(f"  consolidation: {len(consolidation.merged)} merge(s), {len(consolidation.parented)} parent link(s), {len(consolidation.rejected)} rejected")
            for rejected in consolidation.rejected:
                print(f"  ! {rejected}")

        path = codebook.save()
        print(f"\ncodebook version {codebook.version} with {len(codebook.active())} active code(s), induced from {len(codebook.induced)} move(s) -> {path}")
        for code in codebook.active():
            parent = f" (specialises {code.parent_id})" if code.parent_id else ""
            print(f"  {code.id:32s} {code.name}{parent}")

    # --- etalons ----------------------------------------------------------------------------------
    async def _etalons(self, judge: Judge, selection: Selection) -> None:
        experiment = selection[0][0].experiment
        codebook = self.stores.codebook(experiment)
        if not codebook.active():
            raise SystemExit(f"no codebook for experiment {experiment!r} -- run --do induce first")

        # the codebook names moves by uid, so every cell's moves are offered, not just the sampled ones:
        # an example was recorded during induction and may sit outside whatever this run selected
        moves = {move.uid: move for funnel, _ in selection for move in funnel.analyzable}
        run = await elect_etalons(judge, codebook, moves, self.cache)
        self.ledger.record(ETALONS, self.model, run.stats, len(run.chosen))

        print(f"  {len(run.chosen)} etalon(s) elected over {len(codebook.active())} active code(s)")
        for reason in run.rejected_evidence:
            print(f"  ! quote not found in the trace it was attributed to -- {reason}")
        for reason in run.rejected_move:
            print(f"  ! move number outside the candidates shown -- {reason}")
        if run.skipped_no_examples:
            print(f"  {len(run.skipped_no_examples)} code(s) carry no example to pick from: {run.skipped_no_examples[:5]}")

        path = codebook.save()
        for code in codebook.active():
            if code.etalon is not None:
                print(f"  {code.id:32s} {code.etalon.evidence[:90]}")
        print(f"codebook -> {path}")

    # --- annotation -------------------------------------------------------------------------------
    async def _annotate(self, judge: Judge, selection: Selection, informed: bool = False) -> None:
        experiment = selection[0][0].experiment
        codebook = self.stores.codebook(experiment)
        if not codebook.active():
            raise SystemExit(f"no codebook for experiment {experiment!r} -- run --do induce first")
        print(f"codebook version {codebook.version} with {len(codebook.active())} active code(s)")

        store = self.stores.annotations(experiment)
        consistency, joined = consistency_join(self.stores, experiment, judge.annotator)
        print(f"decomposing suboptimal moves against consistency verdicts from {joined}" if joined else "no single consistency judge to join -- skipping the decomposition")

        for funnel, moves in selection:
            run = await run_annotation(judge, moves, codebook, store, self.cache, informed=informed)
            self.ledger.record(INFORMED if informed else ANNOTATE, self.model, run.stats)
            report = prevalence_report(Scope.of(funnel), funnel.analyzable, store.by_move(judge.annotator), codebook, judge.annotator, consistency)
            print(f"\n{cell(funnel)}")
            if run.stats is not None:
                print(f"  annotated {run.stats.n} ({run.stats.n_cached} cached, {run.stats.n_failed} without a verdict)")
                for error in run.stats.errors:
                    print(f"  ! {error}")
            print(
                f"  {run.n_labels} label(s) kept ({run.n_self_corrected} self-corrected, {run.n_other} uncovered by any code, "
                f"{run.n_trimmed} quote(s) trimmed to the verbatim span), {run.n_rejected} rejected"
            )
            for rejected in run.rejected_evidence[:3]:
                print(f"  ! evidence not found in trace -- {rejected}")
            for rejected in sorted(set(run.rejected_unknown_code))[:3]:
                print(f"  ! label used an unknown code id: {rejected}")
            print(f"  {report.n_annotated}/{report.n_moves} moves annotated, {report.n_clean} clean, any-error rate {report.any_error.value:.3f}")
            holdout = "games induction read" if report.game_level_holdout else "moves induction read (no game provenance)"
            print(
                f"  uncovered {report.uncovered.value:.3f} over all {report.n_annotated} annotated move(s); "
                f"{report.n_uncovered_naive}/{report.n_naive} = {report.uncovered_naive.fmt(5, interval=True)} holding back {holdout}"
            )
            for code in report.codes[:8]:
                print(f"    {code.code_id:28s} {code.n_uncorrected:4d} uncorrected  rate {code.rate.value:.3f}  (+{code.n_self_corrected} self-corrected)")
            if report.accounts:
                accounts = ", ".join(f"{account.value}={count}" for account, count in report.accounts.items() if count)
                print(f"    suboptimal moves accounted for: {accounts}")

        self._completeness(judge, selection, codebook, store)

    def _completeness(self, judge: Judge, selection: Selection, codebook: Codebook, store: AnnotationStore) -> None:
        """The uncovered rate per model, which is the figure the freeze decision rests on.

        Printed unprompted rather than left to a pooling flag, because the failure it exists to catch is
        invisible in every other view: a pooled 4% hiding one model at 20% means that model's errors are
        not in the codebook and none of its rates are comparable with the others'."""
        by_model: dict[Scope, list[TracedMove]] = {}
        for funnel, _ in selection:
            by_model.setdefault(Scope.of(funnel).only("model"), []).extend(funnel.analyzable)
        reports = [prevalence_report(scope, moves, store.by_move(judge.annotator), codebook, judge.annotator) for scope, moves in by_model.items()]
        reports = [report for report in reports if report.n_naive]
        if not reports:
            print("\nno annotated move comes from a game induction never read -- the uncovered rate cannot measure completeness yet")
            return

        print(f"\ncodebook completeness per model, on games induction never read (freeze threshold {FREEZE_THRESHOLD:.0%})")
        for report in sorted(reports, key=lambda report: report.uncovered_naive.value, reverse=True):
            flag = "  <-- blocks the freeze" if report.uncovered_naive.value > FREEZE_THRESHOLD else ""
            print(f"  {str(report.scope):48s} {report.n_uncovered_naive:4d}/{report.n_naive:<6d} {report.uncovered_naive.fmt(8, interval=True)}{flag}")
        worst = max(report.uncovered_naive.value for report in reports)
        if worst > FREEZE_THRESHOLD:
            print("  the codebook is not complete for every model -- read the uncovered descriptions, extend it, and run another wave")
