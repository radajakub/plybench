"""Pooling cells into one row, and the per-move export the pooled numbers cannot replace.

The property worth pinning is that pooling happens over the moves rather than over the cells' own rates:
the two agree whenever the cells are the same size, so every test here makes them different sizes."""

from __future__ import annotations

import json

from plybench.analysis.errors.aggregate import pooled_reports
from plybench.analysis.errors.analysis import Analysis
from plybench.analysis.errors.consistency.verdicts import ConsistencyRecord, ConsistencyVerdict
from plybench.analysis.errors.export import write_moves
from plybench.analysis.errors.facets import FACETS, facet_values, parse_axis
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.moves import MatchupId, TracedMove
from plybench.analysis.errors.procedural.detection import MoveDiagnosis, MoveLabel
from plybench.analysis.errors.procedural.refutation import Refutation
from plybench.analysis.errors.stores import AnalysisStores
from plybench.analysis.statistics.standardization import reference_mix, standardized_rate
from plybench.analysis.stats.moves import MoveRecord
from plybench.app import PlyBench
from plybench.common.enums import StateClass
from plybench.llm import LLMConfig

op = PlyBench(LLMConfig())


def _traced(seq: int, game: str, player: str, is_optimal: bool = True) -> TracedMove:
    record = MoveRecord(StateClass.DECISION, is_optimal, 0.0 if is_optimal else 2.0, None, 100, None, 3, 1)
    matchup = MatchupId("exp", game, player, "random:")
    return TracedMove(matchup, 1, seq, record, "reasoning...", "board", "<A1>", ("<A1>", "<A2>"), ("<A1>",))


def _cell(game: str, player: str, n_optimal: int, n_blunders: int, depth: int = 4, stores: AnalysisStores | None = None) -> Analysis:
    moves = [_traced(seq, game, player) for seq in range(n_optimal)]
    moves += [_traced(n_optimal + seq, game, player, is_optimal=False) for seq in range(n_blunders)]
    funnel = FunnelResult("exp", op.registry.game_config(game), op.registry.player_config(player), moves)
    labels = {
        move.uid: MoveDiagnosis(
            frozenset({MoveLabel.CLEAN}) if move.record.is_optimal else frozenset({MoveLabel.DEEP_ERROR}),
            None if move.record.is_optimal else Refutation(depth, 1, 1, 2),
        )
        for move in moves
    }
    return Analysis(funnel, stores if stores is not None else AnalysisStores(), labels)


PLAYER = "llm:actions:text:openai:gpt-5-nano:thinking_enabled=True,reasoning_effort=low"
OTHER = "random:distribution=uniform"


# --- pooling -------------------------------------------------------------------------------------
def test_pooling_is_over_the_moves_and_not_over_the_cells_own_rates():
    # one big clean cell and one small bad one: the mean of the two rates is 0.25, the pooled rate is 0.01
    cells = [_cell("tic_tac_toe:", PLAYER, 99, 1), _cell("nim:", PLAYER, 2, 8)]
    (pooled,) = pooled_reports(cells, ("model",))

    assert pooled.n_cells == 2
    assert pooled.labels.n_moves == 110 and pooled.labels.n_suboptimal == 9
    assert pooled.labels.prevalence[0].n == 101  # clean, summed across both cells
    mean_of_rates = (1 / 100 + 8 / 10) / 2
    assert pooled.labels.explained.value == 0.0  # every blunder here is the residual
    deep = next(row for row in pooled.labels.prevalence if row.label is MoveLabel.DEEP_ERROR)
    assert deep.rate.value == 9 / 110 != mean_of_rates


def test_each_axis_keeps_the_thing_it_pools_over_and_drops_the_other():
    cells = [_cell("tic_tac_toe:", PLAYER, 4, 1), _cell("nim:", PLAYER, 4, 1), _cell("nim:", OTHER, 4, 1)]

    by_player = {report.scope.get("player"): report for report in pooled_reports(cells, ("player",))}
    assert set(by_player) == {PLAYER, OTHER}
    assert all(report.scope.get("game") == "" for report in by_player.values()), "a pooled facet is absent, not blank"
    assert by_player[PLAYER].n_cells == 2 and by_player[OTHER].n_cells == 1

    by_game = {report.scope.get("game"): report for report in pooled_reports(cells, ("game",))}
    assert set(by_game) == {"tic_tac_toe:", "nim:"}
    assert all(report.scope.get("player") == "" for report in by_game.values())

    (everything,) = pooled_reports(cells, ())
    assert (everything.scope.get("game"), everything.scope.get("player"), everything.n_cells) == ("", "", 3)

    # the point of facets: one row per model under each obfuscation, which no single-axis pool can express
    crossed = pooled_reports(cells, ("player", "game"))
    assert len(crossed) == 3 and all(report.n_cells == 1 for report in crossed)


def test_a_pooled_row_totals_the_cells_it_was_built_from():
    cells = [_cell("tic_tac_toe:", PLAYER, 7, 3), _cell("nim:", PLAYER, 11, 2)]
    (pooled,) = pooled_reports(cells, ("player",))
    assert pooled.funnel.n_moves == sum(len(cell.funnel.moves) for cell in cells) == 23
    assert json.dumps(pooled.to_dict())  # the pooled row is a JSON row like any other


def test_the_facet_registry_covers_the_axes_reports_are_split_by():
    assert {"experiment", "family", "presentation", "game", "model", "effort", "player"} <= set(FACETS)
    assert parse_axis("model,presentation") == ("model", "presentation")
    for bad in ("", "moddel", "model,nope"):
        try:
            parse_axis(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not parse -- a typo must fail, not pool over everything")


def test_consistency_pools_across_cells_and_keeps_the_judges_apart(tmp_path, monkeypatch):
    # the per-cell reports are what make this necessary: 425 verdicts spread over 85 ttt cells leave most
    # cells with nothing to print, while the pool has every verdict the judge produced
    monkeypatch.chdir(tmp_path)
    stores = AnalysisStores()
    cells = [_cell("tic_tac_toe:", PLAYER, 4, 1, stores=stores), _cell("nim:", PLAYER, 4, 1, stores=stores)]

    store = stores.consistency("exp")
    for index, move in enumerate(move for cell in cells for move in cell.funnel.analyzable):
        judge = "judge:a" if index % 2 else "judge:b"
        verdict = ConsistencyVerdict.INCONSISTENT if index == 0 else ConsistencyVerdict.CONSISTENT
        store.add(ConsistencyRecord(move.uid, judge, verdict, move.move, "<A2>" if index == 0 else move.move))

    (pooled,) = pooled_reports(cells, ("player",))
    assert {report.annotator for report in pooled.consistency} == {"judge:a", "judge:b"}
    assert sum(report.n_scored for report in pooled.consistency) == 10, "every cell's verdicts, not just the first"
    assert all(report.n_moves == 10 for report in pooled.consistency), "denominator is the pooled analysable moves"
    assert json.dumps(pooled.to_dict())


# --- the per-move export -------------------------------------------------------------------------
def test_the_export_round_trips_and_resolves_its_cells(tmp_path):
    cells = [_cell("tic_tac_toe:", PLAYER, 3, 2), _cell("nim:", OTHER, 1, 1)]
    path = tmp_path / "moves.jsonl"
    written, index = write_moves(path, cells)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert written == len(rows) == 7
    assert [row["uid"] for row in rows] == [move.uid for cell in cells for move in cell.funnel.moves]

    table = {entry["cell"]: entry for entry in json.loads(index.read_text())}
    assert len(table) == 2, "one entry per matchup, written once rather than on every row"
    assert {table[row["cell"]]["player"] for row in rows} == {PLAYER, OTHER}


def test_the_refutation_keys_are_absent_on_moves_the_solver_agreed_with(tmp_path):
    path = tmp_path / "moves.jsonl"
    write_moves(path, [_cell("tic_tac_toe:", PLAYER, 2, 1, depth=6)])
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    optimal = [row for row in rows if row["optimal"]]
    assert optimal and all("depth" not in row for row in optimal), "a null depth would read as 'seen immediately'"
    (blunder,) = [row for row in rows if not row["optimal"]]
    assert (blunder["depth"], blunder["n_replies"], blunder["width"]) == (6, 2, 0.5)
    assert blunder["labels"] == ["deep_error"]


# --- facets --------------------------------------------------------------------------------------
def test_facets_read_the_typed_configs_and_separate_the_game_from_its_obfuscation():
    funnel = _cell("magic_square:sample=False,magic_constant_add=0", PLAYER, 1, 0).funnel
    values = dict(facet_values(funnel))

    # family and presentation are only informative held against each other: the obfuscation is a different
    # presentation of the same game, which is the whole comparison the experiment exists to make
    assert values["family"] == "tic tac toe" and values["presentation"] == "magic_square"
    assert values["model"] == "openai:gpt-5-nano", "read off the typed config, not sliced out of the string"
    assert values["effort"] == "low"

    bot = _cell("tic_tac_toe:", OTHER, 1, 0).funnel
    assert dict(facet_values(bot))["model"] == "random", "a bot carries no model, so the key stands in for one"


def test_a_pooled_scope_keeps_the_experiment_whatever_was_asked_for():
    # two experiments are two corpora; pooling them into one row would be a category error, not a summary
    scope = _cell("tic_tac_toe:", PLAYER, 1, 0).scope
    assert scope.only("model").get("experiment") == "exp"
    assert scope.only().items == (("experiment", "exp"),)


# --- standardisation -----------------------------------------------------------------------------
def test_standardisation_removes_the_position_mix_from_a_comparison():
    """A weak model faces mostly forced positions, where it makes few load-bearing errors; a strong one
    faces mostly real decisions. The raw rates understate the gap because the weak model's denominator is
    padded with easy moves, which is exactly what reweighting to a shared mix takes out."""
    reference = reference_mix({"forced": 45, "decision": 50, "lost": 5})
    weak = {"forced": (10, 300), "decision": (60, 120), "lost": (20, 40)}
    strong = {"forced": (5, 100), "decision": (30, 300), "lost": (2, 10)}

    raw = {name: sum(h for h, _ in r.values()) / sum(n for _, n in r.values()) for name, r in (("weak", weak), ("strong", strong))}
    std = {name: standardized_rate(r, reference) for name, r in (("weak", weak), ("strong", strong))}

    assert raw["weak"] / raw["strong"] < std["weak"].value / std["strong"].value
    assert std["weak"].sem is not None and std["weak"].sem.lower > std["strong"].sem.upper
    assert std["weak"].n == 460, "the interval rests on every move in the group, not on the reweighted total"


def test_a_group_missing_a_stratum_is_renormalised_rather_than_imputed():
    reference = reference_mix({"forced": 50, "decision": 50})
    only_decisions = {"decision": (25, 100)}
    bundle = standardized_rate(only_decisions, reference)
    assert bundle is not None and bundle.value == 0.25, "the unseen stratum is dropped, never filled in with a guess"
    assert standardized_rate({}, reference) is None, "no coverage at all is no rate, which is not a rate of zero"
