from __future__ import annotations

import asyncio
import json
import sys
import weakref

import pytest
from test_learnable import _Stub
from test_status import OPTIMAL, UNIFORM, _op

from plybench.analysis.training import TrainingAnalysis
from plybench.callbacks.training_callbacks import TrainingCallbacks
from plybench.cli import train
from plybench.common.enums import MetricName
from plybench.harness.report import TrainingReporter
from plybench.harness.training.training import TrainingHarness
from plybench.trackers.result_tracker import ResultTracker


def _harness() -> TrainingHarness:
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    return TrainingHarness("reporting", _op(), ["tic_tac_toe:"], ["stub:"], [UNIFORM], [UNIFORM, OPTIMAL], ["num_epochs=2,num_training_games=3,num_test_games=4"], 2, seed=0)


def test_incremental_curves_match_full_results_and_release_each_phase(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = _harness()
    full = asyncio.run(harness.run(sync=True))
    expected = TrainingAnalysis(full, harness.op.registry)
    metadata = harness.get_results(progress=False, games=False)
    analysis = TrainingAnalysis(metadata, harness.op.registry)
    loaded: list[weakref.ReferenceType[ResultTracker]] = []
    original_load = ResultTracker.load

    def load(cls, filepath, parser, path_builder):
        assert all(reference() is None for reference in loaded), "the previous evaluation phase is still retained"
        tracker = original_load(filepath, parser, path_builder)
        assert tracker.place is not None and tracker.place["phase"] == "test"
        loaded.append(weakref.ref(tracker))
        return tracker

    monkeypatch.setattr(ResultTracker, "load", classmethod(load))
    for run in full.runs:
        for config in harness.tester_configs:
            tester = harness.op.registry.player_config(config)
            # Compare every metric and interval, including undefined values encoded as NaN.
            assert json.dumps(analysis.curve(run, tester, load_games=True).to_dict(), sort_keys=True) == json.dumps(expected.curve(run, tester).to_dict(), sort_keys=True)
    assert len(loaded) == 12 and all(reference() is None for reference in loaded)
    assert all(game is None for epoch in metadata.epochs for tracker in [*epoch.tests, *([epoch.train] if epoch.train is not None else [])] for game in tracker.games)


def test_training_cli_finalizes_without_loading_all_games(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    harness = _harness()
    original_get_results = harness.get_results
    ended = []

    def get_results(progress=None, games=True):
        assert games is False
        with monkeypatch.context() as context:

            def refuse(self, game_round):
                raise AssertionError("finalization loaded a game")

            context.setattr(ResultTracker, "load_game", refuse)
            return original_get_results(progress=progress, games=games)

    original_callbacks = train.console_training_callbacks

    def callbacks(progress):
        return TrainingCallbacks.combine(original_callbacks(progress), TrainingCallbacks(training_end_callback=ended.append))

    monkeypatch.setattr(harness, "get_results", get_results)
    monkeypatch.setattr(train, "training_from_args", lambda op, args: harness)
    monkeypatch.setattr(train, "console_training_callbacks", callbacks)
    monkeypatch.setattr(sys, "argv", ["plybench-train", "--experiment", "reporting", "--sync", "--quiet"])
    train.main()
    output = capsys.readouterr().out
    assert "[training] done: 6 epoch(s) across 2 run(s)" in output
    assert output.count("  delta:") == 4 and output.count("(n=4)") == 12
    assert "results under results/training/reporting/" in output
    assert len(ended) == 1 and all(game is None for epoch in ended[0].epochs for tracker in epoch.tests for game in tracker.games)
    # The incremental printer keeps the same output when reading these metadata-only results again.
    train.report_curves(ended[0], harness.op.registry, harness.tester_configs, MetricName.WIN_RATE)
    curves = capsys.readouterr().out
    assert curves in output


@pytest.mark.parametrize("moves", [False, True])
def test_report_rebuild_reads_games_only_when_moves_are_requested(tmp_path, monkeypatch, moves):
    monkeypatch.chdir(tmp_path)
    harness = _harness()
    asyncio.run(harness.run(sync=True))
    expected = TrainingReporter(harness, tmp_path / "unused.json", include_complete=True).build().to_dict()
    if not moves:
        # the solver's rating (optimal move rate, regret) needs the games replayed, so a rebuild without moves leaves it out
        rated = {"optimal move rate", "regret per game", "cumulative regret"}
        expected["blocks"] = [block for block in expected["blocks"] if block.get("measure") not in rated]
        assert len(expected["blocks"]) < len(TrainingReporter(harness, tmp_path / "unused.json", include_complete=True).build().to_dict()["blocks"])
        for block in expected["blocks"]:
            if block["type"] == "line":
                for series in block["series"]:
                    for point in series["points"]:
                        point.pop("moves", None)
    loaded = []
    original_load = ResultTracker.load_game

    def load(self, game_round):
        assert moves, "the default report rebuild read a game"
        loaded.append(game_round)
        return original_load(self, game_round)

    monkeypatch.setattr(ResultTracker, "load_game", load)
    monkeypatch.setattr(train, "training_from_args", lambda op, args: harness)
    monkeypatch.setattr(train, "build_op", lambda **kwargs: harness.op)
    argv = ["plybench-train", "--experiment", "reporting", "--report-only", "rebuilt.json"]
    monkeypatch.setattr(sys, "argv", [*argv, *(["--report-moves"] if moves else [])])
    train.main()
    actual = json.loads((tmp_path / "rebuilt.json").read_text())
    assert actual["finished"]
    assert actual["items"] == expected["items"] and actual["progress"] == expected["progress"]
    assert actual["blocks"] == expected["blocks"]
    assert bool(loaded) == moves


def test_report_moves_requires_a_report_rebuild(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["plybench-train", "--report-moves"])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
