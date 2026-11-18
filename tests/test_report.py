"""The report of plybench-run and plybench-train (`--job-report`): the matchups or training runs the process plays and nothing else, their progress, and
results that equal the experiment's `--status`; what an earlier run saved is not shown."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import pytest
from plyllm import LLMConfig
from test_learnable import _Stub, _StubParams

from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.statistics.intervals import t_ci
from plybench.app import PlyBench
from plybench.cli import run, train
from plybench.cli.shared import write_saved_report
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.plan import benchmark_plan
from plybench.harness.report import BenchmarkReporter, TrainingReporter
from plybench.harness.status import benchmark_status, training_status
from plybench.harness.training.training import TrainingHarness
from plybench.player.spec import PlayerSpec
from plybench.report import Item, Report, ReportFile, replicate_series

UNIFORM = "random:distribution=uniform"
OPTIMAL = "optimal:stochastic=True"


def _op() -> PlyBench:
    op = PlyBench(LLMConfig())
    op.registry.register_player(PlayerSpec("stub", _StubParams, lambda game, cfg, pid: _Stub(cfg, pid)))
    return op


def _blocks(report, kind):
    return [block for block in report["blocks"] if block["type"] == kind]


def test_a_benchmark_report_has_only_the_matchups_left_and_their_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    op = _op()
    # one matchup played before: a later run of the experiment does not show it
    asyncio.run(Benchmark("report", op, ["nim:"], [UNIFORM], [OPTIMAL], 4, seed=0).run(sync=True))
    benchmark = Benchmark("report", op, ["nim:", "tic_tac_toe:"], [UNIFORM], [OPTIMAL, UNIFORM], 4, seed=0)
    path = tmp_path / "report.json"
    reporter = BenchmarkReporter(benchmark, path)
    reporter.file.write()
    before = json.loads(path.read_text())
    plan = benchmark_plan(benchmark)
    assert [item["key"] for item in before["items"]] == [item.key for item in plan.items if item.saved_games < item.planned_games]
    assert len(before["items"]) == 3 and before["progress"] == {"done": 0, "total": 12, "unit": "games"} and not before["finished"]

    asyncio.run(benchmark.run(sync=True, benchmark_callbacks=reporter.callbacks()))
    reporter.file.close(finished=True)
    report = json.loads(path.read_text())
    assert report["finished"] and report["progress"]["done"] == 12 and report["version"] == 1
    status = {(m["game"], m["player"], m["opponent"]): m for m in benchmark_status(benchmark)["matchups"]}
    keys = {item["key"]: item for item in report["items"]}
    [table] = _blocks(report, "table")
    for row in table["rows"]:
        configs = keys[row["item"]]["configs"]
        win = status[(configs["game"], configs["player"], configs["opponent"])]["metrics"]["win_rate"]
        assert row["win_rate"] == {"value": win["value"], "lower": win["lower"], "upper": win["upper"]} and row["games"] == "4/4"
    [bars] = _blocks(report, "bars")
    assert all(abs(sum(row["values"].values()) - 1.0) < 1e-9 for row in bars["rows"])
    json.dumps(report, allow_nan=False)


def test_a_training_report_follows_the_status_of_its_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    schedule = "num_epochs=2,num_training_games=3,num_test_games=4"
    harness = TrainingHarness("report", _op(), ["tic_tac_toe:"], ["stub:"], [UNIFORM], [UNIFORM, OPTIMAL], [schedule], 2, seed=0)
    path = tmp_path / "report.json"
    reporter = TrainingReporter(harness, path)
    initial = reporter.build().to_dict()
    [initial_grid] = _blocks(initial, "grid")
    [initial_row] = initial_grid["rows"]
    assert initial_row["cells"] == [{"done": 0, "total": 16}, {"done": 0, "total": 22}, {"done": 0, "total": 22}]
    assert len(initial_row["rows"]) == 2
    assert all(len(block["series"]) == 1 and block["series"][0]["replicates"] == 2 and block["series"][0]["points"] == [] for block in _blocks(initial, "line"))
    asyncio.run(harness.run(sync=True, training_callbacks=reporter.callbacks()))
    reporter.file.close(finished=True)
    report = json.loads(path.read_text())
    # two replicates of one run: one group, named by their replicate
    assert [item["name"] for item in report["items"]] == ["rep 0", "rep 1"] and len({item["group"] for item in report["items"]}) == 1
    assert report["progress"]["done"] == report["progress"]["total"] == 2 * (2 * 3 + 3 * 4 * 2)
    [grid] = _blocks(report, "grid")
    assert grid["columns"] == ["0", "1", "2"] and all(cell["done"] == cell["total"] for row in grid["rows"] for cell in row["cells"])

    status = training_status(harness)
    win = next(b for b in _blocks(report, "line") if b["group"] == f"tests vs {OPTIMAL}" and b["measure"] == "win rate")
    [series] = win["series"]
    assert series["group"] == report["items"][0]["group"] and series["replicates"] == 2 and "item" not in series
    raw = [reporter._series(item["key"], epochs, reporter._test_of(OPTIMAL), "win_rate") for (_, epochs), item in zip(reporter.runs, report["items"], strict=True)]
    for point in series["points"]:
        values = [run["epochs"][point["x"]]["tests"][OPTIMAL]["metrics"]["win_rate"]["value"] for run in status["runs"]]
        interval = t_ci(Distribution(values))
        assert point["value"] == interval.value and point["n"] == 2
        assert point["lower"] == max(0, interval.lower) and point["upper"] == min(1, interval.upper)
        assert sum(move["count"] for move in point["moves"]) == sum(
            move["count"] for entry in raw for original in entry["points"] if original["x"] == point["x"] for move in original["moves"]
        )
    assert len(grid["rows"]) == 1 and len(grid["rows"][0]["rows"]) == 2
    assert grid["rows"][0]["group"] == series["group"]
    for index, cell in enumerate(grid["rows"][0]["cells"]):
        assert cell == {field: sum(row["cells"][index][field] for row in grid["rows"][0]["rows"]) for field in ("done", "total")}
    train = next(b for b in _blocks(report, "line") if b["group"] == "training games vs the trainer" and b["measure"] == "win rate")
    assert all("lower" in point and point["n"] == 2 and [p["x"] for p in series["points"]] == [1, 2] for series in train["series"] for point in series["points"])

    # a later run of the finished experiment plays nothing: an empty report
    again = TrainingReporter(harness, tmp_path / "again.json")
    again.file.write()
    assert json.loads((tmp_path / "again.json").read_text())["items"] == []


def test_a_training_report_counts_the_moves_of_each_phase_against_the_game_of_its_epoch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    # the stub presses red: never optimal before the switch (0 against 1), always after it (4/5 against 1/5)
    game = "bandit:p=0;1,switch=4/5;1/5,switch_epoch=2,rules=hidden"
    harness = TrainingHarness("moves", _op(), [game], ["stub:"], ["none:"], ["none:"], ["num_epochs=3,num_training_games=3,num_test_games=4"], 1, seed=0)
    path = tmp_path / "report.json"
    reporter = TrainingReporter(harness, path)
    asyncio.run(harness.run(sync=True, training_callbacks=reporter.callbacks()))
    reporter.file.close(finished=True)
    report = json.loads(path.read_text())

    lines = {(block["group"], block["measure"]): block for block in _blocks(report, "line") if "group" in block}
    test, train = lines[("tests vs none", "win rate")], lines[("training games vs the trainer", "win rate")]
    for block, games, epochs in ((test, 4, [0, 1, 2, 3]), (train, 3, [1, 2, 3])):
        [series] = block["series"]
        assert [point["x"] for point in series["points"]] == epochs
        assert all(point["moves"] == [{"move": "<button:red>", "count": games, "optimal": games if point["x"] >= 2 else 0}] for point in series["points"])
        # the optimum of both games the run plays
        assert sorted(reference["value"] for reference in block["references"]) == [0.8, 1.0]
        # both apply to the run's game: the first before the switch, the second after it
        assert all(reference["games"] == [report["items"][0]["configs"]["game"]] for reference in block["references"])
    # the report keeps no game in memory once a phase is complete: each phase a copy with its counts and summary only
    assert reporter.live and all(tracker.is_complete() and tracker.games.count(None) == tracker.n for tracker in reporter.live.values())
    json.dumps(report, allow_nan=False)


def test_a_training_report_rates_the_optimal_play_of_tests_and_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    # the stub presses red: regret V* - Q*(red) = 1 - (-1) = 2 per game before the switch, 0.6 - 0.6 = 0 after it
    game = "bandit:p=0;1,switch=4/5;1/5,switch_epoch=2,rules=hidden"
    harness = TrainingHarness("optimal", _op(), [game], ["stub:"], ["none:"], ["none:"], ["num_epochs=3,num_training_games=3,num_test_games=4"], 1, seed=0)
    path = tmp_path / "report.json"
    reporter = TrainingReporter(harness, path)
    asyncio.run(harness.run(sync=True, training_callbacks=reporter.callbacks()))
    reporter.file.close(finished=True)
    report = json.loads(path.read_text())

    lines = {(block["group"], block["measure"]): block for block in _blocks(report, "line") if "group" in block}
    assert {block.get("panel") for block in _blocks(report, "line")} == {"tests", "optimal", "training", "tokens"}
    assert lines[("optimal play, tests vs none", "optimal move rate")]["panel"] == "optimal"
    assert lines[("training games vs the trainer", "cumulative regret")]["panel"] == "training"

    def points(group, measure):
        [series] = lines[(group, measure)]["series"]
        return series["points"]

    rate = points("optimal play, tests vs none", "optimal move rate")
    assert [(point["x"], point["value"]) for point in rate] == [(0, 0.0), (1, 0.0), (2, 1.0), (3, 1.0)]
    # Wilson over the 4 moves of each test phase
    assert all(0.0 <= point["lower"] <= point["value"] <= point["upper"] <= 1.0 and point["upper"] > point["lower"] for point in rate)
    regret = points("optimal play, tests vs none", "regret per game")
    assert [(point["x"], point["value"]) for point in regret] == [(0, pytest.approx(2.0)), (1, pytest.approx(2.0)), (2, pytest.approx(0.0)), (3, pytest.approx(0.0))]
    train_rate = points("training games vs the trainer", "optimal move rate")
    assert [(point["x"], point["value"]) for point in train_rate] == [(1, 0.0), (2, 1.0), (3, 1.0)] and all("lower" not in point for point in train_rate)
    # epoch 0 trains nothing; epoch 1 trains 3 games of regret 2; after the switch red is optimal
    cumulative = points("training games vs the trainer", "cumulative regret")
    assert [(point["x"], point["value"]) for point in cumulative] == [(0, 0.0), (1, pytest.approx(6.0)), (2, pytest.approx(6.0)), (3, pytest.approx(6.0))]
    assert [reference["value"] for reference in lines[("training games vs the trainer", "cumulative regret")]["references"]] == [0.0]
    assert [reference["value"] for reference in lines[("optimal play, tests vs none", "optimal move rate")]["references"]] == [1.0]
    json.dumps(report, allow_nan=False)


def test_a_training_report_has_no_optimal_play_without_a_solver(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    harness = TrainingHarness("unsolved", _op(), ["connect_four:"], ["stub:"], [UNIFORM], [UNIFORM], ["num_epochs=1,num_training_games=1,num_test_games=1"], 1, seed=0)
    path = tmp_path / "report.json"
    reporter = TrainingReporter(harness, path)
    asyncio.run(harness.run(sync=True, training_callbacks=reporter.callbacks()))
    reporter.file.close(finished=True)
    report = json.loads(path.read_text())
    measures = {block.get("measure") for block in _blocks(report, "line")}
    assert "win rate" in measures and not measures & {"optimal move rate", "regret per game", "cumulative regret"}
    assert {block.get("panel") for block in _blocks(report, "line")} == {"tests", "training", "tokens"}


def test_run_writes_the_report_and_a_finished_experiment_reports_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = ["plybench-run", "--name", "cli", "--games", "nim:", "--players", UNIFORM, "--opponents", OPTIMAL, "--quiet", "--seed", "0"]
    monkeypatch.setattr(sys, "argv", [*base, "--num-games", "2", "--job-report", "first.json"])
    run.main()
    first = json.loads((tmp_path / "first.json").read_text())
    assert first["finished"] and first["progress"] == {"done": 2, "total": 2, "unit": "games"}
    # the same matchup again: everything is saved, so the run plays and reports nothing
    monkeypatch.setattr(sys, "argv", [*base, "--num-games", "2", "--job-report", "second.json"])
    run.main()
    assert json.loads((tmp_path / "second.json").read_text())["items"] == []


def _counting(tmp_path, interval):
    # a ReportFile whose report has one item, done as `done[0]`, and the dones of every build it made
    done, built = [0], []

    def build():
        built.append(done[0])
        return Report(items=[Item("a", {"player": UNIFORM}, 10, done[0])])

    return ReportFile(tmp_path / "report.json", build, interval), done, built


def test_a_change_within_the_interval_is_written_when_it_ends(tmp_path):
    file, done, built = _counting(tmp_path, 0.2)
    file.changed()
    # three changes within the interval: none written now, one write of the last when it ends
    for value in (1, 2, 3):
        done[0] = value
        file.changed()
    assert built == [0]
    time.sleep(0.4)
    assert built == [0, 3]
    assert json.loads((tmp_path / "report.json").read_text())["progress"]["done"] == 3


def test_close_writes_at_once_and_no_change_is_written_after(tmp_path):
    file, done, built = _counting(tmp_path, 0.2)
    file.changed()
    done[0] = 5
    file.changed()
    file.close(True)
    time.sleep(0.4)
    assert built == [0, 5]
    assert json.loads((tmp_path / "report.json").read_text())["finished"]


def test_replicate_series_uses_between_run_intervals_spread_and_pooled_moves():
    entries = [
        {"item": f"rep{index}", "points": [{"x": 0, "value": value, "lower": -100, "upper": 100, "moves": [{"move": "red", "count": index + 1, "optimal": index}]}]}
        for index, value in enumerate([0.2, 0.4, 0.6])
    ]
    [series] = replicate_series(entries, {entry["item"]: "group" for entry in entries}, rate=True)
    [point] = series["points"]
    interval = t_ci(Distribution([0.2, 0.4, 0.6]))
    assert series["group"] == "group" and series["replicates"] == 3
    assert point["value"] == pytest.approx(0.4) and point["n"] == 3
    assert (point["lower"], point["upper"]) == pytest.approx((max(0, interval.lower), min(1, interval.upper)))
    assert (point["q10"], point["q90"]) == pytest.approx((0.24, 0.56))
    assert point["moves"] == [{"move": "red", "count": 6, "optimal": 3}]


def test_replicate_series_handles_partial_empty_and_single_runs():
    entries = [
        {"item": "a0", "points": [{"x": 0, "value": 10}, {"x": 1, "value": 20}]},
        {"item": "a1", "points": [{"x": 0, "value": 30}, {"x": 1, "value": None}, {"x": 2, "value": float("nan")}]},
        {"item": "b0", "points": []},
        {"item": "b1", "points": []},
        {"item": "single", "points": [{"x": 0, "value": 0.5, "lower": 0.1, "upper": 0.9}]},
    ]
    first, empty, single = replicate_series(entries, {"a0": "a", "a1": "a", "b0": "b", "b1": "b"}, rate=False)
    assert [point["n"] for point in first["points"]] == [2, 1]
    assert first["points"][0]["lower"] < 0 and first["points"][0]["upper"] > 30
    assert first["points"][1] == {"x": 1, "value": 20, "n": 1, "q10": 20, "q90": 20}
    assert empty == {"group": "b", "replicates": 2, "points": []} and single == entries[-1]
    assert replicate_series([], {}, rate=True) == []


def test_pooled_moves_leave_out_ratings_unless_every_move_is_rated():
    entries = [
        {"item": "a", "points": [{"x": 0, "value": 1, "moves": [{"move": "red", "count": 2, "optimal": 2}]}]},
        {"item": "b", "points": [{"x": 0, "value": 0, "moves": [{"move": "red", "count": 1}, {"move": "blue", "count": 4}]}]},
    ]
    [series] = replicate_series(entries, {"a": "g", "b": "g"}, rate=True)
    assert series["points"][0]["moves"] == [{"move": "blue", "count": 4}, {"move": "red", "count": 3}]


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_report_only_includes_completed_results_without_playing_or_changing_them(tmp_path, monkeypatch, capsys, complete, training):
    monkeypatch.chdir(tmp_path)
    op = _op()
    harness = None
    if training:
        _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
        schedule = "num_epochs=1,num_training_games=2,num_test_games=3"
        args = ("saved", op, ["bandit:p=4/5;1/5"], ["stub:"], ["none:"], ["none:"], [schedule])
        asyncio.run(TrainingHarness(*args, 2 if complete else 1, seed=0).run(sync=True))
        harness = TrainingHarness(*args, 2, seed=0)
        monkeypatch.setattr(train, "training_from_args", lambda op, args: harness)
        cli = train
        command = "plybench-train"
    else:
        games = ["nim:", "tic_tac_toe:"]
        asyncio.run(Benchmark("saved", op, games if complete else games[:1], [UNIFORM], [OPTIMAL], 2, seed=0).run(sync=True))
        benchmark = Benchmark("saved", op, games, [UNIFORM], [OPTIMAL], 2, seed=0)
        monkeypatch.setattr(run, "benchmark_from_args", lambda op, args: benchmark)
        cli = run
        command = "plybench-run"

    def refuse(*args, **kwargs):
        raise AssertionError("report-only attempted to run games")

    monkeypatch.setattr(TrainingHarness, "run", refuse)
    monkeypatch.setattr(Benchmark, "run", refuse)
    monkeypatch.setattr(cli, "build_op", lambda **kwargs: op)
    saved = {path: path.read_bytes() for path in (tmp_path / "results").rglob("*") if path.is_file()}
    monkeypatch.setattr(sys, "argv", [command, "--experiment", "saved", "--report-only", "rebuilt.json"])
    cli.main()
    report = json.loads((tmp_path / "rebuilt.json").read_text())
    assert "report written to rebuilt.json" in capsys.readouterr().out
    assert report["finished"] == complete and len(report["items"]) == 2
    assert (report["progress"]["done"] == report["progress"]["total"]) == complete
    assert report["progress"]["done"] > 0
    assert saved == {path: path.read_bytes() for path in (tmp_path / "results").rglob("*") if path.is_file()}
    if training:
        [win] = [block for block in _blocks(report, "line") if block.get("group") == "tests vs none" and block["measure"] == "win rate"]
        [series] = win["series"]
        assert series["replicates"] == 2 and [point["x"] for point in series["points"]] == [0, 1]
        assert all(point["n"] == (2 if complete else 1) and ("lower" in point) == complete for point in series["points"])
        assert harness is not None
        status = training_status(harness)
        for point in series["points"]:
            values = [
                entry["epochs"][point["x"]]["tests"]["none:"]["metrics"]["win_rate"]["value"]
                for entry in status["runs"]
                if entry["epochs"][point["x"]]["tests"]["none:"]["metrics"]
            ]
            assert point["value"] == pytest.approx(sum(values) / len(values))
    else:
        [table] = _blocks(report, "table")
        assert sum(row["win_rate"] is not None for row in table["rows"]) == (2 if complete else 1)


@pytest.mark.parametrize("cli", [run, train])
@pytest.mark.parametrize("option", ["--check", "--status", "--summaries", "--job-report"])
def test_report_only_rejects_conflicting_modes(monkeypatch, cli, option):
    extra = ["check"] if option == "--summaries" else ["live.json"] if option == "--job-report" else []
    monkeypatch.setattr(sys, "argv", ["plybench", "--report-only", "rebuilt.json", option, *extra])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_report_only_propagates_write_failures(tmp_path, monkeypatch):
    def refuse(self, path):
        raise OSError("cannot write report")

    monkeypatch.setattr(Report, "save", refuse)
    with pytest.raises(OSError, match="cannot write report"):
        write_saved_report(Report, tmp_path / "report.json", notify=False)


@pytest.mark.parametrize("training", [False, True])
def test_report_source_keeps_exact_item_scope_and_rejects_unknown_keys(tmp_path, monkeypatch, training):
    monkeypatch.chdir(tmp_path)
    op = _op()
    if training:
        _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
        harness = TrainingHarness("source", op, ["bandit:p=4/5;1/5"], ["stub:"], ["none:"], ["none:"], ["num_epochs=1,num_training_games=1,num_test_games=2"], 2, seed=0)
        asyncio.run(harness.run(sync=True))
        initial = TrainingReporter(harness, tmp_path / "unused.json", include_complete=True).build().to_dict()
        monkeypatch.setattr(train, "training_from_args", lambda op, args: harness)
        cli = train
    else:
        benchmark = Benchmark("source", op, ["nim:", "tic_tac_toe:"], [UNIFORM], [OPTIMAL], 2, seed=0)
        asyncio.run(benchmark.run(sync=True))
        initial = BenchmarkReporter(benchmark, tmp_path / "unused.json", include_complete=True).build().to_dict()
        monkeypatch.setattr(run, "benchmark_from_args", lambda op, args: benchmark)
        cli = run
    source = tmp_path / "source.json"
    selected = initial["items"][1]
    source.write_text(json.dumps(initial | {"items": [selected]}))
    monkeypatch.setattr(cli, "build_op", lambda **kwargs: op)
    monkeypatch.setattr(sys, "argv", ["plybench", "--experiment", "source", "--report-only", "rebuilt.json", "--report-source", str(source)])
    cli.main()
    rebuilt = json.loads((tmp_path / "rebuilt.json").read_text())
    assert len(rebuilt["items"]) == 1 and rebuilt["progress"]["done"] == selected["done"]
    assert all(rebuilt["items"][0][field] == selected[field] for field in ("key", "configs", "planned", "done"))
    if training:
        # A single selected replicate must not be averaged with the excluded replicate.
        assert all(series.get("item") == selected["key"] for block in _blocks(rebuilt, "line") for series in block["series"])
    source.write_text(json.dumps({"items": [{"key": "unknown"}]}))
    with pytest.raises(ValueError, match="absent from the selection"):
        cli.main()


@pytest.mark.parametrize("cli", [run, train])
def test_report_source_requires_report_only(monkeypatch, cli):
    monkeypatch.setattr(sys, "argv", ["plybench", "--report-source", "source.json"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
