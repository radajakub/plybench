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
from plybench.cli import run
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
    json.dumps(report, allow_nan=False)


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
