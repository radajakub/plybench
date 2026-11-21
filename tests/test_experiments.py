from __future__ import annotations

import json
import os
import sys

import pytest
from plyllm import LLMConfig
from test_store import _MemoryStore

from plybench.app import PlyBench
from plybench.cli import experiment as cli
from plybench.common.store import use_store
from plybench.configs.benchmark_config import BenchmarkConfig, ToggleItem
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.experiments import list_experiments, load_experiment, save_experiment

# Experiments are records of the store: the harness reads them from it, plybench-experiment writes them, and an item of an experiment with results is
# disabled, never removed.

UNIFORM, NORMAL = "random:distribution=uniform", "random:distribution=normal"


def _config(*players: ToggleItem) -> dict:
    return BenchmarkConfig([ToggleItem("tic_tac_toe:", True)], list(players), [ToggleItem(UNIFORM, True)], 2).to_dict()


@pytest.fixture
def memory():
    store = _MemoryStore()
    use_store(store)
    yield store
    use_store(None)


def test_a_benchmark_reads_its_experiment_from_the_store(memory, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_experiment("benchmarks", "ttt", _config(ToggleItem(UNIFORM, True), ToggleItem(NORMAL, False)))
    assert "experiments/benchmarks/ttt.json" in memory.records and not (tmp_path / "experiments/benchmarks/ttt.json").exists()
    benchmark = Benchmark.load_experiment(PlyBench(LLMConfig()), "ttt")
    assert benchmark.experiment == "ttt" and benchmark.player_configs == [UNIFORM]
    with pytest.raises(FileNotFoundError, match="no experiment experiments/benchmarks/none.json"):
        Benchmark.load_experiment(PlyBench(LLMConfig()), "none")


def test_an_item_of_an_experiment_with_results_is_disabled_not_removed(memory):
    save_experiment("benchmarks", "ttt", _config(ToggleItem(UNIFORM, True), ToggleItem(NORMAL, True)))
    # no results yet: an item may still go
    save_experiment("benchmarks", "ttt", _config(ToggleItem(UNIFORM, True)))
    save_experiment("benchmarks", "ttt", _config(ToggleItem(UNIFORM, True), ToggleItem(NORMAL, True)))
    memory.put("results/benchmarks/ttt/tic_tac_toe_2/x/game_1.json", {})
    with pytest.raises(ValueError, match=f"disable items .* Removed: player_configs: {NORMAL}"):
        save_experiment("benchmarks", "ttt", _config(ToggleItem(UNIFORM, True)))
    save_experiment("benchmarks", "ttt", _config(ToggleItem(UNIFORM, True), ToggleItem(NORMAL, False), ToggleItem("random:distribution=x", True)))
    assert [item["enabled"] for item in load_experiment("benchmarks", "ttt")["player_configs"]] == [True, False, True]
    # another experiment's results do not count
    save_experiment("benchmarks", "ttt2", _config(ToggleItem(UNIFORM, True), ToggleItem(NORMAL, True)))
    save_experiment("benchmarks", "ttt2", _config(ToggleItem(UNIFORM, True)))


@pytest.mark.parametrize(
    ("kind", "name", "data", "problem"),
    [
        ("benchmarks", "ttt", {"game_configs": []}, "not a benchmarks experiment"),
        ("benchmarks", "ttt", [], "not a benchmarks experiment"),
        ("benchmarks", "ttt", _config() | {"colour_rule": "nobody"}, "unknown colour_rule"),
        ("training", "ttt", _config(), "not a training experiment"),
        ("exploration", "ttt", _config(), "kind must be one of"),
        ("benchmarks", "../ttt", _config(), "invalid experiment name"),
    ],
)
def test_what_the_harness_would_refuse_is_not_saved(memory, kind, name, data, problem):
    with pytest.raises(ValueError, match=problem):
        save_experiment(kind, name, data)
    assert memory.records == {}


def _run(monkeypatch, capsys, *argv: str) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["plybench-experiment", *argv])
    code = 0
    try:
        cli.main()
    except SystemExit as exit:
        code = int(exit.code or 0)
    out, err = capsys.readouterr()
    return code, out, err


def test_the_command_lists_shows_and_saves_experiments(memory, tmp_path, monkeypatch, capsys):
    file = tmp_path / "ttt.json"
    file.write_text(json.dumps(_config(ToggleItem(UNIFORM, True))))
    assert _run(monkeypatch, capsys, "put", "benchmarks/ttt", str(file))[:2] == (0, "saved experiments/benchmarks/ttt.json\n")
    assert json.loads(_run(monkeypatch, capsys, "show", "benchmarks/ttt")[1]) == _config(ToggleItem(UNIFORM, True))
    assert _run(monkeypatch, capsys, "list")[1] == "benchmarks/ttt\n"
    assert json.loads(_run(monkeypatch, capsys, "list", "--json")[1]) == {"benchmarks": {"ttt": _config(ToggleItem(UNIFORM, True))}, "training": {}}
    code, _, err = _run(monkeypatch, capsys, "show", "benchmarks/none")
    assert code == 2 and "does not exist" in err
    file.write_text("{not json")
    assert _run(monkeypatch, capsys, "put", "benchmarks/ttt", str(file))[0] == 1
    assert list_experiments("benchmarks") == {"ttt": _config(ToggleItem(UNIFORM, True))}


def test_an_edit_the_store_refuses_is_kept(memory, tmp_path, monkeypatch, capsys):
    save_experiment("benchmarks", "ttt", _config(ToggleItem(UNIFORM, True), ToggleItem(NORMAL, True)))
    memory.put("results/benchmarks/ttt/game_1.json", {})
    # an "editor" that removes the second player
    editor = tmp_path / "editor.py"
    editor.write_text("import json, sys\npath = sys.argv[1]\ndata = json.load(open(path))\ndata['player_configs'] = data['player_configs'][:1]\njson.dump(data, open(path, 'w'))\n")
    monkeypatch.setenv("VISUAL", f"{sys.executable} {editor}")
    code, _, err = _run(monkeypatch, capsys, "edit", "benchmarks/ttt")
    assert code == 1 and "not saved; the edit is in" in err
    kept = err.rsplit("plybench-experiment put benchmarks/ttt ", 1)[1].rstrip(")\n")
    assert len(json.loads(open(kept).read())["player_configs"]) == 1
    os.unlink(kept)
    assert len(load_experiment("benchmarks", "ttt")["player_configs"]) == 2
