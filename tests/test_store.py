from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from plyllm import LLMConfig
from test_learnable import EPOCHS, GAMES, _Stub, _StubParams

from plybench.app import PlyBench
from plybench.common.store import STORE_ENV, FileStore, RecordExists, open_store, use_store
from plybench.harness.training.training import TrainingHarness
from plybench.player.spec import PlayerSpec

# The store every result goes through: files by default, as they were written before; and a run on another store (here one in memory) writes no
# result file of its own and resumes from what that store holds.


class _MemoryStore:
    # the protocol over a dict, as a database store would be: nothing on disk
    def __init__(self) -> None:
        self.records: dict[str, Any] = {}

    def put(self, key: str, value: Any, *, replace: bool = False) -> None:
        if key in self.records and not replace:
            raise RecordExists(key)
        self.records[key] = json.loads(json.dumps(value, allow_nan=False)) if key.endswith(".json") else value

    def put_many(self, values: Mapping[str, Any]) -> None:
        if there := [key for key in values if key in self.records]:
            raise RecordExists(", ".join(there))
        for key, value in values.items():
            self.put(key, value)

    def get(self, key: str) -> Any | None:
        return self.records.get(key)

    def exists(self, key: str) -> bool:
        return key in self.records

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(key for key in self.records if not prefix or key == prefix or key.startswith(prefix + "/"))

    def append(self, key: str, record: Any) -> None:
        self.records.setdefault(key, []).append(record)

    def read_log(self, key: str) -> list[Any]:
        return list(self.records.get(key, []))


@pytest.fixture
def files(tmp_path):
    return FileStore(tmp_path)


def test_a_record_is_written_once_unless_replaced(files, tmp_path):
    files.put("results/a/game_1.json", {"moves": [1, 2]})
    with pytest.raises(RecordExists):
        files.put("results/a/game_1.json", {"moves": [3]})
    files.put("results/a/game_1.json", {"moves": [3]}, replace=True)
    assert files.get("results/a/game_1.json") == {"moves": [3]}
    # compact JSON, as the game files were written before
    assert (tmp_path / "results/a/game_1.json").read_text() == json.dumps({"moves": [3]})
    assert sorted(path.name for path in (tmp_path / "results/a").iterdir()) == ["game_1.json"]


def test_several_records_are_written_all_or_none(files):
    files.put("learner/game_1/llm_calls.json", [])
    with pytest.raises(RecordExists):
        files.put_many({"learner/game_1/update.json": 1.0, "learner/game_1/llm_calls.json": []})
    assert files.keys("learner") == ["learner/game_1/llm_calls.json"]
    files.put_many({"learner/game_2/update.json": 2.0, "learner/game_2/no_update": ""})
    assert files.keys("learner/game_2") == ["learner/game_2/no_update", "learner/game_2/update.json"]
    assert files.get("learner/game_2/no_update") == ""


def test_keys_are_listed_by_whole_parts(files):
    for key in ["results/ttt/game_1.json", "results/ttt2/game_1.json", "results/ttt/sub/game_2.json"]:
        files.put(key, {})
    assert files.keys("results/ttt") == ["results/ttt/game_1.json", "results/ttt/sub/game_2.json"]
    assert files.keys("results/none") == []


def test_a_log_keeps_its_records_in_order(files, tmp_path):
    files.append("results/llm_blocks.jsonl", {"reason": "é"})
    files.append("results/llm_blocks.jsonl", {"reason": "b"})
    assert files.read_log("results/llm_blocks.jsonl") == [{"reason": "é"}, {"reason": "b"}]
    # one line per record, its text unescaped, as the log was written before
    assert (tmp_path / "results/llm_blocks.jsonl").read_text() == '{"reason": "é"}\n{"reason": "b"}\n'


@pytest.mark.parametrize("value", [math.nan, math.inf, {"x": [-math.inf]}])
def test_what_is_not_json_is_refused(files, value):
    with pytest.raises(ValueError, match="not storable JSON"):
        files.put("results/a/game_1.json", value)
    assert files.keys() == []


@pytest.mark.parametrize("key", ["/abs/game_1.json", "results//game_1.json", "results/.hidden/x.json", "results/x.json.tmp", ""])
def test_a_key_is_a_relative_name(files, key):
    with pytest.raises(ValueError, match="invalid key"):
        files.put(key, {})


def test_a_store_named_but_not_installed_is_an_error(monkeypatch):
    monkeypatch.setenv(STORE_ENV, "nowhere")
    with pytest.raises(ValueError, match="no such store is installed"):
        open_store()


def test_without_a_store_named_results_are_files_under_the_working_directory(monkeypatch):
    monkeypatch.delenv(STORE_ENV, raising=False)
    store = open_store()
    assert isinstance(store, FileStore) and store.root == Path(".")


def test_a_training_run_on_another_store_writes_no_result_file_and_resumes_from_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    op = PlyBench(LLMConfig())
    op.registry.register_player(PlayerSpec("stub", _StubParams, lambda game, cfg, pid: _Stub(cfg, pid)))
    uniform = "random:distribution=uniform"
    harness = TrainingHarness("learnable", op, ["tic_tac_toe:"], ["stub:"], [uniform], [uniform], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2"], 1)
    memory = _MemoryStore()
    use_store(memory)
    try:
        asyncio.run(harness.run(sync=True))
        proposed = list(_Stub.proposed)
        run = harness.path_builder.run_dir(harness.experiment, harness.runs[0]).as_posix()
        kept = memory.keys(run)
        assert f"{run}/run.json" in kept
        assert f"{harness.path_builder.checkpoint(harness.experiment, harness.runs[0], EPOCHS).as_posix()}/state.json" in kept
        assert sum(key.endswith("/update.json") for key in kept) == EPOCHS * GAMES
        assert not (tmp_path / "results").exists()

        # a second run finds every game, update and checkpoint in the store: it proposes and writes nothing
        asyncio.run(harness.run(sync=True))
        assert _Stub.proposed == proposed and memory.keys(run) == kept
    finally:
        use_store(None)
