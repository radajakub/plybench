"""A training run's folder names its game, players and replicate, not its schedule or seed: two runs that would share a folder, and a resume into a
folder that holds another run, are refused before any game is played."""

from __future__ import annotations

import asyncio
import json

import pytest
from plyllm import LLMConfig

from plybench.app import PlyBench
from plybench.harness.training.training import TrainingHarness

UNIFORM = "random:distribution=uniform"
SHORT, LONG = "num_epochs=1,num_training_games=2,num_test_games=2", "num_epochs=2,num_training_games=2,num_test_games=2"


def _harness(schedules: list[str], seed: int | None = 0, games: list[str] | None = None) -> TrainingHarness:
    return TrainingHarness("folders", PlyBench(LLMConfig()), games or ["tic_tac_toe:"], [UNIFORM], [UNIFORM], [UNIFORM], schedules, 1, seed=seed)


def _write_manifest(harness: TrainingHarness) -> None:
    for run in harness.runs:
        manifest = harness.path_builder.manifest(harness.experiment, run)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({**run.to_dict(), "seed": harness.seed}))


def test_two_schedules_of_one_run_are_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="would share"):
        _harness([SHORT, LONG]).check_runs()


def test_runs_with_their_own_folders_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _harness([SHORT], games=["tic_tac_toe:", "nim:"]).check_runs()


def test_a_resume_with_another_schedule_or_seed_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_manifest(_harness([SHORT]))
    _harness([SHORT]).check_runs()  # the same run resumes
    with pytest.raises(ValueError, match="training_config: 'num_epochs=1"):
        _harness([LONG]).check_runs()
    with pytest.raises(ValueError, match="seed: 0 there, 1 now"):
        _harness([SHORT], seed=1).check_runs()


def test_a_field_an_older_manifest_lacks_is_not_compared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = _harness([SHORT])
    run = harness.runs[0]
    manifest = harness.path_builder.manifest(harness.experiment, run)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"training_config": SHORT}))
    harness.check_runs()


def test_the_check_runs_before_any_game(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = _harness([SHORT, LONG])
    with pytest.raises(ValueError, match="would share"):
        asyncio.run(harness.run(sync=True))
    assert not list(tmp_path.rglob("game_*.json"))
