"""Provenance: every run that plays games records what produced them (package versions, commits of checkouts, the
command) as a session, with the rounds it played, in the matchup's metadata and the training run's manifest."""

from __future__ import annotations

import asyncio
import json

from plyllm import LLMConfig

from plybench.app import PlyBench
from plybench.common.provenance import package_of, provenance
from plybench.configs.matchup import Matchup
from plybench.harness.matchup import run_matchup_concurrent
from plybench.harness.training.training import TrainingHarness
from plybench.player.simple.random_player import RandomParams

op = PlyBench(LLMConfig())
registry = op.registry
MATCHUP = Matchup(registry.game_config("tic_tac_toe:"), registry.player_config("random:distribution=uniform"), registry.player_config("random:distribution=normal"), 3)


def _metadata(tracker) -> dict:
    return json.loads(tracker.metadata_path.read_text())


def test_provenance_names_the_packages_and_the_command():
    record = provenance(["plybench"])

    assert set(record["packages"]) == {"plybench", "plyllm"}
    assert record["packages"]["plyllm"]["version"] is not None
    assert {"started", "working_directory", "command", "python"} <= set(record)
    assert package_of(RandomParams()) == "plybench"


def test_a_matchup_records_one_session_per_run_that_played_games(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = asyncio.run(run_matchup_concurrent(op, MATCHUP, experiment="exp", max_concurrent=1))

    sessions = _metadata(tracker)["sessions"]
    assert [sorted(session["rounds"]) for session in sessions] == [[1, 2, 3]]
    # a temporary directory is not a checkout
    assert sessions[0]["working_directory"] is None

    # a run that finds the matchup complete plays nothing and adds no session
    asyncio.run(run_matchup_concurrent(op, MATCHUP, experiment="exp", max_concurrent=1))
    assert len(_metadata(tracker)["sessions"]) == 1

    # a run that resumes it files only the round it played under its own session
    tracker.path_builder.game_file(tracker.base_path, 2).unlink()
    metadata = _metadata(tracker)
    tracker.metadata_path.write_text(json.dumps({**metadata, "completed": [1, 3]}))
    asyncio.run(run_matchup_concurrent(op, MATCHUP, experiment="exp", max_concurrent=1))
    assert [session["rounds"] for session in _metadata(tracker)["sessions"]] == [sessions[0]["rounds"], [2]]


def test_a_training_run_adds_a_session_to_its_manifest_each_time_it_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uniform = "random:distribution=uniform"
    harness = TrainingHarness("prov", op, ["tic_tac_toe:"], [uniform], [uniform], [uniform], ["num_epochs=1,num_training_games=2,num_test_games=2"], 1)
    run = harness.runs[0]

    harness._write_manifest(run)
    harness._write_manifest(run)

    manifest = json.loads(harness.path_builder.manifest(harness.experiment, run).read_text())
    assert len(manifest["sessions"]) == 2
    assert set(manifest["sessions"][0]["packages"]) == {"plybench", "plyllm"}
