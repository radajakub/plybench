"""Console progress reporting: the round-range compressor and the per-matchup status lines
printed by `console_benchmark_callbacks` (offline, bots only)."""

from __future__ import annotations

import asyncio

import pytest
from plyllm import LLMConfig

from plybench.app import PlyBench
from plybench.callbacks.console_callbacks import console_benchmark_callbacks, console_training_callbacks
from plybench.configs.matchup import Matchup
from plybench.harness.matchup import run_matchup_concurrent
from plybench.utils.text import compress_ranges

op = PlyBench(LLMConfig())
registry = op.registry


@pytest.mark.parametrize(
    "numbers,expected",
    [
        ([], ""),
        ([4], "4"),
        ([1, 2, 3, 5, 9, 10], "1-3,5,9-10"),
        ([10, 1, 3, 2], "1-3,10"),
        ([2, 2, 3], "2-3"),
    ],
)
def test_compress_ranges(numbers, expected):
    assert compress_ranges(numbers) == expected


def test_console_status_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    game = registry.game_config("tic_tac_toe:")
    i = registry.player_config("random:distribution=uniform")
    o = registry.player_config("random:distribution=normal")

    asyncio.run(run_matchup_concurrent(op, Matchup(game, i, o, 3), matchup_callbacks=console_benchmark_callbacks(), experiment="exp", max_concurrent=1))
    out = capsys.readouterr().out

    assert "start tic_tac_toe: random_uniform vs random_normal (0/3)" in out
    assert "0/3 done · playing 1 · queued 2-3 · 0 moves" in out
    assert "3/3 done · " in out
    assert "moves played)" in out
    # the end line carries the outcomes of the matchup, as a training phase's does
    assert any(f"{outcome}=" in out for outcome in ("win", "draw", "loss"))

    # a completed matchup replays no rounds, so no status line is emitted on resume
    capsys.readouterr()
    asyncio.run(run_matchup_concurrent(op, Matchup(game, i, o, 3), matchup_callbacks=console_benchmark_callbacks(), experiment="exp", max_concurrent=1))
    resumed = capsys.readouterr().out
    assert "start tic_tac_toe: random_uniform vs random_normal (3/3)" in resumed
    assert "done" in resumed and "queued" not in resumed


def test_quiet_training_callbacks_report_phases_and_checkpoints_but_not_games():
    quiet, full = console_training_callbacks(progress=False), console_training_callbacks()
    for name in ("phase_start_callback", "round_start_callback", "round_complete_callback", "move_complete_callback"):
        assert getattr(quiet, name) is None and getattr(full, name) is not None
    for name in ("training_start_callback", "run_start_callback", "phase_end_callback", "checkpoint_callback", "run_end_callback", "training_end_callback"):
        assert getattr(quiet, name) is not None


def test_quiet_benchmark_callbacks_print_the_start_and_the_outcomes_only(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    game = registry.game_config("tic_tac_toe:")
    i = registry.player_config("random:distribution=uniform")
    o = registry.player_config("random:distribution=normal")
    asyncio.run(run_matchup_concurrent(op, Matchup(game, i, o, 3), matchup_callbacks=console_benchmark_callbacks(progress=False), experiment="exp", max_concurrent=1))
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == "[benchmark] start tic_tac_toe: random_uniform vs random_normal (0/3)"
    assert lines[1].startswith("[benchmark] done tic_tac_toe: random_uniform vs random_normal (") and "moves" not in lines[1]
    assert len(lines) == 2
