"""Console progress reporting: the round-range compressor and the per-matchup status lines
printed by `console_benchmark_callbacks` (offline, bots only)."""

from __future__ import annotations

import asyncio

import pytest

from plybench.app import PlyBench
from plybench.callbacks.console_callbacks import console_benchmark_callbacks
from plybench.configs.matchup import Matchup
from plybench.harness.matchup import run_matchup_concurrent
from plybench.llm import LLMConfig
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

    # a completed matchup replays no rounds, so no status line is emitted on resume
    capsys.readouterr()
    asyncio.run(run_matchup_concurrent(op, Matchup(game, i, o, 3), matchup_callbacks=console_benchmark_callbacks(), experiment="exp", max_concurrent=1))
    resumed = capsys.readouterr().out
    assert "start tic_tac_toe: random_uniform vs random_normal (3/3)" in resumed
    assert "done" in resumed and "queued" not in resumed
