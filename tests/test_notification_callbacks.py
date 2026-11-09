"""Notifications: an epoch or a matchup is reported when it played a round in this run; one whose rounds were all on disk (a resumed experiment)
ends silently, while still counting towards the progress."""

from __future__ import annotations

import pytest
from plyllm import LLMConfig

from plybench.app import PlyBench
from plybench.callbacks import notification_callbacks
from plybench.callbacks.notification_callbacks import notification_benchmark_callbacks, notification_training_callbacks
from plybench.common.paths import BenchmarkPathBuilder
from plybench.configs.training_config import TrainingConfig
from plybench.configs.training_run import Epoch, TrainingRun
from plybench.trackers.result_tracker import ResultTracker

op = PlyBench(LLMConfig())
PATHS = BenchmarkPathBuilder()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(notification_callbacks.clankers, "blastthem", lambda message, duration=None: messages.append(message))
    return messages


def _tracker(n: int, completed: set[int], player: str = "random:") -> ResultTracker:
    game, opponent = op.registry.game_config("tic_tac_toe:"), op.registry.player_config("random:")
    player_config = op.registry.player_config(player)
    return ResultTracker("exp", player_config, opponent, game, n, completed, op.registry, PATHS, save_on_record=False)


def test_an_epoch_whose_rounds_were_all_on_disk_is_not_reported(sent: list[str]) -> None:
    callbacks, describe = notification_training_callbacks("exp")
    run = TrainingRun(op.registry.game_config("tic_tac_toe:"), op.registry.player_config("random:"), op.registry.player_config("random:"), TrainingConfig(2, 2, 2), 0)
    assert callbacks.training_start_callback is not None and callbacks.run_start_callback is not None
    assert callbacks.phase_start_callback is not None and callbacks.round_complete_callback is not None and callbacks.epoch_end_callback is not None
    callbacks.training_start_callback(["tic_tac_toe:"], ["random:"], ["random:"], ["random:"])
    callbacks.run_start_callback(run)
    # epoch 1 was saved before (both rounds on disk), epoch 2 plays one round of two
    for epoch, completed in ((Epoch(run, 1), {1, 2}), (Epoch(run, 2), {1})):
        phase = epoch.train
        callbacks.phase_start_callback(_tracker(2, completed), phase)
        for game_round in (1, 2):
            callbacks.round_complete_callback(phase, game_round)
        callbacks.epoch_end_callback(epoch)
    assert len(sent) == 1 and "epoch 2/2" in sent[0]
    # the run plans 2 x 2 training and 3 x 1 x 2 test rounds; 3 were on disk
    assert describe() == "2 epoch(s), 1/7 rounds"


def test_a_matchup_whose_rounds_were_all_on_disk_is_not_reported(sent: list[str]) -> None:
    callbacks, describe = notification_benchmark_callbacks("exp")
    assert callbacks.benchmark_start_callback is not None and callbacks.matchup_start_callback is not None
    assert callbacks.round_complete_callback is not None and callbacks.matchup_end_callback is not None
    callbacks.benchmark_start_callback(["tic_tac_toe:"], ["random:", "mcts:"], ["random:"])
    # random's matchup was saved before, mcts's plays one round of two
    for player, completed in (("random:", {1, 2}), ("mcts:", {1})):
        tracker = _tracker(2, completed, player)
        callbacks.matchup_start_callback(tracker, tracker.game, tracker.i, tracker.o)
        for game_round in (1, 2):
            callbacks.round_complete_callback(tracker.game, tracker.i, tracker.o, game_round)
        callbacks.matchup_end_callback(tracker)
    assert len(sent) == 1 and "mcts" in sent[0]
    assert describe() == "2/2 matchups, 1/1 rounds"
