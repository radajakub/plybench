"""The LearnablePlayer contract, through the real training harness: a learner is built frozen, learns only
in a training phase, and a resumed epoch commits the updates it already proposed instead of proposing them
again -- so a learner whose proposals are sampled still lands on the same state."""

from __future__ import annotations

import asyncio
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
from plyllm import LLM, LLMConfig, OpenAIProviderConfig

from plybench.analysis.usage import calls_cost, calls_usage, learning_calls
from plybench.app import PlyBench
from plybench.callbacks.training_callbacks import TrainingCallbacks
from plybench.common.enums import Mode
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import CheckpointedParams
from plybench.configs.training_run import Epoch
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_STATE, LearnablePlayer
from plybench.player.player import PlayerOutput
from plybench.player.spec import PlayerSpec
from plybench.trace.load import load_game
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.llm_call import LLMCall

EPOCHS, GAMES = 2, 4


class _Interrupted(Exception):
    pass


@dataclass(frozen=True, eq=True)
class _StubParams(CheckpointedParams):
    path: Path | None = None

    @classmethod
    def from_string(cls, params_string: str) -> _StubParams:
        return cls(Path(params_string) if params_string else None)

    def to_string(self) -> str:
        return str(self.path) if self.path else ""

    @property
    def path_suffix(self) -> str:
        return ""

    @property
    def checkpoint(self) -> Path | None:
        return self.path

    def with_checkpoint(self, checkpoint: Path) -> _StubParams:
        return _StubParams(checkpoint)


class _Stub(LearnablePlayer[float]):
    """Plays the first legal move; an update is a random number, the state their sum and count.

    The harness builds one instance per epoch, so what the test inspects is kept on the class."""

    proposed: ClassVar[list[tuple[str, int]]] = []
    # the round whose proposal raises once, as an interrupted run would
    fail_at: ClassVar[int | None] = None
    # the rounds whose proposal is None
    nothing_at: ClassVar[set[int]] = set()

    def __init__(self, player_config: PlayerConfig, identifier) -> None:
        super().__init__(player_config, identifier)
        self.total = 0.0
        self.commits = 0

    def _initialize_policy(self, game, prompt_adapter_template) -> None:
        pass

    async def __call__(self, game, observation, legal_moves) -> PlayerOutput:
        return PlayerOutput(action=legal_moves[0])

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""

    async def propose(self, game: GameTracker) -> float | None:
        if game.game_round == _Stub.fail_at:
            _Stub.fail_at = None
            raise _Interrupted
        assert self.workdir is not None
        _Stub.proposed.append((self.workdir.parent.name, game.game_round))
        # one LLM call per proposal, as a learner that judges the game would make, a game that teaches nothing included
        self.record_call(LLMCall("judge", "openai", "gpt-5.4", "gpt-5.4-2026", "system", f"game {game.game_round}", "", "lesson", 100, 20, 5))
        return None if game.game_round in _Stub.nothing_at else random.random()

    def commit(self, update: float) -> None:
        self.total += update
        self.commits += 1

    def save_update(self, update: float, directory: Path) -> None:
        (directory / "update.json").write_text(json.dumps(update))

    def load_update(self, directory: Path) -> float:
        return json.loads((directory / "update.json").read_text())

    def save_checkpoint(self, checkpoint: Path) -> None:
        (checkpoint / "state.json").write_text(json.dumps({"total": self.total, "commits": self.commits}))

    def load_checkpoint(self, checkpoint: Path) -> None:
        state = json.loads((checkpoint / "state.json").read_text())
        self.total, self.commits = state["total"], state["commits"]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the harness writes results/ under the working directory
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    op = PlyBench(LLMConfig())
    op.registry.register_player(PlayerSpec("stub", _StubParams, lambda game, cfg, pid: _Stub(cfg, pid)))
    uniform = "random:distribution=uniform"
    return TrainingHarness("learnable", op, ["tic_tac_toe:"], ["stub:"], [uniform], [uniform], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2"], 1)


def _learner_dir(harness: TrainingHarness, epoch: int) -> Path:
    return harness.path_builder.learner_dir(harness.experiment, harness.runs[0], epoch)


def _state(harness: TrainingHarness, epoch: int) -> dict:
    return json.loads((harness.path_builder.checkpoint(harness.experiment, harness.runs[0], epoch) / "state.json").read_text())


def _cached(harness: TrainingHarness) -> list[float]:
    # every update the run saved, over all epochs; a None proposal saves none
    return [json.loads(path.read_text()) for epoch in range(1, EPOCHS + 1) for path in sorted(_learner_dir(harness, epoch).glob("game_*/update.json"))]


def _run(harness: TrainingHarness) -> None:
    asyncio.run(harness.run(sync=True))


def test_a_learner_is_built_frozen():
    learner = _Stub(PlayerConfig("stub", _StubParams()), "i")

    assert learner.mode is Mode.EVAL and learner.is_eval and not learner.is_training
    with pytest.raises(RuntimeError, match="frozen"):
        asyncio.run(learner.observe(None))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="frozen"):
        asyncio.run(learner.update([]))


def test_a_learner_needs_a_workdir_to_learn():
    learner = _Stub(PlayerConfig("stub", _StubParams()), "i")
    learner.set_mode(Mode.TRAIN)

    with pytest.raises(RuntimeError, match="begin_epoch"):
        asyncio.run(learner.update([]))


def test_a_training_run_caches_one_update_per_game_and_commits_them_all(harness):
    _run(harness)

    for epoch in range(1, EPOCHS + 1):
        assert sorted(path.name for path in _learner_dir(harness, epoch).iterdir()) == ["epoch", *(f"game_{r}" for r in range(1, GAMES + 1))]
        # the default propose_epoch has nothing to say, and that is cached too
        assert (_learner_dir(harness, epoch) / "epoch" / LearnablePlayer._NO_UPDATE).exists()
    # epoch e starts from the checkpoint of epoch e-1, so the last one holds every update of the run
    state = _state(harness, EPOCHS)
    assert state["commits"] == EPOCHS * GAMES
    assert state["total"] == pytest.approx(sum(_cached(harness)))
    assert _Stub.proposed == [(f"epoch_{e}", r) for e in range(1, EPOCHS + 1) for r in range(1, GAMES + 1)]


def test_a_resumed_epoch_commits_the_cached_updates_and_proposes_only_the_rest(harness):
    _Stub.fail_at = 3
    with pytest.raises(_Interrupted):
        _run(harness)
    assert _Stub.proposed == [("epoch_1", 1), ("epoch_1", 2)]
    before = _cached(harness)
    assert len(before) == 2

    _Stub.proposed = []
    _run(harness)

    # rounds 1 and 2 were replayed from their records and their updates loaded, not proposed again
    assert _Stub.proposed == [("epoch_1", 3), ("epoch_1", 4), *(("epoch_2", r) for r in range(1, GAMES + 1))]
    after = _cached(harness)
    assert after[:2] == before
    state = _state(harness, EPOCHS)
    assert state["commits"] == EPOCHS * GAMES
    assert state["total"] == pytest.approx(sum(after))


def test_a_game_that_teaches_nothing_is_cached_and_not_committed(harness):
    _Stub.nothing_at, _Stub.fail_at = {2}, 3
    with pytest.raises(_Interrupted):
        _run(harness)
    assert (_learner_dir(harness, 1) / "game_2" / LearnablePlayer._NO_UPDATE).exists()

    _Stub.proposed = []
    _run(harness)

    # the None of round 2 is loaded on the replay, not proposed again
    assert ("epoch_1", 2) not in _Stub.proposed
    state = _state(harness, EPOCHS)
    assert state["commits"] == EPOCHS * (GAMES - 1)
    assert state["total"] == pytest.approx(sum(_cached(harness)))


def test_an_update_that_does_not_load_back_unchanged_is_refused_before_it_is_cached(harness, monkeypatch):
    # stores the update rounded, so a resumed epoch would commit something else than this one did
    monkeypatch.setattr(_Stub, "save_update", lambda self, update, directory: (directory / "update.json").write_text(json.dumps(round(update, 2))))

    with pytest.raises(ValueError, match="load_update does not give back"):
        _run(harness)
    assert not (_learner_dir(harness, 1) / "game_1").exists()


def test_the_calls_of_each_proposal_are_stored_with_it_and_kept_across_a_resume(harness):
    _Stub.nothing_at, _Stub.fail_at = {2}, 3
    with pytest.raises(_Interrupted):
        _run(harness)
    _run(harness)

    # one call per training game, the one that taught nothing included, and none twice after the resume
    for epoch in range(1, EPOCHS + 1):
        calls = learning_calls(_learner_dir(harness, epoch))
        assert sorted(call.prompt_message for call in calls) == sorted(f"game {r}" for r in range(1, GAMES + 1))
        assert calls_usage(calls)["judge"].calls == GAMES
        assert calls_usage(calls)["judge"].tokens.input_tokens == 100 * GAMES


def test_learning_calls_are_priced_where_the_provider_is_configured(harness):
    _run(harness)
    calls = learning_calls(_learner_dir(harness, 1))

    cost, unpriced = calls_cost(LLM(LLMConfig(openai=OpenAIProviderConfig(api_key="offline"))), calls)
    assert cost > 0 and unpriced == 0
    assert calls_cost(LLM(LLMConfig()), calls) == (0.0, GAMES)


def test_a_call_cannot_be_recorded_outside_a_proposal():
    learner = _Stub(PlayerConfig("stub", _StubParams()), "i")

    with pytest.raises(RuntimeError, match="only legal inside propose"):
        learner.record_call(LLMCall("judge", "openai", "gpt-5.4", "gpt-5.4", "", "", "", "", 0, 0, 0))


def _kept(harness: TrainingHarness, epoch: int, name: str) -> dict:
    return json.loads((_learner_dir(harness, epoch) / name / LEARNER_STATE / "state.json").read_text())


def _endings(harness: TrainingHarness, epoch: int) -> list[dict]:
    # what the learner kept at the end of each training game of an epoch, in round order
    run = harness.runs[0]
    folder = harness.path_builder.epoch_dir(harness.experiment, run, epoch)
    games = sorted(folder.rglob("train*/**/game_*.json"), key=lambda path: int(path.stem.split("_")[1]))
    return [next(iter(json.loads(path.read_text())["ending"]["players"].values())) for path in games]


def test_the_state_after_each_game_is_kept_and_the_last_one_is_the_checkpoint(harness):
    _run(harness)

    for epoch in range(1, EPOCHS + 1):
        commits = [_kept(harness, epoch, f"game_{r}")["commits"] for r in range(1, GAMES + 1)]
        assert commits == [(epoch - 1) * GAMES + r for r in range(1, GAMES + 1)]
        assert _kept(harness, epoch, f"game_{GAMES}") == _kept(harness, epoch, "epoch") == _state(harness, epoch)


def test_each_training_game_links_the_state_it_started_from(harness):
    _run(harness)

    endings = _endings(harness, 2)
    checkpoint = harness.path_builder.checkpoint(harness.experiment, harness.runs[0], 1).as_posix()
    assert [ending["state"]["checkpoint"] for ending in endings] == [checkpoint] * GAMES
    befores = [ending["state"]["before"] for ending in endings]
    assert befores[0] == checkpoint
    assert befores[1:] == [(_learner_dir(harness, 2) / f"game_{r}" / LEARNER_STATE).as_posix() for r in range(1, GAMES)]


def test_a_replay_that_lands_on_another_state_is_refused(harness):
    _Stub.fail_at = 3
    with pytest.raises(_Interrupted):
        _run(harness)
    kept = _learner_dir(harness, 1) / "game_1" / LEARNER_STATE / "state.json"
    kept.write_text(json.dumps({"total": -1.0, "commits": 1}))

    with pytest.raises(ValueError, match=r"state replayed after game_1 differs .*state\.json"):
        _run(harness)


def test_a_state_lost_before_it_was_written_is_written_on_the_replay(harness):
    _Stub.fail_at = 3
    with pytest.raises(_Interrupted):
        _run(harness)
    shutil.rmtree(_learner_dir(harness, 1) / "game_2" / LEARNER_STATE)

    _run(harness)

    assert _kept(harness, 1, "game_2")["commits"] == 2


def test_a_finished_training_run_is_not_checked_against_its_sessions(harness):
    _run(harness)
    manifest = harness.path_builder.manifest(harness.experiment, harness.runs[0])
    stored = json.loads(manifest.read_text())
    stored["sessions"][-1].pop("players")
    manifest.write_text(json.dumps(stored))

    harness.check_runs()


def test_a_training_game_reads_back_with_the_learner_folder_it_was_learned_into(harness):
    _run(harness)
    folder = harness.path_builder.epoch_dir(harness.experiment, harness.runs[0], 1)
    game_file = next(folder.rglob("train*/**/game_2.json"))

    trace = load_game(game_file, harness.op.registry)

    # the seats alternate in training, so the learner is found by its key
    [learner] = [player for player in trace.players.values() if player.key == "stub"]
    assert trace.place is not None and trace.place["phase"] == "train"
    assert learner.ending is not None and learner.ending["learned"] == (_learner_dir(harness, 1) / "game_2").as_posix()
    assert learner.description is not None and learner.description["key"] == "stub"
    assert len(learner.events) == sum(1 for event in learner.events if event.phase == "move")


def test_a_game_that_changes_at_an_epoch_is_played_from_that_epoch_on_and_resumed_there(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    op = PlyBench(LLMConfig())
    op.registry.register_player(PlayerSpec("stub", _StubParams, lambda game, cfg, pid: _Stub(cfg, pid)))
    # the stub presses red, which never wins before the switch and always wins after it
    game = "bandit:p=0;1,switch=1;0,switch_epoch=2,rules=hidden,stationarity=omitted"
    harness = TrainingHarness("switch", op, [game], ["stub:"], ["none:"], ["none:"], [f"num_epochs=3,num_training_games={GAMES},num_test_games=2"], 1, seed=0)

    def interrupt_after_the_switch(epoch: Epoch) -> None:
        if epoch.index == 3:
            _Stub.fail_at = 2

    with pytest.raises(_Interrupted):
        asyncio.run(harness.run(sync=True, training_callbacks=TrainingCallbacks(epoch_start_callback=interrupt_after_the_switch)))
    _run(harness)

    [run] = harness.runs
    after = "bandit:p=1;0,shift=0,rules=hidden,stationarity=omitted"
    for epoch in harness.get_results().epochs:
        phases = epoch.tests + ([epoch.train] if epoch.train is not None else [])
        assert {tracker.game.to_string() for tracker in phases} == {run.game.to_string() if epoch.epoch.index < 2 else after}
        assert all(tracker.summary is not None and tracker.summary["win_rate"]["value"] == (0.0 if epoch.epoch.index < 2 else 1.0) for tracker in phases)
    # one run, named by its own game, whose learner kept learning across the switch and was resumed after it
    assert len(list((tmp_path / "results").rglob("run.json"))) == 1 and run.game.to_string() == "bandit:p=0;1,shift=0,rules=hidden,switch=1;0,switch_epoch=2,stationarity=omitted"
    assert _state(harness, 3)["commits"] == 3 * GAMES
