from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import cast

from plybench.app import PlyBench
from plybench.callbacks.game_callbacks import GameCallbacks
from plybench.callbacks.training_callbacks import TrainingCallbacks
from plybench.common.paths import TrainingHarnessPathBuilder
from plybench.common.progress import track
from plybench.configs.matchup import Matchup
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import CheckpointedParams
from plybench.configs.training_config import TrainingConfig
from plybench.configs.training_harness_config import TrainingHarnessConfig
from plybench.configs.training_run import Epoch, EpochPhase, TrainingRun
from plybench.core.game import TurnBasedGame
from plybench.harness.matchup import first_starts_by_parity, run_matchup_concurrent, run_matchup_sequential
from plybench.harness.training.results import EpochResults, TrainingResults
from plybench.player.learnable import LearnablePlayer
from plybench.player.player import Player, PlayerIdentifier
from plybench.trackers.result_tracker import ResultTracker


class TrainingHarness:
    @classmethod
    def load_experiment(
        cls,
        op: PlyBench,
        filename: str,
        game_override: list[str] | None = None,
        player_override: list[str] | None = None,
        trainer_override: list[str] | None = None,
        tester_override: list[str] | None = None,
        training_override: list[str] | None = None,
    ) -> TrainingHarness:
        path_builder = TrainingHarnessPathBuilder()
        path = path_builder.experiment_path(filename)

        with open(path) as f:
            config = TrainingHarnessConfig.from_dict(json.load(f))

        return cls(
            path.stem,
            op,
            game_override if game_override is not None else config.get_game_configs(),
            player_override if player_override is not None else config.get_player_configs(),
            trainer_override if trainer_override is not None else config.get_trainer_configs(),
            tester_override if tester_override is not None else config.get_tester_configs(),
            training_override if training_override is not None else config.get_training_configs(),
            config.num_replicates,
            path_builder,
        )

    def __init__(
        self,
        experiment: str,
        op: PlyBench,
        game_configs: list[str],
        player_configs: list[str],
        trainer_configs: list[str],
        tester_configs: list[str],
        training_configs: list[str],
        num_replicates: int,
        path_builder: TrainingHarnessPathBuilder | None = None,
    ) -> None:
        self.experiment = experiment
        self.op = op
        self.game_configs = game_configs
        self.player_configs = player_configs
        self.trainer_configs = trainer_configs
        self.tester_configs = tester_configs
        self.training_configs = training_configs
        self.num_replicates = num_replicates
        self.path_builder = path_builder if path_builder is not None else TrainingHarnessPathBuilder()

    def _testers(self) -> list[PlayerConfig]:
        return [self.op.registry.player_config(tester) for tester in self.tester_configs]

    def _matrix(self) -> list[TrainingRun]:
        # instead of a tensor of options, just build a flat list of jobs, it's the job's job to write the results correctly
        return [
            TrainingRun(
                self.op.registry.game_config(game),
                self.op.registry.player_config(player),
                self.op.registry.player_config(trainer),
                training_config,
                replicate,
            )
            for player in self.player_configs
            for trainer in self.trainer_configs
            for game in self.game_configs
            for training_config in map(TrainingConfig.from_string, self.training_configs)
            for replicate in range(self.num_replicates)
        ]

    async def _run_sync(self, concurrency: int | None, game_callbacks: GameCallbacks | None, training_callbacks: TrainingCallbacks) -> None:
        for run in self._matrix():
            await self.single_run(run, True, concurrency, game_callbacks, training_callbacks)

    async def _run_async(self, concurrency: int | None, runs_concurrency: int | None, game_callbacks: GameCallbacks | None, training_callbacks: TrainingCallbacks) -> None:
        # get all the jobs to execute
        runs = self._matrix()
        # throttle the number of concurrent runs that can run at a single time
        semaphore = asyncio.Semaphore(runs_concurrency if runs_concurrency is not None else len(runs))

        async def _guarded(run: TrainingRun) -> None:
            async with semaphore:
                await self.single_run(run, False, concurrency, game_callbacks, training_callbacks)

        await asyncio.gather(*(asyncio.create_task(_guarded(run)) for run in runs))

    async def run(
        self,
        sync: bool = False,
        concurrency: int | None = None,  # per-provider quota
        runs_concurrency: int | None = None,  # how many games can be started at a single point in time
        game_callbacks: GameCallbacks | None = None,
        training_callbacks: TrainingCallbacks | None = None,
    ) -> TrainingResults:
        training_callbacks = training_callbacks if training_callbacks is not None else TrainingCallbacks()
        training_callbacks.on_training_start(self.game_configs, self.player_configs, self.trainer_configs, self.tester_configs)

        if sync:
            await self._run_sync(concurrency, game_callbacks, training_callbacks)
        else:
            await self._run_async(concurrency, runs_concurrency, game_callbacks, training_callbacks)

        results = self.get_results()
        training_callbacks.on_training_end(results)
        return results

    async def single_run(self, run: TrainingRun, sync: bool, concurrency: int | None, game_callbacks: GameCallbacks | None, training_callbacks: TrainingCallbacks) -> str:
        # start the run with the callback
        training_callbacks.on_run_start(run)
        # write its manifest
        self._write_manifest(run)

        # epochs are sequential by construction: epoch e plays the checkpoint epoch e-1 produced, and
        # epoch 0 is the untrained baseline -- it is only evaluated, never trained
        for index in range(run.training_config.num_epochs + 1):
            await self._run_epoch(Epoch(run, index), sync, concurrency, game_callbacks, training_callbacks)

        training_callbacks.on_run_end(run)
        return str(self.path_builder.run_dir(self.experiment, run))

    async def _run_epoch(self, epoch: Epoch, sync: bool, concurrency: int | None, game_callbacks: GameCallbacks | None, training_callbacks: TrainingCallbacks) -> None:
        # start an epoch with a callback
        training_callbacks.on_epoch_start(epoch)

        # check if there is a checkpoint
        checkpoint = self.path_builder.checkpoint(self.experiment, epoch.run, epoch.index)
        # if there is no checkpoint, we have to learn it
        if not checkpoint.exists():
            await self._learn(epoch, checkpoint, game_callbacks, training_callbacks)
        # otherwise test the epoch checkpoint
        await self._test(epoch, sync, concurrency, game_callbacks, training_callbacks)

        # end the epoch with a callback
        training_callbacks.on_epoch_end(epoch)

    async def _learn(self, epoch: Epoch, checkpoint: Path, game_callbacks: GameCallbacks | None, training_callbacks: TrainingCallbacks) -> None:
        # case if there is no checkpoint
        run = epoch.run
        learner = self._build_learner(run, run.trainee if epoch.index == 0 else self._policy_config(run, epoch.index - 1))
        # first epoch does not train, it is evaluated on empty memory, i.e. no state
        if epoch.index > 0:
            learner.begin_epoch(self.path_builder.learner_dir(self.experiment, run, epoch.index))
            await self._train(epoch, learner, game_callbacks, training_callbacks)

        # the directory's existence is what marks the epoch done, so stage the write and rename it into
        # place: a checkpoint appears only once save_checkpoint returned, and an empty one means the
        # player stores nothing rather than that the epoch died halfway through writing
        staged = checkpoint.with_name(f"{checkpoint.name}.tmp")
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir(parents=True, exist_ok=True)
        learner.save_checkpoint(staged)
        os.replace(staged, checkpoint)

        training_callbacks.on_checkpoint(epoch, checkpoint)

    async def _train(self, epoch: Epoch, learner: LearnablePlayer, game_callbacks: GameCallbacks | None, training_callbacks: TrainingCallbacks) -> ResultTracker:
        phase = epoch.train
        matchup = Matchup(phase.run.game, self._policy_config(phase.run, phase.player_epoch), phase.opponent, phase.num_games)

        def build_player(game: TurnBasedGame, player_config: PlayerConfig, identifier: PlayerIdentifier) -> Player:
            # the learner is the one thing an epoch carries across its games; the trainer is rebuilt per round
            return learner if identifier == "i" else self.op.registry.build_player(game, player_config, identifier)

        learner.train()
        try:
            tracker = await run_matchup_sequential(
                self.op,
                matchup,
                game_callbacks=game_callbacks,
                matchup_callbacks=training_callbacks.for_phase(phase),
                path_builder=self.path_builder,
                experiment=self.path_builder.scope(self.experiment, phase),
                player_builder=build_player,
                after_game=learner.observe,
                colour_rule=first_starts_by_parity,
            )
            await learner.update([game for game in tracker.games if game is not None])
        finally:
            # whatever happens in the phase, the checkpoint that leaves it is frozen
            learner.eval()
        return tracker

    async def _test(self, epoch: Epoch, sync: bool, concurrency: int | None, game_callbacks: GameCallbacks | None, training_callbacks: TrainingCallbacks) -> list[ResultTracker]:
        # the checkpoint is frozen here: every tester plays the same policy and none of them touch it
        phases = [epoch.test(tester) for tester in self._testers()]

        if sync:
            return [await self._play_phase(phase, concurrency, game_callbacks, training_callbacks) for phase in phases]
        return list(await asyncio.gather(*(asyncio.create_task(self._play_phase(phase, concurrency, game_callbacks, training_callbacks)) for phase in phases)))

    async def _play_phase(self, phase: EpochPhase, max_concurrent: int | None, game_callbacks: GameCallbacks | None, training_callbacks: TrainingCallbacks) -> ResultTracker:
        matchup = Matchup(phase.run.game, self._policy_config(phase.run, phase.player_epoch), phase.opponent, phase.num_games)
        return await run_matchup_concurrent(
            self.op,
            matchup,
            game_callbacks=game_callbacks,
            matchup_callbacks=training_callbacks.for_phase(phase),
            path_builder=self.path_builder,
            experiment=self.path_builder.scope(self.experiment, phase),
            max_concurrent=max_concurrent,
            colour_rule=first_starts_by_parity,
        )

    def _policy_config(self, run: TrainingRun, epoch: int) -> PlayerConfig:
        # a LearnablePlayer requires CheckpointedParams of itself, so a trainee that builds at all has them
        params = cast(CheckpointedParams, run.trainee.params)
        return PlayerConfig(run.trainee.key, params.with_checkpoint(self.path_builder.checkpoint(self.experiment, run, epoch)))

    def _build_learner(self, run: TrainingRun, config: PlayerConfig) -> LearnablePlayer:
        # building an engine is the registry's only route to a TurnBasedGame; the rounds build their own
        engine = self.op.registry.build_engine(run.game)
        player = self.op.registry.build_player(engine.game, config, "i")
        if not isinstance(player, LearnablePlayer):
            raise TypeError(f"trainee {config.to_string()} is not a LearnablePlayer and cannot be trained")
        # nothing calls initialize_policy on a replayed round, so the checkpoint is restored here
        player.restore()
        return player

    def _write_manifest(self, run: TrainingRun) -> None:
        # the paths carry identity only, so the counts behind a run live here
        manifest = self.path_builder.manifest(self.experiment, run)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w") as f:
            json.dump({**run.to_dict(), "testers": self.tester_configs}, f, indent=2)

    def _load_phase(self, phase: EpochPhase) -> ResultTracker:
        tracker = ResultTracker.new(
            self.path_builder.scope(self.experiment, phase),
            self._policy_config(phase.run, phase.player_epoch),
            phase.opponent,
            phase.run.game,
            phase.num_games,
            self.op.registry,
            path_builder=self.path_builder,
        )
        tracker.load_if_exists()
        return tracker

    def _load_epoch(self, epoch: Epoch) -> EpochResults:
        train = self._load_phase(epoch.train) if epoch.index > 0 else None
        return EpochResults(epoch, train, [self._load_phase(epoch.test(tester)) for tester in self._testers()])

    def get_results(self, progress: bool | None = None) -> TrainingResults:
        runs = self._matrix()
        epochs: list[EpochResults] = []
        for run in track(runs, "Loading results", len(runs), progress):
            epochs.extend(self._load_epoch(Epoch(run, index)) for index in range(run.training_config.num_epochs + 1))
        return TrainingResults(runs, epochs)
