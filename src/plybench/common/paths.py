from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.configs.training_run import EpochPhase, TrainingRun


def _build_dir(dir_name: str | Path) -> Path:
    directory = Path(dir_name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class BasePathBuilder(ABC):
    def __init__(self) -> None:
        self.base_experiments_dir = _build_dir("experiments")
        self.base_results_dir = _build_dir("results")
        self.base_plots_dir = _build_dir("plots")
        self.base_analysis_dir = _build_dir("analysis")
        self.base_cache_dir = _build_dir("cache")


class ExperimentPathBuilder(BasePathBuilder):
    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self.experiments_dir = _build_dir(self.base_experiments_dir / self.key)
        self.results_dir = _build_dir(self.base_results_dir / self.key)
        self.plots_dir = _build_dir(self.base_plots_dir / self.key)

    def experiment_path(self, experiment_name: str) -> Path:
        return self.experiments_dir / f"{experiment_name}.json"

    def game_base(self, experiment: str, game: GameConfig, i: PlayerConfig, o: PlayerConfig, num_games: int) -> Path:
        game_string = f"{game.path}_{num_games}"
        player_string = f"{i.path}_{o.path}"
        return self.results_dir / experiment / game_string / player_string

    def game_file(self, base: Path, game_round: int) -> Path:
        return base / f"game_{game_round}.json"

    def metadata(self, base: Path) -> Path:
        return base / "metadata.json"


class BenchmarkPathBuilder(ExperimentPathBuilder):
    def __init__(self) -> None:
        super().__init__("benchmarks")


class TrainingHarnessPathBuilder(ExperimentPathBuilder):
    def __init__(self) -> None:
        super().__init__("training")

    def run_dir(self, experiment: str, run: TrainingRun) -> Path:
        return self.results_dir / experiment / run.game.path / f"{run.trainee.path}_vs_{run.trainer.path}" / f"replicate_{run.replicate}"

    def manifest(self, experiment: str, run: TrainingRun) -> Path:
        return self.run_dir(experiment, run) / "run.json"

    def epoch_dir(self, experiment: str, run: TrainingRun, epoch: int) -> Path:
        return self.run_dir(experiment, run) / f"epoch_{epoch}"

    def learner_dir(self, experiment: str, run: TrainingRun, epoch: int) -> Path:
        return self.epoch_dir(experiment, run, epoch) / "learner"

    def checkpoint(self, experiment: str, run: TrainingRun, epoch: int) -> Path:
        return self.epoch_dir(experiment, run, epoch) / "checkpoint"

    def scope(self, experiment: str, phase: EpochPhase) -> str:
        # every phase of every epoch is an ordinary matchup; the run coordinates ride along in the
        # experiment segment so ResultTracker keeps its per-game resume without knowing about training
        return str((self.epoch_dir(experiment, phase.run, phase.epoch.index) / phase.dir_name).relative_to(self.results_dir))

    def game_base(self, experiment: str, game: GameConfig, i: PlayerConfig, o: PlayerConfig, num_games: int) -> Path:
        # `experiment` is a scope(), which already names the game, the players and the phase
        return self.results_dir / experiment


class ReasoningPathBuilder(BasePathBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.key = "reasoning"

    def experiment_dir(self, experiment: str) -> Path:
        return _build_dir(self.base_analysis_dir / experiment / self.key)

    def codebook(self, experiment: str, label: str = "") -> Path:
        return self.experiment_dir(experiment) / (f"codebook.{label}.json" if label else "codebook.json")

    def annotations(self, experiment: str) -> Path:
        return self.experiment_dir(experiment) / "annotations.jsonl"

    def consistency(self, experiment: str) -> Path:
        return self.experiment_dir(experiment) / "consistency.jsonl"

    def responses(self) -> Path:
        return _build_dir(self.base_cache_dir / self.key / "responses")


class MinimaxPathBuilder(BasePathBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.key = "minimax"
        self.cache_dir = _build_dir(self.base_cache_dir / "minimax")

    def cache(self, game_name: str, params: dict[str, Any] | None = None) -> Path:
        game_string = f"{game_name}"
        params = params if params is not None else {}
        for key, value in params.items():
            game_string += f"_{key}={str(value).replace(';', '_')}"
        return self.cache_dir / f"minimax_{game_string}.json"
