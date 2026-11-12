from __future__ import annotations

from dataclasses import dataclass

from plybench.utils.text import extract_params


@dataclass(frozen=True, eq=True)
class TrainingConfig:
    num_epochs: int = 10
    num_training_games: int = 20
    num_test_games: int = 20
    # a learner's batch: after every update_every training games (and after the last of a phase) the harness hands it those games
    # (LearnablePlayer.observe_batch); None: no batches
    update_every: int | None = None

    def __post_init__(self) -> None:
        for name in ("num_epochs", "num_training_games", "num_test_games"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.update_every is not None and self.update_every <= 0:
            raise ValueError(f"update_every must be positive, got {self.update_every}")

    @classmethod
    def from_string(cls, params_string: str) -> TrainingConfig:
        params = extract_params(params_string)
        return cls(
            num_epochs=int(params.get("num_epochs", 10)),
            num_training_games=int(params.get("num_training_games", 20)),
            num_test_games=int(params.get("num_test_games", 20)),
            update_every=int(params["update_every"]) if "update_every" in params else None,
        )

    def to_string(self) -> str:
        # update_every only when set: the string is part of a run's identity, so a run without batches keeps its records
        batches = f",update_every={self.update_every}" if self.update_every is not None else ""
        return f"num_epochs={self.num_epochs},num_training_games={self.num_training_games},num_test_games={self.num_test_games}{batches}"

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.__str__()
