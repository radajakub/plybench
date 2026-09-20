from __future__ import annotations

from dataclasses import dataclass

from plybench.utils.text import extract_params


@dataclass(frozen=True, eq=True)
class TrainingConfig:
    num_epochs: int = 10
    num_training_games: int = 20
    num_test_games: int = 20

    def __post_init__(self) -> None:
        for name in ("num_epochs", "num_training_games", "num_test_games"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @classmethod
    def from_string(cls, params_string: str) -> TrainingConfig:
        params = extract_params(params_string)
        return cls(
            num_epochs=int(params.get("num_epochs", 10)),
            num_training_games=int(params.get("num_training_games", 20)),
            num_test_games=int(params.get("num_test_games", 20)),
        )

    def to_string(self) -> str:
        return f"num_epochs={self.num_epochs},num_training_games={self.num_training_games},num_test_games={self.num_test_games}"

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.__str__()
