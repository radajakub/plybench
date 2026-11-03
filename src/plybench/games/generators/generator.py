from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

GameInstanceT = TypeVar("GameInstanceT")


class InstanceGenerator(Generic[GameInstanceT], ABC):
    def __init__(self, sample: bool = False) -> None:
        self.should_sample = sample
        self.generator = np.random.default_rng()

    def seed(self, seed: int) -> None:
        self.generator = np.random.default_rng(seed)

    def new(self) -> GameInstanceT:
        return self.sample() if self.should_sample else self.get_normal()

    @abstractmethod
    def get_normal(self) -> GameInstanceT:
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> GameInstanceT:
        raise NotImplementedError
