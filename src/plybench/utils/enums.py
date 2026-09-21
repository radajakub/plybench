from enum import Enum
from typing import Self


class ExtendedEnum(Enum):
    @classmethod
    def from_value(cls, value: str | int) -> Self | None:
        return next((e for e in cls if str(e.value).lower() == str(value).lower()), None)

    @classmethod
    def values(cls) -> list[str]:
        return [item.value for item in cls]

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return self.__str__()
