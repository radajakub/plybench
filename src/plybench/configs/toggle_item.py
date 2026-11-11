from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plybench.common.serializable import Serializable
from plybench.utils.text import to_bool


@dataclass(frozen=True, eq=True)
class ToggleItem(Serializable):
    value: str
    enabled: bool
    # a name for plots and tools in place of the one built from the config; not part of the config, so renaming it changes no path and no result
    label: str | None = None

    @staticmethod
    def extract_values(items: list[ToggleItem]) -> list[str]:
        return [item.value for item in items if item.enabled]

    @staticmethod
    def extract_labels(items: list[ToggleItem]) -> dict[str, str]:
        # value -> label of every enabled item with a label (a disabled one may name a config that no longer parses)
        return {item.value: item.label for item in items if item.enabled and item.label is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToggleItem:
        return cls(data["value"], to_bool(data["enabled"]), data.get("label"))

    def to_dict(self) -> dict[str, Any]:
        # the label only when set, so the files of unlabelled experiments stay as they are
        return {"value": self.value, **({"label": self.label} if self.label is not None else {}), "enabled": self.enabled}
