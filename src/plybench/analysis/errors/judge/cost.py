from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from plybench.analysis.errors.judge.runner import RunStats
from plybench.llm import LLMModel, LLMTokens, ModelConfig, Provider


class Pricer(Protocol):
    """The slice of `LLM` the ledger needs: what a call cost, and whether the model carries a price at
    all. Locally hosted models are registered with zero cost, which is not the same as a free API call —
    the ledger reports those as unpriced rather than as $0.00."""

    def calculate_cost(self, model_config: ModelConfig, tokens: LLMTokens) -> float: ...

    def resolve_model(self, provider: Provider, model_name: str) -> LLMModel: ...


@dataclass(frozen=True)
class StepCost:
    step: str
    model: str
    n_calls: int = 0
    n_cached: int = 0
    n_failed: int = 0
    n_units: int = 0  # moves the step covered; batched steps cover more moves than they make calls
    tokens: LLMTokens = field(default_factory=LLMTokens)
    cost: float | None = None  # None means the model carries no pricing, not that it was free

    def __add__(self, other: StepCost) -> StepCost:
        cost = None if self.cost is None and other.cost is None else (self.cost or 0.0) + (other.cost or 0.0)
        return StepCost(
            step=self.step if self.step == other.step else "total",
            model=self.model if self.model == other.model else "",
            n_calls=self.n_calls + other.n_calls,
            n_cached=self.n_cached + other.n_cached,
            n_failed=self.n_failed + other.n_failed,
            n_units=self.n_units + other.n_units,
            tokens=self.tokens + other.tokens,
            cost=cost,
        )

    @property
    def cost_per_unit(self) -> float | None:
        return None if self.cost is None or not self.n_units else self.cost / self.n_units

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model,
            "n_calls": self.n_calls,
            "n_cached": self.n_cached,
            "n_failed": self.n_failed,
            "n_units": self.n_units,
            "input_tokens": self.tokens.input_tokens,
            "cached_input_tokens": self.tokens.cached_input_tokens,
            "output_tokens": self.tokens.output_tokens,
            "reasoning_tokens": self.tokens.reasoning_tokens,
            "cost_usd": self.cost,
            "cost_usd_per_move": self.cost_per_unit,
        }


class CostLedger:
    """What a run actually spent. Cache hits carry no tokens, so a resumed run reports the calls it made
    this time rather than the calls the analysis has cost in total — the figure is real spend, not a
    notional price for the artefacts on disk."""

    def __init__(self, pricer: Pricer) -> None:
        self._pricer = pricer
        self._rows: dict[tuple[str, str], StepCost] = {}

    def _priced(self, model: ModelConfig) -> bool:
        try:
            resolved = self._pricer.resolve_model(model.provider, model.model_name)
        except ValueError:
            return False
        return bool(resolved.input_cost or resolved.output_cost or resolved.cached_input_cost)

    def record(self, step: str, model: ModelConfig, stats: RunStats | None, n_units: int | None = None) -> None:
        """`n_units` is how many moves the step covered, which is the call count for the per-move passes
        and the batch's worth of moves for induction — so cost-per-move stays comparable across steps."""
        if stats is None or not stats.n:  # a fully resumed pass did nothing; a zero row would only be noise
            return
        name = f"{model.provider.value}:{model.model_name}"
        row = StepCost(
            step=step,
            model=name,
            n_calls=stats.n,
            n_cached=stats.n_cached,
            n_failed=stats.n_failed,
            n_units=stats.n if n_units is None else n_units,
            tokens=stats.tokens,
            cost=self._pricer.calculate_cost(model, stats.tokens) if self._priced(model) else None,
        )
        key = (step, name)
        self._rows[key] = self._rows[key] + row if key in self._rows else row

    def rows(self) -> list[StepCost]:
        return list(self._rows.values())

    def total(self) -> StepCost:
        rows = self.rows()
        if not rows:
            return StepCost("total", "")
        total = rows[0]
        for row in rows[1:]:
            total = total + row
        return StepCost(
            "total",
            total.model,
            total.n_calls,
            total.n_cached,
            total.n_failed,
            total.n_units,
            total.tokens,
            total.cost,
        )

    def __bool__(self) -> bool:
        return bool(self._rows)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [row.to_dict() for row in self.rows()], "total": self.total().to_dict()}
