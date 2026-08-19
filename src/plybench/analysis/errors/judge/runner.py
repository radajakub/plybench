from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, ValidationError

from plybench.analysis.errors.judge.prompts import Prompt
from plybench.common.paths import ReasoningPathBuilder
from plybench.common.progress import track
from plybench.llm import LLMMessage, LLMResponse, LLMTokens, ModelConfig


class Generator(Protocol):
    """The slice of `LLM` the analysis passes use. Narrow on purpose: a pass can be exercised against a
    stub generator without the provider SDKs, which is what keeps these tests offline."""

    async def generate(
        self,
        model_config: ModelConfig,
        system: LLMMessage,
        messages: list[LLMMessage],
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse: ...


@dataclass(frozen=True)
class Judge:
    """One configured judge: the model plus the prompt revision it is being used under. `annotator` is
    what lands in the stored verdicts, so re-running under a changed prompt writes a new annotator rather
    than silently mixing two protocols in one column."""

    generator: Generator
    model: ModelConfig
    revision: str

    @property
    def annotator(self) -> str:
        return f"{self.model.provider.value}:{self.model.model_name}|{self.revision}"

    async def generate[T: BaseModel](self, prompt: Prompt, schema: type[T]) -> LLMResponse:
        return await self.generator.generate(self.model, LLMMessage.system(prompt.system), [LLMMessage.user(prompt.user)], schema)


class ResponseCache:
    """Raw judge responses keyed by (judge, schema, prompt). Unlike the verdict stores this *is* cache:
    the key pins everything that determines the answer, so a re-run of an unchanged pass costs nothing,
    while any change to the prompt or the model misses and re-asks."""

    def __init__(self, directory: Path | None = None, enabled: bool = True) -> None:
        self.directory = directory if directory is not None else ReasoningPathBuilder().responses()
        self.enabled = enabled

    def key(self, annotator: str, schema: str, prompt: Prompt) -> str:
        payload = "\x00".join((annotator, schema, prompt.system, prompt.user))
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def get(self, key: str) -> str | None:
        path = self.directory / f"{key}.json"
        if not self.enabled or not path.exists():
            return None
        return cast(str, json.loads(path.read_text())["output_text"])

    def put(self, key: str, annotator: str, output_text: str) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{key}.json").write_text(json.dumps({"annotator": annotator, "output_text": output_text}))


@dataclass(frozen=True)
class Completion[T: BaseModel]:
    """One judge call. `parsed is None` means this move produced no verdict — a provider error or an
    unparseable answer — which is reported rather than raised: one bad move must not sink a whole pass."""

    parsed: T | None
    error: str | None = None
    cached: bool = False
    tokens: LLMTokens = field(default_factory=LLMTokens)


MAX_ERRORS = 3


@dataclass(frozen=True)
class RunStats:
    n: int
    n_parsed: int
    n_cached: int
    n_failed: int
    tokens: LLMTokens
    errors: tuple[str, ...]  # distinct error strings, for a one-line report of what went wrong

    def __add__(self, other: RunStats) -> RunStats:
        errors = tuple(dict.fromkeys(self.errors + other.errors))
        return RunStats(
            n=self.n + other.n,
            n_parsed=self.n_parsed + other.n_parsed,
            n_cached=self.n_cached + other.n_cached,
            n_failed=self.n_failed + other.n_failed,
            tokens=self.tokens + other.tokens,
            errors=errors[:MAX_ERRORS],
        )


def run_stats[T: BaseModel](completions: Sequence[Completion[T]]) -> RunStats:
    tokens = LLMTokens()
    for completion in completions:
        tokens = tokens + completion.tokens
    errors = list(dict.fromkeys(completion.error for completion in completions if completion.error is not None))
    return RunStats(
        n=len(completions),
        n_parsed=sum(completion.parsed is not None for completion in completions),
        n_cached=sum(completion.cached for completion in completions),
        n_failed=sum(completion.parsed is None for completion in completions),
        tokens=tokens,
        errors=tuple(cast(list[str], errors)[:MAX_ERRORS]),
    )


async def _complete[T: BaseModel](judge: Judge, schema: type[T], prompt: Prompt, cache: ResponseCache) -> Completion[T]:
    key = cache.key(judge.annotator, schema.__name__, prompt)
    hit = cache.get(key)
    if hit is not None:
        try:
            return Completion(schema.model_validate_json(hit), cached=True)
        except ValidationError as error:
            return Completion(None, f"cached response no longer matches {schema.__name__}: {error}", cached=True)

    try:
        response = await judge.generate(prompt, schema)
        parsed = response.resolve_structured_output(schema)
    except Exception as error:  # noqa: BLE001 -- a provider failure on one move is data, not a crash
        return Completion(None, f"{type(error).__name__}: {error}")

    if parsed is None:
        return Completion(None, "judge returned no structured output", tokens=response.tokens)
    cache.put(key, judge.annotator, response.output_text)  # only a parsed answer is worth remembering
    return Completion(parsed, tokens=response.tokens)


async def run_prompts[T: BaseModel](
    judge: Judge,
    schema: type[T],
    prompts: Sequence[Prompt],
    desc: str,
    cache: ResponseCache | None = None,
    progress: bool | None = None,
) -> list[Completion[T]]:
    """Every prompt in flight at once; the provider's own semaphore and rate gates do the pacing, so the
    same quotas that shape a benchmark run shape this too. Results keep the input order."""
    cache = cache if cache is not None else ResponseCache()
    results: list[Completion[T] | None] = [None] * len(prompts)

    async def one(index: int, prompt: Prompt) -> None:
        results[index] = await _complete(judge, schema, prompt, cache)

    tasks = [asyncio.create_task(one(index, prompt)) for index, prompt in enumerate(prompts)]
    for task in track(asyncio.as_completed(tasks), desc, len(tasks), progress):
        await task
    return cast("list[Completion[T]]", results)
