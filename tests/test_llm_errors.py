"""Failure classification, and the budget that stops two retry layers multiplying.

Six provider SDKs reach this package with no exception base in common. Callers used to either import
several SDKs or match on strings -- the judge runner matched exception class names, and
`TracedMove.output_failure` still matches substrings of a message. The clients already import these types
for `_should_retry_on_error`, so they classify, and the kind travels out on the exception.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from plybench.llm.concurrency import safe_call
from plybench.llm.errors import FailureKind, LLMCallError, LLMRateLimited, LLMTimedOut, as_call_error
from plybench.llm.providers.metacentrum.client import MetacentrumLLMClient

REQUEST = httpx.Request("POST", "https://example.invalid/v1/responses")


def _client() -> MetacentrumLLMClient:
    return MetacentrumLLMClient.__new__(MetacentrumLLMClient)  # only error_kind is under test


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (APITimeoutError(request=REQUEST), FailureKind.TIMEOUT),
        (RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None), FailureKind.RATE_LIMIT),
        (APIConnectionError(request=REQUEST), FailureKind.CONNECTION),
        (APIStatusError("boom", response=httpx.Response(500, request=REQUEST), body=None), FailureKind.PROVIDER),
        (ValueError("not from the sdk at all"), FailureKind.OTHER),
    ],
)
def test_the_client_names_its_own_sdks_failures(error: Exception, expected: FailureKind):
    # APITimeoutError subclasses APIConnectionError, so order inside error_kind is load-bearing
    assert _client().error_kind(error) is expected


def test_a_wrapped_error_is_catchable_without_importing_anyones_sdk():
    """The point of wrapping. Catching a rate limit generically used to mean naming four SDKs' types at
    the call site; a kind that travels on the exception works for every provider at once."""
    original = RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)

    wrapped = as_call_error(original, FailureKind.RATE_LIMIT)

    assert isinstance(wrapped, LLMRateLimited) and isinstance(wrapped, LLMCallError)
    assert wrapped.kind is FailureKind.RATE_LIMIT
    assert wrapped.__cause__ is original, "the SDK's own object stays reachable for anything that wants it"
    assert isinstance(as_call_error(APITimeoutError(request=REQUEST), FailureKind.TIMEOUT), LLMTimedOut)
    assert as_call_error(ValueError("x"), FailureKind.OTHER).kind is FailureKind.OTHER


def test_an_already_wrapped_error_is_not_wrapped_twice():
    wrapped = as_call_error(ValueError("x"), FailureKind.PROVIDER)
    assert as_call_error(wrapped, FailureKind.OTHER) is wrapped


def test_a_client_can_lower_its_own_retry_count():
    """Our retries wrap the SDK's, so the two multiply: ten attempts around a 600s timeout that itself
    retries is hours on one stalled call. A provider that stalls gets fewer attempts rather than a
    special mechanism."""
    attempts = 0

    async def always_fails():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("never comes back")

    with pytest.raises(TimeoutError):
        asyncio.run(safe_call(always_fails, retry_errors=(TimeoutError,), retries=3, delay_base=0.01, delay_max=0.01))

    assert attempts == 3
