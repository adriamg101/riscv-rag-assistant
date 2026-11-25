"""Generation through any OpenAI-compatible endpoint, with rate-limit state per provider."""

import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

from openai import OpenAI, OpenAIError, RateLimitError

from app.config import ProviderConfig
from app.prompts import SYSTEM_PROMPT

TIMEOUT_SECONDS = 120.0  # default; slow local models can raise it with LLM_TIMEOUT_SECONDS
DEFAULT_COOLDOWN_SECONDS = 60.0  # when a 429 has no useful headers
STATUS_AVAILABLE = "available"

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

_limited_until: dict[str, datetime] = {}
_lock = threading.Lock()


class LLMError(Exception):
    """The provider failed or returned an unusable response."""


class ProviderRateLimitedError(LLMError):
    """The provider is rate limited until `until`."""

    def __init__(self, provider: str, until: datetime):
        super().__init__(f"{provider}: rate limited until {_format_time(until)}")
        self.provider = provider
        self.until = until


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    latency_ms: int


def _now() -> datetime:
    return datetime.now()


def _format_time(moment: datetime) -> str:
    # Round up to the next minute so the announced time is never still inside the limit.
    return (moment + timedelta(seconds=59)).strftime("%H:%M")


def _parse_duration(value: str) -> float | None:
    """Convert '30', '7.66s', '2m59.56s' or '120ms' to seconds."""
    value = value.strip().lower()
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    parts = _DURATION_PART.findall(value)
    if not parts or "".join(number + unit for number, unit in parts) != value:
        return None
    return sum(float(number) * _UNIT_SECONDS[unit] for number, unit in parts)


def _parse_retry_after(value: str, now: datetime) -> float | None:
    """Retry-After holds seconds or an HTTP date."""
    seconds = _parse_duration(value)
    if seconds is not None:
        return seconds
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    local_moment = moment.astimezone().replace(tzinfo=None)
    return max((local_moment - now).total_seconds(), 0.0)


def _cooldown_seconds(headers: Mapping[str, str], now: datetime) -> float:
    """Prefer retry-after, otherwise wait for the latest reset among exhausted limits."""
    retry_after = headers.get("retry-after")
    if retry_after:
        seconds = _parse_retry_after(retry_after, now)
        if seconds is not None:
            return seconds

    resets = []
    for kind in ("requests", "tokens"):
        if headers.get(f"x-ratelimit-remaining-{kind}") == "0":
            reset = _parse_duration(headers.get(f"x-ratelimit-reset-{kind}", ""))
            if reset is not None:
                resets.append(reset)
    return max(resets, default=DEFAULT_COOLDOWN_SECONDS)


def _mark_limited(provider: str, headers: Mapping[str, str]) -> datetime:
    now = _now()
    until = now + timedelta(seconds=_cooldown_seconds(headers, now))
    with _lock:
        _limited_until[provider] = until
    return until


def _active_limit(provider: str) -> datetime | None:
    with _lock:
        until = _limited_until.get(provider)
        if until is not None and until <= _now():
            del _limited_until[provider]
            return None
        return until


def provider_status(provider: str) -> str:
    """Return 'available' or 'rate limited until HH:MM'."""
    until = _active_limit(provider)
    return STATUS_AVAILABLE if until is None else f"rate limited until {_format_time(until)}"


def timeout_seconds() -> float:
    """LLM_TIMEOUT_SECONDS if it is a positive number, otherwise the default."""
    try:
        value = float(os.environ.get("LLM_TIMEOUT_SECONDS", ""))
    except ValueError:
        return TIMEOUT_SECONDS
    return value if value > 0 else TIMEOUT_SECONDS


def make_client(provider: ProviderConfig) -> OpenAI:
    # max_retries=0: a 429 must update the status instead of being retried silently.
    # The SDK requires a non-empty key even though endpoints like Ollama ignore it.
    return OpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key or "not-needed",
        timeout=timeout_seconds(),
        max_retries=0,
    )


def build_user_message(question: str, chunks: Sequence[Mapping[str, object]]) -> str:
    """Number the excerpts [S1]..[Sk] so citations can be verified."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        section = chunk.get("section")
        header = f"[S{number}] ({section})" if section else f"[S{number}]"
        blocks.append(f"{header}\n{chunk['text']}")
    context = "\n\n".join(blocks)
    return f"Excerpts:\n\n{context}\n\nQuestion: {question}"


def _describe_error(exc: OpenAIError) -> str:
    """Error class, HTTP status and cause class. Never the message, which may hold a URL."""
    description = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if status is not None:
        description += f" (HTTP {status})"
    if exc.__cause__ is not None:
        description += f" (cause: {type(exc.__cause__).__name__})"
    return description


def generate(
    question: str, chunks: Sequence[Mapping[str, object]], provider: ProviderConfig
) -> GenerationResult:
    """Generate an answer. Raise LLMError or ProviderRateLimitedError on failure."""
    until = _active_limit(provider.name)
    if until is not None:
        raise ProviderRateLimitedError(provider.name, until)

    start = time.perf_counter()
    try:
        response = make_client(provider).chat.completions.create(
            model=provider.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, chunks)},
            ],
        )
    except RateLimitError as exc:
        until = _mark_limited(provider.name, exc.response.headers)
        raise ProviderRateLimitedError(provider.name, until) from exc
    except OpenAIError as exc:
        raise LLMError(f"{provider.name}: {_describe_error(exc)}") from exc

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise LLMError(f"{provider.name}: empty response")
    latency_ms = round((time.perf_counter() - start) * 1000)
    return GenerationResult(content.strip(), provider.model, latency_ms)
