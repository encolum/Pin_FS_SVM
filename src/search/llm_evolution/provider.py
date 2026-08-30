"""Replayable LLM provider boundary with no embedded credentials."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from time import perf_counter
from typing import Any, Protocol
from urllib import request


@dataclass(frozen=True)
class GenerationRecord:
    provider: str
    model: str
    prompt_hash: str
    prompt: str
    response: str
    seed: int | None
    temperature: float
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    estimated_cost: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, seed: int | None = None) -> str: ...


class MockProvider:
    """Deterministic provider for tests and offline evolution rehearsals."""

    def __init__(self, responses: list[str], *, model: str = "mock-verapin") -> None:
        if not responses:
            raise ValueError("MockProvider requires at least one response")
        self.responses = list(responses)
        self.model = model
        self.records: list[GenerationRecord] = []

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        selector = 0 if seed is None else int(seed)
        response = self.responses[selector % len(self.responses)]
        self.records.append(
            GenerationRecord(
                provider="mock",
                model=self.model,
                prompt_hash=_text_hash(prompt),
                prompt=prompt,
                response=response,
                seed=seed,
                temperature=0.0,
                input_tokens=None,
                output_tokens=None,
                latency_seconds=0.0,
                estimated_cost=0.0,
            )
        )
        return response


class ReplayProvider:
    """Return previously persisted responses without any network or LLM call."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = list(records)
        self._position = 0
        self.records: list[GenerationRecord] = []

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        if self._position >= len(self._records):
            raise RuntimeError("replay provider has no remaining response")
        source = self._records[self._position]
        self._position += 1
        expected_hash = source.get("prompt_hash")
        if expected_hash is not None and expected_hash != _text_hash(prompt):
            raise ValueError("replay prompt hash does not match the recorded evolution prompt")
        response = str(source["response"])
        self.records.append(
            GenerationRecord(
                provider="replay",
                model=str(source.get("model", "recorded")),
                prompt_hash=_text_hash(prompt),
                prompt=prompt,
                response=response,
                seed=seed,
                temperature=float(source.get("temperature", 0.0)),
                input_tokens=_optional_int(source.get("input_tokens")),
                output_tokens=_optional_int(source.get("output_tokens")),
                latency_seconds=0.0,
                estimated_cost=_optional_float(source.get("estimated_cost")),
            )
        )
        return response


class EnvironmentLLMProvider:
    """OpenAI-compatible HTTP provider configured only through environment variables."""

    def __init__(
        self,
        *,
        temperature: float,
        timeout_seconds: float,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        api_url_env: str = "VERAPIN_LLM_API_URL",
        api_key_env: str = "VERAPIN_LLM_API_KEY",
        model_env: str = "VERAPIN_LLM_MODEL",
    ) -> None:
        self.api_url = _required_environment(api_url_env)
        self.api_key = _required_environment(api_key_env)
        self.model = _required_environment(model_env)
        self.temperature = float(temperature)
        self.timeout_seconds = float(timeout_seconds)
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must lie in [0, 2]")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.input_cost_per_million = _nonnegative_optional(input_cost_per_million)
        self.output_cost_per_million = _nonnegative_optional(output_cost_per_million)
        self.records: list[GenerationRecord] = []

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            payload["seed"] = int(seed)
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = perf_counter()
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        latency = perf_counter() - started
        try:
            content = str(decoded["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response does not contain choices[0].message.content") from exc
        usage = decoded.get("usage", {})
        input_tokens = _optional_int(usage.get("prompt_tokens"))
        output_tokens = _optional_int(usage.get("completion_tokens"))
        cost = _estimated_cost(
            input_tokens,
            output_tokens,
            input_rate=self.input_cost_per_million,
            output_rate=self.output_cost_per_million,
        )
        self.records.append(
            GenerationRecord(
                provider="environment-openai-compatible",
                model=self.model,
                prompt_hash=_text_hash(prompt),
                prompt=prompt,
                response=content,
                seed=seed,
                temperature=self.temperature,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=latency,
                estimated_cost=cost,
            )
        )
        return content


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    return value


def _text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _nonnegative_optional(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value < 0:
        raise ValueError("token-cost rates must be non-negative")
    return value


def _estimated_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    input_rate: float | None,
    output_rate: float | None,
) -> float | None:
    if None in {input_tokens, output_tokens, input_rate, output_rate}:
        return None
    return float(input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000)
