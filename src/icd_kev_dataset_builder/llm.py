from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import openai

from .errors import LLMConfigurationError, LLMRefusalError, LLMRequestError

ALLOWED_ENDPOINT_SUFFIXES = (
    ".openai.azure.com",
    ".services.ai.azure.com",
    ".cognitiveservices.azure.com",
)
OPTIONAL_PARAMETERS = ("response_format", "max_completion_tokens", "temperature")


def load_dotenv(path: Path, environ: dict[str, str] | None = None) -> None:
    """Load KEY=VALUE pairs without overriding the environment. Values are never logged."""
    target = os.environ if environ is None else environ
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            target.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class LLMSettings:
    endpoint: str
    api_key: str = field(repr=False)
    api_version: str | None
    generation_deployment: str
    review_deployment: str
    mode: str = "classic"
    timeout_seconds: float = 90.0
    max_retries: int = 4
    max_output_tokens: int | None = None
    temperature: float | None = None
    json_mode: str = "auto"

    @property
    def review_shares_generation_deployment(self) -> bool:
        return self.review_deployment == self.generation_deployment

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> LLMSettings:
        env = os.environ if environ is None else environ
        endpoint = env.get("AZURE_OPENAI_ENDPOINT", "")
        api_key = env.get("AZURE_OPENAI_API_KEY", "")
        generation = env.get("AZURE_OPENAI_GENERATION_DEPLOYMENT", "")
        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_API_KEY", api_key),
                ("AZURE_OPENAI_GENERATION_DEPLOYMENT", generation),
            )
            if not value
        ]
        if missing:
            raise LLMConfigurationError("Missing required settings: " + ", ".join(missing))
        parsed = urlparse(endpoint)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not host.endswith(ALLOWED_ENDPOINT_SUFFIXES):
            raise LLMConfigurationError(
                "AZURE_OPENAI_ENDPOINT must be an https Azure OpenAI / Foundry endpoint."
            )
        api_version = env.get("AZURE_OPENAI_API_VERSION") or None
        mode = env.get("AZURE_OPENAI_MODE") or ("classic" if api_version else "v1")
        if mode not in {"classic", "v1"}:
            raise LLMConfigurationError("AZURE_OPENAI_MODE must be 'classic' or 'v1'.")
        if mode == "classic" and not api_version:
            raise LLMConfigurationError("Classic mode requires AZURE_OPENAI_API_VERSION.")
        json_mode = env.get("AZURE_OPENAI_JSON_MODE", "auto")
        if json_mode not in {"auto", "on", "off"}:
            raise LLMConfigurationError("AZURE_OPENAI_JSON_MODE must be auto, on, or off.")
        try:
            timeout = float(env.get("AZURE_OPENAI_TIMEOUT_SECONDS", "90"))
            max_tokens = env.get("AZURE_OPENAI_MAX_OUTPUT_TOKENS")
            temperature = env.get("AZURE_OPENAI_TEMPERATURE")
            return cls(
                endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                generation_deployment=generation,
                review_deployment=env.get("AZURE_OPENAI_REVIEW_DEPLOYMENT") or generation,
                mode=mode,
                timeout_seconds=timeout,
                max_retries=int(env.get("AZURE_OPENAI_MAX_RETRIES", "4")),
                max_output_tokens=int(max_tokens) if max_tokens else None,
                temperature=float(temperature) if temperature else None,
                json_mode=json_mode,
            )
        except ValueError as exc:
            raise LLMConfigurationError(f"Invalid numeric Azure OpenAI setting: {exc}") from exc


@dataclass(frozen=True)
class LLMResult:
    text: str
    finish_reason: str | None
    request_id: str | None
    deployment: str
    prompt_tokens: int
    completion_tokens: int
    attempts: int
    started_at: float
    finished_at: float
    parameters_sent: tuple[str, ...]


def _build_sdk_client(settings: LLMSettings) -> Any:
    if settings.mode == "v1":
        base = settings.endpoint.rstrip("/") + "/openai/v1/"
        return openai.OpenAI(
            base_url=base, api_key=settings.api_key, timeout=settings.timeout_seconds, max_retries=0
        )
    return openai.AzureOpenAI(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        api_version=settings.api_version,
        timeout=settings.timeout_seconds,
        max_retries=0,
    )


class LLMClient:
    def __init__(
        self,
        settings: LLMSettings,
        sdk_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._client = sdk_client if sdk_client is not None else _build_sdk_client(settings)
        self._sleep = sleep
        self._disabled: set[str] = set()
        if settings.json_mode == "off":
            self._disabled.add("response_format")

    @property
    def disabled_parameters(self) -> frozenset[str]:
        return frozenset(self._disabled)

    def probe(self, deployment: str | None = None) -> dict[str, Any]:
        """Tiny capability/connection probe. Reports which optional parameters were accepted."""
        result = self.complete(
            deployment or self.settings.generation_deployment,
            "Reply with a JSON object only.",
            'Return {"ok": true}.',
        )
        return {
            "deployment": result.deployment,
            "finish_reason": result.finish_reason,
            "request_id": result.request_id,
            "parameters_sent": list(result.parameters_sent),
            "disabled_parameters": sorted(self._disabled),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }

    def complete(self, deployment: str, system: str, user: str) -> LLMResult:
        started = time.time()
        attempts = 0
        transient_failures = 0
        while True:
            attempts += 1
            kwargs, sent = self._request_kwargs(deployment, system, user)
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # classified below; unknown errors are not swallowed
                action, delay = self._classify(exc, transient_failures)
                if action == "retry_params":
                    continue
                if action == "retry":
                    transient_failures += 1
                    if transient_failures > self.settings.max_retries:
                        raise LLMRequestError(
                            f"Request failed after {attempts} attempts: {type(exc).__name__}"
                        ) from exc
                    self._sleep(delay)
                    continue
                raise self._permanent(exc) from exc
            return self._to_result(response, deployment, attempts, started, sent)

    def _request_kwargs(self, deployment: str, system: str, user: str) -> tuple[dict[str, Any], tuple[str, ...]]:
        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        sent: list[str] = []
        if "response_format" not in self._disabled:
            kwargs["response_format"] = {"type": "json_object"}
            sent.append("response_format")
        limit = self.settings.max_output_tokens
        if limit is not None:
            if "max_completion_tokens" not in self._disabled:
                kwargs["max_completion_tokens"] = limit
                sent.append("max_completion_tokens")
            elif "max_tokens" not in self._disabled:
                kwargs["max_tokens"] = limit
                sent.append("max_tokens")
        if self.settings.temperature is not None and "temperature" not in self._disabled:
            kwargs["temperature"] = self.settings.temperature
            sent.append("temperature")
        return kwargs, tuple(sent)

    def _classify(self, exc: Exception, transient_failures: int) -> tuple[str, float]:
        if isinstance(exc, openai.BadRequestError):
            text = str(exc).lower()
            if getattr(exc, "code", None) == "content_filter" or "content_filter" in text:
                return "permanent", 0.0
            for name in OPTIONAL_PARAMETERS + ("max_tokens",):
                if name in text and name not in self._disabled:
                    self._disabled.add(name)
                    return "retry_params", 0.0
            return "permanent", 0.0
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
            return "permanent", 0.0
        retryable = isinstance(
            exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
        ) or (isinstance(exc, openai.APIStatusError) and exc.status_code >= 500)
        if not retryable:
            return "permanent", 0.0
        retry_after = self._retry_after(exc)
        base = retry_after if retry_after is not None else float(2**transient_failures)
        delay = min(base, 30.0)
        return "retry", delay + random.uniform(0.0, min(delay * 0.1, 1.0))

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        header = getattr(response, "headers", {}).get("retry-after") if response is not None else None
        try:
            return float(header) if header is not None else None
        except ValueError:
            return None

    @staticmethod
    def _permanent(exc: Exception) -> Exception:
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
            return LLMConfigurationError(
                f"Azure OpenAI rejected the configuration ({type(exc).__name__}); check endpoint, "
                "credentials, and deployment name."
            )
        if isinstance(exc, openai.BadRequestError) and (
            getattr(exc, "code", None) == "content_filter" or "content_filter" in str(exc).lower()
        ):
            return LLMRefusalError("The request or response was blocked by content filtering.")
        return LLMRequestError(f"Permanent model request failure: {type(exc).__name__}: {exc}")

    @staticmethod
    def _to_result(
        response: Any, deployment: str, attempts: int, started: float, sent: tuple[str, ...]
    ) -> LLMResult:
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMRefusalError("The model refused the request.")
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            request_id=getattr(response, "_request_id", None),
            deployment=deployment,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            attempts=attempts,
            started_at=started,
            finished_at=time.time(),
            parameters_sent=sent,
        )
