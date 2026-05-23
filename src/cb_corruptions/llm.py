"""LLM helper: OpenAI SDK wrapper with structured output and cost tracking.

Supports both sync (single call) and async batch (parallel calls) modes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import openai
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLM:
    """OpenAI-compatible client with function calling and cost tracking."""

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        temperature: float = 0.7,
        api_key: str | None = None,
        base_url: str | None = None,
        pricing: tuple[float, float] | None = None,
        batch_size: int = 10,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        tool_choice_mode: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        # OpenRouter-specific knobs go in extra_body, e.g.
        #   {"provider": {"ignore": ["AtlasCloud"]}} to route around a bad provider
        #   {"reasoning": {"effort": "high"}} to force reasoning on
        self.extra_body: dict[str, Any] = dict(extra_body or {})
        # tool_choice override: None → explicit function-object form (default),
        # "auto" / "required" / "none" → pass the string verbatim. Some
        # providers (e.g. AtlasCloud for Qwen3.5-27B) only accept "auto".
        self.tool_choice_mode: str | None = tool_choice_mode
        # Hard per-request timeout. Without this, hung connections on a flaky
        # provider (e.g. AtlasCloud silently dropping mid-stream) freeze the
        # whole batch. 180s is generous enough for reasoning models that emit
        # several thousand thinking tokens.
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
        }
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self._client = openai.OpenAI(**client_kwargs)
        self._async_client = openai.AsyncOpenAI(**client_kwargs)
        self._pricing = pricing or (0.0, 0.0)
        self.total_cost = 0.0

        # Ollama doesn't support tool_choice
        resolved_url = str(self._client.base_url)
        self._supports_tool_choice = "localhost:11434" not in resolved_url and "127.0.0.1:11434" not in resolved_url
        if not self._supports_tool_choice:
            logger.info("Ollama detected — tool_choice disabled")

    def _track_cost(self, usage: Any) -> float:
        if usage is None:
            return 0.0
        input_price, output_price = self._pricing
        cost = (
            usage.prompt_tokens * input_price / 1_000_000
            + usage.completion_tokens * output_price / 1_000_000
        )
        self.total_cost += cost
        return cost

    @staticmethod
    def _extract_usage(usage: Any) -> dict[str, Any]:
        """Extract token + cost details from an OpenAI/OpenRouter usage object.

        OpenRouter exposes reasoning_tokens at usage.completion_tokens_details.reasoning_tokens
        and the per-call settled cost at usage.cost. Fields missing on a given provider
        are returned as None. See https://openrouter.ai/docs/cookbook/administration/usage-accounting.
        """
        if usage is None:
            return {}
        completion_details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
            "cached_tokens": getattr(prompt_details, "cached_tokens", None),
            "cost_openrouter": getattr(usage, "cost", None),
        }

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Plain text completion."""
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body
        response = self._client.chat.completions.create(**create_kwargs)
        self._track_cost(response.usage)
        return response.choices[0].message.content or ""

    def structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        """Function calling -> parsed Pydantic model. Discards usage; use
        ``structured_with_usage`` when token/cost telemetry is needed."""
        parsed, _ = self.structured_with_usage(messages, response_model, **kwargs)
        return parsed

    def structured_with_usage(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        **kwargs: Any,
    ) -> tuple[BaseModel, dict[str, Any]]:
        """Like ``structured()`` but also returns the token+cost usage dict."""
        schema = response_model.model_json_schema()
        tool = {
            "type": "function",
            "function": {
                "name": response_model.__name__,
                "description": schema.get("description", ""),
                "parameters": schema,
            },
        }
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "tools": [tool],
        }
        if self._supports_tool_choice:
            if self.tool_choice_mode in ("auto", "required", "none"):
                create_kwargs["tool_choice"] = self.tool_choice_mode
            else:
                create_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": response_model.__name__},
                }
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body

        response = self._client.chat.completions.create(**create_kwargs)
        self._track_cost(response.usage)
        parsed = self._parse_structured_response(response, response_model)
        meta = self._extract_usage(response.usage)
        meta["reasoning"] = getattr(response.choices[0].message, "reasoning", None)
        return parsed, meta

    @staticmethod
    def _parse_structured_response(response: Any, response_model: type[BaseModel]) -> BaseModel:
        msg = response.choices[0].message
        if msg.tool_calls:
            return response_model.model_validate_json(msg.tool_calls[0].function.arguments)

        # Fallback: model returned plain text instead of a tool call — parse as JSON
        content = msg.content or ""
        # Strip markdown code fences if present
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]
        return response_model.model_validate_json(content.strip())

    # ------------------------------------------------------------------
    # Async / batch support
    # ------------------------------------------------------------------

    async def _async_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        """Async version of structured()."""
        schema = response_model.model_json_schema()
        tool = {
            "type": "function",
            "function": {
                "name": response_model.__name__,
                "description": schema.get("description", ""),
                "parameters": schema,
            },
        }
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "tools": [tool],
        }
        if self._supports_tool_choice:
            if self.tool_choice_mode in ("auto", "required", "none"):
                create_kwargs["tool_choice"] = self.tool_choice_mode
            else:
                create_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": response_model.__name__},
                }
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body

        response = await self._async_client.chat.completions.create(**create_kwargs)
        self._track_cost(response.usage)

        msg = response.choices[0].message
        if msg.tool_calls:
            return response_model.model_validate_json(msg.tool_calls[0].function.arguments)

        content = msg.content or ""
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]
        return response_model.model_validate_json(content.strip())

    def structured_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        response_model: type[BaseModel],
        **kwargs: Any,
    ) -> list[BaseModel | Exception]:
        """Run multiple structured calls in parallel batches.

        Returns a list of results (BaseModel) or Exceptions for failed calls,
        preserving order.
        """

        async def _run_batch(batch: list[list[dict[str, str]]]) -> list[BaseModel | Exception]:
            tasks = [
                self._async_structured(msgs, response_model, **kwargs)
                for msgs in batch
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results: list[BaseModel | Exception] = []
        for i in range(0, len(messages_list), self.batch_size):
            batch = messages_list[i : i + self.batch_size]
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already inside an event loop (e.g. Jupyter) — use nest_asyncio or thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    batch_results = pool.submit(asyncio.run, _run_batch(batch)).result()
            else:
                batch_results = asyncio.run(_run_batch(batch))

            results.extend(batch_results)

        return results
