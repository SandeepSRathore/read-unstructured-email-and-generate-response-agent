"""The three Claude calls: extract details, draft a grounded reply, verify the draft."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

import anthropic

from .prompts import (
    DRAFT_SYSTEM,
    EXTRACT_SYSTEM,
    VERIFY_SYSTEM,
    draft_user_message,
    render_email,
    verify_user_message,
)
from .schemas import DraftReply, ExtractedDetails, InboundEmail, Verification, strict_json_schema
from .tools import TOOL_DEFINITIONS, ToolExecutor

# Models that accept server-side refusal fallbacks ("default" routing).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOOL_TURNS = 8


class LLMError(RuntimeError):
    """A Claude call ended without a usable answer (refusal, truncation, bad JSON)."""


class LLM(Protocol):
    def extract(self, email: InboundEmail) -> ExtractedDetails: ...
    def draft(self, email: InboundEmail, details: ExtractedDetails, tools: ToolExecutor) -> DraftReply: ...
    def verify(
        self, email: InboundEmail, details: ExtractedDetails, draft: DraftReply, tool_log: list[dict[str, Any]]
    ) -> Verification: ...


class ClaudeLLM:
    def __init__(
        self,
        model: str,
        company_name: str,
        signature: str,
        policies: str,
        client: anthropic.Anthropic | None = None,
        today: date | None = None,
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.today = today or date.today()
        self.extract_system = EXTRACT_SYSTEM.format(company=company_name)
        self.draft_system = DRAFT_SYSTEM.format(company=company_name, signature=signature, policies=policies)
        self.verify_system = VERIFY_SYSTEM.format(company=company_name, policies=policies)

    # ---- plumbing ----

    def _create(self, *, system: str, messages: list[dict[str, Any]], schema: type, effort: str, tools=None):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            # System prompts are frozen per run, so cache them across emails.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": strict_json_schema(schema)},
            },
        )
        if tools:
            kwargs["tools"] = tools
        if self.model in _FALLBACK_MODELS:
            # If a safety classifier declines, retry server-side on Anthropic's recommended model.
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        response = self.client.beta.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise LLMError(f"Model declined the request (category: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMError("Model output was cut off at max_tokens")
        return response

    @staticmethod
    def _parse(response, schema: type):
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return schema.model_validate_json(text)
        except Exception as e:
            raise LLMError(f"Could not parse {schema.__name__} from model output: {e}") from e

    # ---- stage 1: extraction ----

    def extract(self, email: InboundEmail) -> ExtractedDetails:
        response = self._create(
            system=self.extract_system,
            messages=[{"role": "user", "content": render_email(email)}],
            schema=ExtractedDetails,
            effort="medium",
        )
        return self._parse(response, ExtractedDetails)

    # ---- stage 2: grounded drafting with order-lookup tools ----

    def draft(self, email: InboundEmail, details: ExtractedDetails, tools: ToolExecutor) -> DraftReply:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": draft_user_message(email, details, self.today)}
        ]
        for _ in range(MAX_TOOL_TURNS):
            response = self._create(
                system=self.draft_system,
                messages=messages,
                schema=DraftReply,
                effort="high",
                tools=TOOL_DEFINITIONS,
            )
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_calls:
                return self._parse(response, DraftReply)

            # Keep the whole assistant turn (thinking + tool calls) in history, then answer every call at once.
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for call in tool_calls:
                content, is_error = tools.run(call.name, call.input)
                results.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})
        raise LLMError(f"Drafting did not finish within {MAX_TOOL_TURNS} tool turns")

    # ---- stage 3: independent verification ----

    def verify(
        self, email: InboundEmail, details: ExtractedDetails, draft: DraftReply, tool_log: list[dict[str, Any]]
    ) -> Verification:
        content = verify_user_message(email, details, draft, tool_log, self.today)
        response = self._create(
            system=self.verify_system,
            messages=[{"role": "user", "content": content}],
            schema=Verification,
            effort="high",
        )
        return self._parse(response, Verification)
