"""OpenAI backend (Responses API). Same three stages and prompts as the Claude backend."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import openai

from .llm import MAX_TOOL_TURNS, LLMError
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

# Same tools, in the Responses API's function-tool shape.
OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": t["name"],
        "description": t["description"],
        "parameters": t["input_schema"],
        "strict": True,
    }
    for t in TOOL_DEFINITIONS
]


class OpenAILLM:
    def __init__(
        self,
        model: str,
        company_name: str,
        signature: str,
        policies: str,
        client: openai.OpenAI | None = None,
        today: date | None = None,
    ):
        self.client = client or openai.OpenAI()
        self.model = model
        self.today = today or date.today()
        self.extract_system = EXTRACT_SYSTEM.format(company=company_name)
        self.draft_system = DRAFT_SYSTEM.format(company=company_name, signature=signature, policies=policies)
        self.verify_system = VERIFY_SYSTEM.format(company=company_name, policies=policies)

    # ---- plumbing ----

    def _create(self, *, instructions: str, input_items: list[dict[str, Any]], schema: type, effort: str, tools=None):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            instructions=instructions,
            input=input_items,
            reasoning={"effort": effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": strict_json_schema(schema),
                    "strict": True,
                }
            },
            # Stateless: nothing is stored server-side; reasoning is carried forward encrypted.
            store=False,
            include=["reasoning.encrypted_content"],
        )
        if tools:
            kwargs["tools"] = tools
        response = self.client.responses.create(**kwargs)

        if response.status != "completed":
            raise LLMError(f"Model response {response.status}: {getattr(response, 'incomplete_details', None)}")
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        raise LLMError(f"Model declined the request: {part.refusal}")
        return response

    @staticmethod
    def _parse(response, schema: type):
        try:
            return schema.model_validate_json(response.output_text)
        except Exception as e:
            raise LLMError(f"Could not parse {schema.__name__} from model output: {e}") from e

    # ---- stages ----

    def extract(self, email: InboundEmail) -> ExtractedDetails:
        response = self._create(
            instructions=self.extract_system,
            input_items=[{"role": "user", "content": render_email(email)}],
            schema=ExtractedDetails,
            effort="medium",
        )
        return self._parse(response, ExtractedDetails)

    def draft(self, email: InboundEmail, details: ExtractedDetails, tools: ToolExecutor) -> DraftReply:
        input_items: list[dict[str, Any]] = [
            {"role": "user", "content": draft_user_message(email, details, self.today)}
        ]
        for _ in range(MAX_TOOL_TURNS):
            response = self._create(
                instructions=self.draft_system,
                input_items=input_items,
                schema=DraftReply,
                effort="high",
                tools=OPENAI_TOOLS,
            )
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                return self._parse(response, DraftReply)

            # Replay every output item (reasoning included), then one output per call.
            input_items.extend(item.model_dump(exclude_none=True) for item in response.output)
            for call in calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except ValueError:
                    content = f"Error: arguments were not valid JSON: {call.arguments!r}"
                else:
                    content, _ = tools.run(call.name, args)
                input_items.append({"type": "function_call_output", "call_id": call.call_id, "output": content})
        raise LLMError(f"Drafting did not finish within {MAX_TOOL_TURNS} tool turns")

    def verify(
        self, email: InboundEmail, details: ExtractedDetails, draft: DraftReply, tool_log: list[dict[str, Any]]
    ) -> Verification:
        response = self._create(
            instructions=self.verify_system,
            input_items=[
                {"role": "user", "content": verify_user_message(email, details, draft, tool_log, self.today)}
            ],
            schema=Verification,
            effort="high",
        )
        return self._parse(response, Verification)
