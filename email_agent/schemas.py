"""Data shapes passed between pipeline stages.

The three Claude-facing models (ExtractedDetails, DraftReply, Verification) double as
JSON schemas for structured outputs, so Claude's responses are guaranteed to parse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Intent = Literal[
    "order_status",
    "return_or_exchange",
    "refund_request",
    "damaged_or_wrong_item",
    "billing_question",
    "product_question",
    "shipping_address_change",
    "cancellation",
    "account_help",
    "complaint",
    "legal_or_chargeback",
    "spam_or_not_customer",
    "other",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedDetails(_Strict):
    customer_name: str
    """Customer's name as written in the email; empty string if not given."""
    primary_intent: Intent
    secondary_intents: list[Intent]
    order_ids: list[str]
    products_mentioned: list[str]
    refund_amount_requested: float | None
    summary: str
    customer_questions: list[str]
    sentiment: Literal["positive", "neutral", "frustrated", "angry"]
    urgency: Literal["low", "normal", "high"]
    threatens_legal_action_or_chargeback: bool
    language: str
    missing_information: list[str]


class DraftReply(_Strict):
    subject: str
    body: str
    addressed_questions: list[str]
    facts_used: list[str]
    confidence: float
    needs_human: bool
    needs_human_reason: str


class Verification(_Strict):
    approved: bool
    unsupported_claims: list[str]
    unanswered_questions: list[str]
    tone_or_safety_issues: list[str]
    notes: str


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema, minus decorative keys structured outputs doesn't need."""

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k not in {"title", "default"}}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(model.model_json_schema())


@dataclass
class InboundEmail:
    message_id: str
    sender_email: str
    sender_name: str
    subject: str
    body: str
    received_at: str = ""
    references: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    # Opaque handle the mailbox uses to find this message again (file path, IMAP UID...).
    source_ref: str = ""


@dataclass
class Decision:
    action: Literal["send", "review", "skip"]
    reasons: list[str] = field(default_factory=list)


@dataclass
class ProcessResult:
    email: InboundEmail
    decision: Decision
    details: ExtractedDetails | None = None
    draft: DraftReply | None = None
    verification: Verification | None = None
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
