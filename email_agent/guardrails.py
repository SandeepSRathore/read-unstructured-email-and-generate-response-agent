"""Deterministic rules that decide whether a reply is sent automatically.

The model proposes; these rules dispose. A reply is auto-sent only when every check passes.
"""

from __future__ import annotations

import re

from .config import Settings
from .schemas import Decision, DraftReply, ExtractedDetails, InboundEmail, Verification

_NO_REPLY_SENDER = re.compile(r"(no-?reply|do-?not-?reply|mailer-daemon|postmaster|bounce)", re.IGNORECASE)

# Intents where the customer needs something *done* (money moved, order changed). The agent can
# explain policy and next steps, but a person must perform the action, so these never auto-send.
ACTION_INTENTS = {"refund_request", "cancellation", "shipping_address_change"}
ALWAYS_REVIEW_INTENTS = {"legal_or_chargeback", "spam_or_not_customer"}


def should_skip(email: InboundEmail, settings: Settings) -> str | None:
    """Return a reason to ignore this email entirely (no reply, no review), or None.

    Guards against mail loops and replying to machines.
    """
    h = {k.lower(): v for k, v in email.headers.items()}
    if not email.sender_email:
        return "no sender address"
    if email.sender_email.lower() == settings.support_address.lower():
        return "sent from our own support address (loop guard)"
    if _NO_REPLY_SENDER.search(email.sender_email):
        return f"automated sender address ({email.sender_email})"
    if h.get("auto-submitted", "no").lower() != "no":
        return f"auto-submitted message ({h['auto-submitted']})"
    if h.get("precedence", "").lower() in {"bulk", "list", "junk", "auto_reply"}:
        return f"bulk/list precedence ({h['precedence']})"
    if "list-id" in h or "list-unsubscribe" in h:
        return "mailing-list or newsletter message"
    if any(k in h for k in ("x-autoreply", "x-autorespond")):
        return "auto-responder message"
    return None


def decide(
    details: ExtractedDetails,
    draft: DraftReply,
    verification: Verification,
    settings: Settings,
) -> Decision:
    reasons: list[str] = []

    intents = {details.primary_intent, *details.secondary_intents}
    for intent in sorted(intents & ALWAYS_REVIEW_INTENTS):
        reasons.append(f"intent '{intent}' always needs a person")
    for intent in sorted(intents & ACTION_INTENTS):
        reasons.append(f"intent '{intent}' needs an account action the agent cannot perform")
    if details.threatens_legal_action_or_chargeback:
        reasons.append("customer mentions legal action or a chargeback")
    if details.refund_amount_requested is not None and details.refund_amount_requested > settings.max_auto_refund:
        reasons.append(
            f"refund of {details.refund_amount_requested:.2f} exceeds auto limit {settings.max_auto_refund:.2f}"
        )

    if draft.needs_human:
        reasons.append(f"drafter flagged for a person: {draft.needs_human_reason or 'no reason given'}")
    if draft.confidence < settings.min_confidence:
        reasons.append(f"drafter confidence {draft.confidence:.2f} below threshold {settings.min_confidence:.2f}")

    if not verification.approved:
        reasons.append("verifier rejected the draft")
    for claim in verification.unsupported_claims:
        reasons.append(f"unsupported claim: {claim}")
    for q in verification.unanswered_questions:
        reasons.append(f"unanswered question: {q}")
    for issue in verification.tone_or_safety_issues:
        reasons.append(f"tone/safety issue: {issue}")

    if not settings.auto_send:
        reasons.append("AUTO_SEND is off; all replies are held for review")

    return Decision(action="review" if reasons else "send", reasons=reasons)
