"""Per-email pipeline: skip-check -> extract -> draft (with tools) -> verify -> decide -> send or hold."""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
import openai

from .config import Settings
from .guardrails import decide, should_skip
from .llm import LLM, LLMError
from .mailbox import Mailbox, build_reply, utc_now, write_review_file
from .schemas import Decision, InboundEmail, ProcessResult
from .tools import OrderStore, ToolExecutor

log = logging.getLogger("email_agent")

# Temporary API problems: leave the email untouched so the next run retries it.
TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class EmailAgent:
    def __init__(self, settings: Settings, llm: LLM, store: OrderStore, mailbox: Mailbox):
        self.settings = settings
        self.llm = llm
        self.store = store
        self.mailbox = mailbox
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        self._processed = self._load_processed()

    # ---- state ----

    def _load_processed(self) -> set[str]:
        p = self.settings.processed_ids_path
        return set(p.read_text(encoding="utf-8").split()) if p.exists() else set()

    def _remember(self, message_id: str) -> None:
        self._processed.add(message_id)
        with open(self.settings.processed_ids_path, "a", encoding="utf-8") as f:
            f.write(message_id + "\n")

    def _audit(self, result: ProcessResult) -> None:
        entry: dict[str, Any] = {
            "at": utc_now(),
            "message_id": result.email.message_id,
            "from": result.email.sender_email,
            "subject": result.email.subject,
            "action": result.decision.action,
            "reasons": result.decision.reasons,
            "error": result.error,
            "details": result.details.model_dump() if result.details else None,
            "draft": result.draft.model_dump() if result.draft else None,
            "verification": result.verification.model_dump() if result.verification else None,
            "tool_calls": result.tool_log,
        }
        with open(self.settings.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # ---- pipeline ----

    def run_once(self) -> list[ProcessResult]:
        results = []
        for email in self.mailbox.fetch():
            if email.message_id in self._processed:
                log.info("Already handled %s; marking done", email.message_id)
                self.mailbox.mark_done(email)
                continue
            result = self.process(email)
            if result is not None:
                results.append(result)
        return results

    def process(self, email: InboundEmail) -> ProcessResult | None:
        log.info("Processing %s from %s: %r", email.message_id, email.sender_email, email.subject)

        skip_reason = should_skip(email, self.settings)
        if skip_reason:
            result = ProcessResult(email=email, decision=Decision("skip", [skip_reason]))
            self._finish(result)
            return result

        result = ProcessResult(email=email, decision=Decision("review"))
        tools = ToolExecutor(self.store, email.sender_email)
        try:
            result.details = self.llm.extract(email)
            result.draft = self.llm.draft(email, result.details, tools)
            result.tool_log = tools.log
            result.verification = self.llm.verify(email, result.details, result.draft, tools.log)
            result.decision = decide(result.details, result.draft, result.verification, self.settings)
        except TRANSIENT_ERRORS as e:
            log.warning("Temporary API error on %s, will retry next run: %s", email.message_id, e)
            return None
        except LLMError as e:
            # Never auto-send after a failure: hand whatever we have to a person.
            result.tool_log = tools.log
            result.error = str(e)
            result.decision = Decision("review", [f"pipeline error: {e}"])

        self._finish(result)
        return result

    def _finish(self, result: ProcessResult) -> None:
        email, decision = result.email, result.decision
        reply = build_reply(email, result.draft, self.settings) if result.draft else None

        if decision.action == "send" and reply is not None:
            self.mailbox.send(reply, email)
            log.info("SENT reply to %s", email.sender_email)
        elif decision.action == "review":
            path = write_review_file(
                self.settings.review_dir,
                email,
                {
                    "message_id": email.message_id,
                    "from": f"{email.sender_name} <{email.sender_email}>",
                    "subject": email.subject,
                    "original_body": email.body,
                    "reasons": decision.reasons,
                    "error": result.error,
                    "details": result.details.model_dump() if result.details else None,
                    "draft": result.draft.model_dump() if result.draft else None,
                    "verification": result.verification.model_dump() if result.verification else None,
                    "tool_calls": result.tool_log,
                },
                reply,
            )
            self.mailbox.flag_for_review(email)
            log.info("HELD for review (%s): %s", path.name, "; ".join(decision.reasons))
        else:
            log.info("SKIPPED %s: %s", email.message_id, "; ".join(decision.reasons))

        self._audit(result)
        self._remember(email.message_id)
        self.mailbox.mark_done(email)
