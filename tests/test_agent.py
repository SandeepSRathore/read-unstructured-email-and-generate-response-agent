from __future__ import annotations

import json
import shutil
from datetime import date
from email import message_from_bytes, policy
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from email_agent.agent import EmailAgent
from email_agent.config import PROJECT_ROOT, Settings
from email_agent.guardrails import decide, should_skip
from email_agent.llm import ClaudeLLM, LLMError
from email_agent.mailbox import FolderMailbox, parse_email
from email_agent.schemas import DraftReply, ExtractedDetails, InboundEmail, Verification, strict_json_schema
from email_agent.tools import OrderStore, ToolExecutor

SAMPLES = PROJECT_ROOT / "inbox"
TODAY = date(2026, 11, 2)


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.inbox_dir = tmp_path / "inbox"
    s.outbox_dir = tmp_path / "outbox"
    s.review_dir = tmp_path / "review"
    s.state_dir = tmp_path / "state"
    s.auto_send = True
    s.min_confidence = 0.8
    s.max_auto_refund = 100
    s.inbox_dir.mkdir()
    return s


@pytest.fixture
def store(settings: Settings) -> OrderStore:
    return OrderStore(settings.orders_path, today=TODAY)


def details(**overrides) -> ExtractedDetails:
    base = dict(
        customer_name="Maria",
        primary_intent="order_status",
        secondary_intents=[],
        order_ids=["A1001"],
        products_mentioned=["floor lamp"],
        refund_amount_requested=None,
        summary="Asks where her lamp is.",
        customer_questions=["When will order A1001 arrive?"],
        sentiment="neutral",
        urgency="normal",
        threatens_legal_action_or_chargeback=False,
        language="English",
        missing_information=[],
    )
    return ExtractedDetails(**{**base, **overrides})


def draft(**overrides) -> DraftReply:
    base = dict(
        subject="Re: where is my lamp??",
        body="Hi Maria,\n\nYour lamp shipped via UPS.\n\nThe Acme Home Goods Support Team",
        addressed_questions=["When will order A1001 arrive?"],
        facts_used=["Shipped via UPS [order A1001]"],
        confidence=0.95,
        needs_human=False,
        needs_human_reason="",
    )
    return DraftReply(**{**base, **overrides})


def verification(**overrides) -> Verification:
    base = dict(approved=True, unsupported_claims=[], unanswered_questions=[], tone_or_safety_issues=[], notes="")
    return Verification(**{**base, **overrides})


class FakeLLM:
    def __init__(self, d=None, r=None, v=None, raise_on=None, exc=None):
        self.d, self.r, self.v = d or details(), r or draft(), v or verification()
        self.raise_on, self.exc = raise_on, exc
        self.calls: list[str] = []

    def _maybe_raise(self, stage):
        self.calls.append(stage)
        if stage == self.raise_on:
            raise self.exc

    def extract(self, email):
        self._maybe_raise("extract")
        return self.d

    def draft(self, email, d, tools):
        self._maybe_raise("draft")
        tools.run("lookup_order", {"order_id": "A1001"})
        return self.r

    def verify(self, email, d, r, tool_log):
        self._maybe_raise("verify")
        return self.v


def copy_sample(settings: Settings, name: str) -> None:
    shutil.copy(SAMPLES / name, settings.inbox_dir / name)


# ------------------------------------------------------------------ parsing


def test_parse_plain_email():
    e = parse_email((SAMPLES / "01_order_status.eml").read_bytes())
    assert e.sender_email == "maria.gonzalez@example.com"
    assert e.sender_name == "Maria Gonzalez"
    assert e.message_id == "<cafe0001@mail.example.com>"
    assert "#A1001" in e.body


def test_parse_html_only_email_and_encoded_subject():
    e = parse_email((SAMPLES / "03_damaged_angry.eml").read_bytes())
    assert e.subject == "DAMAGED table — unacceptable"
    assert "<div>" not in e.body and "cracked" in e.body and "A1003" in e.body


def test_parse_generates_message_id_when_missing():
    e = parse_email(b"From: a@b.com\nSubject: hi\n\nhello")
    assert e.message_id.startswith("<generated-") and e.body == "hello"


# ------------------------------------------------------------------ skip rules


@pytest.mark.parametrize(
    "headers,sender,expect_skip",
    [
        ({}, "customer@example.com", False),
        ({"Auto-Submitted": "auto-replied"}, "customer@example.com", True),
        ({"Precedence": "bulk"}, "customer@example.com", True),
        ({"List-Unsubscribe": "<mailto:x>"}, "news@shop.com", True),
        ({}, "no-reply@carrier.com", True),
        ({}, "MAILER-DAEMON@mx.example.com", True),
        ({}, "support@acmehomegoods.example", True),
    ],
)
def test_should_skip(settings, headers, sender, expect_skip):
    e = InboundEmail(message_id="<x>", sender_email=sender, sender_name="", subject="", body="", headers=headers)
    assert (should_skip(e, settings) is not None) == expect_skip


# ------------------------------------------------------------------ tools / privacy


def test_lookup_own_order_includes_day_counts(store):
    r = store.lookup("#a1001", "Maria.Gonzalez@example.com")
    assert r["found"] and r["order"]["tracking_number"] == "1Z999AA10123456784"
    assert r["order"]["days_since_shipped"] == 6


def test_lookup_other_customers_order_reveals_nothing(store):
    r = store.lookup("A1003", "alex.random@example.net")
    assert r["found"] is False
    blob = json.dumps(r)
    assert "Walnut" not in blob and "9400" not in blob and "priya" not in blob.lower()


def test_unknown_order_and_unknown_tool(store):
    assert store.lookup("ZZZ999", "maria.gonzalez@example.com")["found"] is False
    out, is_error = ToolExecutor(store, "maria.gonzalez@example.com").run("delete_everything", {})
    assert is_error and "Unknown tool" in out


def test_list_sender_orders_is_scoped(store):
    r = store.for_sender("james.okafor@example.com")
    assert r["count"] == 1 and r["orders"][0]["order_id"] == "A1002"


# ------------------------------------------------------------------ decision rules


def test_clean_reply_is_sent(settings):
    assert decide(details(), draft(), verification(), settings).action == "send"


@pytest.mark.parametrize(
    "d_over,r_over,v_over,fragment",
    [
        ({"primary_intent": "legal_or_chargeback"}, {}, {}, "legal_or_chargeback"),
        ({"secondary_intents": ["cancellation"]}, {}, {}, "cancellation"),
        ({"threatens_legal_action_or_chargeback": True}, {}, {}, "chargeback"),
        ({"refund_amount_requested": 649.0}, {}, {}, "exceeds auto limit"),
        ({}, {"needs_human": True, "needs_human_reason": "exception"}, {}, "drafter flagged"),
        ({}, {"confidence": 0.5}, {}, "confidence 0.50"),
        ({}, {}, {"approved": False}, "verifier rejected"),
        ({}, {}, {"unsupported_claims": ["arrives Friday"]}, "unsupported claim: arrives Friday"),
    ],
)
def test_risky_cases_are_held(settings, d_over, r_over, v_over, fragment):
    dec = decide(details(**d_over), draft(**r_over), verification(**v_over), settings)
    assert dec.action == "review"
    assert any(fragment in reason for reason in dec.reasons), dec.reasons


def test_auto_send_off_holds_everything(settings):
    settings.auto_send = False
    assert decide(details(), draft(), verification(), settings).action == "review"


def test_small_refund_within_limit_not_blocked_by_amount(settings):
    dec = decide(details(refund_amount_requested=40.0), draft(), verification(), settings)
    assert dec.action == "send"


# ------------------------------------------------------------------ end to end (folder transport)


def test_approved_reply_is_sent_threaded_and_marked_auto(settings, store):
    copy_sample(settings, "01_order_status.eml")
    agent = EmailAgent(settings, FakeLLM(), store, FolderMailbox(settings))
    [result] = agent.run_once()

    assert result.decision.action == "send"
    [sent] = list(settings.outbox_dir.glob("*.eml"))
    msg = message_from_bytes(sent.read_bytes(), policy=policy.default)
    assert msg["To"] == "Maria Gonzalez <maria.gonzalez@example.com>"
    assert msg["In-Reply-To"] == "<cafe0001@mail.example.com>"
    assert msg["Auto-Submitted"] == "auto-replied"
    assert "shipped via UPS" in msg.get_content()
    assert (settings.inbox_dir / "processed" / "01_order_status.eml").exists()
    audit = [json.loads(l) for l in settings.audit_log_path.read_text().splitlines()]
    assert audit[0]["action"] == "send" and audit[0]["tool_calls"][0]["tool"] == "lookup_order"


def test_rejected_reply_is_held_with_draft_and_reasons(settings, store):
    copy_sample(settings, "01_order_status.eml")
    llm = FakeLLM(v=verification(approved=False, unsupported_claims=["arrives Saturday"]))
    [result] = EmailAgent(settings, llm, store, FolderMailbox(settings)).run_once()

    assert result.decision.action == "review"
    assert not settings.outbox_dir.exists() or not list(settings.outbox_dir.iterdir())
    [review_json] = list(settings.review_dir.glob("*.json"))
    data = json.loads(review_json.read_text())
    assert "unsupported claim: arrives Saturday" in data["reasons"]
    assert review_json.with_suffix(".eml").exists()  # ready to edit + approve


def test_auto_reply_is_skipped_without_calling_claude(settings, store):
    copy_sample(settings, "07_out_of_office.eml")
    llm = FakeLLM()
    [result] = EmailAgent(settings, llm, store, FolderMailbox(settings)).run_once()
    assert result.decision.action == "skip" and llm.calls == []


def test_model_failure_goes_to_review_never_sends(settings, store):
    copy_sample(settings, "01_order_status.eml")
    llm = FakeLLM(raise_on="verify", exc=LLMError("declined"))
    [result] = EmailAgent(settings, llm, store, FolderMailbox(settings)).run_once()
    assert result.decision.action == "review" and "pipeline error: declined" in result.decision.reasons
    assert not settings.outbox_dir.exists() or not list(settings.outbox_dir.iterdir())


def test_transient_api_error_leaves_email_for_retry(settings, store):
    copy_sample(settings, "01_order_status.eml")
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    llm = FakeLLM(raise_on="extract", exc=anthropic.APIConnectionError(request=request))
    assert EmailAgent(settings, llm, store, FolderMailbox(settings)).run_once() == []
    assert (settings.inbox_dir / "01_order_status.eml").exists()
    assert not settings.processed_ids_path.exists() or not settings.processed_ids_path.read_text().strip()


def test_same_message_is_never_answered_twice(settings, store):
    copy_sample(settings, "01_order_status.eml")
    EmailAgent(settings, FakeLLM(), store, FolderMailbox(settings)).run_once()
    copy_sample(settings, "01_order_status.eml")  # same Message-ID shows up again
    llm = FakeLLM()
    assert EmailAgent(settings, llm, store, FolderMailbox(settings)).run_once() == []
    assert llm.calls == [] and len(list(settings.outbox_dir.iterdir())) == 1


# ------------------------------------------------------------------ ClaudeLLM against a fake API client


def _text(s):
    return SimpleNamespace(type="text", text=s)


def _resp(content, stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=stop_details)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _llm(client):
    return ClaudeLLM("claude-opus-5-5", "Acme", "Acme Team", "POLICY TEXT", client=client, today=TODAY)


def test_draft_runs_tool_loop_then_parses_reply(store):
    tool_call = SimpleNamespace(type="tool_use", id="tu_1", name="lookup_order", input={"order_id": "A1001"})
    client = FakeClient([_resp([tool_call], stop_reason="tool_use"), _resp([_text(draft().model_dump_json())])])
    executor = ToolExecutor(store, "maria.gonzalez@example.com")
    email = parse_email((SAMPLES / "01_order_status.eml").read_bytes())

    result = _llm(client).draft(email, details(), executor)

    assert result.subject == "Re: where is my lamp??"
    second = client.requests[1]
    tool_result = second["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tu_1" and "1Z999AA10123456784" in tool_result["content"]
    assert second["fallbacks"] == "default" and second["betas"] == ["server-side-fallback-2026-07-01"]
    assert second["output_config"]["format"]["type"] == "json_schema"
    assert "POLICY TEXT" in second["system"][0]["text"]
    assert "Today's date: 2026-11-02" in second["messages"][0]["content"]


def test_refusal_raises_llm_error():
    client = FakeClient([_resp([], stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))])
    email = InboundEmail(message_id="<x>", sender_email="a@b.com", sender_name="", subject="s", body="b")
    with pytest.raises(LLMError, match="declined"):
        _llm(client).extract(email)


def test_garbage_output_raises_llm_error():
    client = FakeClient([_resp([_text("not json")])])
    email = InboundEmail(message_id="<x>", sender_email="a@b.com", sender_name="", subject="s", body="b")
    with pytest.raises(LLMError, match="Could not parse"):
        _llm(client).extract(email)


def test_schemas_are_strict():
    for model in (ExtractedDetails, DraftReply, Verification):
        schema = strict_json_schema(model)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_add_business_days_skips_weekends(store):
    ex = ToolExecutor(store, "maria.gonzalez@example.com")
    out, err = ex.run("add_business_days", {"start_date": "2026-10-27", "business_days": 7})  # Tue + 7 bd
    assert not err and json.loads(out)["result_date"] == "2026-11-05 (Thursday)"
    out, err = ex.run("add_business_days", {"start_date": "2026-10-30", "business_days": 1})  # Fri + 1 bd
    assert json.loads(out)["result_date"] == "2026-11-02 (Monday)"
    _, err = ex.run("add_business_days", {"start_date": "2026-10-30", "business_days": 999})
    assert err
