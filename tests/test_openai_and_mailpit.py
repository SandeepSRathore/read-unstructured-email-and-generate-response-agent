from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from email_agent.llm import LLMError
from email_agent.llm_openai import OPENAI_TOOLS, OpenAILLM
from email_agent.mailbox import MailpitMailbox
from email_agent.schemas import InboundEmail
from email_agent.tools import ToolExecutor

from .test_agent import SAMPLES, TODAY, details, draft, settings, store  # noqa: F401  (fixtures)
from email_agent.mailbox import parse_email


class Item(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


def _resp(output, text="", status="completed"):
    return SimpleNamespace(status=status, output=output, output_text=text, incomplete_details=None)


class FakeOpenAI:
    def __init__(self, queued):
        self._queue = list(queued)
        self.requests: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        return self._queue.pop(0)


def _llm(client):
    return OpenAILLM("gpt-6.1-sol", "Acme", "Acme Team", "POLICY TEXT", client=client, today=TODAY)


def test_openai_tools_are_strict_function_tools():
    assert {t["name"] for t in OPENAI_TOOLS} == {"lookup_order", "list_sender_orders", "add_business_days"}
    assert all(t["type"] == "function" and t["strict"] is True for t in OPENAI_TOOLS)


def test_openai_draft_runs_tool_loop(store):  # noqa: F811
    reasoning = Item(type="reasoning", id="rs_1", encrypted_content="xyz", summary=[])
    call = Item(type="function_call", id="fc_1", call_id="call_1", name="lookup_order",
                arguments='{"order_id": "A1001"}', status="completed")
    client = FakeOpenAI([_resp([reasoning, call]), _resp([Item(type="message", content=[])], draft().model_dump_json())])
    email = parse_email((SAMPLES / "01_order_status.eml").read_bytes())

    result = _llm(client).draft(email, details(), ToolExecutor(store, "maria.gonzalez@example.com"))

    assert result.subject == "Re: where is my lamp??"
    second = client.requests[1]
    replayed_types = [i.get("type") for i in second["input"]]
    assert replayed_types[1:] == ["reasoning", "function_call", "function_call_output"]
    out = second["input"][-1]
    assert out["call_id"] == "call_1" and "1Z999AA10123456784" in out["output"]
    assert second["text"]["format"]["strict"] is True and second["store"] is False
    assert second["reasoning"] == {"effort": "high"} and "POLICY TEXT" in second["instructions"]


def test_openai_refusal_and_incomplete_raise():
    email = InboundEmail(message_id="<x>", sender_email="a@b.com", sender_name="", subject="s", body="b")
    refusal = Item(type="message", content=[SimpleNamespace(type="refusal", refusal="no")])
    with pytest.raises(LLMError, match="declined"):
        _llm(FakeOpenAI([_resp([refusal])])).extract(email)
    with pytest.raises(LLMError, match="incomplete"):
        _llm(FakeOpenAI([_resp([], status="incomplete")])).extract(email)


def test_openai_bad_tool_arguments_are_reported_not_crashing(store):  # noqa: F811
    call = Item(type="function_call", call_id="c1", name="lookup_order", arguments="{not json")
    client = FakeOpenAI([_resp([call]), _resp([], draft().model_dump_json())])
    email = InboundEmail(message_id="<x>", sender_email="maria.gonzalez@example.com", sender_name="", subject="s", body="b")
    _llm(client).draft(email, details(), ToolExecutor(store, email.sender_email))
    assert client.requests[1]["input"][-1]["output"].startswith("Error: arguments were not valid JSON")


# ---------------------------------------------------------------- Mailpit transport


class RecordingMailpit(MailpitMailbox):
    def __init__(self, settings, search_pages, raws):
        super().__init__(settings)
        self.search_pages, self.raws, self.calls = list(search_pages), raws, []

    def _request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path.startswith("/api/v1/search"):
            return json.dumps(self.search_pages.pop(0)).encode()
        if path.endswith("/raw"):
            return self.raws[path.split("/")[4]]
        return b"ok"


def test_mailpit_fetch_filters_to_support_inbox_and_returns_oldest_first(settings):  # noqa: F811
    raws = {"new": (SAMPLES / "02_return.eml").read_bytes(), "old": (SAMPLES / "01_order_status.eml").read_bytes()}
    mb = RecordingMailpit(settings, [{"total": 2, "messages": [{"ID": "new"}, {"ID": "old"}]}], raws)
    emails = mb.fetch()
    assert [e.source_ref for e in emails] == ["old", "new"]
    assert "is%3Aunread%20to%3Asupport%40acmehomegoods.example" in mb.calls[0][1]


def test_mailpit_mark_done_tags_and_marks_read_only_that_message(settings):  # noqa: F811
    mb = RecordingMailpit(settings, [], {})
    e = InboundEmail(message_id="<x>", sender_email="a@b.com", sender_name="", subject="", body="", source_ref="ID1")
    mb.flag_for_review(e)
    mb.mark_done(e)
    assert ("PUT", "/api/v1/tags", {"Tags": ["needs-review"], "IDs": ["ID1"]}) in mb.calls
    assert ("PUT", "/api/v1/messages", {"Read": True, "IDs": ["ID1"]}) in mb.calls


def test_mailpit_never_sends_empty_id_list(settings):  # noqa: F811
    mb = RecordingMailpit(settings, [], {})
    mb.mark_done(InboundEmail(message_id="<x>", sender_email="", sender_name="", subject="", body=""))
    assert mb.calls == []  # an empty IDs list would mark the WHOLE mailbox read
