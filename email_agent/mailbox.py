"""Reading inbound mail and sending replies.

FolderMailbox  - .eml/.txt files in a folder; "sent" replies are written to an outbox folder.
                 Use it to test safely, or to plug into any system that can drop files.
ImapSmtpMailbox - a real mailbox (Gmail, Outlook, Fastmail...) over IMAP + SMTP.
"""

from __future__ import annotations

import hashlib
import imaplib
import json
import re
import shutil
import smtplib
import urllib.request
from datetime import datetime, timezone
from email import message_from_bytes, policy
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from html import unescape
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from .config import Settings
from .schemas import DraftReply, InboundEmail

# ---------------------------------------------------------------- parsing


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", "", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", html)
    text = unescape(re.sub(r"<[^>]+>", "", html))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_email(raw: bytes, source_ref: str = "") -> InboundEmail:
    msg = message_from_bytes(raw, policy=policy.default)
    sender_name, sender_email = parseaddr(str(msg.get("Reply-To") or msg.get("From") or ""))
    if not sender_name:
        sender_name = parseaddr(str(msg.get("From") or ""))[0]

    body = ""
    part = msg.get_body(preferencelist=("plain",))
    if part is not None:
        body = part.get_content()
    else:
        part = msg.get_body(preferencelist=("html",))
        if part is not None:
            body = _html_to_text(part.get_content())
    if not body and not msg.is_multipart():
        body = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="replace")

    message_id = str(msg.get("Message-ID") or "").strip()
    if not message_id:
        digest = hashlib.sha256(raw).hexdigest()[:16]
        message_id = f"<generated-{digest}@email-agent>"

    return InboundEmail(
        message_id=message_id,
        sender_email=sender_email.strip(),
        sender_name=sender_name.strip(),
        subject=str(msg.get("Subject") or "").strip(),
        body=body.strip(),
        received_at=str(msg.get("Date") or ""),
        references=str(msg.get("References") or "").strip(),
        headers={k: str(v) for k, v in msg.items()},
        source_ref=source_ref,
    )


def build_reply(email: InboundEmail, draft: DraftReply, settings: Settings) -> EmailMessage:
    reply = EmailMessage()
    reply["From"] = f"{settings.company_name} Support <{settings.support_address}>"
    reply["To"] = f"{email.sender_name} <{email.sender_email}>" if email.sender_name else email.sender_email
    subject = draft.subject.strip() or f"Re: {email.subject}"
    reply["Subject"] = subject
    reply["Date"] = formatdate(localtime=True)
    reply["Message-ID"] = make_msgid(domain=settings.support_address.split("@")[-1])
    # Threading: keeps the reply in the customer's existing conversation.
    if not email.message_id.startswith("<generated-"):
        reply["In-Reply-To"] = email.message_id
        reply["References"] = f"{email.references} {email.message_id}".strip()
    # RFC 3834: tells other auto-responders not to answer us (prevents mail loops).
    reply["Auto-Submitted"] = "auto-replied"
    reply.set_content(draft.body)
    return reply


def _safe_name(message_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", message_id.strip("<>"))[:80]
    return f"{stem}-{hashlib.sha256(message_id.encode()).hexdigest()[:8]}"


def write_review_file(
    review_dir: Path, email: InboundEmail, payload: dict[str, Any], reply: EmailMessage | None
) -> Path:
    """Save everything a person needs to review (and optionally approve) a held reply."""
    review_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_name(email.message_id)
    if reply is not None:
        (review_dir / f"{name}.eml").write_bytes(bytes(reply))
    path = review_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------- transports


class Mailbox(Protocol):
    def fetch(self) -> list[InboundEmail]: ...
    def send(self, reply: EmailMessage, email: InboundEmail) -> None: ...
    def flag_for_review(self, email: InboundEmail) -> None: ...
    def mark_done(self, email: InboundEmail) -> None: ...


class FolderMailbox:
    def __init__(self, settings: Settings):
        self.inbox = settings.inbox_dir
        self.outbox = settings.outbox_dir
        self.processed = settings.inbox_dir / "processed"

    def fetch(self) -> list[InboundEmail]:
        if not self.inbox.exists():
            return []
        files = sorted(p for p in self.inbox.iterdir() if p.is_file() and p.suffix.lower() in {".eml", ".txt"})
        return [parse_email(p.read_bytes(), source_ref=str(p)) for p in files]

    def send(self, reply: EmailMessage, email: InboundEmail) -> None:
        self.outbox.mkdir(parents=True, exist_ok=True)
        (self.outbox / f"{_safe_name(email.message_id)}.eml").write_bytes(bytes(reply))

    def flag_for_review(self, email: InboundEmail) -> None:
        pass  # the review file written by the agent is the flag

    def mark_done(self, email: InboundEmail) -> None:
        src = Path(email.source_ref)
        if src.exists():
            self.processed.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), self.processed / src.name)


class MailpitMailbox:
    """Local test mail server. Reads via Mailpit's REST API, sends via its SMTP port.

    Only unread mail addressed to SUPPORT_ADDRESS is picked up, so the agent's own replies
    (addressed to customers) are never read back in. Handled mail is marked read and tagged
    "auto-replied", "needs-review" or "skipped" so you can filter for it in the Mailpit UI.
    """

    def __init__(self, settings: Settings):
        self.s = settings
        self.api = settings.mailpit_api_url.rstrip("/")
        self._tags: dict[str, str] = {}

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> bytes:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.api}{path}", data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def fetch(self) -> list[InboundEmail]:
        query = quote(f"is:unread to:{self.s.support_address}")
        emails = []
        start = 0
        while True:
            page = json.loads(self._request("GET", f"/api/v1/search?query={query}&start={start}&limit=100"))
            messages = page.get("messages") or []
            for m in messages:
                raw = self._request("GET", f"/api/v1/message/{m['ID']}/raw")
                emails.append(parse_email(raw, source_ref=m["ID"]))
            start += len(messages)
            if not messages or start >= page.get("total", 0):
                break
        emails.reverse()  # search returns newest first; answer oldest first
        return emails

    def send(self, reply: EmailMessage, email: InboundEmail) -> None:
        with smtplib.SMTP(self.s.mailpit_smtp_host, self.s.mailpit_smtp_port) as server:
            server.send_message(reply)
        if email.source_ref:
            self._tags[email.source_ref] = "auto-replied"

    def flag_for_review(self, email: InboundEmail) -> None:
        if email.source_ref:
            self._tags[email.source_ref] = "needs-review"

    def mark_done(self, email: InboundEmail) -> None:
        msg_id = email.source_ref
        if not msg_id:
            return  # never send an empty ID list: Mailpit would apply it to the whole mailbox
        tag = self._tags.pop(msg_id, "skipped")
        self._request("PUT", "/api/v1/tags", {"Tags": [tag], "IDs": [msg_id]})
        self._request("PUT", "/api/v1/messages", {"Read": True, "IDs": [msg_id]})

    def deliver(self, raw: bytes) -> None:
        """Inject a customer email into Mailpit (used by the `seed` command)."""
        msg = message_from_bytes(raw, policy=policy.default)
        with smtplib.SMTP(self.s.mailpit_smtp_host, self.s.mailpit_smtp_port) as server:
            server.send_message(msg)


class ImapSmtpMailbox:
    """Fetches UNSEEN mail; marks a message \\Seen only after it has been replied to or held.

    Messages are read with BODY.PEEK so a crash mid-run leaves them unread for the next run.
    """

    def __init__(self, settings: Settings):
        missing = [k for k in ("imap_host", "imap_user", "imap_password", "smtp_host") if not getattr(settings, k)]
        if missing:
            raise ValueError(f"IMAP/SMTP mode needs these settings: {', '.join(m.upper() for m in missing)}")
        self.s = settings
        self.imap = imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port)
        self.imap.login(settings.imap_user, settings.imap_password)
        self.imap.select(settings.imap_folder)

    def fetch(self) -> list[InboundEmail]:
        status, data = self.imap.uid("SEARCH", None, "UNSEEN")
        if status != "OK":
            raise RuntimeError(f"IMAP search failed: {data}")
        emails = []
        for uid in data[0].split():
            status, msg_data = self.imap.uid("FETCH", uid, "(BODY.PEEK[])")
            if status != "OK" or not msg_data or msg_data[0] is None:
                continue
            emails.append(parse_email(msg_data[0][1], source_ref=uid.decode()))
        return emails

    def send(self, reply: EmailMessage, email: InboundEmail) -> None:
        if self.s.smtp_port == 465:
            server: smtplib.SMTP = smtplib.SMTP_SSL(self.s.smtp_host, self.s.smtp_port)
        else:
            server = smtplib.SMTP(self.s.smtp_host, self.s.smtp_port)
            server.starttls()
        with server:
            server.login(self.s.smtp_user, self.s.smtp_password)
            server.send_message(reply)

    def flag_for_review(self, email: InboundEmail) -> None:
        folder = self.s.imap_review_folder
        self.imap.create(folder)  # harmless "already exists" error if present
        self.imap.uid("COPY", email.source_ref, folder)
        self.imap.uid("STORE", email.source_ref, "+FLAGS", r"(\Flagged)")

    def mark_done(self, email: InboundEmail) -> None:
        self.imap.uid("STORE", email.source_ref, "+FLAGS", r"(\Seen)")

    def close(self) -> None:
        try:
            self.imap.logout()
        except Exception:
            pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
