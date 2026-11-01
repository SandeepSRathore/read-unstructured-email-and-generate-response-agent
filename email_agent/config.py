"""Runtime settings, read from environment variables (and a local .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class Settings:
    # --- Model ---
    # "openai" or "anthropic"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").strip().lower())
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"))

    # --- Where mail comes from: "folder", "mailpit", or "imap" ---
    mail_source: str = field(default_factory=lambda: os.getenv("MAIL_SOURCE", "folder").strip().lower())

    # --- Business identity used in replies ---
    company_name: str = field(default_factory=lambda: os.getenv("COMPANY_NAME", "Acme Home Goods"))
    agent_signature: str = field(
        default_factory=lambda: os.getenv("AGENT_SIGNATURE", "The Acme Home Goods Support Team")
    )
    support_address: str = field(
        default_factory=lambda: os.getenv("SUPPORT_ADDRESS", "support@acmehomegoods.example")
    )

    # --- Grounding data ---
    policies_path: Path = field(default_factory=lambda: _path("POLICIES_PATH", "data/policies.md"))
    orders_path: Path = field(default_factory=lambda: _path("ORDERS_PATH", "data/orders.json"))

    # --- Decision rules ---
    # When False, every reply is held for review even if it passes all checks.
    auto_send: bool = field(default_factory=lambda: _bool("AUTO_SEND", True))
    min_confidence: float = field(default_factory=lambda: float(os.getenv("MIN_CONFIDENCE", "0.8")))
    # Refund/credit requests above this amount always go to a person.
    max_auto_refund: float = field(default_factory=lambda: float(os.getenv("MAX_AUTO_REFUND", "100")))

    # --- Folder transport (local testing) ---
    inbox_dir: Path = field(default_factory=lambda: _path("INBOX_DIR", "inbox"))
    outbox_dir: Path = field(default_factory=lambda: _path("OUTBOX_DIR", "outbox"))
    review_dir: Path = field(default_factory=lambda: _path("REVIEW_DIR", "review"))
    state_dir: Path = field(default_factory=lambda: _path("STATE_DIR", "state"))

    # --- Mailpit transport (local test mail server: https://mailpit.axllent.org) ---
    mailpit_api_url: str = field(default_factory=lambda: os.getenv("MAILPIT_API_URL", "http://localhost:8025"))
    mailpit_smtp_host: str = field(default_factory=lambda: os.getenv("MAILPIT_SMTP_HOST", "localhost"))
    mailpit_smtp_port: int = field(default_factory=lambda: int(os.getenv("MAILPIT_SMTP_PORT", "1025")))

    # --- IMAP/SMTP transport (real mailbox) ---
    imap_host: str = field(default_factory=lambda: os.getenv("IMAP_HOST", ""))
    imap_port: int = field(default_factory=lambda: int(os.getenv("IMAP_PORT", "993")))
    imap_user: str = field(default_factory=lambda: os.getenv("IMAP_USER", ""))
    imap_password: str = field(default_factory=lambda: os.getenv("IMAP_PASSWORD", ""))
    imap_folder: str = field(default_factory=lambda: os.getenv("IMAP_FOLDER", "INBOX"))
    # Mailbox folder that held-for-review emails are copied to (created if missing).
    imap_review_folder: str = field(default_factory=lambda: os.getenv("IMAP_REVIEW_FOLDER", "Needs-Review"))
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", "") or os.getenv("IMAP_USER", ""))
    smtp_password: str = field(
        default_factory=lambda: os.getenv("SMTP_PASSWORD", "") or os.getenv("IMAP_PASSWORD", "")
    )

    @property
    def audit_log_path(self) -> Path:
        return self.state_dir / "audit.jsonl"

    @property
    def processed_ids_path(self) -> Path:
        return self.state_dir / "processed_ids.txt"
