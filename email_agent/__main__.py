"""Command line entry point.

    python -m email_agent seed                       # send the sample inbox/*.eml emails into Mailpit
    python -m email_agent run --source mailpit       # answer unread mail in Mailpit once
    python -m email_agent run --source mailpit --watch 30   # keep polling every 30 seconds
    python -m email_agent run --source folder        # process the local inbox/ folder once
    python -m email_agent run --source imap          # process unread mail in a real mailbox
    python -m email_agent run --review-only          # draft everything, send nothing
    python -m email_agent approve review/<file>.json # send a held reply (edit its .eml first if needed)

Defaults for --source and the model provider come from MAIL_SOURCE and LLM_PROVIDER (see .env.example).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from email import message_from_bytes, policy
from pathlib import Path

import anthropic
import openai

from .agent import EmailAgent
from .config import Settings
from .mailbox import FolderMailbox, ImapSmtpMailbox, MailpitMailbox
from .schemas import InboundEmail
from .tools import OrderStore

SOURCES = ["folder", "mailpit", "imap"]


def _mailbox(source: str, settings: Settings):
    if source == "imap":
        return ImapSmtpMailbox(settings)
    if source == "mailpit":
        return MailpitMailbox(settings)
    return FolderMailbox(settings)


def _llm(settings: Settings):
    policies = settings.policies_path.read_text(encoding="utf-8")
    common = dict(company_name=settings.company_name, signature=settings.agent_signature, policies=policies)
    if settings.llm_provider == "anthropic":
        from .llm import ClaudeLLM

        return ClaudeLLM(model=settings.anthropic_model, **common)
    if settings.llm_provider == "openai":
        from .llm_openai import OpenAILLM

        return OpenAILLM(model=settings.openai_model, **common)
    raise SystemExit(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (use 'openai' or 'anthropic')")


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    if args.review_only:
        settings.auto_send = False
    llm = _llm(settings)
    store = OrderStore(settings.orders_path)
    model = settings.openai_model if settings.llm_provider == "openai" else settings.anthropic_model
    logging.info("Using %s (%s), mail source: %s", settings.llm_provider, model, args.source)

    while True:
        mailbox = _mailbox(args.source, settings)
        try:
            results = EmailAgent(settings, llm, store, mailbox).run_once()
        finally:
            if hasattr(mailbox, "close"):
                mailbox.close()
        counts = {a: sum(r.decision.action == a for r in results) for a in ("send", "review", "skip")}
        logging.info(
            "Done: %d sent, %d held for review, %d skipped", counts["send"], counts["review"], counts["skip"]
        )
        if not args.watch:
            return 0
        time.sleep(args.watch)


def cmd_seed(args: argparse.Namespace, settings: Settings) -> int:
    mailbox = MailpitMailbox(settings)
    files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in {".eml", ".txt"})
    for p in files:
        mailbox.deliver(p.read_bytes())
        print(f"Delivered {p.name}")
    print(f"\n{len(files)} emails sent to Mailpit. Open {settings.mailpit_api_url} to see them.")
    return 0


def cmd_approve(args: argparse.Namespace, settings: Settings) -> int:
    review_json = Path(args.review_file)
    eml = review_json.with_suffix(".eml")
    if not eml.exists():
        print(f"No drafted reply at {eml}. Write one there (or reply manually).", file=sys.stderr)
        return 1
    reply = message_from_bytes(eml.read_bytes(), policy=policy.default)
    data = json.loads(review_json.read_text(encoding="utf-8"))
    original = InboundEmail(
        message_id=data["message_id"], sender_email="", sender_name="", subject=data["subject"], body=""
    )
    mailbox = _mailbox(args.source, settings)
    try:
        mailbox.send(reply, original)
    finally:
        if hasattr(mailbox, "close"):
            mailbox.close()
    review_json.rename(review_json.with_suffix(".approved.json"))
    eml.rename(eml.with_suffix(".approved.eml"))
    print(f"Sent reply to {reply['To']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(prog="email_agent", description="Customer email auto-responder")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="Process inbound email")
    run.add_argument("--source", choices=SOURCES, default=settings.mail_source)
    run.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="Poll forever at this interval")
    run.add_argument("--review-only", action="store_true", help="Hold every reply for review; send nothing")

    seed = sub.add_parser("seed", parents=[common], help="Send sample customer emails into Mailpit")
    seed.add_argument("folder", nargs="?", default=str(settings.inbox_dir))

    approve = sub.add_parser("approve", parents=[common], help="Send a reply that was held for review")
    approve.add_argument("review_file")
    approve.add_argument("--source", choices=SOURCES, default=settings.mail_source)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    commands = {"run": cmd_run, "seed": cmd_seed, "approve": cmd_approve}
    try:
        return commands[args.command](args, settings)
    except ConnectionRefusedError:
        print(
            "Could not connect to the mail server. If you're using Mailpit, start it with: mailpit",
            file=sys.stderr,
        )
    except OSError as e:
        if "Connection refused" not in str(e):
            raise
        print("Could not reach Mailpit's API. Start it with: mailpit", file=sys.stderr)
    except (anthropic.AuthenticationError, openai.AuthenticationError):
        print("The model provider rejected the API key. Check OPENAI_API_KEY / ANTHROPIC_API_KEY.", file=sys.stderr)
    except openai.OpenAIError as e:
        if "api_key" not in str(e):
            raise
        print("No OpenAI API key found. Set OPENAI_API_KEY (see .env.example).", file=sys.stderr)
    except TypeError as e:
        if "authentication" not in str(e):
            raise
        print("No Anthropic credentials found. Set ANTHROPIC_API_KEY (see .env.example).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
