# Customer Email Agent

Reads messy, unstructured customer emails, extracts the key details, looks up the customer's
orders, writes a reply grounded in your policies, checks that reply for accuracy, and sends it
automatically when every check passes. Anything risky or uncertain is held for a person.

## How it works

```
inbound email
  │
  ├─ skip?        auto-replies, no-reply senders, newsletters, our own address  → ignored (no mail loops)
  │
  ├─ 1. extract   LLM → structured JSON: intent(s), order IDs, questions asked,
  │               refund amount, sentiment, legal threats, missing info
  │
  ├─ 2. draft     LLM + tools (lookup_order, list_sender_orders, add_business_days)
  │               writes the reply using ONLY your policy doc + order records,
  │               and lists every fact it used with its source
  │
  ├─ 3. verify    a separate LLM call checks every claim in the draft against
  │               the policy doc and the actual tool results, and checks that every
  │               customer question was answered
  │
  └─ 4. decide    deterministic rules (guardrails.py) → SEND or HOLD FOR REVIEW
```

A reply is auto-sent only if **all** of these hold:

- The verifier approved it: no unsupported claims, no unanswered questions, no tone or safety issues.
- The drafter's confidence is at least `MIN_CONFIDENCE`, and the drafter didn't flag it for a person.
- The intent isn't one that needs a human action the agent can't perform: refund, cancellation, address change.
- There's no legal threat or chargeback, and no refund request above `MAX_AUTO_REFUND`.

The other safeguards:

- **Privacy:** order lookups only return orders placed with the sender's own email address.
- **Prompt injection:** customer text is treated as data. Even if a model were fooled, the verifier and the rules above still have to pass.
- **Failures:** if a model call fails, the email is held for review. Temporary API errors leave it in place for the next run. Each Message-ID is answered at most once.
- **Threading:** replies thread correctly (`In-Reply-To`/`References`) and carry `Auto-Submitted: auto-replied`, so other auto-responders don't answer back.
- **Audit:** every decision goes to `state/audit.jsonl` with the extracted details, the draft, the verification and the tool calls.

Works with **OpenAI** (default, `LLM_PROVIDER=openai`) or **Anthropic Claude** (`LLM_PROVIDER=anthropic`).
Both backends use the same prompts, tools and checks.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set LLM_PROVIDER and its API key (OPENAI_API_KEY may already be in your shell)
brew install mailpit          # local test mail server (or: docker run -p 8025:8025 -p 1025:1025 axllent/mailpit)
```

## Try it with Mailpit (a fake mail server on your machine)

```bash
mailpit                                  # terminal 1: SMTP on :1025, web UI on http://localhost:8025
python -m email_agent seed               # terminal 2: deliver the 8 sample customer emails to Mailpit
python -m email_agent run                # answer them (MAIL_SOURCE=mailpit in .env)
python -m email_agent run --watch 30     # or keep polling
```

Open http://localhost:8025 to see the customer emails and the agent's replies, which thread
under the originals. Handled emails are marked read and tagged `auto-replied`,
`needs-review` or `skipped`. You can also send your own test email to
`support@acmehomegoods.example` through `localhost:1025` from any mail client or script.

The 8 samples in `inbox/` are: a messy order-status question, a return, an angry damaged-item
email with a chargeback threat, a cancellation, a stranger asking about someone else's order,
a prompt-injection attempt, an out-of-office auto-reply, and a pre-sales question.
Expected result: 4 sent (order status, return, stranger, pre-sales), 3 held for review
(chargeback, cancellation, prompt injection), and 1 skipped (out-of-office).

- Held replies go to `review/*.json`: the original email, the extracted details, the draft, the verifier's findings, and the reasons it was held. The drafted reply sits next to it as `review/*.eml`.
- To re-run from scratch: `curl -X DELETE localhost:8025/api/v1/messages; rm -rf review state/processed_ids.txt`, then seed again.

Review a held reply, edit its `.eml` if needed, then send it:

```bash
python -m email_agent approve review/<name>.json
```

Without Mailpit, `python -m email_agent run --source folder` reads `inbox/*.eml` and writes replies to `outbox/`.

## Connect a real mailbox

Fill in the IMAP/SMTP block in `.env`. For Gmail, use an App Password. Then:

```bash
python -m email_agent run --source imap --review-only   # start here: drafts everything, sends nothing
python -m email_agent run --source imap                 # auto-send replies that pass every check
python -m email_agent run --source imap --watch 60      # keep polling every 60s
```

In IMAP mode the agent reads UNSEEN mail without marking it read. Once an email has been handled
it is marked read. Held emails are also copied to the `Needs-Review` folder and flagged.

**Recommended rollout:** run with `--review-only` (or `AUTO_SEND=false`) on real traffic for a
while, read the drafts in `review/`, and tune `data/policies.md` until you trust them before
turning on auto-send.

## Make it yours

| What | Where |
|---|---|
| Your policies, FAQs, shipping/return rules | `data/policies.md`: the agent can only state facts written here |
| Your order system | `data/orders.json`, or replace `OrderStore` in `email_agent/tools.py` with calls to Shopify / your DB / your API (keep the sender-scoping) |
| Auto-send rules | `email_agent/guardrails.py` and `MIN_CONFIDENCE` / `MAX_AUTO_REFUND` in `.env` |
| Intents extracted | `Intent` in `email_agent/schemas.py` |
| Tone and reply style | `DRAFT_SYSTEM` in `email_agent/llm.py` |

## Tests

```bash
python -m pytest -q
```

The tests stub out the model (both providers) and Mailpit, so they need no API key or server. They cover email parsing (HTML-only, encoded
subjects), the skip and loop guards, order privacy, every review trigger, the end-to-end
send/hold/retry/dedupe flow, and the tool-calling loop against a fake API client.

## Cost and model

Each email takes three model calls (extract, draft, verify). The drafting call can add a turn or
two for order lookups and date calculations.

- **OpenAI** (default): `OPENAI_MODEL=gpt-6-luna`, the cheapest option ($0.10 input / $0.50 output
  per 1M tokens). Use `gpt-6.1-sol` for more capability, or `gpt-6-astra` for the most. It uses the Responses API with strict JSON-schema output, and
  `store=False`, so nothing is kept on OpenAI's side.
- **Anthropic**: `LLM_PROVIDER=anthropic`, `ANTHROPIC_MODEL=claude-opus-5-5`. System prompts,
  including your policy doc, are prompt-cached, and refusals retry server-side on a fallback model.
