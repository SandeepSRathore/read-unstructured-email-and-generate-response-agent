"""Read-only business-data tools the drafting model can call.

Every lookup is scoped to the email's sender: a customer can only ever learn about
orders placed with their own email address, no matter what the email text asks for.
Swap OrderStore for your real order system (Shopify, a DB, an internal API) by keeping
the same two methods.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "lookup_order",
        "description": (
            "Look up one order by its order ID and return its status, items, totals, shipping "
            "and tracking details, and day counts since shipping/delivery. Only orders that belong "
            "to the email's sender are returned. Call this for every order ID the customer mentions."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID as written by the customer, e.g. A1001 or #A1001."}
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_sender_orders",
        "description": (
            "List all orders placed with the sender's email address, most recent first. Use this when "
            "the customer refers to an order without giving its ID, or to confirm which order they mean."
        ),
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "add_business_days",
        "description": (
            "Return the date that is N business days (Monday-Friday, excluding weekends) after a start date. "
            "Use it for every delivery-window or deadline calculation, e.g. 'arrives 3-7 business days after "
            "shipping' -> call it with N=3 and N=7."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "ISO date, YYYY-MM-DD."},
                "business_days": {"type": "integer", "description": "Number of business days to add (0-60)."},
            },
            "required": ["start_date", "business_days"],
            "additionalProperties": False,
        },
    },
]


def add_business_days(start: date, n: int) -> date:
    if not 0 <= n <= 60:
        raise ValueError("business_days must be between 0 and 60")
    d = start
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def normalize_order_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", raw).upper()


class OrderStore:
    def __init__(self, path: Path, today: date | None = None):
        self.path = path
        self.today = today or date.today()
        with open(path, encoding="utf-8") as f:
            orders = json.load(f)
        self._by_id = {normalize_order_id(o["order_id"]): o for o in orders}

    def _enrich(self, order: dict[str, Any]) -> dict[str, Any]:
        out = dict(order)
        for key in ("shipped_at", "delivered_at", "placed_at"):
            if out.get(key):
                days = (self.today - date.fromisoformat(out[key][:10])).days
                out[f"days_since_{key.removesuffix('_at')}"] = days
        return out

    def lookup(self, order_id: str, sender_email: str) -> dict[str, Any]:
        order = self._by_id.get(normalize_order_id(order_id))
        if order is None:
            return {"found": False, "message": f"No order exists with ID '{order_id}'."}
        if order["customer_email"].strip().lower() != sender_email.strip().lower():
            return {
                "found": False,
                "message": (
                    f"Order '{order_id}' is not associated with the sender's email address. Do not "
                    "confirm or deny that it exists and do not share any details. Ask the customer to "
                    "write from the email address used at checkout."
                ),
            }
        return {"found": True, "order": self._enrich(order)}

    def for_sender(self, sender_email: str) -> dict[str, Any]:
        email = sender_email.strip().lower()
        orders = [self._enrich(o) for o in self._by_id.values() if o["customer_email"].strip().lower() == email]
        orders.sort(key=lambda o: o.get("placed_at", ""), reverse=True)
        return {"count": len(orders), "orders": orders}


class ToolExecutor:
    """Runs tool calls for one email, bound to that email's sender, and logs every call."""

    def __init__(self, store: OrderStore, sender_email: str):
        self.store = store
        self.sender_email = sender_email
        self.log: list[dict[str, Any]] = []

    def run(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Returns (json_result, is_error)."""
        try:
            if name == "lookup_order":
                result = self.store.lookup(str(tool_input["order_id"]), self.sender_email)
            elif name == "list_sender_orders":
                result = self.store.for_sender(self.sender_email)
            elif name == "add_business_days":
                start = date.fromisoformat(str(tool_input["start_date"])[:10])
                n = int(tool_input["business_days"])
                end = add_business_days(start, n)
                result = {
                    "start_date": f"{start.isoformat()} ({start:%A})",
                    "business_days": n,
                    "result_date": f"{end.isoformat()} ({end:%A})",
                }
            else:
                raise ValueError(f"Unknown tool '{name}'")
        except Exception as e:  # report the failure to the model rather than crashing the email
            self.log.append({"tool": name, "input": tool_input, "error": str(e)})
            return f"Error: {e}", True
        self.log.append({"tool": name, "input": tool_input, "result": result})
        return json.dumps(result, sort_keys=True), False
