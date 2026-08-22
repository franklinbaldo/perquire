"""Explicit OpenRouter provider-routing configuration.

OpenRouter automatically routes/fails over across upstream providers when a request
does not constrain the `provider` object. Scientific runs that require one stable
inference path use these helpers to make that request-body constraint explicit and
auditable.
"""

from __future__ import annotations

from typing import Any


def normalize_provider_order(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = list(value)
    order = tuple(str(item).strip() for item in raw if str(item).strip())
    if len(set(order)) != len(order):
        raise ValueError("provider_order must not contain duplicates")
    return order


def provider_routing(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the OpenRouter `provider` request object, if explicitly configured."""
    order = normalize_provider_order(config.get("provider_order"))
    if not order:
        return None
    return {
        "order": list(order),
        # A configured order is treated as a scientific pin by default: do not
        # silently escape to another provider unless a caller opts in explicitly.
        "allow_fallbacks": bool(config.get("allow_fallbacks", False)),
    }


def provider_extra_body(config: dict[str, Any]) -> dict[str, Any] | None:
    routing = provider_routing(config)
    return {"provider": routing} if routing is not None else None


def routing_identity(config: dict[str, Any]) -> str | None:
    routing = provider_routing(config)
    if routing is None:
        return None
    order = ",".join(routing["order"])
    fallbacks = "true" if routing["allow_fallbacks"] else "false"
    return f"provider_order={order};allow_fallbacks={fallbacks}"
