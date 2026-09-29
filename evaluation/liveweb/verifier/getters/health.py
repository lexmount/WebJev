"""Page health gate -- judges the environment, never the task.

An anti-bot wall, a network error page or an empty page at evidence time is the environment's failure, not the
agent's. Such an episode goes to the `infra` bucket (excluded from the success-rate denominator and retried) instead of
being scored 0.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["get_page_health", "BLOCK_STRONG", "BLOCK_WEAK"]


# Verification / blocking phrases. Bare "captcha" and bare "429" are deliberately absent: normal pages say "protected
# by reCAPTCHA" in their footer and contain "429" in ordinary text. The \u escapes are Chinese-language
# verification prompts.
BLOCK_STRONG = re.compile(
    r"you have been blocked|performing security verification"
    r"|checking your browser before accessing|enable javascript and cookies to continue"
    r"|attention required!|automated access is not permitted"
    r"|our systems have detected unusual traffic"
    r"|is not available from your region or network"
    r"|you don't have permission to access|we're checking your connection"
    r"|just a quick check|why we occasionally ask for a quick verification"
    r"|verify you are human|verifying you are human|are you a human"
    r"|please re-puzzle|choose all the|access to this page has been denied"
    r"|request unsuccessful.*incapsula|incapsula incident|sorry, you have been blocked"
    r"|pardon our interruption|bot detection|human verification|prove you are human"
    r"|vercel security checkpoint|we're verifying your browser|verifying your browser"
    r"|\u4f7f\u7528\u8005\u9a57\u8b49|\u8bf7\u5b8c\u6210\u9a8c\u8bc1|\u8bf7\u8f93\u5165\u9a8c\u8bc1\u7801"
    r"|please check the box below"
    r"|too many requests|rate limit exceeded|error 429|http 429"
    r"|this site can.t be reached|this page isn.t working|took too long to respond",
    re.I)

# Weak signals: only a wall when the page is also short. On their own they can be part of a normal page.
BLOCK_WEAK = re.compile(
    r"ray id\s*:|just a moment|access denied|403 forbidden|404 not found"
    r"|reference #\d|errors\.edgesuite\.net|px-captcha|datadome|site not found"
    r"|service unavailable|error 5\d\d|not found on this server", re.I)

_SHORT = 2500        # length gate for weak signals
_MIN_CHARS = 1       # below this many characters the page counts as empty


def _main_document_status(events: Any) -> int | None:
    status = None
    for event in (events or []):
        if not isinstance(event, dict):
            continue
        kind = str(event.get("resourceType") or event.get("type") or "")
        if kind.lower() != "document":
            continue
        resp = event.get("response") if isinstance(event.get("response"), dict) else event
        code = resp.get("status", resp.get("statusCode"))
        if isinstance(code, int):
            status = code
    return status


def get_page_health(ctx: Any, config: dict) -> dict:
    """Returns {ok, http_status, blocked, empty, signals, chars}; `blocked` sends the episode to the infra bucket.

    Two conditions, because length alone misses walls (a 372-character "Performing security verification" page looks
    normal by length): the page is non-empty AND carries no verification copy.

    This getter never raises GetterError -- it is the one that reports "could not get it"; raising would turn an
    environment problem into a scoring problem.
    """
    signals: list[str] = []
    text, url, title = "", "", ""

    try:
        from .navigation import get_final_url
        from .page import get_page_text, get_page_title
        url = get_final_url(ctx, {})
        text = get_page_text(ctx, {"max_chars": 40_000})
        title = get_page_title(ctx, {})
    except Exception as exc:                                   # noqa: BLE001
        signals.append(f"page_unreachable:{type(exc).__name__}")

    if url.startswith("chrome-error://"):
        signals.append("nav_error")
    if "ERR_BLOCKED_BY_ADMINISTRATOR" in text or "ERR_BLOCKED_BY_ADMINISTRATOR" in url:
        signals.append("blocked_by_policy")

    haystack = f"{title}\n{text}"
    strong = BLOCK_STRONG.search(haystack)
    if strong:
        signals.append(f"bot_wall:{strong.group(0)[:60]}")
    elif BLOCK_WEAK.search(haystack) and len(text.strip()) < _SHORT:
        weak = BLOCK_WEAK.search(haystack)
        signals.append(f"bot_wall_weak:{weak.group(0)[:60]}")

    min_chars = int(config.get("min_chars", _MIN_CHARS))
    empty = len(text.strip()) < min_chars
    if empty:
        signals.append("empty_page")

    for extra in (config.get("patterns") or []):
        if re.search(str(extra), haystack, re.I):
            signals.append(f"custom:{extra}")

    status = config.get("http_status")
    if status is None:
        status = _main_document_status(getattr(ctx, "network", None))
    if isinstance(status, int) and (status in (403, 429) or status >= 500):
        signals.append(f"http_{status}")

    blocked = any(s.startswith(("bot_wall", "blocked_by_policy", "http_", "custom:"))
                  for s in signals)
    return {"ok": not signals, "http_status": status, "blocked": blocked,
            "empty": empty, "signals": signals, "chars": len(text)}
