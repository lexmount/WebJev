"""Getters that read files and need no browser session."""

from __future__ import annotations

from typing import Any

from .base import GetterError, first_str

__all__ = ["get_const", "get_agent_answer"]


def get_const(ctx: Any, config: dict) -> Any:
    """A constant -- the usual form of `expected`."""
    if "value" not in config:
        raise GetterError("const needs a value")
    return config["value"]


#: Configuration keys that were removed. They raise instead of being ignored, because a silently changed meaning is
#: worse than an error: an old task would score the same evidence differently without any notice.
_RETIRED = ("format", "path", "types")


def get_agent_answer(ctx: Any, config: dict) -> Any:
    """The agent's final answer, as raw text.

    `done_text` first, then `final_answer`. Both empty -> MISSING (not a GetterError): not answering is a legitimate
    agent failure, not broken evidence capture.

    No JSON parsing happens here. Judging the content is `answer_match`'s job, which runs a structured channel (JSON
    found anywhere in the reply) and a text channel (the whole reply) in parallel, so a correct answer is not rejected
    because of its wrapping.
    """
    retired = sorted(k for k in _RETIRED if k in config)
    if retired:
        raise GetterError(
            f"agent_answer no longer supports {retired}: final answers are judged by content, not wrapping. "
            f"Use the answer_match metric and put the paths into the keys of `expected`.")
    return first_str(ctx.record, "done_text", "final_answer")
