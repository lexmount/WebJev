"""Page content and control state.

This layer rots when a site is redesigned, so every field-extracting getter REQUIRES a canary (`must_exist`): if a
listed name cannot be extracted, the getter raises GetterError (-> judge_error) instead of silently scoring 0.
Without it, a redesign would drop every task to 0 on the same day and look like a model regression.
"""

from __future__ import annotations

import re
from typing import Any

from ..metrics.rules import MISSING
from .base import GetterError, cdp_call, js_json, js_text, quote

__all__ = ["get_page_text", "get_page_title",
           "get_dom_extract", "get_dom_count", "get_control_state",
           "get_visible_text", "get_axtree_query"]

_DEFAULT_MAX_CHARS = 200_000


def _truncate(text: str, config: dict) -> str:
    limit = int(config.get("max_chars", _DEFAULT_MAX_CHARS))
    return text if limit <= 0 or len(text) <= limit else text[:limit]


def get_page_text(ctx: Any, config: dict) -> str:
    """Page body text (innerText, not textContent, which would include script/style content)."""
    text = js_text(ctx, "document.body ? document.body.innerText : ''",
                   what="page_text")
    return _truncate(text, config)


def get_page_title(ctx: Any, config: dict) -> str:
    return js_text(ctx, "document.title", what="page_title")


# -- field extraction --------------------------------------------------------------------------------------------

_EXTRACT_JS = """
(() => {
  const sel = %(selectors)s, attr = %(attr)s, all = %(all)s;
  const read = (el) => {
    if (!el) return null;
    if (attr === null) return (el.innerText || el.textContent || '').trim();
    if (attr === 'value') return el.value === undefined ? null : String(el.value);
    const v = el.getAttribute(attr);
    return v === null ? null : String(v);
  };
  const out = {};
  for (const [name, css] of Object.entries(sel)) {
    try {
      if (all) out[name] = Array.from(document.querySelectorAll(css)).map(read);
      else out[name] = read(document.querySelector(css));
    } catch (e) { out[name] = {__selector_error__: String(e)}; }
  }
  return JSON.stringify(out);
})()
"""


def _check_canary(values: dict, config: dict, what: str) -> None:
    """Canary: a name that must exist but cannot be extracted means broken capture, not a wrong agent."""
    must = config.get("must_exist")
    if must is None:
        raise GetterError(
            f"{what}: must_exist (the canary) is required. Selectors rot when sites change; without a canary "
            f"'not on the page' and 'selector broken' cannot be told apart")
    for name in must:
        got = values.get(name, MISSING)
        if isinstance(got, dict) and "__selector_error__" in got:
            raise GetterError(
                f"{what}: the selector of canary {name!r} is invalid: {got['__selector_error__']}")
        if got is MISSING or got is None or got == [] or got == "":
            raise GetterError(
                f"{what}: canary {name!r} was not found -- the selector has most likely rotted; "
                f"this is a judge_error, not a 0")


def get_dom_extract(ctx: Any, config: dict) -> dict:
    """Extract fields by CSS selector -- {name: value}.

    config: `selectors` {name: css}; `attr` (default: text; "value": form value; anything else: that attribute);
    `all` (every match instead of the first); `must_exist` (required).
    """
    selectors = config.get("selectors")
    if not isinstance(selectors, dict) or not selectors:
        raise GetterError("dom_extract needs non-empty selectors")
    expr = _EXTRACT_JS % {
        "selectors": quote(selectors),
        "attr": quote(config["attr"]) if config.get("attr") else "null",
        "all": "true" if config.get("all") else "false",
    }
    values = js_json(ctx, expr, what="dom_extract")
    if not isinstance(values, dict):
        raise GetterError("dom_extract: JS did not return an object")
    _check_canary(values, config, "dom_extract")
    return {k: (MISSING if v is None else v) for k, v in values.items()}


def get_dom_count(ctx: Any, config: dict) -> int:
    """Count selector matches. With `must_exist: true`, zero matches means the selector is broken."""
    selector = config.get("selector")
    if not selector:
        raise GetterError("dom_count needs a selector")
    expr = ("(() => { try { return String(document.querySelectorAll(%s).length); }"
            " catch (e) { return 'ERR:' + e; } })()" % quote(selector))
    raw = js_text(ctx, expr, what="dom_count")
    if raw.startswith("ERR:"):
        raise GetterError(f"dom_count: invalid selector: {raw[4:]}")
    count = int(raw)
    if count == 0 and config.get("must_exist"):
        raise GetterError(
            f"dom_count: selector {selector!r} matched nothing although must_exist is set -- "
            f"this is a judge_error, not a 0")
    return count


# -- control state -----------------------------------------------------------------------------------------------

_CONTROL_JS = """
(() => {
  const sel = %(selectors)s;
  const out = {};
  for (const [name, css] of Object.entries(sel)) {
    try {
      const el = document.querySelector(css);
      if (!el) { out[name] = null; continue; }
      let selected = null;
      if (el.tagName === 'SELECT') {
        selected = Array.from(el.selectedOptions || []).map(
          o => (o.textContent || '').trim());
      }
      out[name] = {
        tag: el.tagName.toLowerCase(),
        value: el.value === undefined ? null : String(el.value),
        checked: el.checked === undefined ? null : !!el.checked,
        selected: selected,
        text: (el.innerText || el.textContent || '').trim(),
        disabled: !!el.disabled,
        aria: {
          checked: el.getAttribute('aria-checked'),
          selected: el.getAttribute('aria-selected'),
          expanded: el.getAttribute('aria-expanded'),
          pressed: el.getAttribute('aria-pressed'),
        },
      };
    } catch (e) { out[name] = {__selector_error__: String(e)}; }
  }
  return JSON.stringify(out);
})()
"""


def get_control_state(ctx: Any, config: dict) -> dict:
    """Current state of controls -- {name: {tag, value, checked, selected, text, disabled, aria}}.

    For "the sort is set to lowest price" or "this filter is ticked". Native controls expose value/checked, custom
    widgets expose aria-*; both are returned and the task picks what the site uses.
    """
    selectors = config.get("selectors")
    if not isinstance(selectors, dict) or not selectors:
        raise GetterError("control_state needs non-empty selectors")
    values = js_json(ctx, _CONTROL_JS % {"selectors": quote(selectors)},
                     what="control_state")
    if not isinstance(values, dict):
        raise GetterError("control_state: JS did not return an object")
    _check_canary(values, config, "control_state")
    return {k: (MISSING if v is None else v) for k, v in values.items()}


# -- visible text ------------------------------------------------------------------------------------------------

_VISIBLE_JS = """
(() => {
  const root = %(selector)s ? document.querySelector(%(selector)s) : document.body;
  if (!root) return JSON.stringify('');
  const out = [];
  const walk = (n) => {
    if (n.nodeType === 3) { const t = n.textContent.trim(); if (t) out.push(t); return; }
    if (n.nodeType !== 1) return;
    const cs = getComputedStyle(n);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return;
    if (n.offsetParent === null && cs.position !== 'fixed' && n.tagName !== 'BODY') return;
    for (const c of n.childNodes) walk(c);
  };
  walk(root);
  return JSON.stringify(out.join(' '));
})()
"""


def get_visible_text(ctx: Any, config: dict) -> str:
    """Only text that is actually visible.

    innerText already drops `display:none`, but not `opacity:0`, off-screen positioning or pre-rendered content in
    collapsed panels, which would cause false positives for "X appears in the filtered results".
    """
    sel = quote(config["selector"]) if config.get("selector") else "null"
    text = js_json(ctx, _VISIBLE_JS % {"selector": sel}, what="visible_text")
    if not isinstance(text, str):
        raise GetterError("visible_text: JS did not return a string")
    return _truncate(text, config)


# -- accessibility tree ------------------------------------------------------------------------------------------

def _ax_field(node: dict, key: str) -> str:
    value = node.get(key)
    if isinstance(value, dict):
        return str(value.get("value", "") or "")
    return str(value or "")


def get_axtree_query(ctx: Any, config: dict) -> list:
    """Accessibility-tree query -- [{role, name, value}].

    `role` and `name_pattern` are optional; when given, both must hold. More robust than CSS for structural questions
    ("how many checked checkboxes"): a role is semantics, not markup.
    """
    out = cdp_call(ctx, "Accessibility.getFullAXTree", what="axtree_query")
    nodes = out.get("nodes")
    if not isinstance(nodes, list):
        raise GetterError("axtree_query: Accessibility.getFullAXTree returned no nodes")

    want_role = config.get("role")
    roles = {str(r).lower() for r in (want_role if isinstance(want_role, (list, tuple))
                                      else [want_role])} if want_role else None
    pattern = re.compile(config["name_pattern"], re.IGNORECASE) \
        if config.get("name_pattern") else None

    hits = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("ignored"):
            continue
        role, name = _ax_field(node, "role"), _ax_field(node, "name")
        if roles is not None and role.lower() not in roles:
            continue
        if pattern is not None and not pattern.search(name):
            continue
        hits.append({"role": role, "name": name, "value": _ax_field(node, "value")})
    return hits
