"""Which browser a run uses; the same rule as vendor/jev_ultrafast/browser_cdp.browser_backend.

  - "lexmount": a Lexmount Browser cloud session (https://browser.lexmount.com). Needs LEXMOUNT_API_KEY and
    LEXMOUNT_PROJECT_ID.
  - "local": your own Chrome started with --remote-debugging-port (CHROME_CDP_URL, default http://127.0.0.1:9222).

BROWSER picks one explicitly; without it, lexmount is used when LEXMOUNT_API_KEY is set and local otherwise. Other
BROWSER values are ignored, because many systems already use BROWSER for the default web browser.
"""

import os

LABELS = {"lexmount": "Lexmount Browser", "local": "Local Chrome"}


def browser_backend(environ=None):
    environ = os.environ if environ is None else environ
    choice = (environ.get("BROWSER") or "").strip().lower()
    if choice in LABELS:
        return choice
    return "lexmount" if environ.get("LEXMOUNT_API_KEY") else "local"
