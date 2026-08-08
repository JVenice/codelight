"""Ollama Cloud usage meter.

Usage only: Ollama Cloud is a model host, not a coding agent, so this module
has no hooks, no listener, no transcripts, and deliberately declares no
``executables`` — a local ``ollama`` binary is a model server, not a session,
and probing for it would give every Ollama user a permanently-idle card with
nothing to report. The meter appears only once a credential is configured.

``GET /api/usage`` is undocumented (absent from Ollama's OpenAPI spec) and
returns no reset timestamps for either window, so this fetcher reports
percentages alone and lets the state layer default ``reset`` to ``"--"``. The
4-week window in ``activity.period`` is an activity range, not a quota reset,
and is never turned into a countdown: it would be wrong by up to three weeks
on weekly and always wrong on session. Like Cursor's equally undocumented
endpoint, everything here fails closed — any error hides the meter rather than
guessing.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

from codelight_core.agents import base


USAGE_API = "https://ollama.com/api/usage"

# A conventional credential file the companion reads without any shell
# environment or config entry. A key that lives only in an interactive shell
# (e.g. ~/.zshrc sourcing a secrets file) never reaches a companion launched
# by the desktop session or a service manager, so the meter is silently
# absent after a reboot. Dropping the key here once makes the meter work from
# any launch path. Resolved after OLLAMA_API_KEY and agents.ollama.api_key_file.
DEFAULT_KEY_PATH = os.path.expanduser("~/.config/codelight/ollama-api-key")

# Said once per process so a missing credential is legible in the daemon log
# instead of a silently-absent card. Reset for tests by setting this False.
_no_key_warned = False
NO_KEY_HINT = (
    "[ollama-usage] meter hidden: no API key configured. Set OLLAMA_API_KEY "
    "in the companion's launch environment, configure "
    "agents.ollama.api_key_file, or write the key to "
    "~/.config/codelight/ollama-api-key."
)

# Ollama's llama mark (simple-icons), fills with currentColor.
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    '<path fill="currentColor" d="M16.36 10.26a.89.89 0 0 0-.56.47l-.07.15 0 .21c0 .19 0 .22.06.35.08.19.15.31.29.45.24.24.51.3.87.2a.86.86 0 0 0 .52-.44.75.75 0 0 0 .08-.5c-.06-.45-.33-.78-.72-.9a1.06 1.06 0 0 0-.47 0zm-9.2.01c-.3.1-.53.32-.65.64a1.19 1.19 0 0 0-.06.52c.06.31.31.59.6.67.36.1.63.03.87-.2.14-.14.21-.26.29-.45.06-.14.06-.16.06-.35l0-.21-.07-.15a.89.89 0 0 0-.56-.47 1.02 1.02 0 0 0-.47.01Zm4.18 2c-.13.07-.22.25-.2.38.03.14.16.29.35.41.1.06.11.07.12.14 0 .04-.01.15-.03.24-.02.09-.04.19-.04.22 0 .07.07.2.14.25.06.05.08.05.26.06.16.01.2 0 .26-.03.17-.08.21-.23.15-.53-.05-.24-.04-.28.09-.35.14-.08.28-.22.32-.31a.36.36 0 0 0-.17-.48.39.39 0 0 0-.18-.03c-.13 0-.21.03-.35.12l-.09.05-.05-.03c-.22-.13-.26-.14-.39-.14a.4.4 0 0 0-.19.03zm.39-2.19c-.37.04-.47.05-.65.09-.29.06-.68.2-.95.33-.94.46-1.59 1.23-1.79 2.11-.04.18-.04.23-.04.53 0 .29.01.36.04.52.26 1.16 1.33 2.02 2.71 2.17.3.03 1.6.03 1.9 0 1.11-.12 2.06-.73 2.49-1.57.11-.23.17-.37.22-.6.04-.17.04-.23.04-.52 0-.3-.01-.35-.04-.53-.29-1.29-1.54-2.3-3.07-2.5a6.87 6.87 0 0 0-.85-.03zm.65.94a3.28 3.28 0 0 1 1.44.51c.22.15.54.46.67.66.17.25.26.51.3.82.02.14.01.25-.04.48-.08.34-.33.7-.67.96a3.12 3.12 0 0 1-.69.35c-.38.12-.63.14-1.52.14-.58-.01-.69-.01-.85-.04-.57-.11-1.02-.33-1.35-.68-.26-.28-.39-.54-.45-.95-.03-.19.03-.51.14-.78.14-.33.49-.73.84-.96.4-.27.93-.46 1.42-.51.19-.02.59-.02.77 0zm-5.5-11a1.65 1.65 0 0 0-.68.3C5.62.74 5.17 1.67 4.99 2.82c-.07.44-.12 1.04-.12 1.5 0 .54.06 1.24.15 1.72.02.11.03.2.02.21a8.12 8.12 0 0 1-.19.15 5.32 5.32 0 0 0-.95 1.02 5.49 5.49 0 0 0-.94 2.34 6.62 6.62 0 0 0-.02 1.36c.09.78.33 1.44.73 2.04l.13.2-.04.06c-.27.45-.5 1.1-.6 1.73-.08.5-.1.63-.1 1.29 0 .67.01.8.09 1.27.1.56.29 1.14.5 1.53.07.13.24.39.26.41.01 0-.01.07-.05.14a7.41 7.41 0 0 0-.55 1.87c-.06.42-.07.55-.07.99 0 .56.03.83.15 1.28L3.42 24h1.48l-.05-.09c-.3-.55-.33-1.57-.07-2.6.12-.47.25-.82.5-1.3l.15-.29v-.18c0-.17 0-.18-.06-.29a.92.92 0 0 0-.19-.25 1.74 1.74 0 0 1-.39-.54c-.42-.92-.51-2.29-.21-3.45.12-.49.33-.92.54-1.15a.79.79 0 0 0 .22-.53c0-.2-.07-.35-.22-.52a3.14 3.14 0 0 1-.82-1.73c-.14-.96.11-2 .69-2.83.56-.81 1.35-1.34 2.24-1.48.2-.03.57-.03.78.01.23.04.37.03.51-.04.18-.09.27-.19.37-.43.09-.21.17-.33.36-.58.23-.29.46-.49.82-.73.41-.27.88-.47 1.35-.56.17-.04.25-.04.57-.04.32 0 .4.01.57.04a4.07 4.07 0 0 1 1.91 1c.12.11.4.46.49.6.03.06.1.18.13.27.1.24.2.35.37.43.14.07.29.08.5.04.34-.06.61-.05.94.02 1.14.23 2.14 1.17 2.58 2.44.39 1.11.28 2.27-.3 3.15-.1.15-.19.27-.33.42-.3.32-.3.72 0 1.05.49.54.8 1.87.71 3.04-.06.77-.26 1.46-.53 1.85a2.1 2.1 0 0 1-.22.26.92.92 0 0 0-.19.25c-.05.11-.06.13-.06.29v.18l.15.29c.25.48.38.82.5 1.29.25 1.01.23 2.01-.06 2.58a.84.84 0 0 0-.04.1c0 .01.33.01.73.01h.73l.02-.07.04-.13c.02-.08.06-.3.09-.52.03-.22.03-1.02 0-1.26-.11-.88-.29-1.57-.6-2.23-.03-.07-.05-.14-.05-.14.01-.01.06-.07.11-.15.38-.57.61-1.28.72-2.23.03-.26.03-1.38 0-1.63-.08-.65-.18-1.08-.35-1.52a6.08 6.08 0 0 0-.33-.7l-.04-.06.13-.19c.4-.6.64-1.26.73-2.04a6.62 6.62 0 0 0-.02-1.36 5.51 5.51 0 0 0-.94-2.34 5.33 5.33 0 0 0-.95-1.02 8.1 8.1 0 0 1-.19-.15.69.69 0 0 1 .02-.21c.21-1.09.2-2.44-.02-3.5-.19-.92-.54-1.66-.98-2.08-.35-.34-.72-.48-1.15-.46-1 .06-1.8 1.21-2.12 3.01a6.8 6.8 0 0 0-.1.73c0 .04-.01.07-.01.07a.96.96 0 0 1-.15-.08A4.86 4.86 0 0 0 12 3.03c-.83 0-1.69.24-2.46.7a.96.96 0 0 1-.15.08c-.01 0-.01-.03-.01-.07a6.71 6.71 0 0 0-.1-.72C9 1.39 8.34.32 7.46.05a2.1 2.1 0 0 0-.58-.04Zm.29 1.4c.25.2.52.76.68 1.39.03.11.06.24.07.29.01.05.03.15.04.23.07.36.1.76.1 1.24l0 .47-.12.17-.12.18h-.28c-.32 0-.65.04-.95.12l-.24.06c-.03.01-.04 0-.06-.14a8.44 8.44 0 0 1 .02-2.32c.12-.79.41-1.5.7-1.71.07-.05.08-.05.16.01zm9.82-.01c.17.13.36.46.5.89.28.85.36 2.03.21 3.15-.02.14-.02.15-.06.14l-.24-.06a3.69 3.69 0 0 0-.95-.12h-.28l-.12-.18-.12-.17 0-.47c0-.67.07-1.19.21-1.77.16-.62.43-1.19.68-1.38.08-.06.09-.06.16-.01z"/>'
    '</svg>'
)

# 48x48 1-bit render of LOGO_SVG for the ESP8266 screen.
LOGO_BITMAP = (
    "AA8AAPAAAB+AAfgAAB/AA/gAAD/AA/wAADngB5wAADngB5wAAHjv9xwAAHj//x4AAHj//x"
    "4AAHj8Px4AAHvwD94AAD/gB/wAAD/AA/wAAP/AA/4AAPgAAB8AAeAAAAeAA8AAAAPAA8AA"
    "AAPAA4AAAAHAA4AAAAHAB4MP8MHAB4ef+eHAA4e+feHAA4fwD+HAA8JjxkPAA+DjxwfAAe"
    "DhhweAAeDhhweAA8BwDgPAA8B//gPAA4A//AHAA4AP8AHAA4AAAAHAA4AAAAHAA4AAAAHA"
    "A4AAAAPAA8AAAAPAA8AAAAOAAeAAAAeAAeAAAAeAAeAAAAeAAcAAAAOAA8AAAAPAA8AAAA"
    "PAA8AAAAPAA8AAAAPAA8AAAAPAA8AAAAPA"
)

SPEC = base.AgentSpec(
    "ollama",
    "Ollama",
    # No executables on purpose: see the module docstring. A local `ollama`
    # server is not a coding-agent session, so there is no status to detect.
    color="#FFFFFF",
    logo_svg=LOGO_SVG,
    logo_bitmap=LOGO_BITMAP,
)


def api_key(api_key_file: str = "") -> str:
    """Ollama Cloud API key: ``OLLAMA_API_KEY`` first, then a configured key
    file, then the conventional ``~/.config/codelight/ollama-api-key``.
    Env-or-file only — no inline secret in config. Never logged.

    The conventional default exists so a companion launched outside an
    interactive shell (desktop autostart, a service manager) can still find a
    key the user dropped once, without editing shell files or config. It is
    re-read on every poll, so creating the file later starts the meter without
    a restart."""
    env = os.environ.get("OLLAMA_API_KEY", "").strip()
    if env:
        return env
    for path in (api_key_file, DEFAULT_KEY_PATH):
        if not path:
            continue
        try:
            with open(os.path.expanduser(path)) as stream:
                value = stream.read().strip()
        except Exception:
            continue
        if value:
            return value
    return ""


def _window_pct(limits: dict, window: str) -> float | None:
    """One window's already-0-to-1 usage fraction, or None if absent or not a
    number. Ollama reports a fraction, not a percentage — dividing by 100 here
    (as the Claude fetcher must) would render 17% as 0.17%."""
    entry = limits.get(window)
    if not isinstance(entry, dict):
        return None
    usage = entry.get("usage")
    if isinstance(usage, bool) or not isinstance(usage, (int, float)):
        return None
    return max(0.0, min(1.0, float(usage)))


def get_usage(
    key: str,
    *,
    usage_api: str = USAGE_API,
    log: Callable[[str], None] | None = None,
) -> dict | None:
    """Ollama Cloud's session and weekly usage fractions.

    Returns ``{"session_pct": …, "weekly_pct": …}`` — no reset keys, because
    the endpoint reports none. A window that is missing or unreadable is left
    out so its bar disappears instead of reading 0%; when neither window
    survives, the whole meter is hidden. Returns None (meter hidden) on any
    issue — no key, rejected key, endpoint changed, offline."""
    if not key:
        # The one silent failure: with no credential the meter simply does
        # not appear, and a usage-only agent (no status) then vanishes from
        # every client. Say why, once, so this is legible in the daemon log
        # rather than a mystery missing card. Never logs the key or response.
        global _no_key_warned
        if log is not None and not _no_key_warned:
            _no_key_warned = True
            log(NO_KEY_HINT)
        return None
    req = urllib.request.Request(usage_api, headers={
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": "codelight",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as e:
        # Reason only: an error body from an undocumented endpoint is not ours
        # to put in the log.
        if log:
            log(f"[ollama-usage] HTTP {e.code}: {e.reason}")
        return None
    except (json.JSONDecodeError, UnicodeDecodeError):
        if log:
            log("[ollama-usage] malformed response")
        return None
    except Exception as e:
        if log:
            log(f"[ollama-usage] request failed: {e}")
        return None

    limits = data.get("limits") if isinstance(data, dict) else None
    if not isinstance(limits, dict):
        if log:
            log("[ollama-usage] no limits in response")
        return None

    usage: dict[str, float] = {}
    for window in ("session", "weekly"):
        pct = _window_pct(limits, window)
        if pct is not None:
            usage[f"{window}_pct"] = pct
    if not usage:
        if log:
            log("[ollama-usage] no readable window")
        return None
    if log:
        log("[ollama-usage] " + " ".join(
            f"{window}={usage[f'{window}_pct']:.0%}"
            for window in ("session", "weekly")
            if f"{window}_pct" in usage))
    # No `limits` list: the state layer derives and overwrites it.
    return usage


class OllamaAgent:
    """Usage-only agent. The credential is resolved on every poll so adding a
    key file (or exporting OLLAMA_API_KEY before a restart) starts the meter
    without any further wiring."""

    def __init__(self, api_key_file: str = "",
                 log: Callable[[str], None] | None = None) -> None:
        self.api_key_file = api_key_file
        self.log = log

    def get_usage(self) -> dict | None:
        return get_usage(api_key(self.api_key_file), log=self.log)


def build_integration(config: dict, *,
                      log: Callable[[str], None] | None = None) -> base.AgentIntegration:
    """Config keys (~/.config/codelight/config.json, agents.ollama):
    api_key_file (Ollama Cloud API key file; or set OLLAMA_API_KEY, or drop
    the key in ~/.config/codelight/ollama-api-key),
    usage (default true)."""
    api_key_file = os.path.expanduser(str(config.get("api_key_file") or "").strip())
    usage_enabled = bool(config.get("usage", True))
    agent = OllamaAgent(api_key_file, log=log)

    return base.AgentIntegration(
        spec=SPEC,
        agent=agent,
        # Session + weekly usage fractions from Ollama Cloud's account API.
        # Hidden entirely when no credential is configured.
        usage_fetcher=agent.get_usage if usage_enabled else None,
    )
