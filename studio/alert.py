"""Telegram alerts for the Shorts engine: a read-back mismatch, a double-down that needs new premises, the daily
summary. Configured by TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID; without them the message is printed instead.

The token is never printed, logged or written anywhere: errors report only the HTTP status.
"""
from __future__ import annotations

import os

API = "https://api.telegram.org/bot{token}/sendMessage"
CHUNK = 3900  # Telegram's limit is 4096 characters per message


def chunks(text: str, size: int = CHUNK) -> list[str]:
    pieces = [line[i:i + size] for line in text.splitlines(keepends=True) for i in range(0, len(line), size)]
    out, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) > size:
            out.append(cur)
            cur = ""
        cur += piece
    if cur:
        out.append(cur)
    return out or [""]


def send(text: str) -> bool:
    """True when Telegram accepted every part. Never raises: an alert must not break the run that sends it."""
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("[telegram not configured]\n" + text)
        return False
    import requests

    ok = True
    for part in chunks(text):
        try:
            r = requests.post(API.format(token=token), data={"chat_id": chat, "text": part,
                                                             "disable_web_page_preview": "true"}, timeout=30)
            if r.status_code != 200:
                print(f"telegram: HTTP {r.status_code}")
                ok = False
        except Exception as exc:  # the exception text can contain the URL, and so the token: never print it
            print(f"telegram: {type(exc).__name__}")
            ok = False
    return ok
