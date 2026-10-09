"""Pick tonight's topic: never a published one, never the same series as either of the last two episodes, and the
series that has rested longest goes first. Titles rotate through each series' patterns and never near-duplicate.
"""
from __future__ import annotations

import re

from . import config, ledger


class BacklogExhausted(RuntimeError):
    pass


def next_topic(data: dict, backlog: list[dict] | None = None, series: dict | None = None,
               override: str | None = None) -> dict:
    backlog = backlog if backlog is not None else config.backlog()
    series = series if series is not None else config.series()
    used = ledger.used_topics(data)
    if override:
        topic = next((t for t in backlog if t["id"] == override), None)
        if topic is None:
            raise ValueError(f"no topic '{override}' in the backlog")
        return topic
    free = [t for t in backlog if t["id"] not in used]
    if not free:
        raise BacklogExhausted("every backlog topic has been published; add topics to bible/backlog.json")
    history = [e["series"] for e in data["episodes"]]
    recent = set(history[-2:])
    order = list(series)

    def rested(s: str) -> tuple:
        last = max((i for i, h in enumerate(history) if h == s), default=-1)
        return (last, order.index(s) if s in order else len(order))

    open_series = sorted({t["series"] for t in free}, key=rested)
    choices = [s for s in open_series if s not in recent] or open_series
    chosen = choices[0]
    return next(t for t in free if t["series"] == chosen)


def _words(title: str) -> set[str]:
    return set(re.findall(r"[a-z]+", title.lower())) - {"the", "of", "a", "in", "to", "and", "for", "hours", "hour"}


def similar(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, len(wa | wb))


def title(topic: dict, core: str, hours: int, data: dict, series: dict | None = None) -> str:
    """Rotate the series' title patterns and avoid any title too close to one already used (or over 100 chars)."""
    series = series if series is not None else config.series()
    patterns = series[topic["series"]]["titles"]
    done = sum(1 for e in data["episodes"] if e["series"] == topic["series"])
    previous = [e.get("title", "") for e in data["episodes"]]
    core = core.strip()
    options = []
    for k in range(len(patterns)):
        pattern = patterns[(done + k) % len(patterns)]
        text = pattern.format(title=core, hours=hours)
        text = re.sub(r"\b(in|on|to|with) The\b", r"\1 the", text)
        if len(text) <= 100:
            options.append(text)
    for t in options:
        if all(similar(t, p) < 0.75 for p in previous):
            return t
    return options[0] if options else f"{core} | Calm Sleep Documentary"[:100]
