"""The record of everything made and published, kept as JSON on the orphan `state` branch.

Published episodes consume their topic forever; dry and test runs are recorded separately and consume nothing.
Shorts live in their own list and never touch `last_upload`, so a Short can never hold back a long-form episode.
`ladder` holds what the Shorts learner keeps: weights, the pt.2 queue, per-build gate results, alerts and summaries,
plus the proven-topics premises (studio/topics.py). `topics` holds that weekly scan's competitor outliers (public
metadata only), the outliers already tried, and the long-form backlog boosts the planner reads.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from . import config

EMPTY = {"episodes": [], "dry_runs": [], "last_upload": None, "shorts": [],
         "ladder": {"weights": {}, "streaks": {}, "queue": [], "needs_bank": [], "builds": [], "alerts": [],
                    "summaries": [], "tests": [], "proven": []},
         "topics": {"outliers": [], "tried": [], "boosts": {}, "scans": []}}


def path(state: Path | None = None) -> Path:
    return (state or config.STATE) / "ledger.json"


def load(state: Path | None = None) -> dict:
    p = path(state)
    if not p.exists():
        return json.loads(json.dumps(EMPTY))
    data = json.loads(p.read_text(encoding="utf-8"))
    for k, v in EMPTY.items():
        data.setdefault(k, json.loads(json.dumps(v)))
    for group in ("ladder", "topics"):
        for k, v in EMPTY[group].items():
            data[group].setdefault(k, json.loads(json.dumps(v)))
    return data


def save(data: dict, state: Path | None = None) -> None:
    p = path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")  # whole or not at all: a job killed mid-save never leaves half a ledger
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def used_topics(data: dict) -> set[str]:
    return {e["topic"] for e in data["episodes"]}


def record_episode(data: dict, entry: dict) -> dict:
    """Add (or update, by id) a published episode. A topic can never be published twice."""
    for e in data["episodes"]:
        if e["id"] == entry["id"]:
            e.update(entry)
            break
    else:
        if entry["topic"] in used_topics(data):
            raise ValueError(f"topic {entry['topic']} was already published")
        data["episodes"].append(entry)
    if entry.get("video_id"):
        data["last_upload"] = entry.get("uploaded_at") or now()
    return data


def record_dry_run(data: dict, entry: dict) -> dict:
    data["dry_runs"].append(entry)
    data["dry_runs"] = data["dry_runs"][-60:]
    return data


def hours_since_upload(data: dict, at: datetime | None = None) -> float | None:
    if not data.get("last_upload"):
        return None
    last = datetime.fromisoformat(data["last_upload"].replace("Z", "+00:00"))
    return ((at or datetime.now(timezone.utc)) - last).total_seconds() / 3600


def record_short(data: dict, entry: dict) -> dict:
    """Add (or update, by video id) a published Short. A premise can never be published twice."""
    for s in data["shorts"]:
        if s.get("video_id") and s["video_id"] == entry.get("video_id"):
            s.update(entry)
            return data
    if any(s["premise"] == entry["premise"] and s.get("video_id") for s in data["shorts"]):
        raise ValueError(f"premise {entry['premise']} was already published")
    data["shorts"].append(entry)
    return data


def ladder_log(data: dict, key: str, entry: dict, keep: int = 120, unique: tuple[str, ...] = ()) -> dict:
    """Append to one of the ladder's rolling logs (builds, alerts, summaries, tests). unique: fields that identify an
    entry (e.g. run and slot); an entry with the same values replaces the earlier one, so a re-run of the record job
    never counts a build twice."""
    log = data["ladder"].setdefault(key, [])
    row = {"at": now(), **entry}
    if unique:
        ident = tuple(entry.get(k) for k in unique)
        for i, old in enumerate(log):
            if tuple(old.get(k) for k in unique) == ident:
                log[i] = {**row, "at": old.get("at") or row["at"]}
                break
        else:
            log.append(row)
    else:
        log.append(row)
    data["ladder"][key] = log[-keep:]
    return data
