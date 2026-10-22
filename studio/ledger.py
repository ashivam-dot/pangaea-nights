"""The record of everything made and published, kept as JSON on the orphan `state` branch.

Published episodes consume their topic forever; dry and test runs are recorded separately and consume nothing.
Shorts live in their own list and never touch `last_upload`, so a Short can never hold back a long-form episode.
`ladder` holds what the Shorts learner keeps: weights, the pt.2 queue, per-build gate results, alerts and summaries.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config

EMPTY = {"episodes": [], "dry_runs": [], "last_upload": None, "shorts": [],
         "ladder": {"weights": {}, "streaks": {}, "queue": [], "needs_bank": [], "builds": [], "alerts": [],
                    "summaries": [], "tests": []}}


def path(state: Path | None = None) -> Path:
    return (state or config.STATE) / "ledger.json"


def load(state: Path | None = None) -> dict:
    p = path(state)
    if not p.exists():
        return json.loads(json.dumps(EMPTY))
    data = json.loads(p.read_text(encoding="utf-8"))
    for k, v in EMPTY.items():
        data.setdefault(k, json.loads(json.dumps(v)))
    for k, v in EMPTY["ladder"].items():
        data["ladder"].setdefault(k, json.loads(json.dumps(v)))
    return data


def save(data: dict, state: Path | None = None) -> None:
    p = path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


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


def ladder_log(data: dict, key: str, entry: dict, keep: int = 120) -> dict:
    """Append to one of the ladder's rolling logs (builds, alerts, summaries, tests)."""
    log = data["ladder"].setdefault(key, [])
    log.append({"at": now(), **entry})
    data["ladder"][key] = log[-keep:]
    return data
