"""What the channel's own numbers say about each series, fed back into tonight's topic.

The publish step (the one that holds the YouTube sign-in) reads every upload's views into out/channel_stats.json,
whatever the mode. The record step folds them into the ledger: a snapshot per published episode, its views at
SCORE_AT_H hours after the premiere, and a lift per series. A lift needs MIN_SAMPLES scored episodes, is shrunk
toward 1 by n / (n + PRIOR) and stays within LIFT_RANGE; the planner only uses it to choose among the series that
have rested longest, so the rotation and the never-twice-in-three-nights rule still hold.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

SCORE_AT_H = 72.0
MIN_SAMPLES = 4
PRIOR = 4.0
LIFT_RANGE = (0.75, 1.35)
KEEP_SNAPSHOTS = 16


def _when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace(".000Z", "Z").replace("Z", "+00:00"))


def fetch(api, channel: dict, limit: int = 50) -> dict:
    """Views, likes and comments of the channel's latest uploads, by video id."""
    uploads = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    rows = api.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=limit).execute()
    ids = [r["contentDetails"]["videoId"] for r in rows.get("items", [])]
    if not ids:
        return {}
    items = api.videos().list(part="statistics", id=",".join(ids)).execute().get("items", [])
    return {v["id"]: {k: int(v.get("statistics", {}).get(f"{k}Count", 0)) for k in ("view", "like", "comment")}
            for v in items}


def views_at(snapshots: list[dict], hours: float) -> float | None:
    """Views `hours` after the premiere: interpolated between snapshots, or scaled back from the first later one
    (views grow roughly with the square root of age after the first push). None while too young."""
    snaps = sorted(snapshots, key=lambda s: s["age_h"])
    if not snaps or snaps[-1]["age_h"] < hours:
        return None
    later = next(s for s in snaps if s["age_h"] >= hours)
    earlier = [s for s in snaps if s["age_h"] < hours]
    if not earlier:
        return later["views"] * math.sqrt(hours / later["age_h"]) if later["age_h"] > 0 else float(later["views"])
    a = earlier[-1]
    span = later["age_h"] - a["age_h"]
    return a["views"] + (later["views"] - a["views"]) * ((hours - a["age_h"]) / span if span else 1.0)


def lifts(episodes: list[dict]) -> dict:
    scored = [e for e in episodes if e.get("views_at_72h") is not None]
    out = {"scored": len(scored), "min_samples": MIN_SAMPLES, "lifts": {}}
    if len(scored) < MIN_SAMPLES:
        return out
    logs = [math.log1p(e["views_at_72h"]) for e in scored]
    base = sum(logs) / len(logs)
    out["baseline_views_at_72h"] = round(math.expm1(base))
    groups: dict[str, list[float]] = {}
    for e, score in zip(scored, logs):
        groups.setdefault(e["series"], []).append(score)
    for series, scores in groups.items():
        n = len(scores)
        if n < MIN_SAMPLES:
            continue
        lift = math.exp((sum(scores) / n - base) * n / (n + PRIOR))
        out["lifts"][series] = {"n": n, "lift": round(min(max(lift, LIFT_RANGE[0]), LIFT_RANGE[1]), 3),
                                "typical_views_at_72h": round(math.expm1(sum(scores) / n))}
    return out


def lengths(episodes: list[dict]) -> dict:
    """The length experiment's scoreboard: scored episodes and their typical views at 72 h, per shape. Report only;
    studio/length.py keeps alternating whatever it says, so both arms keep collecting samples."""
    out: dict[str, dict] = {}
    for e in episodes:
        if e.get("views_at_72h") is None or not e.get("shape"):
            continue
        out.setdefault(e["shape"], {"n": 0, "logs": []})
        out[e["shape"]]["n"] += 1
        out[e["shape"]]["logs"].append(math.log1p(e["views_at_72h"]))
    return {s: {"n": g["n"], "typical_views_at_72h": round(math.expm1(sum(g["logs"]) / g["n"]))}
            for s, g in sorted(out.items())}


def update(data: dict, stats: dict, at: datetime | None = None) -> dict:
    """Snapshot each published episode that has gone public, rescore, relearn. Changes `data` in place."""
    at = at or datetime.now(timezone.utc)
    for e in data["episodes"]:
        vid = e.get("video_id")
        if not vid or vid not in stats or not e.get("publish_at"):
            continue
        age = (at - _when(e["publish_at"])).total_seconds() / 3600
        if age < 0:
            continue
        s = stats[vid]
        snaps = e.get("snapshots", []) + [{"age_h": round(age, 2), "views": s["view"], "likes": s["like"],
                                            "comments": s["comment"]}]
        e["snapshots"] = snaps[-KEEP_SNAPSHOTS:]
        e["views_at_72h"] = views_at(e["snapshots"], SCORE_AT_H)
    data["learned"] = {"at": at.replace(microsecond=0).isoformat(), **lifts(data["episodes"]),
                       "lengths": lengths(data["episodes"])}
    return data


def prefer(choices: list[str], data: dict, window: int = 3) -> list[str]:
    """Among the `window` series that have rested longest, lead with the best learned lift (ties keep rest order)."""
    learned = (data.get("learned") or {}).get("lifts", {})
    if not learned:
        return choices
    head = sorted(choices[:window], key=lambda s: -learned.get(s, {}).get("lift", 1.0))
    return head + choices[window:]
