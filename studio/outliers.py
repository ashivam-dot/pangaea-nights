"""Which of the competitor channels' recent uploads beat their own channel by a wide margin (bible/competitors.json).

Only public metadata is ever read or kept: video id, title, view count, publish date, duration and channel id. Every
request names its `fields`, so a description, thumbnail, caption or tag never even reaches us, and public_metadata()
keeps only those six fields of whatever comes back. Nothing is downloaded.

Quota: one channels.list for every channel at once, then one playlistItems.list (the latest RECENT uploads) and one
videos.list per channel, 1 unit each: about 40 units a scan against the 10,000-unit day. Budget refuses to go past
QUOTA_MAX. search.list (100 units) is never used.

An outlier is a video with at least OUTLIER_RATIO x the median views of the same channel's recent uploads of the same
kind (Shorts against Shorts, long-form against long-form) and at least that kind's absolute FLOOR of views.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from . import config

CHANNELS = config.BIBLE / "competitors.json"
RECENT = 50  # uploads read per channel: one playlistItems page
SHORT_MAX_S = 180  # YouTube counts vertical videos up to 3 minutes as Shorts
OUTLIER_RATIO = 3.0
FLOOR = {"short": 100_000, "long": 20_000}
MIN_SAMPLES = 5  # videos of one kind a channel needs before its median means anything
QUOTA_MAX = 300
KEEP = ("video_id", "channel_id", "title", "views", "published", "duration_s")

# Partial responses: the API returns only these, so nothing else is ever downloaded.
CHANNEL_FIELDS = "items(id,contentDetails/relatedPlaylists/uploads)"
PLAYLIST_FIELDS = "items(contentDetails/videoId)"
VIDEO_FIELDS = "items(id,snippet(title,publishedAt,channelId),statistics/viewCount,contentDetails/duration)"


class QuotaExceeded(RuntimeError):
    pass


class Budget:
    """Data API units spent in one scan; spend() refuses the call that would pass the limit."""

    def __init__(self, limit: int = QUOTA_MAX):
        self.limit, self.units = limit, 0

    def spend(self, units: int = 1) -> None:
        if self.units + units > self.limit:
            raise QuotaExceeded(f"a scan may spend {self.limit} units; {self.units} spent")
        self.units += units


def load(path: Path = CHANNELS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def competitors(path: Path = CHANNELS) -> list[dict]:
    return load(path)["channels"]


def do_not_copy(path: Path = CHANNELS) -> list[str]:
    """Competitor channel, character and brand names: never in anything we publish."""
    doc = load(path)
    return list(dict.fromkeys(doc["do_not_copy"] + [c["name"] for c in doc["channels"]]))


def duration_s(iso: str) -> int:
    """Seconds in an ISO 8601 duration such as PT1M5S or P1DT2H (0 when unreadable, e.g. an upcoming premiere)."""
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", iso or "")
    if not m:
        return 0
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def kind(seconds: int) -> str:
    return "short" if seconds <= SHORT_MAX_S else "long"


def public_metadata(item: dict) -> dict | None:
    """The six public fields of one videos.list item, and nothing else (None: no view count or no length yet)."""
    sn, st, cd = item.get("snippet") or {}, item.get("statistics") or {}, item.get("contentDetails") or {}
    secs = duration_s(cd.get("duration", ""))
    if "viewCount" not in st or not secs or not item.get("id"):
        return None
    return {"video_id": item["id"], "channel_id": sn.get("channelId", ""), "title": str(sn.get("title", ""))[:200],
            "views": int(st["viewCount"]), "published": sn.get("publishedAt", ""), "duration_s": secs}


def _chunks(seq: list, n: int = 50) -> list[list]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def scan(api, channels: list[dict], budget: Budget | None = None) -> dict:
    """{videos: [public metadata], channels: {id: {uploads, read} or {error}}, units}. One channel's error is noted
    and the scan goes on; running out of budget stops it."""
    budget = budget or Budget()
    report: dict[str, dict] = {}
    ids = [c["id"] for c in channels]
    uploads: dict[str, str] = {}
    for batch in _chunks(ids):
        budget.spend()
        resp = api.channels().list(part="contentDetails", id=",".join(batch), maxResults=50,
                                   fields=CHANNEL_FIELDS).execute()
        for it in resp.get("items", []):
            uploads[it["id"]] = it["contentDetails"]["relatedPlaylists"]["uploads"]
    videos = []
    for cid in ids:
        if cid not in uploads:
            report[cid] = {"error": "channel not found"}
            continue
        try:
            budget.spend(2)
            items = api.playlistItems().list(part="contentDetails", playlistId=uploads[cid], maxResults=RECENT,
                                             fields=PLAYLIST_FIELDS).execute().get("items", [])
            vids = [i["contentDetails"]["videoId"] for i in items]
            got = api.videos().list(part="snippet,statistics,contentDetails", id=",".join(vids), maxResults=50,
                                    fields=VIDEO_FIELDS).execute().get("items", []) if vids else []
        except QuotaExceeded:
            raise
        except Exception as exc:  # a removed or private channel must not stop the other eighteen
            report[cid] = {"error": type(exc).__name__}
            continue
        rows = [r for r in map(public_metadata, got) if r and r["channel_id"] == cid]
        videos += rows
        report[cid] = {"uploads": len(vids), "read": len(rows)}
    return {"videos": videos, "channels": report, "units": budget.units}


def flag(videos: list[dict], ratio: float = OUTLIER_RATIO, floors: dict | None = None,
         min_samples: int = MIN_SAMPLES) -> list[dict]:
    """The outliers, best ratio first: each with its kind, its channel's median for that kind and the ratio."""
    floors = floors or FLOOR
    groups: dict[tuple[str, str], list[dict]] = {}
    for v in videos:
        groups.setdefault((v["channel_id"], kind(v["duration_s"])), []).append(v)
    out = []
    for (_, k), rows in groups.items():
        if len(rows) < min_samples:
            continue
        median = float(statistics.median(r["views"] for r in rows))
        if median <= 0:
            continue
        for r in rows:
            x = r["views"] / median
            if x >= ratio and r["views"] >= floors[k]:
                out.append({**{f: r[f] for f in KEEP}, "kind": k, "median": round(median), "ratio": round(x, 2)})
    return sorted(out, key=lambda r: (-r["ratio"], r["video_id"]))
