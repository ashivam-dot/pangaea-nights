"""What the ladder Shorts' own numbers say, fed back into which premise runs next (cron 06:00 UTC).

Each published Short is read twice: 24 h and 72 h after release (YouTube Analytics: views and average percentage
viewed; the Data API's view count when Analytics has no row yet). The 72 h reading decides:

- double down: views at least 2x the median of the Shorts released in the 14 days before it, AND at least 70%
  viewed. Its series x era weight goes up x1.5, and a pt.2 (the next unused premise of the same series and era) is
  queued to run next; when the bank has none, the pair is flagged needs_bank and a Telegram message asks for one;
- below the median: one more in that series x era's streak; six in a row halve its weight (and reset the streak);
- otherwise the streak resets.

The median needs MIN_SAMPLES earlier Shorts, so the first few Shorts only record numbers. Weights stay within
WEIGHT_RANGE; the rotation rules in ladder.pick still hold whatever the weights say. Every run also retries any
first comment that was refused while the video was private, and writes one summary to the ledger and Telegram.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

CHECKPOINTS = (24, 72)
TRAILING_DAYS = 14
MIN_SAMPLES = 3
DOUBLE_VIEWS = 2.0
DOUBLE_VIEWED = 70.0
UP, DOWN = 1.5, 0.5
STREAK = 6
WEIGHT_RANGE = (0.125, 4.0)


def _when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace(".000Z", "Z").replace("Z", "+00:00"))


def age_h(short: dict, at: datetime) -> float:
    return (at - _when(short["publish_at"])).total_seconds() / 3600


def due(short: dict, at: datetime) -> list[int]:
    """Checkpoints this Short has reached but not yet recorded."""
    if not short.get("video_id") or not short.get("publish_at"):
        return []
    done = short.get("checkpoints", {})
    return [h for h in CHECKPOINTS if age_h(short, at) >= h and str(h) not in done]


def key(short: dict) -> str:
    return f"{short['series']}|{short['era']}"


def fetch(analytics, api, ids: list[str], start: str, end: str) -> dict:
    """{video id: {views, viewed_pct, source}} from Analytics, falling back to the Data API's view count."""
    out: dict[str, dict] = {}
    if not ids:
        return out
    if analytics is not None:
        try:
            rep = analytics.reports().query(ids="channel==MINE", startDate=start, endDate=end,
                                            metrics="views,averageViewPercentage,averageViewDuration",
                                            dimensions="video", filters="video==" + ",".join(ids)).execute()
            for row in rep.get("rows") or []:
                out[row[0]] = {"views": int(row[1]), "viewed_pct": float(row[2]), "view_s": float(row[3]),
                               "source": "analytics"}
        except Exception as exc:  # Analytics lags by a day or two; the Data API still has the view count
            print(f"analytics: {type(exc).__name__}"[:200])
    missing = [i for i in ids if i not in out]
    if missing and api is not None:
        items = api.videos().list(part="statistics", id=",".join(missing)).execute().get("items", [])
        for v in items:
            out[v["id"]] = {"views": int(v.get("statistics", {}).get("viewCount", 0)), "viewed_pct": None,
                            "source": "data-api"}
    return out


def record_checkpoints(data: dict, metrics: dict, at: datetime) -> list[tuple[dict, int]]:
    """Write each due checkpoint that has numbers; returns the (short, hours) pairs written."""
    out = []
    for s in data.get("shorts", []):
        m = metrics.get(s.get("video_id"))
        if not m:
            continue
        for h in due(s, at):
            s.setdefault("checkpoints", {})[str(h)] = {**m, "age_h": round(age_h(s, at), 1),
                                                       "at": at.strftime("%Y-%m-%dT%H:%M:%SZ")}
            out.append((s, h))
    return out


def trailing_median(data: dict, short: dict) -> float | None:
    """Median 72 h views of the Shorts released in the TRAILING_DAYS before this one (None below MIN_SAMPLES)."""
    t = _when(short["publish_at"])
    views = [s["checkpoints"]["72"]["views"] for s in data.get("shorts", [])
             if s is not short and s.get("publish_at") and "72" in s.get("checkpoints", {})
             and t - timedelta(days=TRAILING_DAYS) <= _when(s["publish_at"]) < t]
    return float(statistics.median(views)) if len(views) >= MIN_SAMPLES else None


def decide(views: float, viewed_pct: float | None, median: float | None) -> str:
    """'double', 'below', 'ok', or 'learning' (no median yet)."""
    if median is None:
        return "learning"
    if views >= DOUBLE_VIEWS * median and viewed_pct is not None and viewed_pct >= DOUBLE_VIEWED:
        return "double"
    if views < median:
        return "below"
    return "ok"


def _scale(data: dict, k: str, factor: float) -> float:
    w = data["ladder"]["weights"]
    w[k] = round(min(max(float(w.get(k, 1.0)) * factor, WEIGHT_RANGE[0]), WEIGHT_RANGE[1]), 4)
    return w[k]


def part_two(banks: dict, data: dict, short: dict) -> dict | None:
    """The next unused premise of the same series and era, if the bank has one."""
    used = {s["premise"] for s in data.get("shorts", []) if s.get("video_id")} | set(data["ladder"]["queue"])
    return next((p for p in banks.get(short["series"], []) if p["era"] == short["era"] and p["id"] not in used),
                None)


def apply(data: dict, short: dict, banks: dict) -> dict:
    """Fold one 72 h result into weights, streaks and the queue. Returns what happened."""
    c = short["checkpoints"]["72"]
    median = trailing_median(data, short)
    verdict = decide(c["views"], c.get("viewed_pct"), median)
    k = key(short)
    lad = data["ladder"]
    event = {"premise": short["premise"], "video_id": short["video_id"], "key": k, "views_72h": c["views"],
             "viewed_pct": c.get("viewed_pct"), "median": median, "verdict": verdict}
    if verdict == "double":
        event["weight"] = _scale(data, k, UP)
        lad["streaks"][k] = 0
        nxt = part_two(banks, data, short)
        if nxt:
            lad["queue"].append(nxt["id"])
            event["queued"] = nxt["id"]
        elif k not in lad["needs_bank"]:
            lad["needs_bank"].append(k)
            event["needs_bank"] = k
    elif verdict == "below":
        lad["streaks"][k] = lad["streaks"].get(k, 0) + 1
        if lad["streaks"][k] >= STREAK:
            event["weight"] = _scale(data, k, DOWN)
            lad["streaks"][k] = 0
    elif verdict == "ok":
        lad["streaks"][k] = 0
    short["verdict"] = verdict
    return event


def summary(data: dict, events: list[dict], written: list[tuple[dict, int]], comments: list[dict],
            at: datetime) -> dict:
    lad = data["ladder"]
    day = at.date().isoformat()
    released = [s for s in data.get("shorts", []) if s.get("publish_at", "")[:10] == day]
    return {"day": day, "checkpoints": [{"premise": s["premise"], "h": h, **s["checkpoints"][str(h)]}
                                        for s, h in written],
            "decisions": events, "comments": comments, "released_today": [s["premise"] for s in released],
            "weights": dict(lad["weights"]), "queue": list(lad["queue"]), "needs_bank": list(lad["needs_bank"])}


def message(sm: dict) -> str:
    lines = [f"Pangaea Nights Shorts, {sm['day']}"]
    for c in sm["checkpoints"]:
        pct = f", {c['viewed_pct']:.0f}% viewed" if c.get("viewed_pct") is not None else ""
        lines.append(f"- {c['premise']} at {c['h']}h: {c['views']} views{pct}")
    for e in sm["decisions"]:
        extra = f" -> queued {e['queued']}" if e.get("queued") else ""
        extra += f" -> NEEDS BANK: write a pt.2 for {e['needs_bank']}" if e.get("needs_bank") else ""
        extra += f" (weight {e['weight']})" if "weight" in e else ""
        lines.append(f"- {e['premise']}: {e['verdict']}{extra}")
    for c in sm["comments"]:
        lines.append(f"- first comment on {c['premise']}: {c['status']}")
    if not sm["checkpoints"] and not sm["decisions"]:
        lines.append("- nothing new to read")
    return "\n".join(lines)


def run(data: dict, banks: dict, analytics, api, at: datetime | None = None) -> dict:
    """One learner pass. Changes `data` in place; returns the summary (also logged under ladder.summaries)."""
    from . import ladder, ledger, youtube

    at = at or datetime.now(timezone.utc)
    shorts = [s for s in data.get("shorts", []) if due(s, at)]
    metrics = {}
    if shorts:
        start = min(s["publish_at"][:10] for s in shorts)
        metrics = fetch(analytics, api, [s["video_id"] for s in shorts], start, at.date().isoformat())
    written = record_checkpoints(data, metrics, at)
    events = [apply(data, s, banks) for s, h in written if h == 72]
    comments = []
    if api is not None:
        for s in data.get("shorts", []):
            if str(s.get("comment", "")).startswith("pending") and s.get("publish_at") and age_h(s, at) >= 0:
                s["comment"] = youtube.first_comment(api, s["video_id"], ladder.FIRST_COMMENT)
                comments.append({"premise": s["premise"], "status": s["comment"]})
    sm = summary(data, events, written, comments, at)
    ledger.ladder_log(data, "summaries", sm, keep=60)
    return sm
