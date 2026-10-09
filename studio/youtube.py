"""Publishing to the Pangaea Nights channel, and the guard that refuses everything else.

Nothing is uploaded unless: the run is a real publish run, the channel is not paused, the script came from the real
writer and passed every gate, the media passed QA, and the signed-in account's channel id is exactly CHANNEL_ID.
Uploads are idempotent: each carries a marker tag, and a rerun finds the earlier upload instead of duplicating it.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config

MARKER = "pangaea-nights:"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]


class Refused(RuntimeError):
    pass


def refusals(episode: dict, mode: str, paused: bool, qa: dict) -> list[str]:
    """Why this episode must not be published (empty list = may publish)."""
    out = []
    if mode != "publish":
        out.append(f"mode is {mode!r}, not 'publish'")
    if paused:
        out.append("the channel is paused (repository variable PAUSED)")
    if episode.get("writer") != "gemini" or episode.get("publishable") is not True:
        out.append("the script did not come from the real writer (stub scripts are never publishable)")
    if any("stub" in (c.get("warnings") or []) for c in episode.get("chapters", [])):
        out.append("a chapter is marked as stub text")
    if not qa.get("ok"):
        out.append("media QA failed: " + "; ".join(qa.get("problems", [])[:5]))
    return out


def check_identity(channels_response: dict) -> dict:
    items = channels_response.get("items") or []
    if len(items) != 1 or items[0].get("id") != config.CHANNEL_ID:
        raise Refused("the YouTube sign-in is not the Pangaea Nights channel: refusing every write")
    return items[0]


def client():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    raw = os.environ.get("YOUTUBE_OAUTH_JSON", "")
    if not raw:
        raise Refused("YOUTUBE_OAUTH_JSON is not set")
    creds = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)
    if not creds.valid:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def identity(api) -> dict:
    return check_identity(api.channels().list(part="id,snippet,contentDetails", mine=True).execute())


def _clean(text: str) -> str:
    return text.replace("<", "").replace(">", "")


def description(episode: dict, chapters: list[tuple[str, str]], docs: list[dict], credits: list[dict],
                image_model: str, voice_name: str) -> str:
    lines = [episode["summary"], "",
             "A slow, calm journey for falling asleep: soft narration, gentle images and quiet nature sound.", ""]
    lines += [f"{stamp} {title}" for stamp, title in chapters] + [""]
    lines += ["Sources (Wikipedia, the revisions the script was checked against):"]
    lines += [f"• {d['title']}: {d['page_url']}" for d in docs] + [""]
    lines += ["How this was made: the script was written with AI (Gemini) from the sources above and checked "
              "automatically, number by number and name by name, against them. The images are AI-generated "
              f"artistic reconstructions ({image_model}), not photographs or scientific illustrations; the narration "
              f"is a synthetic voice (Kokoro, {voice_name}). The music is generated in code.", ""]
    if credits:
        lines += ["Nature recordings (CC0 public domain):"]
        lines += [f"• {c['title']} by {c['author']}: {c['page']}" for c in credits] + [""]
    lines += ["Spotted an error? Leave a comment with a source and we will note it.", "",
              "#sleepdocumentary #prehistoric #deeptime"]
    text = _clean("\n".join(lines))
    return text[:4900]


def tags(episode: dict) -> list[str]:
    marker = MARKER + episode["id"]
    out, total = [], len(marker) + 2
    for t in [*episode.get("tags", []), "sleep documentary", "Pangaea Nights"]:
        t = _clean(t).strip()
        if not t or t in out or total + len(t) + 3 > 480:
            continue
        out.append(t)
        total += len(t) + 3
    return out + [marker]


def find(api, channel: dict, marker: str) -> dict | None:
    uploads = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    rows = api.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=50).execute()
    ids = [r["contentDetails"]["videoId"] for r in rows.get("items", [])]
    if not ids:
        return None
    vids = api.videos().list(part="snippet,status", id=",".join(ids)).execute().get("items", [])
    hits = [v for v in vids if marker in (v["snippet"].get("tags") or [])]
    if len(hits) > 1:
        raise Refused(f"{marker} is on the channel twice: resolve by hand before publishing again")
    return hits[0] if hits else None


def _insert(api, path: Path, body: dict) -> str:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=16 * 1024 * 1024, resumable=True)
    request = api.videos().insert(part="snippet,status", body=body, notifySubscribers=True, media_body=media)
    response, errors = None, 0
    while response is None:
        try:
            _, response = request.next_chunk(num_retries=5)
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and errors < 8:
                errors += 1
                time.sleep(min(60, 2 ** errors))
                continue
            raise
    return response["id"]


def publish_at(day_offset: int = 0, hour: int = 0, minute: int = 30, now: datetime | None = None) -> str:
    """The next HH:MM UTC at least two hours away, plus day_offset days, as RFC 3339."""
    now = now or datetime.now(timezone.utc)
    at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    while at < now + timedelta(hours=2):
        at += timedelta(days=1)
    return (at + timedelta(days=day_offset)).strftime("%Y-%m-%dT%H:%M:%SZ")


def after(when: str, hours: float) -> str:
    at = datetime.strptime(when, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) + timedelta(hours=hours)
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


def upload(api, channel: dict, path: Path, title: str, desc: str, tag_list: list[str], when: str,
           thumbnail: Path | None = None) -> dict:
    marker = tag_list[-1]
    existing = find(api, channel, marker)
    if existing:
        video_id = existing["id"]
    else:
        body = {"snippet": {"title": _clean(title)[:100], "description": desc, "tags": tag_list,
                            "categoryId": config.CATEGORY_EDUCATION, "defaultLanguage": "en",
                            "defaultAudioLanguage": "en"},
                "status": {"privacyStatus": "private", "publishAt": when, "selfDeclaredMadeForKids": False,
                           "containsSyntheticMedia": True, "embeddable": True, "license": "youtube",
                           "publicStatsViewable": True}}
        video_id = _insert(api, path, body)
    video = api.videos().list(part="snippet,status", id=video_id).execute()["items"][0]
    if video["snippet"]["channelId"] != config.CHANNEL_ID:
        raise Refused("the uploaded video is on another channel")
    if not video["status"].get("publishAt"):
        api.videos().update(part="status", body={"id": video_id, "status": {
            "privacyStatus": "private", "publishAt": when, "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": True, "embeddable": True, "publicStatsViewable": True}}).execute()
        video = api.videos().list(part="snippet,status", id=video_id).execute()["items"][0]
    result = {"video_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}",
              "title": video["snippet"]["title"], "privacy": video["status"]["privacyStatus"],
              "publish_at": video["status"].get("publishAt"), "existing": bool(existing)}
    if thumbnail and thumbnail.exists():
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        try:
            api.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(thumbnail),
                                                                              mimetype="image/jpeg")).execute()
            result["thumbnail"] = "set"
        except HttpError as e:  # custom thumbnails need a phone-verified channel
            result["thumbnail"] = f"not set ({e.resp.status})"
    return result


def short_description(episode_url: str, title: str, summary: str) -> str:
    return _clean(f"{summary}\n\nThe full sleep documentary: {title}\n{episode_url}\n\n"
                  "AI-generated reconstruction images and synthetic narration.\n#shorts #prehistoric #sleep")[:4900]


def short_title(hook: str) -> str:
    t = re.sub(r"\s+", " ", _clean(hook)).strip()
    return (t[:1].upper() + t[1:])[:90] + " #shorts"
