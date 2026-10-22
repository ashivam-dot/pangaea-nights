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
# What the channel's OAuth grant holds; asking for any other scope on refresh fails with invalid_scope.
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.force-ssl",
          "https://www.googleapis.com/auth/yt-analytics.readonly"]


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
    info = json.loads(raw)
    creds = Credentials.from_authorized_user_info(info, info.get("scopes") or SCOPES)
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
    """The next HH:MM UTC at least twenty minutes away, plus day_offset days, as RFC 3339."""
    now = now or datetime.now(timezone.utc)
    at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    while at < now + timedelta(minutes=20):
        at += timedelta(days=1)
    return (at + timedelta(days=day_offset)).strftime("%Y-%m-%dT%H:%M:%SZ")


def episode_slot(now: datetime | None = None) -> str:
    """The long episode's premiere: the next config.EPISODE_SLOT_UTC (21:00 UTC), as RFC 3339."""
    return publish_at(0, *config.EPISODE_SLOT_UTC, now=now)


def soon(minutes: int = 20, now: datetime | None = None) -> str:
    """A release time a few minutes ahead (publishAt must be in the future once processing ends), as RFC 3339."""
    at = (now or datetime.now(timezone.utc)) + timedelta(minutes=minutes)
    return at.replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    # the same read-back as every Short: made-for-kids (as YouTube classified it), privacy and the release time
    want = _instant(when) if not existing else (_instant(video["status"].get("publishAt")) or _instant(when))
    result = {"video_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}",
              "title": video["snippet"]["title"], "privacy": video["status"]["privacyStatus"],
              "publish_at": video["status"].get("publishAt"), "existing": bool(existing),
              "made_for_kids": video["status"].get("madeForKids"),
              "self_declared_made_for_kids": video["status"].get("selfDeclaredMadeForKids"),
              "readback_problems": readback_problems(video, want)}
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


# ---------------------------------------------------------------------------------------------------- ladder Shorts

CHANNEL_URL = "https://www.youtube.com/@PangaeaNights"
LADDER_DISCLOSURE = ("How this was made: the images are AI-generated artistic reconstructions, not photographs; the "
                     "narration is a synthetic voice and the music is generated in code. The real answer is checked "
                     "against the Wikipedia revision linked below.")


def slot_publish_at(slot: str, build_day, now: datetime | None = None, late_ok: bool = False) -> str | None:
    """The slot's release time for a build on build_day (slot A 14:30 UTC the same day, slot B 00:30 UTC the next),
    or None when it is less than twenty minutes away: a late build skips its slot rather than drift into the next.
    late_ok: the time regardless (an upload that already exists keeps its slot)."""
    from . import ladder

    hour, minute = ladder.SLOTS[slot]
    at = datetime(build_day.year, build_day.month, build_day.day, hour, minute, tzinfo=timezone.utc)
    if slot == "B":
        at += timedelta(days=1)
    if not late_ok and at < (now or datetime.now(timezone.utc)) + timedelta(minutes=20):
        return None
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "day", "last", "one", "night", "with", "from", "sleep", "hours", "documentary"}
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in stop}


def related_episode(data: dict, hint: str) -> dict | None:
    """The published long-form episode closest to a premise's related_topic_hint (word overlap with its title, topic
    and series), the most recent on a tie; None before the first episode."""
    eps = [e for e in data.get("episodes", []) if e.get("video_id") and e.get("url")]
    if not eps:
        return None
    want = _tokens(hint)
    scored = [(len(want & _tokens(" ".join(str(e.get(k, "")).replace("_", " ") for k in ("title", "topic", "series")))),
               i, e) for i, e in enumerate(eps)]
    return max(scored, key=lambda x: (x[0], x[1]))[2]


def ladder_marker(p: dict, test_run: str | None = None) -> str:
    base = f"{MARKER}short-{p['id']}"
    return f"{base}-test-{test_run}" if test_run else base


def ladder_tags(p: dict, marker: str) -> list[str]:
    out, total = [], len(marker) + 2
    for t in [*p.get("tags", []), "Pangaea Nights"]:
        t = _clean(t).strip()
        if not t or t in out or total + len(t) + 3 > 480:
            continue
        out.append(t)
        total += len(t) + 3
    return out + [marker]


def ladder_description(p: dict, episode: dict | None) -> str:
    """Premise, the most related documentary (or the channel), the AI disclosure, and the sourced revision."""
    f = p["fact"]
    src = f"https://en.wikipedia.org/w/index.php?oldid={f['wiki_revid']}"
    doc = f"Full documentary: {episode['url']}" if episode else f"Full documentaries: {CHANNEL_URL}"
    lines = [p["description"], "", doc, "", LADDER_DISCLOSURE, "",
             f"Source: Wikipedia, {f['wiki_title']} (revision {f['wiki_revid']}): {src}", "",
             "#shorts #prehistoric #deeptime"]
    return _clean("\n".join(lines))[:4900]


def ladder_body(p: dict, desc: str, tag_list: list[str], when: str | None) -> dict:
    """The videos.insert body. Always category 27, synthetic media declared, not made for kids, private until
    publishAt. when=None is a private test upload that never releases."""
    status = {"privacyStatus": "private", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True,
              "embeddable": True, "license": "youtube", "publicStatsViewable": True}
    if when:
        status["publishAt"] = when
    return {"snippet": {"title": _clean(p["title"])[:100], "description": desc, "tags": tag_list,
                        "categoryId": config.CATEGORY_EDUCATION, "defaultLanguage": "en",
                        "defaultAudioLanguage": "en"}, "status": status}


def _instant(text: str | None) -> str | None:
    return text.replace(".000Z", "Z") if text else None


def _released(when: str | None, now: datetime | None) -> bool:
    if not when:
        return False
    at = datetime.strptime(_instant(when), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return at <= (now or datetime.now(timezone.utc))


def readback_problems(video: dict, when: str | None, now: datetime | None = None) -> list[str]:
    """Where the uploaded video's status (from videos.list) differs from what was asked (empty = it matches). Once
    the release time has passed (a re-run that found the earlier upload), public is the expected privacy."""
    st = video.get("status", {})
    out = []
    if video.get("snippet", {}).get("channelId") != config.CHANNEL_ID:
        out.append("the video is on another channel")
    if st.get("madeForKids") is not False:
        out.append(f"madeForKids is {st.get('madeForKids')!r}, not False")
    if st.get("selfDeclaredMadeForKids") is not False:
        out.append(f"selfDeclaredMadeForKids is {st.get('selfDeclaredMadeForKids')!r}, not False")
    if _released(when, now):
        if st.get("privacyStatus") not in ("private", "public"):
            out.append(f"privacyStatus is {st.get('privacyStatus')!r} after the release time")
    elif st.get("privacyStatus") != "private":
        out.append(f"privacyStatus is {st.get('privacyStatus')!r}, not 'private' before release")
    if _instant(st.get("publishAt")) != when and not (_released(when, now) and st.get("privacyStatus") == "public"):
        out.append(f"publishAt is {st.get('publishAt')!r}, not {when!r}")
    if "containsSyntheticMedia" in st and st["containsSyntheticMedia"] is not True:
        out.append("containsSyntheticMedia is not True")
    return out


def upload_ladder(api, channel: dict, path: Path, p: dict, desc: str, tag_list: list[str], when: str | None) -> dict:
    """Upload (or find again, by marker tag) one ladder Short, then read its status back with videos.list."""
    marker = tag_list[-1]
    existing = find(api, channel, marker)
    video_id = existing["id"] if existing else _insert(api, path, ladder_body(p, desc, tag_list, when))
    video = api.videos().list(part="snippet,status", id=video_id).execute()["items"][0]
    st = video["status"]
    return {"video_id": video_id, "url": f"https://www.youtube.com/shorts/{video_id}", "existing": bool(existing),
            "title": video["snippet"]["title"], "privacy": st.get("privacyStatus"),
            "publish_at": _instant(st.get("publishAt")), "made_for_kids": st.get("madeForKids"),
            "self_declared_made_for_kids": st.get("selfDeclaredMadeForKids"),
            "readback_problems": readback_problems(video, when)}


def first_comment(api, video_id: str, text: str) -> str:
    """Post the channel's own first comment once: 'posted', 'exists', or 'pending: <reason>' (comments may be
    refused while a video is private; the learner retries after release)."""
    from googleapiclient.errors import HttpError

    try:
        rows = api.commentThreads().list(part="snippet", videoId=video_id, maxResults=50,
                                         textFormat="plainText").execute().get("items", [])
        for r in rows:
            top = r["snippet"]["topLevelComment"]["snippet"]
            mine = top.get("authorChannelId", {}).get("value") == config.CHANNEL_ID
            if mine and top.get("textOriginal", top.get("textDisplay", "")).strip() == text:
                return "exists"
        api.commentThreads().insert(part="snippet", body={"snippet": {
            "videoId": video_id, "topLevelComment": {"snippet": {"textOriginal": text}}}}).execute()
        return "posted"
    except HttpError as e:
        return f"pending: HTTP {e.resp.status}"


def analytics_client():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    raw = os.environ.get("YOUTUBE_OAUTH_JSON", "")
    if not raw:
        raise Refused("YOUTUBE_OAUTH_JSON is not set")
    info = json.loads(raw)
    creds = Credentials.from_authorized_user_info(info, info.get("scopes") or SCOPES)
    if not creds.valid:
        creds.refresh(Request())
    return build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)
