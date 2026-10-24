"""Release exact reviewed choice-ladder previews; sleep documentaries are never touched."""
import json
import os
import time
from datetime import datetime, timezone

from . import config, ladder, ladder_cli, ledger

REPO = "ashivam-dot/pangaea-nights"
WORKFLOW = "shorts.yml"


def paused():
    return ladder_cli.paused()


def artifact_names(job):
    return ["ladder-plan", "ladder-out-A"]


def _read(folder, name):
    files = list(folder.rglob(name))
    if len(files) != 1:
        raise ValueError(f"expected one {name}, found {len(files)}")
    return json.loads(files[0].read_text())


def validate(job, folder):
    plan, gates = _read(folder, "plan.json"), _read(folder, "gates.json")
    p = plan["premise"]
    if p["id"] != job["premise"] or plan.get("mode") != "dry":
        raise ValueError("artifact differs from the intended dry preview")
    if gates.get("ok") is not True or gates.get("failures"):
        raise ValueError("preview failed original image/voice/render gates")
    current = ladder.find(ladder.with_proven(ladder.load_banks(strict=False), ledger.load()), p["id"])
    if current != p:
        raise ValueError("the pinned premise changed after this render; rerender it")
    data = ledger.load()
    own = next((r for r in data["shorts"] if r.get("premise") == p["id"]), None)
    if own:
        # A committed upload from this exact reviewed run is recovery, never permission for a second video.
        if str(own.get("run")) != str(job["run_id"]):
            raise ValueError("premise is already owned by a different published run")
        data["shorts"] = [r for r in data["shorts"] if r is not own]
    from . import qa

    if not gates.get("frame0"):
        raise ValueError("preview has no opening-image fingerprint")
    similarity = max((qa.cosine(qa.unpack(gates["frame0"]), qa.unpack(r["frame0"]))
                      for r in data["shorts"][-ladder.HISTORY:] if r.get("frame0")), default=0)
    if similarity > 0.93:
        raise ValueError(f"opening image is too close to release history ({similarity:.3f})")
    day = (datetime.now(timezone.utc) if job["target"] == "now-after-review" else
           datetime.fromisoformat(job["target"].replace("Z", "+00:00"))).date()
    fresh = ladder.plan_problems(p, data, day)
    bad = [k for k, v in fresh.items() if not v.get("ok")]
    if bad:
        raise ValueError("fresh editorial gates failed: " + ", ".join(bad))
    videos = list(folder.rglob("short.mp4"))
    if len(videos) != 1:
        raise ValueError("no unique finished MP4")
    return videos[0], {"channel": "Pangaea Nights choice-ladder Shorts", "audience": "English general audience",
                       "premise": p, "spoken_script": ladder.script_text(p),
                       "look": "Earth deep time and oceans. No humans. Cinematic foreground lighting may illuminate "
                               "deep-sea animals, but kilometre-deep water has no sunlight or bright surface sky.",
                       "expected_narrator": "the channel's existing English narrator"}


def taken():
    return {r["publish_at"] for r in ledger.load()["shorts"] if r.get("publish_at")}


def release_ready(data, job):
    return all(j["stage"] in ("public", "scheduled") for j in data["jobs"][:data["jobs"].index(job)])


def publish(job, folder, when):
    from . import youtube

    validate(job, folder)
    plan, gates = _read(folder, "plan.json"), _read(folder, "gates.json")
    p = plan["premise"]
    api, data = youtube.client(), ledger.load()
    channel = youtube.identity(api)
    marker = youtube.ladder_marker(p)
    tags = youtube.ladder_tags(p, marker)
    episode = youtube.related_episode(data, p["related_topic_hint"])
    # Always upload privately first. No unverified metadata or processing failure can become an immediate release.
    result = youtube.upload_ladder(api, channel, next(folder.rglob("short.mp4")), p,
                                   youtube.ladder_description(p, episode), tags, when)
    video_id = result["video_id"]
    for _ in range(41):
        video = api.videos().list(part="snippet,status,processingDetails", id=video_id).execute()["items"][0]
        st = video["status"]
        if st.get("uploadStatus") in ("failed", "rejected", "deleted"):
            raise RuntimeError("YouTube rejected this upload; preserve its id and investigate")
        if st.get("uploadStatus") == "processed" or video.get("processingDetails", {}).get("processingStatus") == "succeeded":
            break
        time.sleep(15)
    else:
        raise RuntimeError("upload is still processing; next tick reconciles the same marker")
    if not when and st.get("privacyStatus") != "public":
        # An immediate release uses the very same disclosure and audience settings as scheduled uploads.
        problems = youtube.readback_problems(video, None)
        if problems:
            raise RuntimeError("private pre-release readback: " + "; ".join(problems))
        api.videos().update(part="status", body={"id": video_id, "status": {
            "privacyStatus": "public", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True,
            "embeddable": True, "license": "youtube", "publicStatsViewable": True}}).execute()
    for delay in (0, 5, 10, 20):
        time.sleep(delay)
        video = api.videos().list(part="snippet,status", id=video_id).execute()["items"][0]
        st = video["status"]
        problems = youtube.readback_problems(video, when) if when else [
            issue for issue in youtube.readback_problems(video, None) if not issue.startswith("privacyStatus")]
        if not when and st.get("privacyStatus") != "public":
            problems.append("immediate release is not public")
        if not problems:
            break
    else:
        api.videos().update(part="status", body={"id": video_id, "status": {
            "privacyStatus": "private", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True}}).execute()
        raise RuntimeError("release readback mismatch; kept private: " + "; ".join(problems))
    released_at = when or video["snippet"].get("publishedAt") or ledger.now()
    result.update(privacy=st["privacyStatus"], publish_at=when, readback_problems=[],
                  owner_readback={"channel_id": video["snippet"]["channelId"],
                                  **{k: st.get(k) for k in ("privacyStatus", "publishAt", "madeForKids",
                                                           "selfDeclaredMadeForKids", "containsSyntheticMedia",
                                                           "uploadStatus")}})
    entry = {"premise": p["id"], "series": p["series"], "era": p["era"], "slot": job.get("slot", "A"),
             "video_id": video_id, "url": result["url"], "title": p["title"], "publish_at": released_at,
             "uploaded_at": ledger.now(), "script": plan["script"], "frame0": gates.get("frame0"),
             "related": (episode or {}).get("url"), "run": str(job["run_id"]), "checkpoints": {},
             "av_review_sha256": job["review"]["video_sha256"], "owner_readback": result["owner_readback"]}
    ledger.record_short(data, entry)
    data["ladder"]["queue"] = [q for q in data["ladder"]["queue"] if q != p["id"]]
    ledger.save(data)
    return result
