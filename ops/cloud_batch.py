"""One resumable batch tick: adopt/build previews, review exact media, then release with owner readback.

Run only in the channel's existing youtube environment. A serialized workflow owns this coordinator;
state is committed before dispatch/upload, so ambiguous network failures are reconciled, never blindly retried.
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from . import media_review

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
STATE_FILE = "batch-production.json"
PLAN_FILE = "strategy/batch-2026-10-25.json"
FINAL = {"public", "scheduled"}
MAX_ATTEMPTS = 2


class TransientOperation(RuntimeError):
    """An uncertain GitHub/network result is not an editorial rejection."""


def now():
    return datetime.now(timezone.utc)


def stamp():
    return now().isoformat(timespec="seconds")


def gh(*args, body=None, raw=False):
    command = ["gh", *args]
    if body is not None:
        command += ["--input", "-"]
    p = subprocess.run(command, input=json.dumps(body).encode() if body is not None else None,
                       capture_output=True, timeout=180)
    if p.returncode:
        # No provider payloads or credential-bearing environment values enter the state/log.
        raise TransientOperation(f"GitHub operation {args[0]} {args[1]} failed (exit {p.returncode})")
    return p.stdout if raw else json.loads(p.stdout) if p.stdout.strip() else None


def persist(data):
    data["updated_at"] = stamp()
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / STATE_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    if not (STATE / ".git").exists():
        raise RuntimeError("batch state must be a durable state-branch worktree")
    def git(*args, check=True):
        return subprocess.run(["git", "-C", str(STATE), *args], check=check, capture_output=True, text=True)
    git("config", "user.name", "ashivam-dot")
    git("config", "user.email", "335661989+ashivam-dot@users.noreply.github.com")
    git("add", "-A")
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    git("commit", "-q", "-m", "batch: checkpoint reviewed production and delivery")
    # A conflict must stop before the next external write. The next tick can recover upload markers.
    git("pull", "-q", "--rebase", "origin", "state")
    git("push", "-q", "origin", "HEAD:state")


def unpack(blob: bytes, folder: Path):
    """Only bounded regular files below this directory; artifact paths are never trusted."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        infos = archive.infolist()
        if len(infos) > 500 or sum(i.file_size for i in infos) > 600_000_000:
            raise ValueError("oversized production artifact")
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("unsafe production artifact path")
            if info.is_dir():
                continue
            target = folder.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)


def load():
    if (STATE / STATE_FILE).exists():
        return json.loads((STATE / STATE_FILE).read_text())
    plan = json.loads((ROOT / PLAN_FILE).read_text())
    seed_file = ROOT / "strategy/batch-adopt-2026-10-25.json"
    adopt = json.loads(seed_file.read_text()) if seed_file.exists() else {}
    jobs = []
    for entry in plan["jobs"]:
        job = {k: v for k, v in entry.items() if k not in ("run_id", "stage")}
        job.update(stage="planned", attempts=0)
        job.update(adopt.get(entry["premise"], {}))
        jobs.append(job)
    return {"schema": 1, "enabled": True, "created_at": stamp(), "jobs": jobs, "halted": None}


def release_time(job, taken, at):
    target = job["target"]
    if target == "now-after-review":
        return None
    when = datetime.fromisoformat(target.replace("Z", "+00:00"))
    if when <= at + timedelta(minutes=20):
        raise ValueError("planned release is too close or past; reconcile the calendar, never silently shift it")
    if target in taken:
        raise ValueError("the requested release slot is already occupied")
    return target


def builds(repo, workflow):
    return gh("api", f"repos/{repo}/actions/workflows/{workflow}/runs?per_page=60")["workflow_runs"]


def resolve(repo, workflow, job):
    found = [r for r in builds(repo, workflow) if r["id"] not in job["before_ids"]
             and r["head_sha"] == job["source_sha"] and r["event"] == "workflow_dispatch"]
    if len(found) == 1:
        job.update(run_id=found[0]["id"], stage="building")
        return True
    return False


def archive_review(repo, job, folder):
    """A private, durable review receipt and exact MP4 survive Actions artifact expiry."""
    tag = f"batch-reviewed-{job['run_id']}"
    exists = subprocess.run(["gh", "release", "view", tag, "-R", repo], capture_output=True).returncode == 0
    if not exists:
        gh("release", "create", tag, "--draft", "--title", f"Reviewed batch {job['premise']}", "--notes",
           "Exact rendered media and AI audiovisual review. Original channel gates and upload readback still apply.",
           "-R", repo, raw=True)
    files = [str(p) for p in sorted(folder.rglob("*.json")) if p.name != "review-status.json"]
    files += [str(next(folder.rglob("short.mp4")))]
    # Upload unique basenames; the two production formats have no duplicate report names within one Short.
    if len({Path(p).name for p in files}) != len(files):
        raise ValueError("ambiguous review archive filenames")
    gh("release", "upload", tag, *files, "--clobber", "-R", repo, raw=True)
    return tag


def materialize(repo, adapter, job):
    folder = ROOT / "work/batch" / str(job["run_id"])
    items = gh("api", f"repos/{repo}/actions/runs/{job['run_id']}/artifacts?per_page=100")["artifacts"]
    for name in adapter.artifact_names(job):
        artifact = next((a for a in items if a["name"] == name and not a.get("expired")), None)
        if artifact is None:
            raise ValueError(f"finished run lacks required {name} artifact")
        unpack(gh("api", f"repos/{repo}/actions/artifacts/{artifact['id']}/zip", raw=True), folder)
    if job.get("archive"):
        gh("release", "download", job["archive"], "--pattern", "av-review.json", "--dir", str(folder),
           "--clobber", "-R", repo, raw=True)
    return folder


def tick(adapter, mode, save=persist):
    repo = adapter.REPO
    data = load()
    if adapter.paused():
        print("batch held: channel pause switch is active")
        return data
    if data.get("halted"):
        print("batch held:", data["halted"])
        return data
    save(data)
    reviewed = 0
    for job in data["jobs"]:
        if job["stage"] in FINAL or job["stage"] == "held":
            continue
        if job.get("review_retry_after") and now() < datetime.fromisoformat(job["review_retry_after"]):
            continue
        if job["stage"] == "dispatch_unresolved":
            if not resolve(repo, adapter.WORKFLOW, job):
                # Leave intent pending. Later ticks may see the run after GitHub's eventual-consistency delay.
                if (now() - datetime.fromisoformat(job["submitted_at"])).total_seconds() > 900:
                    data["halted"] = "Uncertain dispatch; reconcile the run before any replacement"
                save(data)
                return data
        if job["stage"] not in ("building", "reviewed", "upload_intent"):
            continue
        if job["stage"] == "building":
            run = gh("api", f"repos/{repo}/actions/runs/{job['run_id']}")
            if job.get("source_sha") and run["head_sha"] != job["source_sha"]:
                raise ValueError("production source differs from the saved dispatch intent")
            job.update(cloud_status=run["status"], source_sha=run["head_sha"])
            if run["status"] != "completed":
                continue
            if run["conclusion"] != "success":
                job.update(stage="held", problems=[f"production run {run['conclusion']}"])
                save(data)
                break
        if reviewed >= 2:
            break
        try:
            had_intent = job["stage"] == "upload_intent"
            folder = materialize(repo, adapter, job)
            mp4, brief = adapter.validate(job, folder)
            receipt = media_review.review(mp4, brief, folder)
            problems = media_review.receipt_problems(receipt, mp4, brief)
            job["review"] = {"model": receipt["model"], "video_sha256": receipt["video_sha256"],
                             "verdict": receipt["verdict"], "duration": receipt["duration"]}
            job["archive"] = archive_review(repo, job, folder)
            reviewed += 1
            if problems:
                raise ValueError("; ".join(problems))
            if not had_intent:
                job["stage"] = "reviewed"
            save(data)
            if mode != "publish":
                continue
            if not adapter.release_ready(data, job):
                continue
            # Fresh ledger, duplicate/rotation and channel checks must still pass after review.
            when = job.get("intended_publish_at") if had_intent else release_time(job, adapter.taken(), now())
            job.update(stage="upload_intent", intended_publish_at=when, upload_intent_at=stamp())
            save(data)  # durable intent BEFORE the only YouTube write
            result = adapter.publish(job, folder, when)
            job.update(delivery=result, stage="public" if result.get("privacy") == "public" else "scheduled")
            save(data)
        except (media_review.ReviewHold, TransientOperation) as exc:
            job["retry_reason"] = str(exc)
            job["review_retry_after"] = (now() + timedelta(minutes=30)).isoformat()
            save(data)
            # A quota outage is not a bad story and never lowers its review requirement.
            break
        except Exception as exc:
            if job["stage"] == "upload_intent":
                job["retry_reason"] = f"{type(exc).__name__}: {str(exc)[:400]}"
                save(data)
                # Keep the intent and marker; the next run finds the same upload rather than inserting twice.
                raise
            job.update(stage="held", problems=[f"{type(exc).__name__}: {str(exc)[:700]}"])
            save(data)
            break
    held = [j for j in data["jobs"] if j["stage"] == "held"]
    if held:
        # Rotation is a sequence. A refused entry must be repaired or deliberately re-planned before successors.
        data["halted"] = f"Repair held entry {held[0]['premise']}: " + "; ".join(held[0].get("problems", []))
    if not data.get("halted") and not any(j["stage"] in ("building", "dispatch_unresolved") for j in data["jobs"]):
        # Avoid competing with an inherited manual or scheduled production run.
        active = [r for r in builds(repo, adapter.WORKFLOW) if r["status"] != "completed"]
        pending = next((j for j in data["jobs"] if j["stage"] == "planned"), None)
        if pending and not active:
            head = gh("api", f"repos/{repo}/commits/main")["sha"]
            pending.update(stage="dispatch_unresolved", source_sha=head,
                           before_ids=[r["id"] for r in builds(repo, adapter.WORKFLOW)], submitted_at=stamp(),
                           attempts=pending.get("attempts", 0) + 1)
            save(data)
            gh("api", f"repos/{repo}/actions/workflows/{adapter.WORKFLOW}/dispatches", body={
                "ref": "main", "inputs": pending["inputs"]}, raw=True)
    data["complete"] = all(j["stage"] in FINAL for j in data["jobs"])
    data["health"] = {"public": sum(j["stage"] == "public" for j in data["jobs"]),
                      "scheduled": sum(j["stage"] == "scheduled" for j in data["jobs"]),
                      "held": len(held), "mode": mode}
    save(data)
    print(json.dumps(data["health"]))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("kahaani", "studio"), required=True)
    parser.add_argument("--mode", choices=("review", "publish"), default="review")
    args = parser.parse_args()
    adapter = importlib.import_module(f"{args.adapter}.batch_release")
    if os.environ.get("GITHUB_REPOSITORY") != adapter.REPO:
        raise SystemExit("refusing a batch coordinator outside its own channel repository")
    if args.mode == "publish" and not os.environ.get("YOUTUBE_OAUTH_JSON"):
        raise SystemExit("publishing requires this channel's existing youtube environment")
    data = tick(adapter, args.mode)
    if data.get("halted"):
        raise SystemExit("batch held: " + data["halted"])


if __name__ == "__main__":
    main()
