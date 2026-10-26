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
MAX_PREVIEW_AHEAD = 4  # held and completed dry previews share a bounded buffer
MAX_BUILDING = 2  # eight concurrent image shards each; leave four runners for the rest of the fleet


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
    try:
        p = subprocess.run(command, input=json.dumps(body).encode() if body is not None else None,
                           capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise TransientOperation(f"GitHub operation {args[0]} {args[1]} timed out; reconcile before retry") from exc
    if p.returncode:
        # No provider payloads or credential-bearing environment values enter the state/log.
        raise TransientOperation(f"GitHub operation {args[0]} {args[1]} failed (exit {p.returncode})")
    try:
        return p.stdout if raw else json.loads(p.stdout) if p.stdout.strip() else None
    except ValueError as exc:
        raise TransientOperation("GitHub returned an unreadable response; retry without changing the story") from exc


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
    # A later job's manual repair can arrive while an earlier upload is being processed.
    # Merge only independent jobs, preserving both the exact delivery and the new hold.
    # A change to the same job, order, global halt or another state file stops all writes.
    from . import batch_checkpoint as batch_state
    for attempt in range(3):
        try:
            git("fetch", "-q", "origin", "state")
            upstream = git("rev-parse", "FETCH_HEAD").stdout.strip()
            ancestor = git("merge-base", "HEAD", upstream).stdout.strip()
            local = json.loads(git("show", f"HEAD:{STATE_FILE}").stdout)
            before = git("show", f"{ancestor}:{STATE_FILE}", check=False)
            incoming = git("show", f"{upstream}:{STATE_FILE}", check=False)
            if before.returncode and incoming.returncode:
                merged = local  # first initialization, with no competing initializer
            elif before.returncode or incoming.returncode:
                raise batch_state.StateConflict('batch initialization changed during checkpoint')
            else:
                base, remote = json.loads(before.stdout), json.loads(incoming.stdout)
                merged = local if remote == base else batch_state.merge(base, local, remote)
            rebased = git("rebase", upstream, check=False)
            if rebased.returncode:
                conflicts = git("diff", "--name-only", "--diff-filter=U").stdout.splitlines()
                if conflicts != [STATE_FILE]:
                    raise batch_state.StateConflict('state changed outside the batch checkpoint')
                (STATE / STATE_FILE).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
                git("add", STATE_FILE)
                git("-c", "core.editor=true", "rebase", "--continue")
            # Even Git's automatic line merge must refresh the in-memory ledger, or
            # the next checkpoint would overwrite an independent repair with stale data.
            (STATE / STATE_FILE).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
            git("add", STATE_FILE)
            if git("diff", "--cached", "--quiet", check=False).returncode:
                git("commit", "-q", "-m", "batch: retain concurrent independent job repair")
            batch_state.refresh_in_place(data, merged)
        except Exception as exc:
            git("rebase", "--abort", check=False)
            raise TransientOperation('concurrent state edit requires reconciliation before another write') from exc
        if git("push", "-q", "origin", "HEAD:state", check=False).returncode == 0:
            return
    raise TransientOperation('state checkpoint push unavailable; reconcile before another write')


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


def poll_due(data, adapter, at):
    """A booked schedule is verified again after its release time, without changing the channel."""
    for job in data["jobs"]:
        when = job.get("intended_publish_at")
        if job["stage"] != "scheduled" or not when:
            continue
        due = datetime.fromisoformat(when.replace("Z", "+00:00"))
        if due > at:
            continue
        video = adapter.poll_delivery(job)
        st = video.get("status", {})
        problems = []
        if video.get("snippet", {}).get("channelId") != adapter.CHANNEL_ID:
            problems.append("wrong or missing channel identity")
        if st.get("madeForKids") is not False or st.get("selfDeclaredMadeForKids") is not False:
            problems.append("audience status is not confirmed as general audience")
        if st.get("uploadStatus") != "processed":
            problems.append("upload is not confirmed processed")
        if st.get("containsSyntheticMedia") is False:
            problems.append("synthetic-media disclosure is explicitly false")
        if st.get("privacyStatus") != "public" and at > due + timedelta(minutes=15):
            problems.append("scheduled release is still not public fifteen minutes after its slot")
        if problems:
            data["halted"] = f"Delivery {job['delivery']['video_id']}: " + "; ".join(problems)
            return
        if st.get("privacyStatus") == "public":
            job["stage"] = "public"
            job["public_verified_at"] = at.isoformat()
            job["delivery"].update(privacy="public", owner_readback={
                "channel_id": video["snippet"]["channelId"], **{k: st.get(k) for k in (
                    "privacyStatus", "publishAt", "madeForKids", "selfDeclaredMadeForKids",
                    "containsSyntheticMedia", "uploadStatus")}})


def builds(repo, workflow):
    return gh("api", f"repos/{repo}/actions/workflows/{workflow}/runs?per_page=60")["workflow_runs"]


def resolve(repo, workflow, job):
    found = [r for r in builds(repo, workflow) if r["id"] not in job["before_ids"]
             and r["head_sha"] == job["source_sha"] and r["event"] == "workflow_dispatch"
             and (not job.get("dispatch_run_id") or r["id"] == job["dispatch_run_id"])]
    if len(found) == 1:
        job.update(run_id=found[0]["id"], stage="building")
        return True
    return False


def archive_review(repo, job, folder):
    """A private, durable review receipt and exact MP4 survive Actions artifact expiry."""
    tag = f"batch-reviewed-{job['run_id']}"
    try:
        exists = subprocess.run(["gh", "release", "view", tag, "-R", repo], capture_output=True,
                                timeout=60).returncode == 0
    except subprocess.TimeoutExpired as exc:
        raise TransientOperation("review archive lookup timed out; retry the same archive") from exc
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


def restore_review(job, mp4, brief, folder):
    """An archive outage must not erase an editorial verdict between runners."""
    receipt = job.get("review") or {}
    if not receipt:
        return
    target = Path(folder) / "av-review.json"
    if receipt.get("schema") != 1:
        # Older state stored a summary only; its complete archived receipt can still be used.
        if receipt.get("verdict", {}).get("decision") == "hold" and not target.exists():
            raise ValueError("restore the complete saved negative review before retrying this media")
        return
    if receipt.get("video_sha256") != media_review.digest(mp4) \
            or receipt.get("brief_sha256") != media_review.brief_digest(brief):
        raise ValueError("durable review differs from this media or brief; reconcile the repair first")
    if not all(receipt.get(k) for k in ("model", "review_video_sha256", "review_audio_sha256")):
        raise ValueError("durable review lacks complete media provenance")
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


def entry_halt(data):
    """Only a recognised sequence hold permits predecessors to make progress."""
    first = next((j for j in data["jobs"] if j["stage"] == "held"), None)
    return bool(first and str(data.get("halted") or "").startswith(
        f"Repair held entry {first['premise']}: "))


def editorial_problems(job):
    return [h.get("reason") or "unresolved direct editorial review"
            for h in job.get("editorial_holds", []) if not h.get("resolved_at")]


def tick(adapter, mode, save=persist):
    repo = adapter.REPO
    data = load()
    if adapter.paused():
        print("batch held: channel pause switch is active")
        return data
    if data.get("halted") and not entry_halt(data):
        print("batch held:", data["halted"])
        return data
    save(data)
    if mode == "publish":
        poll_due(data, adapter, now())
        save(data)
        if data.get("halted") and not entry_halt(data):
            return data
    # Reconcile every reserved dispatch before another write, even behind a sequence hold.
    # A repair can block release, but cannot hide an already submitted dry-preview intent.
    for reserved in data["jobs"]:
        if reserved["stage"] == "dispatch_unresolved":
            if not resolve(repo, adapter.WORKFLOW, reserved):
                if (now() - datetime.fromisoformat(reserved["submitted_at"])).total_seconds() > 900:
                    data["halted"] = "Uncertain dispatch; reconcile the run before any replacement"
                save(data)
                return data
    # Release ordering must not hide producer completion behind an editorial hold.
    # Refresh metadata only: this grants no review, upload or rotation permission.
    production = {}
    for job in data["jobs"]:
        if job["stage"] != "building" or editorial_problems(job):
            continue
        run = gh("api", f"repos/{repo}/actions/runs/{job['run_id']}")
        if job.get("source_sha") and run["head_sha"] != job["source_sha"]:
            raise ValueError("production source differs from the saved dispatch intent")
        job.update(cloud_status=run["status"], source_sha=run["head_sha"])
        production[job["run_id"]] = run
    reviewed = 0
    for job in data["jobs"]:
        if problems := editorial_problems(job):
            if job["stage"] in FINAL or job["stage"] == "upload_intent":
                data["halted"] = "Editorial hold with existing delivery intent: " + "; ".join(problems)
                save(data)
                return data
            job.update(stage="held", problems=problems)
        if job["stage"] == "held":
            # A later refusal blocks itself and successors, never a legitimate predecessor.
            break
        if job["stage"] in FINAL:
            continue
        if job.get("review_retry_after") and now() < datetime.fromisoformat(job["review_retry_after"]):
            continue
        if job["stage"] not in ("building", "reviewed", "upload_intent"):
            continue
        if job["stage"] == "building":
            run = production[job["run_id"]]
            if run["status"] != "completed":
                continue
            if run["conclusion"] != "success":
                job.update(stage="held", problems=[f"production run {run['conclusion']}"])
                save(data)
                break
        if reviewed >= 2:
            break
        if not getattr(adapter, "review_ready", lambda d, j: True)(data, job):
            continue
        try:
            had_intent = job["stage"] == "upload_intent"
            folder = materialize(repo, adapter, job)
            mp4, brief = adapter.validate(job, folder)
            restore_review(job, mp4, brief, folder)
            receipt = media_review.review(mp4, brief, folder)
            problems = media_review.receipt_problems(receipt, mp4, brief)
            job["review"] = receipt
            save(data)  # preserve the complete verdict before an archive upload can fail
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
    pending = next((j for j in data["jobs"] if j["stage"] == "planned"), None)
    buffered = sum(j["stage"] not in FINAL | {"planned"} for j in data["jobs"])
    # A completed future preview can still be "building" while ordered review waits
    # for an earlier repair. It occupies buffer space, but no longer uses a producer.
    in_flight = [j for j in data["jobs"] if j["stage"] == "dispatch_unresolved"
                 or (j["stage"] == "building" and j.get("cloud_status") != "completed")]
    # Independent reserved dry previews may overlap within a small bounded buffer.
    # Editorial holds still stop review/release, and uncertain dispatch/delivery stops all work.
    safe_halt = not data.get("halted") or entry_halt(data)
    dry_inputs = pending and pending.get("inputs", {}).get("dry_run") == "true"
    if (safe_halt and dry_inputs and len(in_flight) < MAX_BUILDING
            and buffered < MAX_PREVIEW_AHEAD):
        # An inherited manual or scheduled production run must still hold new dispatches.
        active = [r for r in builds(repo, adapter.WORKFLOW) if r["status"] != "completed"]
        owned = {j.get("run_id") for j in in_flight if j.get("run_id")}
        # A direct image hold can arrive while its reserved dry producer is still
        # running. Count that actual run against capacity without granting review
        # or release permission; a completed held preview uses buffer space only.
        active_held = {j["run_id"] for j in held if j.get("run_id") and j.get("source_sha")
                       and any(r["id"] == j["run_id"] and r.get("head_sha") == j["source_sha"]
                               for r in active)}
        occupied = len(in_flight) + len(active_held - owned)
        owned |= active_held
        if occupied < MAX_BUILDING and all(r["id"] in owned for r in active):
            head = gh("api", f"repos/{repo}/commits/main")["sha"]
            pending.update(stage="dispatch_unresolved", source_sha=head,
                           before_ids=[r["id"] for r in builds(repo, adapter.WORKFLOW)], submitted_at=stamp(),
                           attempts=pending.get("attempts", 0) + 1)
            save(data)
            response = gh("api", f"repos/{repo}/actions/workflows/{adapter.WORKFLOW}/dispatches", body={
                "ref": "main", "inputs": pending["inputs"], "return_run_details": True})
            if response and response.get("workflow_run_id"):
                pending["dispatch_run_id"] = response["workflow_run_id"]
                save(data)
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
