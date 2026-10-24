"""Exact image inputs for targeted dry repairs; cached pictures still receive current QA."""
import hashlib
import json
import re
from pathlib import Path

from . import config, ladder


def digest(path):
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def load(path, expected_sha=None):
    path = Path(path)
    root = (config.ROOT / ".github/replay").resolve()
    path = (config.ROOT / path).resolve()
    if not path.is_relative_to(root) or path.suffix != ".json":
        raise ValueError("image replays must be JSON files inside .github/replay")
    if expected_sha and digest(path) != expected_sha:
        raise ValueError("the image replay changed after planning")
    data = json.loads(path.read_text())
    source = data.get("source", {})
    if data.get("schema") != 1 or not data.get("premise") or type(source.get("run_id")) is not int \
            or source["run_id"] <= 0 or not source.get("key") \
            or not re.fullmatch(r"[0-9a-f]{40}", source.get("sha", "")):
        raise ValueError("image replay lacks its exact original production source")
    if not isinstance(data.get("images"), dict) or set(data["images"]) - set(ladder.IMAGE_IDS):
        raise ValueError("image replay contains unknown image slots")
    data["_folder"] = path.parent
    return data


def candidates(replay, job, model, size):
    """A changed prompt/model invalidates reuse; missing/corrupt matching inputs stop the repair."""
    if not replay:
        return []
    from PIL import Image

    found = []
    seen_takes, seen_hashes = set(), set()
    for row in replay["images"].get(job["id"], []):
        if row.get("prompt") != job["prompt"] or row.get("model") != model:
            continue
        folder = replay["_folder"].resolve()
        path = (folder / row["file"]).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or digest(path) != row.get("sha256"):
            raise ValueError(f"corrupt or unsafe replay input for {job['id']}")
        if type(row.get("artifact_id")) is not int or row["artifact_id"] <= 0 \
                or type(row.get("take")) is not int or not 0 <= row["take"] < 4 \
                or not all(k in row.get("exposure", {}) for k in ("raw_p99", "spread")):
            raise ValueError("replay image lacks original take, artifact or exposure provenance")
        if row["take"] in seen_takes or row["sha256"] in seen_hashes:
            raise ValueError("replay inputs must be distinct original takes and images")
        seen_takes.add(row["take"])
        seen_hashes.add(row["sha256"])
        with Image.open(path) as src:
            if src.size != size:
                raise ValueError("replay image dimensions differ from the production model")
            img = src.convert("RGB")
        provenance = {k: row[k] for k in ("file", "sha256", "artifact_id", "take")}
        provenance["source_image"] = row.get("source_image", job["id"])
        if provenance["source_image"] not in ladder.IMAGE_IDS:
            raise ValueError("replay input has an unknown original image slot")
        provenance["source"] = replay["source"]
        found.append((img, row["exposure"], row["take"], provenance))
    return found


def exhausted(replay, job, model, key, take_limit):
    """A hashed original shard can prove all deterministic seeds were already attempted.

    One passing picture is valid in the normal producer too. Recheck that picture, but do not
    spend another runner repainting its three already-refused seeds. Current QA can still hold it.
    """
    if not replay or replay["source"]["key"] != key:
        return False
    row = replay.get("completed_images", {}).get(job["id"])
    if not row or row.get("prompt") != job["prompt"] or row.get("model") != model:
        return False
    folder = replay["_folder"].resolve()
    path = (folder / row["file"]).resolve()
    if not path.is_relative_to(folder) or not path.is_file() or digest(path) != row.get("sha256"):
        raise ValueError("corrupt or unsafe original shard report")
    if type(row.get("artifact_id")) is not int or row["artifact_id"] <= 0:
        raise ValueError("original shard report lacks artifact provenance")
    report = json.loads(path.read_text())
    entries = [e for e in report.get("images", []) if e.get("id") == job["id"]]
    if report.get("premise") != replay["premise"] or report.get("model") != model or len(entries) != 1:
        raise ValueError("original shard report differs from this image")
    return len(entries[0].get("takes", [])) == take_limit
