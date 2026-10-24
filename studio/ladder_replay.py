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
        with Image.open(path) as src:
            if src.size != size:
                raise ValueError("replay image dimensions differ from the production model")
            img = src.convert("RGB")
        provenance = {k: row[k] for k in ("file", "sha256", "artifact_id", "take")}
        provenance["source"] = replay["source"]
        found.append((img, row["exposure"], row["take"], provenance))
    return found
