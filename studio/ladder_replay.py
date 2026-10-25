"""Exact image inputs for targeted dry repairs; cached pictures still receive current QA."""
import hashlib
import html
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
    if "cached_only" in data and type(data["cached_only"]) is not bool:
        raise ValueError("cached_only must be an explicit boolean")
    if "art_only" in data:
        slots = data["art_only"]
        if not isinstance(slots, list) or not slots or any(type(i) is not str or i not in ladder.IMAGE_IDS
                                                        for i in slots) or len(set(slots)) != len(slots):
            raise ValueError("art_only must name distinct known image slots")
        if data.get("cached_only"):
            raise ValueError("art_only generation cannot also be cached_only")
    data["_folder"] = path.parent
    return data


def local_generation(folder, row, job, model, size, key):
    """Keep local GPU production honest: exact raw pixels, settings, weights and grade; current QA still runs."""
    from PIL import Image
    from . import ladder_images, sdcpp

    binding = row["generation"]
    def checked(spec):
        path = (folder / spec["file"]).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or digest(path) != spec.get("sha256"):
            raise ValueError("corrupt or unsafe local generation input")
        return path
    path = checked(binding)
    if path.stat().st_size > 1_000_000:
        raise ValueError("oversized local generation receipt")
    proof = json.loads(path.read_text())
    cand = sdcpp.CANDIDATES[model]
    settings = {"model": model, "prompt": job["prompt"], "width": size[0], "height": size[1],
                "steps": cand.steps, "cfg": cand.cfg, "sampler": cand.sampler,
                "seed": ladder_images._seed(key, row.get("source_image", job["id"]), row["take"])}
    reference = proof.get("schema") == 2 and proof.get("origin") == "local-stable-diffusion.cpp-reference"
    text_only = proof.get("schema") == 1 and proof.get("origin") == "local-stable-diffusion.cpp" \
        and "conditioning" not in proof
    if not (reference or text_only) or proof.get("settings") != settings or proof.get("binary_tag") != sdcpp.SD_TAG \
            or not re.fullmatch(r"[0-9a-f]{40}", proof.get("source_sha", "")):
        raise ValueError("local generation settings or source differ from this production")
    references = []
    if reference:
        conditioning = proof.get("conditioning", {})
        extra = {"seed": settings["seed"], "sample_params": {"sample_steps": cand.steps,
                 "sample_method": cand.sampler, "guidance": {"txt_cfg": cand.cfg}}}
        refs = conditioning.get("references", [])
        if conditioning.get("mode") != "reference" or conditioning.get("endpoint") != "/v1/images/edits" \
                or conditioning.get("request_extra_args") != extra or not 1 <= len(refs) <= 4:
            raise ValueError("local reference conditioning lacks its exact request")
        for ref in refs:
            image = checked(ref)
            evidence = checked(ref["license_evidence"])
            if image.stat().st_size > 8_000_000 or evidence.stat().st_size > 1_000_000:
                raise ValueError("oversized reference or license evidence")
            with Image.open(image) as img:
                if img.width * img.height > 16_000_000:
                    raise ValueError("oversized reference dimensions")
                img.verify()
            pages = json.loads(evidence.read_text()).get("query", {}).get("pages", {})
            page = next((p for p in pages.values() if p.get("title") == ref.get("title")), {})
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            artist = html.unescape(re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")))
            if ref.get("license") != "CC BY 3.0" or meta.get("LicenseShortName", {}).get("value") != ref["license"] \
                    or ref.get("license_url", "").rstrip("/") != "https://creativecommons.org/licenses/by/3.0" \
                    or meta.get("LicenseUrl", {}).get("value", "").rstrip("/") != ref["license_url"].rstrip("/") \
                    or not str(ref.get("author", "")).strip() or ref["author"] not in artist \
                    or ref.get("page") != info.get("descriptionurl") or ref.get("url") != info.get("url"):
                raise ValueError("reference attribution differs from saved Commons license evidence")
            credit = ref.get("credit", "")
            if not all(s and s in credit for s in (ref["author"], ref["page"], ref["license"], ref["license_url"])) \
                    or "AI-adapted" not in credit:
                raise ValueError("reference lacks complete adaptation attribution")
            references.append(ref)
    weights = proof.get("weights", [])
    if len(weights) != len(cand.weights) or any(
            w.get("repo") != expected.repo or w.get("file") != expected.file
            or not re.fullmatch(r"[0-9a-f]{64}", w.get("sha256", ""))
            for w, expected in zip(weights, cand.weights)):
        raise ValueError("local generation lacks exact model weight provenance")
    with Image.open(checked(proof["raw"])) as raw:
        if raw.size != size:
            raise ValueError("local raw image dimensions differ from production")
        graded, exposure = ladder_images.expose(raw.convert("RGB"), sum(job["luma"]) / 2)
    with Image.open(checked(row)) as actual:
        if actual.convert("RGB").tobytes() != graded.tobytes() or exposure != row.get("exposure"):
            raise ValueError("local image or exposure differs from its reproducible raw grade")
    result = {"kind": proof["origin"], "receipt_sha256": binding["sha256"],
            "source_sha": proof["source_sha"], "settings": settings, "weights": weights,
            "binary_tag": proof["binary_tag"], "raw_sha256": proof["raw"]["sha256"]}
    if reference:
        result.update(conditioning=proof["conditioning"], required_credits=[r["credit"] for r in references])
    return result


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
        if type(row.get("take")) is not int or not 0 <= row["take"] < 4 \
                or not all(k in row.get("exposure", {}) for k in ("raw_p99", "spread")):
            raise ValueError("replay image lacks original take or exposure provenance")
        source_image = row.get("source_image", job["id"])
        if source_image not in ladder.IMAGE_IDS:
            raise ValueError("replay input has an unknown original image slot")
        if row.get("generation") and source_image != job["id"]:
            fields = ("file", "sha256", "prompt", "model", "take", "exposure", "generation")
            originals = replay["images"].get(source_image, [])
            if job["id"] != "end" or source_image == "end" or not any(
                    all(original.get(k) == row.get(k) for k in fields) for original in originals):
                raise ValueError("local closing reprise differs from its exact original image")
        local = local_generation(folder, row, job, model, size, replay["source"]["key"]) \
            if row.get("generation") else None
        if not local and (type(row.get("artifact_id")) is not int or row["artifact_id"] <= 0):
            raise ValueError("replay image lacks original artifact provenance")
        if row["take"] in seen_takes or row["sha256"] in seen_hashes:
            raise ValueError("replay inputs must be distinct original takes and images")
        seen_takes.add(row["take"])
        seen_hashes.add(row["sha256"])
        with Image.open(path) as src:
            if src.size != size:
                raise ValueError("replay image dimensions differ from the production model")
            img = src.convert("RGB")
        provenance = {k: row[k] for k in ("file", "sha256", "take")}
        if "editorial_refusal" in row:
            refusal = row["editorial_refusal"]
            if not isinstance(refusal, dict) or not str(refusal.get("reason", "")).strip() \
                    or refusal.get("image_sha256") != row["sha256"]:
                raise ValueError("invalid or stale image-bound editorial refusal")
            provenance["editorial_refusal"] = refusal
        if local:
            provenance["generation"] = row["generation"]
        else:
            provenance["artifact_id"] = row["artifact_id"]
        provenance["source_image"] = source_image
        origin = row.get("source", replay["source"])
        if not isinstance(origin, dict) or type(origin.get("run_id")) is not int or origin["run_id"] <= 0 \
                or not re.fullmatch(r"[0-9a-f]{40}", origin.get("sha", "")) \
                or origin.get("key") != replay["source"]["key"]:
            raise ValueError("cached image lacks its exact cloud source and seed key")
        provenance["source"] = local or origin
        found.append((img, row["exposure"], row["take"], provenance))
    return found


def exhausted(replay, job, model, key, take_limit):
    """A hashed original shard can prove all deterministic seeds were already attempted.

    One passing picture is valid in the normal producer too. Recheck that picture, but do not
    spend another runner repainting its three already-refused seeds. Current QA can still hold it.
    """
    if not replay or replay["source"]["key"] != key:
        return False
    local_rows = replay.get("local_attempts", {}).get(job["id"])
    if local_rows:
        from . import sdcpp

        if len(local_rows) != take_limit or sorted(r.get("take", -1) for r in local_rows) != list(range(take_limit)):
            raise ValueError("local attempt history does not prove every allowed deterministic take")
        cand = sdcpp.CANDIDATES[model]
        for attempt in local_rows:
            if attempt.get("model") != model or attempt.get("prompt") != job["prompt"]:
                raise ValueError("local attempt history differs from the current painter job")
            local_generation(replay["_folder"].resolve(), attempt, job, model, (cand.height, cand.width), key)
        return True
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
