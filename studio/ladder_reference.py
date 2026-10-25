"""Pinned, attributed Commons references for isolated dry art repairs; no network fetches."""
import html
import json
import re
from pathlib import Path

from PIL import Image

from . import ladder_replay

ATTRIBUTION_LICENSES = {
    "CC BY 3.0": "https://creativecommons.org/licenses/by/3.0",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0",
}


def inputs(replay, image_id):
    rows = (replay or {}).get("references", {}).get(image_id, [])
    if not rows:
        return []
    if image_id not in replay.get("art_only", []) or replay.get("cached_only"):
        raise ValueError("generation references require isolated art-only repair slots")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
        raise ValueError("reference repair needs one to four pinned inputs")
    folder = replay["_folder"].resolve()

    def bound(row):
        path = (folder / row["file"]).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size > 20_000_000 \
                or ladder_replay.digest(path) != row.get("sha256"):
            raise ValueError("corrupt or unsafe generation reference")
        return path

    out = []
    for row in rows:
        image, parent, licence = bound(row), bound(row["parent"]), bound(row["license_evidence"])
        evidence = json.loads(licence.read_text())
        pages = [p for p in evidence.get("query", {}).get("pages", {}).values() if p.get("title") == row["title"]]
        if len(pages) != 1:
            raise ValueError("reference has no unambiguous Commons licence evidence")
        info = pages[0]["imageinfo"][0]
        meta = info["extmetadata"]
        author = html.unescape(re.sub(r"<[^>]*>", "", meta.get("Artist", {}).get("value", ""))).strip()
        license_name = row.get("license")
        if license_name not in {"CC0", *ATTRIBUTION_LICENSES} \
                or meta["LicenseShortName"]["value"] != license_name \
                or row.get("page") != info["descriptionurl"] or row.get("author") != author \
                or not author or not row.get("credit", "").strip() or row.get("url") not in (info["url"], info.get("thumburl")):
            raise ValueError("reference attribution differs from pinned Commons evidence")
        if license_name in ATTRIBUTION_LICENSES:
            expected = ATTRIBUTION_LICENSES[license_name]
            credit = row["credit"]
            # Initial punctuation may be styled differently in the description;
            # every author word still has to appear in the attribution.
            words = lambda value: re.sub(r"\W+", " ", value).casefold().strip()
            if row.get("license_url", "").rstrip("/") != expected \
                    or meta.get("LicenseUrl", {}).get("value", "").rstrip("/") != expected \
                    or words(author) not in words(credit) \
                    or any(value not in credit for value in (row["page"], license_name, expected, "AI-adapted")):
                raise ValueError("reference lacks complete source, licence and AI adaptation attribution")
        with Image.open(parent) as src, Image.open(image) as actual:
            box = row.get("crop_ltrb")
            if not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box) \
                    or not 0 <= box[0] < box[2] <= src.width or not 0 <= box[1] < box[3] <= src.height \
                    or max(src.size) > 8000:
                raise ValueError("reference crop is unsafe")
            expected = src.convert("RGB").crop(box)
            if expected.size != actual.size or expected.tobytes() != actual.convert("RGB").tobytes():
                raise ValueError("reference crop differs from pinned parent")
        out.append({"path": image, "proof": row})
    return out
