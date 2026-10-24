"""Bind a reviewed original MP4 to upload recovery without replacing the channel's story marker."""
import hashlib
import re


def media_tag(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("reviewed upload requires a complete SHA-256")
    with path.open("rb") as fh:
        actual = hashlib.file_digest(fh, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("upload bytes differ from the reviewed MP4")
    return "reviewed-sha256:" + expected


def bound_tags(tags, binding):
    if not tags:
        raise ValueError("reviewed upload has no existing story marker")
    marker = tags[-1]
    if len(binding) + len(marker) + 4 > 480:
        raise ValueError("review binding and story marker exceed the upload tag allowance")
    others = [tag for tag in tags[:-1] if not tag.startswith("reviewed-sha256:")]
    while sum(len(tag) + 2 for tag in others + [binding, marker]) > 480:
        others.pop(0)
    return others + [binding, marker]


def check_existing(video, binding):
    if video and binding not in video.get("snippet", {}).get("tags", []):
        raise ValueError(f"existing upload {video['id']} is not bound to this reviewed MP4; preserve its id and reconcile")
