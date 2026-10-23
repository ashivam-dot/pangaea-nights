"""The channel's About text and keywords, kept in code so they follow the formats the channel actually publishes.

`python -m studio.cli branding` prints the change; `--apply` writes it (brandingSettings.channel, after the identity
check) and reads it back. The name, handle, avatar and banner are not touched: the API cannot set the first three,
and the current banner and avatar (ammonite, moonlit coast) still fit both formats.
"""
from __future__ import annotations

import json

DESCRIPTION = """Pangaea Nights: Earth's deep past, told two ways.

Daily Shorts, "Survive the Night": eight choices climb from cosy to deadly. Pick your number, then hear the real answer, checked against the science. The first seas, the giant-insect forests, the age of the dinosaurs, the ice-age steppes.

Sleep documentaries every Tuesday and Friday: calm, slow narration about the deep past of our planet, made for falling asleep, told softly over gentle ambience with no sudden sounds and no music stings.

Every video is written from real science: facts are checked against Wikipedia and the research it cites. Some scenes are artistic reconstructions created with AI image tools, and the narration voice is AI generated.

Contact: aksh.shivam17@gmail.com"""

KEYWORDS = ('"deep time" prehistoric "Earth history" paleontology dinosaurs "ice age" Cambrian "prehistoric life" '
            '"survive the night" "sleep documentary" "documentary for sleep" "calm documentary"')


def plan(current: dict) -> dict:
    """The brandingSettings.channel to write: the current one with the description and keywords replaced (update
    replaces the whole part, so every other field is carried over)."""
    return current | {"description": DESCRIPTION, "keywords": KEYWORDS}


def apply(api, write: bool) -> dict:
    from . import youtube

    me = youtube.identity(api)
    got = api.channels().list(part="brandingSettings", id=me["id"]).execute()["items"][0]
    current = got.get("brandingSettings", {}).get("channel", {})
    wanted = plan(current)
    changed = sorted(k for k in wanted if wanted[k] != current.get(k))
    out = {"channel": me["id"], "changed": changed, "write": write, "fields": sorted(current)}
    if write and changed:
        api.channels().update(part="brandingSettings",
                              body={"id": me["id"], "brandingSettings": {"channel": wanted}}).execute()
        back = api.channels().list(part="brandingSettings", id=me["id"]).execute()["items"][0]
        now = back.get("brandingSettings", {}).get("channel", {})
        out["readback_ok"] = all(now.get(k) == wanted[k] for k in ("description", "keywords"))
        if not out["readback_ok"]:
            raise RuntimeError("branding read-back does not match: " + json.dumps(
                {k: now.get(k) for k in ("description", "keywords")}, ensure_ascii=False)[:400])
    return out


def run(args) -> None:
    from . import youtube

    print(json.dumps(apply(youtube.client(), args.apply), indent=1))


def register(sub) -> None:
    s = sub.add_parser("branding")
    s.add_argument("--apply", action="store_true")
    s.set_defaults(fn=run)
