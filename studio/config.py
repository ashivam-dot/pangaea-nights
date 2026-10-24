"""Channel identity, episode shapes and paths shared by every stage."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIBLE = ROOT / "bible"
ASSETS = ROOT / "assets"
WORK = Path(os.environ.get("PANGAEA_WORK", ROOT / "work"))
STATE = Path(os.environ.get("PANGAEA_STATE", ROOT / "state"))

CHANNEL_NAME = "Pangaea Nights"
CHANNEL_HANDLE = "@PangaeaNights"
CHANNEL_ID = "UCtVG78IUTqvP9rH-LLies0A"
CATEGORY_EDUCATION = "27"
WPM = 120  # target narration pace including pauses


@dataclass(frozen=True)
class Shape:
    """How big an episode is. 'full' is the 2-hour sleep documentary and 'full3h' its 3-hour form (the length
    experiment is held during the stable pilot); 'short' proves the pipeline in ~10 minutes."""
    name: str
    chapters: int
    words_per_chapter: int
    scenes_per_chapter: tuple[int, int]
    tail_minutes: float
    tail_scenes: int
    trailers: int
    trailer_scenes: int
    image_shards: int
    voice_shards: int
    render_shards: int
    hedges_per_chapter: int


SHAPES = {
    # 18 image + 2 voice runners = the 20 concurrent jobs of a free public repo; at most 8 images per image runner.
    "full": Shape("full", 14, 1030, (8, 9), 10.0, 6, 2, 4, 18, 2, 15, 2),
    # 3 hours: 21 chapters (14 chapters measured ~7,250 s, so ~10,700 s). 17 image + 3 voice runners stay inside
    # the 20 concurrent jobs: 21 x 6 scenes + 6 tail + 1 thumbnail = 133 images, at most 8 per image runner; each
    # scene holds ~80 s instead of ~55 s. 3 voice runners keep 7 chapters each, as in the 2-hour shape.
    "full3h": Shape("full3h", 21, 1030, (6, 6), 10.0, 6, 2, 4, 17, 3, 20, 2),
    "short": Shape("short", 2, 560, (6, 6), 1.0, 1, 1, 4, 6, 2, 3, 1),
}
# Keep the familiar-topic relaunch at one length through four published episodes and their 7-day review.
# Set STABLE_LONG_LENGTH to None only after that review to resume the deferred alternating-length experiment.
# PANGAEA_LENGTH, set by the plan job, resolves "full"; explicit private length previews remain available.
LENGTHS = {"2h": "full", "3h": "full3h"}
LONG_SHAPES = tuple(LENGTHS.values())
STABLE_LONG_LENGTH = "2h"
LENGTH_EXPERIMENT_FROM = "2026-10-27"  # published long episodes created from this day on take part

# The old calm per-episode trailer Shorts (studio/shorts.py). Off: the choice-ladder Shorts replace them. The code
# stays; turning this on makes the writer, images, voice and finish stages produce and publish them again.
TRAILER_SHORTS = False

# Long-form cadence: built Tue and Fri at 14:10 UTC (episode.yml cron), premiering at this UTC time the same day.
EPISODE_DAYS = ("Tue", "Fri")
EPISODE_SLOT_UTC = (21, 0)


def shape(name: str, length: str | None = None) -> Shape:
    """The shape called `name`; "full" means the 2- or 3-hour form chosen for this run (default 2 h)."""
    if name == "full":
        length = length or os.environ.get("PANGAEA_LENGTH", "").strip() or "2h"
        if length not in LENGTHS:
            raise ValueError(f"PANGAEA_LENGTH must be one of {sorted(LENGTHS)}, not {length!r}")
        name = LENGTHS[length]
    return SHAPES[name]


def trailer_count(s: Shape) -> int:
    """How many trailer Shorts an episode of this shape gets: none while TRAILER_SHORTS is off."""
    return s.trailers if TRAILER_SHORTS else 0


def series() -> dict:
    return json.loads((BIBLE / "series.json").read_text(encoding="utf-8"))


def backlog() -> list[dict]:
    return json.loads((BIBLE / "backlog.json").read_text(encoding="utf-8"))


def episode_dir(episode_id: str) -> Path:
    d = WORK / episode_id
    d.mkdir(parents=True, exist_ok=True)
    return d
