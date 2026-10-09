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
    """How big an episode is. 'full' is the daily long episode; 'short' proves the pipeline in ~10 minutes."""
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
    "full": Shape("full", 14, 1030, (8, 10), 10.0, 8, 2, 5, 15, 5, 15, 2),
    "short": Shape("short", 2, 560, (6, 6), 1.0, 1, 1, 4, 4, 2, 3, 1),
}


def shape(name: str) -> Shape:
    return SHAPES[name]


def series() -> dict:
    return json.loads((BIBLE / "series.json").read_text(encoding="utf-8"))


def backlog() -> list[dict]:
    return json.loads((BIBLE / "backlog.json").read_text(encoding="utf-8"))


def episode_dir(episode_id: str) -> Path:
    d = WORK / episode_id
    d.mkdir(parents=True, exist_ok=True)
    return d
