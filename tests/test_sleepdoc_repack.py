"""The long-form repackage: Tue/Fri 21:00 UTC slot, trailers off, 2 h / 3 h length experiment, bright thumbnails."""
from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

from studio import config, images, length, thumbnail, writer, youtube


# ------------------------------------------------------------------ cadence


def test_episode_slot_is_21_utc():
    assert config.EPISODE_SLOT_UTC == (21, 0)
    tue = datetime(2026, 10, 27, 14, 10, tzinfo=timezone.utc)  # a Tuesday build start
    assert youtube.episode_slot(now=tue) == "2026-10-27T21:00:00Z"
    late = datetime(2026, 10, 27, 20, 50, tzinfo=timezone.utc)  # too close: the next day's slot
    assert youtube.episode_slot(now=late) == "2026-10-28T21:00:00Z"


def test_workflow_builds_tuesday_and_friday_afternoons():
    yml = (config.ROOT / ".github" / "workflows" / "episode.yml").read_text()
    assert 'cron: "10 14 * * 2,5"' in yml
    assert "python -m studio.length" in yml
    assert yml.index("studio.length") < yml.index("studio.cli matrix")
    health = (config.ROOT / ".github" / "workflows" / "health.yml").read_text()
    assert "--hours 108" in health


# ------------------------------------------------------------------ trailers off


def _docs():
    text = "\n".join(["Intro line that is short."] + [
        "The shallow sea was warm and calm, and small animals moved slowly over the soft pale sand near the shore. "
        "Light fell in long green shafts through the clear water while the tide came in and out each day."] * 30)
    return [{"title": f"Doc {k}", "text": text} for k in range(3)]


def test_trailer_shorts_are_off_but_the_shape_keeps_them():
    assert config.TRAILER_SHORTS is False
    assert config.trailer_count(config.shape("full")) == 0 and config.shape("full").trailers == 2
    topic = {"id": "x", "series": "deep-sea", "title": "The Deep Ocean", "lead": "The Deep Ocean",
             "angle": "calm", "articles": []}
    script = writer.write_stub(topic, _docs(), config.shape("short"))
    assert script["trailers"] == []
    assert script["thumbnail"]["words"] == "The Deep Ocean"
    assert not [j for j in images.manifest({**script, "art_direction": "a"}) if j["kind"] == "trailer"]


def test_trailer_flag_turns_them_back_on(monkeypatch):
    monkeypatch.setattr(config, "TRAILER_SHORTS", True)
    assert config.trailer_count(config.shape("short")) == 1


# ------------------------------------------------------------------ length experiment


def test_full_resolves_to_2h_or_3h(monkeypatch):
    monkeypatch.delenv("PANGAEA_LENGTH", raising=False)
    assert config.shape("full").name == "full"
    monkeypatch.setenv("PANGAEA_LENGTH", "3h")
    s = config.shape("full")
    assert s.name == "full3h" and s.chapters == 21
    assert config.shape("full3h").name == "full3h" and config.shape("short").name == "short"
    assert config.shape("full", length="2h").name == "full"
    monkeypatch.setenv("PANGAEA_LENGTH", "4h")
    with pytest.raises(ValueError):
        config.shape("full")


def test_3h_shape_reserves_four_fleet_runners_without_dropping_scenes():
    s = config.SHAPES["full3h"]
    assert s.image_shards + s.voice_shards <= 16 and s.render_shards <= 16
    n_images = s.chapters * writer.scene_count(s) + s.tail_scenes + 1 + config.trailer_count(s) * s.trailer_scenes
    assert n_images <= 11 * s.image_shards
    hours = s.chapters * s.words_per_chapter / config.WPM / 60 + s.tail_minutes / 60
    assert 2.8 <= hours <= 3.3
    full = config.SHAPES["full"]
    full_images = full.chapters * writer.scene_count(full) + full.tail_scenes + 1
    assert full_images <= 10 * full.image_shards
    assert 1.9 <= full.chapters * full.words_per_chapter / config.WPM / 60 + full.tail_minutes / 60 <= 2.3


def _ep(i, shape, created="2026-10-27T14:20:00Z", video=True):
    e = {"id": f"e{i}", "topic": f"t{i}", "series": "s", "shape": shape, "created": created}
    if video:
        e["video_id"] = f"v{i}"
    return e


def test_lengths_alternate_starting_with_2h_when_pilot_is_released(monkeypatch):
    monkeypatch.setattr(config, "STABLE_LONG_LENGTH", None)
    data = {"episodes": [_ep(0, "full", created="2026-10-18T16:20:00Z")]}  # before the experiment: ignored
    assert length.next_length(data) == "2h"
    data["episodes"].append(_ep(1, "full"))
    assert length.next_length(data) == "3h"
    data["episodes"].append(_ep(2, "full3h"))
    assert length.next_length(data) == "2h"
    data["episodes"].append(_ep(3, "full3h", video=False))  # never published: does not count
    assert length.next_length(data) == "2h"
    assert length.choose("3h", data) == "3h" and length.choose("auto", data) == "2h"
    with pytest.raises(ValueError):
        length.choose("90m", data)
    assert length.length_of("full3h") == "3h" and length.length_of("short") is None


def test_length_main_writes_github_env(tmp_path, monkeypatch):
    env = tmp_path / "env"
    monkeypatch.setenv("GITHUB_ENV", str(env))
    # Publishing four episodes alone must not release the hold before their seven-day review.
    data = {"episodes": [_ep(i, "full") for i in range(4)]}
    monkeypatch.setattr(length.ledger, "load", lambda: data)
    assert length.main(["--choice", "auto"]) == "2h"
    assert env.read_text() == "PANGAEA_LENGTH=2h\n"
    assert length.choose("3h", data) == "3h"  # explicit private previews remain possible


# ------------------------------------------------------------------ thumbnail


def _flat(v):
    return Image.fromarray(np.full((thumbnail.H, thumbnail.W, 3), v, np.uint8))


def test_luma_gate_has_a_floor_and_a_ceiling():
    assert thumbnail.mean_luma(_flat(30)) == pytest.approx(30, abs=0.5)
    assert thumbnail.luma_problems(30) and "dark" in thumbnail.luma_problems(30)[0]
    assert thumbnail.luma_problems(230) and "bright" in thumbnail.luma_problems(230)[0]
    assert thumbnail.luma_problems(thumbnail.LUMA_FLOOR) == [] and thumbnail.luma_problems(thumbnail.LUMA_CEIL) == []
    assert thumbnail.LUMA_FLOOR < thumbnail.LUMA_CEIL


def test_brighten_moves_dark_and_blown_images_into_the_band():
    lo, hi = thumbnail.LUMA_TARGET
    rng = np.random.default_rng(0)
    dark = Image.fromarray(rng.integers(0, 70, (thumbnail.H, thumbnail.W, 3), dtype=np.uint8))
    lifted = thumbnail.mean_luma(thumbnail.brighten(dark))
    assert lifted > thumbnail.mean_luma(dark) + 30 and lifted >= thumbnail.LUMA_FLOOR
    assert thumbnail.mean_luma(thumbnail.brighten(_flat(250))) == pytest.approx(hi, abs=3)
    mid = _flat(150)
    assert np.array_equal(np.asarray(thumbnail.brighten(mid)), np.asarray(mid))


def test_fit_words_keeps_two_to_four_big_words():
    assert thumbnail.fit_words("The Last Day of the Dinosaurs".split()) == ["The", "Last", "Day", "Dinosaurs"]
    assert thumbnail.fit_words(["Megalodon", "Rising"]) == ["Megalodon", "Rising"]
    assert len(thumbnail.fit_words("one two three four five six".split())) == 4


def _fake_scene(tmp_path, value):
    img = np.full((360, 640, 3), value, np.uint8)
    img[100:300, 380:600] = np.clip(value + 90, 0, 255)  # one bright focal subject on the right
    p = tmp_path / "scene.png"
    Image.fromarray(img).save(p)
    return p


def test_build_reports_ok_with_luma_when_every_gate_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(thumbnail, "legible", lambda img, words, box=None: (True, [w.lower() for w in words]))
    res = thumbnail.build(_fake_scene(tmp_path, 25), "The Last Day of the Dinosaurs".split(), tmp_path / "t.jpg",
                          upscale=False)
    assert res["ok"], res["attempts"]
    assert thumbnail.LUMA_FLOOR <= res["luma"] <= thumbnail.LUMA_CEIL
    assert res["words"] == ["The", "Last", "Day", "Dinosaurs"] and res["problems"] == []
    assert (tmp_path / "t.jpg").exists()
    assert res["attempts"][0]["size"] >= 170  # larger words than before (150)


def test_build_fails_a_thumbnail_that_stays_too_dark(tmp_path, monkeypatch):
    monkeypatch.setattr(thumbnail, "legible", lambda img, words, box=None: (True, []))
    monkeypatch.setattr(thumbnail, "brighten", lambda base: base)  # no lift: a black night frame stays black
    res = thumbnail.build(_fake_scene(tmp_path, 0), ["Deep", "Ocean"], tmp_path / "t.jpg", upscale=False)
    assert not res["ok"] and any("too dark" in p for p in res["problems"])


def test_build_still_needs_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr(thumbnail, "legible", lambda img, words, box=None: (False, []))
    res = thumbnail.build(_fake_scene(tmp_path, 40), ["Giant", "Insects"], tmp_path / "t.jpg", upscale=False)
    assert not res["ok"] and any("OCR" in p for p in res["problems"])


def test_thumbnail_prompt_asks_for_a_bright_focal_creature():
    job = images.manifest({"art_direction": "dark calm", "chapters": [], "tail_scenes": [], "trailers": [],
                           "thumbnail": {"prompt": "a megalodon"}})[0]
    assert job["id"] == "thumb" and "bright" in job["prompt"] and "single focal creature" in job["prompt"]
