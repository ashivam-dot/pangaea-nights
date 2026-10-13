import json
import shutil
import subprocess
from argparse import Namespace

import numpy as np
import pytest
from PIL import Image

from studio import cli, mix, thumbnail

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
needs_tesseract = pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract not installed")


@needs_ffmpeg
def test_limited_master_keeps_its_true_peak_after_aac(tmp_path):
    hot = tmp_path / "hot.flac"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "anoisesrc=color=pink:amplitude=0.9:sample_rate=48000:duration=12",
                    "-af", "lowpass=f=6000,volume='0.6+0.4*sin(2*PI*1.7*t)':eval=frame,volume=6dB",
                    "-ac", "2", "-c:a", "flac", str(hot)], check=True)
    assert mix.integrated(hot)["true_peak"] > mix.TARGET_TP + 0.5
    limited = tmp_path / "limited.flac"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(hot), "-af", mix.peak_limiter(),
                    "-c:a", "flac", str(limited)], check=True)
    aac = mix.to_aac(limited, tmp_path / "limited.m4a")
    assert mix.integrated(aac)["true_peak"] <= mix.TARGET_TP + 0.5


@needs_ffmpeg
def test_finish_relimits_an_old_hot_mix_and_keeps_a_good_one(tmp_path):
    from studio import finish

    hot = tmp_path / "mix.flac"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "anoisesrc=color=pink:amplitude=0.9:sample_rate=48000:duration=8", "-af", "volume=6dB",
                    "-ac", "2", "-c:a", "flac", str(hot)], check=True)
    hot_aac = mix.to_aac(hot, tmp_path / "mix.m4a")
    fixed = finish.limited_mix(hot_aac)
    assert fixed != hot_aac and mix.integrated(fixed)["true_peak"] <= mix.TARGET_TP + 0.5
    assert finish.limited_mix(fixed) == fixed


@needs_tesseract
@pytest.mark.parametrize("low", [20, 150])
def test_busy_picture_beside_the_words_still_gets_a_legible_thumbnail(tmp_path, low):
    rng = np.random.default_rng(7)
    plate = rng.integers(low, 256, (720, 1280, 3), dtype=np.uint8)
    src = tmp_path / "plate.png"
    Image.fromarray(plate).save(src)
    res = thumbnail.build(src, ["The", "Coal", "Forest"], tmp_path / "t.jpg", upscale=False)
    assert res["ok"], res["attempts"]


def _publish_with(monkeypatch, tmp_path, mode, paused):
    monkeypatch.setattr(cli, "W", tmp_path)
    (tmp_path / "out").mkdir()
    for name, data in (("episode.json", {"id": "x"}), ("timeline.json", {}),
                       ("out/qa.json", {"ok": False, "problems": ["true peak -2.1 dBFS"]})):
        (tmp_path / name).write_text(json.dumps(data))
    monkeypatch.setattr(cli, "_channel_stats", lambda: None)
    monkeypatch.setenv("PAUSED", "true" if paused else "false")
    cli.publish(Namespace(mode=mode))
    return json.loads((tmp_path / "out" / "publish.json").read_text())


def test_a_refused_publish_run_fails_loudly(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _publish_with(monkeypatch, tmp_path, "publish", paused=False)
    assert json.loads((tmp_path / "out" / "publish.json").read_text())["published"] is False


@pytest.mark.parametrize("mode, paused", [("dry", False), ("publish", True)])
def test_dry_runs_and_a_paused_channel_refuse_quietly(monkeypatch, tmp_path, mode, paused):
    assert _publish_with(monkeypatch, tmp_path, mode, paused)["published"] is False
