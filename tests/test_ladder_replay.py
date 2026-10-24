import io
import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from studio import config, ladder_cli, ladder_images as li, ladder_replay as replay, qa


@pytest.fixture
def cached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    folder = tmp_path / ".github/replay"
    folder.mkdir(parents=True)
    job = {"id": "o3", "scene": "the original scene", "prompt": "exact painter prompt", "luma": (62, 198)}
    exposure = {"raw_p99": 200, "spread": 90}
    rows = []
    for take in range(2):
        path = folder / f"o3-{take}.png"
        Image.new("RGB", (8, 8), (180 + take, 0, 0)).save(path)
        rows.append({"file": path.name, "sha256": replay.digest(path), "prompt": job["prompt"],
                     "model": "test", "artifact_id": 99, "take": take, "exposure": exposure})
    source = {"run_id": 123, "sha": "a" * 40, "key": "original-key"}
    path = folder / "repair.json"
    path.write_text(json.dumps({"schema": 1, "premise": "how_deep-01", "source": source, "images": {"o3": rows}}))
    monkeypatch.setitem(li.sdcpp.CANDIDATES, "test", SimpleNamespace(width=8, height=8))
    monkeypatch.setattr(li.ladder, "image_jobs", lambda p: [job])
    monkeypatch.setattr(qa, "figure", lambda img: 0.0)
    monkeypatch.setattr(qa, "embed", lambda img: np.array([1.0, 0.0]))
    monkeypatch.setattr(li.plates, "save_plate", lambda img, stem, **kw: {
        "plate": stem.name + ".jpg", "depth": stem.name + ".depth.png"})
    return path, job, replay.load(".github/replay/repair.json"), exposure


def verdict(reasons=None):
    return qa.Verdict(not reasons, reasons or [], 0.0, 0.0, 0.0, 0.4, 6.0, 120.0, 50.0, 12.0)


def test_cached_alternates_are_reviewed_again_without_loading_a_painter(cached, tmp_path, monkeypatch):
    _, job, data, _ = cached
    seen = []
    def inspect(img, scene):
        seen.append(scene)
        return verdict()
    monkeypatch.setattr(qa, "review", inspect)
    monkeypatch.setattr(li.sdcpp, "Server", lambda *a: pytest.fail("unchanged pictures were repainted"))
    report = li.generate({"id": data["premise"]}, "original-key", tmp_path / "out", 0, 1, model="test", replay=data)
    assert seen == [job["scene"], job["scene"]]
    kept = report["images"][0]["kept"]
    assert [r["cached_from"]["take"] for r in kept] == [0, 1]
    assert all(r["cached_from"]["source"]["run_id"] == 123 for r in kept)
    assert report["load_s"] == 0


def test_cache_hashes_and_prompt_binding_cannot_be_ignored(cached):
    path, job, data, _ = cached
    (path.parent / "o3-0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        replay.candidates(data, job, "test", (8, 8))
    assert replay.candidates(data, {**job, "prompt": "repaired prompt"}, "test", (8, 8)) == []
    assert replay.candidates(data, job, "different model", (8, 8)) == []


def test_a_current_qa_refusal_repaints_without_repeating_cached_seeds(cached, tmp_path, monkeypatch):
    _, _, data, exposure = cached
    seeds = []
    class Painter:
        load_seconds = 1
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def generate(self, prompt, seed, w, h):
            seeds.append(seed)
            blob = io.BytesIO()
            Image.new("RGB", (w, h), (0, 180, 0)).save(blob, format="PNG")
            return blob.getvalue()
    monkeypatch.setattr(li.sdcpp, "Server", lambda *a: Painter())
    monkeypatch.setattr(li, "expose", lambda img, target: (img, exposure))
    monkeypatch.setattr(qa, "review", lambda img, scene: verdict(["text 0.9"] if img.getpixel((0, 0))[0] else []))
    report = li.generate({"id": data["premise"]}, "original-key", tmp_path / "out", 0, 1, model="test", replay=data)
    assert seeds == [li._seed("original-key", "o3", k) for k in (2, 3)]
    entry = report["images"][0]
    assert entry["ok"] and len(entry["takes"]) == 4
    assert all("cached_from" not in row for row in entry["kept"])


def test_replay_manifest_is_pinned_and_confined_to_the_repo(cached, tmp_path):
    path, _, _, _ = cached
    sha = replay.digest(path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after planning"):
        replay.load(".github/replay/repair.json", sha)
    with pytest.raises(ValueError, match="inside .github/replay"):
        replay.load(tmp_path / "elsewhere.json")


def test_an_image_replay_cannot_use_the_direct_publishing_mode(cached):
    with pytest.raises(ValueError, match="dry-run only"):
        ladder_cli.plan(Namespace(mode="publish", replay=".github/replay/repair.json", premise=""))


def test_dry_plan_pins_the_manifest_and_selects_its_premise(tmp_path, monkeypatch):
    from studio import ledger

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "WORK", tmp_path / "work")
    monkeypatch.setattr(ledger, "load", lambda: json.loads(json.dumps(ledger.EMPTY)))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    path = tmp_path / ".github/replay/repair.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": 1, "premise": "how_deep-01", "images": {},
                               "source": {"run_id": 123, "sha": "a" * 40, "key": "source-key"}}))
    ladder_cli.plan(Namespace(mode="dry", replay=".github/replay/repair.json", premise="", slot="A", shards=9))
    plan = json.loads((config.WORK / "ladder/A/plan.json").read_text())
    assert plan["premise"]["id"] == "how_deep-01" and plan["mode"] == "dry"
    assert plan["image_replay"] == {"path": ".github/replay/repair.json", "sha256": replay.digest(path)}
