import base64
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from studio import config, ladder_images as li, ladder_reference, ladder_replay, qa, sdcpp


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    folder = tmp_path / ".github/replay"
    folder.mkdir(parents=True)
    parent = folder / "museum.png"
    Image.fromarray((np.arange(12 * 12 * 3).reshape(12, 12, 3) % 200 + 30).astype(np.uint8)).save(parent)
    cropped = folder / "shell.png"
    with Image.open(parent) as src:
        src.crop((2, 2, 10, 10)).save(cropped)
    page, url = "https://commons.wikimedia.org/wiki/File:Museum.png", "https://upload.wikimedia.org/museum.png"
    licence = folder / "licence.json"
    licence.write_text(json.dumps({"query": {"pages": {"1": {"title": "File:Museum.png", "imageinfo": [{
        "url": url, "descriptionurl": page, "extmetadata": {
            "Artist": {"value": "<a>Museum photographer</a>"}, "LicenseShortName": {"value": "CC0"}}}]}}}}))
    def bound(p):
        return {"file": p.name, "sha256": ladder_replay.digest(p)}
    row = {**bound(cropped), "parent": bound(parent), "license_evidence": bound(licence),
           "title": "File:Museum.png", "author": "Museum photographer", "license": "CC0",
           "page": page, "url": url, "crop_ltrb": [2, 2, 10, 10], "credit": "Museum photo, CC0; AI adaptation."}
    data = {"schema": 1, "premise": "how_deep-04", "source": {"run_id": 123, "sha": "a" * 40, "key": "original"},
            "art_only": ["o3"], "images": {}, "references": {"o3": [row]}}
    path = folder / "repair.json"
    path.write_text(json.dumps(data))
    return path, ladder_replay.load(".github/replay/repair.json")


def test_reference_is_bound_to_its_crop_and_commons_attribution(reference):
    _, data = reference
    refs = ladder_reference.inputs(data, "o3")
    assert refs[0]["path"].name == "shell.png"
    assert refs[0]["proof"]["license"] == "CC0"
    assert ladder_reference.inputs(data, "o4") == []


def attributed_reference(data, license_name="CC BY 4.0"):
    row = data["references"]["o3"][0]
    url = ladder_reference.ATTRIBUTION_LICENSES[license_name]
    evidence_path = data["_folder"] / row["license_evidence"]["file"]
    evidence = json.loads(evidence_path.read_text())
    meta = evidence["query"]["pages"]["1"]["imageinfo"][0]["extmetadata"]
    meta.update(LicenseShortName={"value": license_name}, LicenseUrl={"value": url})
    evidence_path.write_text(json.dumps(evidence))
    row["license_evidence"]["sha256"] = ladder_replay.digest(evidence_path)
    row.update(license=license_name, license_url=url,
               credit=f'Museum photographer, {row["page"]}, {license_name}, {url}, AI-adapted with illustrative colour')
    return row


@pytest.mark.parametrize("license_name", ["CC BY 3.0", "CC BY 4.0"])
def test_attributed_references_require_the_matching_commons_license(reference, license_name):
    _, data = reference
    row = attributed_reference(data, license_name)
    assert ladder_reference.inputs(data, "o3")[0]["proof"] == row
    row["license_url"] = "https://creativecommons.org/licenses/by-nc/4.0"
    with pytest.raises(ValueError, match="attribution"):
        ladder_reference.inputs(data, "o3")


@pytest.mark.parametrize("missing", ["Museum photographer", "https://commons.wikimedia.org/wiki/File:Museum.png",
                                      "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0", "AI-adapted"])
def test_attributed_reference_cannot_omit_required_credit(reference, missing):
    _, data = reference
    row = attributed_reference(data)
    row["credit"] = row["credit"].replace(missing, "")
    with pytest.raises(ValueError, match="attribution"):
        ladder_reference.inputs(data, "o3")


@pytest.mark.parametrize("unsupported", ["CC BY-SA 4.0", "CC BY-NC 4.0", "Public domain", "unknown"])
def test_additional_licences_do_not_silently_enter_reference_generation(reference, unsupported):
    _, data = reference
    row = attributed_reference(data)
    row["license"] = unsupported
    with pytest.raises(ValueError):
        ladder_reference.inputs(data, "o3")


def test_saved_licence_url_must_match_attribution_even_when_file_hash_is_valid(reference):
    _, data = reference
    row = attributed_reference(data)
    evidence_path = data["_folder"] / row["license_evidence"]["file"]
    evidence = json.loads(evidence_path.read_text())
    evidence["query"]["pages"]["1"]["imageinfo"][0]["extmetadata"]["LicenseUrl"]["value"] = "https://creativecommons.org/licenses/by-nc/4.0"
    evidence_path.write_text(json.dumps(evidence))
    row["license_evidence"]["sha256"] = ladder_replay.digest(evidence_path)
    with pytest.raises(ValueError, match="attribution"):
        ladder_reference.inputs(data, "o3")


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("file", "../outside.png"),
    ("author", "Someone else"), ("license", "unverified"), ("crop_ltrb", [0, 0, 8, 8]),
    ("crop_ltrb", [0, 0, 100, 100]), ("page", "https://example.org/unrelated")])
def test_corrupt_or_unlicensed_reference_is_refused(reference, field, value):
    _, data = reference
    data["references"]["o3"][0][field] = value
    with pytest.raises(ValueError):
        ladder_reference.inputs(data, "o3")


def test_generation_references_cannot_enter_an_ordinary_or_cached_replay(reference):
    path, data = reference
    for change in ({"art_only": ["o4"]}, {"cached_only": True}):
        raw = json.loads(path.read_text()) | change
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            ladder_replay.load(".github/replay/repair.json")
        path.write_text(json.dumps({k: v for k, v in data.items() if k != "_folder"}))


def test_reference_uses_normal_quality_checks_and_retry_budget(reference, tmp_path, monkeypatch):
    _, data = reference
    job = {"id": "o3", "scene": "shell profile", "prompt": "exact shell profile", "luma": (62, 198)}
    monkeypatch.setattr(li.ladder, "image_jobs", lambda p: [job])
    monkeypatch.setitem(sdcpp.CANDIDATES, "test", SimpleNamespace(width=8, height=8))
    calls, checks = [], []
    raw = io.BytesIO()
    Image.fromarray(np.repeat(np.linspace(50, 210, 64).reshape(8, 8, 1), 3, axis=2).astype(np.uint8)).save(raw, format="PNG")
    class Server:
        load_seconds = 0
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def generate(self, *a): pytest.fail("reference silently became text-only")
        def edit(self, prompt, seed, paths, w, h):
            calls.append((prompt, seed, paths, w, h))
            return raw.getvalue()
    monkeypatch.setattr(sdcpp, "Server", lambda *a: Server())
    def review(*a):
        checks.append(True)
        return qa.Verdict(len(checks) > 1, [] if len(checks) > 1 else ["person 0.9"],
                          0 if len(checks) > 1 else .9, 0, 0, .4, 6, 120, 50, 12)
    monkeypatch.setattr(qa, "review", review)
    monkeypatch.setattr(qa, "figure", lambda img: .9 if len(checks) == 1 else 0)
    monkeypatch.setattr(qa, "embed", lambda img: np.array([1., 0.]))
    monkeypatch.setattr(li.plates, "save_plate", lambda img, stem, **kw: {
        "plate": stem.name + ".jpg", "depth": stem.name + ".depth.png"})
    with pytest.raises(ValueError, match="attribution"):
        li.generate({"id": data["premise"]}, "original", tmp_path / "no-credit", 0, 1, "test", replay=data)
    report = li.generate({"id": data["premise"], "description": data["references"]["o3"][0]["credit"]},
                         "original", tmp_path / "art", 0, 1, "test", replay=data)
    entry = report["images"][0]
    assert len(calls) == len(checks) == 3 <= li.TAKES
    assert [r["take"] for r in entry["kept"]] == [1, 2]
    assert entry["reference_inputs"] == data["references"]["o3"]
    assert all(c[0] == job["prompt"] and c[2][0].name == "shell.png" and c[3:] == (8, 8) for c in calls)


def test_klein_edit_sends_exact_sampling_settings_and_closes_reference(reference, monkeypatch):
    _, data = reference
    server = object.__new__(sdcpp.Server)
    server.c = SimpleNamespace(name="klein4b-q8-768", steps=4, cfg=1., sampler="euler", prompt_suffix="")
    server.port = 8123
    streams = []
    def post(url, *, files, data, timeout):
        assert url.endswith("/v1/images/edits") and data["size"] == "432x768" and data["n"] == "1"
        extra = json.loads(data["prompt"].split("<sd_cpp_extra_args>")[1].split("</sd_cpp_extra_args>")[0])
        assert extra == {"seed": 17, "sample_params": {"sample_steps": 4, "sample_method": "euler",
                                                    "guidance": {"txt_cfg": 1.}}}
        streams.append(files[0][1][1])
        assert streams[0].read()
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"data": [{"b64_json": base64.b64encode(b"image").decode()}]})
    monkeypatch.setattr(sdcpp.requests, "post", post)
    assert server.edit("exact prompt", 17, [ladder_reference.inputs(data, "o3")[0]["path"]], 432, 768) == b"image"
    assert streams[0].closed
