import pytest

from studio import images
from studio.cli import thumb_source


def test_own_thumbnail_first_then_earliest_chapter_plate(tmp_path):
    for name in ("t0s00.jpg", "tail00.jpg", "c02s00.jpg", "c01s03.jpg", "c01s03.depth.png"):
        (tmp_path / name).write_bytes(b"x")
    assert thumb_source(tmp_path).name == "c01s03.jpg"
    (tmp_path / "thumb.png").write_bytes(b"x")
    assert thumb_source(tmp_path).name == "thumb.png"


def test_no_image_at_all_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="no thumbnail"):
        thumb_source(tmp_path)


def test_thumbnail_is_generated_first():
    episode = {"art_direction": "art", "chapters": [{"n": 1, "scenes": ["a", "b"]}], "tail_scenes": ["t"],
               "trailers": [{"scenes": ["v"]}], "thumbnail": {"prompt": "p"}}
    assert images.manifest(episode)[0]["id"] == "thumb"
    assert images.shard(images.manifest(episode), 0, 3)[0]["id"] == "thumb"
