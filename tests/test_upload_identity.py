import hashlib

import pytest

from ops import upload_identity as identity


def test_recovery_requires_the_original_reviewed_bytes_and_retains_the_story_marker(tmp_path):
    mp4 = tmp_path / "short.mp4"
    mp4.write_bytes(b"approved original MP4")
    digest = hashlib.sha256(mp4.read_bytes()).hexdigest()
    binding = identity.media_tag(mp4, digest)
    tags = identity.bound_tags(["a descriptive tag " * 12, "other tag " * 12, "channel:story"], binding)
    assert tags[-2:] == [binding, "channel:story"]
    assert sum(len(t) + 2 for t in tags) <= 480
    identity.check_existing({"id": "same-video", "snippet": {"tags": tags}}, binding)
    with pytest.raises(ValueError, match="existing upload same-video"):
        identity.check_existing({"id": "same-video", "snippet": {"tags": ["channel:story"]}}, binding)
    mp4.write_bytes(b"unreviewed replacement")
    with pytest.raises(ValueError, match="differ"):
        identity.media_tag(mp4, digest)
