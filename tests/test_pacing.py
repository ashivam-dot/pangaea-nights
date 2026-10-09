import pytest

from studio import narrate, voice


def test_pauses_stretch_to_reach_the_target_pace():
    words, speech = 240, 80.0  # 240 words at ~180 wpm articulation
    s = narrate.pause_scale(words, speech, sentence_weights=20, paragraph_gaps=2)
    total = speech + s * (20 * voice.SENTENCE_GAP + 2 * voice.PARAGRAPH_GAP)
    assert words / total * 60 == pytest.approx(voice.TARGET_WPM, abs=0.5)


def test_pause_scale_is_bounded():
    assert narrate.pause_scale(100, 200.0, 5, 0) == narrate.MIN_PAUSE_SCALE  # already slow: never squash
    assert narrate.pause_scale(400, 30.0, 3, 0) == narrate.MAX_PAUSE_SCALE  # never absurdly long silences
    assert narrate.pause_scale(10, 5.0, 0, 0) == 1.0


def test_dense_text_reads_faster_but_never_slower_than_default():
    want = voice.TARGET_WPM / narrate.SPEECH_SHARE
    fast = narrate.reading_speed(120, 120 / (want * 0.85) * 60, 0.8)
    assert fast == pytest.approx(0.8 / 0.85, abs=0.002)
    assert narrate.reading_speed(200, 60.0, 0.8) == 0.8
    assert narrate.reading_speed(100, 200.0, 0.8) == narrate.MAX_SPEED


def _episode(words):
    return {"chapters": [{"n": i + 1, "paragraphs": ["w " * w]} for i, w in enumerate(words)]}


def test_voice_shards_are_contiguous_balanced_and_never_idle():
    assert narrate.chapter_shards(_episode([530, 540]), 2) == [[1], [2]]
    assert narrate.chapter_shards(_episode([1000] * 14), 2) == [list(range(1, 8)), list(range(8, 15))]
    groups = narrate.chapter_shards(_episode([1000] * 14), 5)
    assert [n for g in groups for n in g] == list(range(1, 15)) and all(groups)
    assert narrate.chapter_shards(_episode([500]), 2) == [[1], []]
