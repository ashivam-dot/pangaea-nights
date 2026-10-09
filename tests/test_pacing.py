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
