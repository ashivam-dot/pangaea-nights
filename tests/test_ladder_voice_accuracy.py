"""Actual Snowball ASR variants must not invent spoken errors or hide wrong content."""
import pytest
from studio.ladder_voice import wer

@pytest.mark.parametrize('expected,heard',[
 ('Three: ten metres down, freezing blue twilight.','3 10 meters down, freezing blue twilight.'),
 ('Four: fifty metres, the last light fades.','4-50 meters the last light fades.'),
 ('Five: two hundred metres. Total black.','5-200 meters total black.'),
 ('Six: one kilometre. Crushing cold, no light.','6-1 km, crushing cold no light.'),
 ('Seven: the seafloor, kilometres under the ice.','7. The seafloor, kilometers under the ice.'),
 ('Eight: beside a superheated vent.','8 beside a super-heated vent.'),
])
def test_orthographic_transcript_variants_are_the_same_words(expected,heard):
 assert wer(expected,heard)==0

@pytest.mark.parametrize('expected,heard',[
 ('Two: right under the ice.','to ride under the ice.'),
 ('Six: one kilometre.','Six: one meter.'),
 ('Four: fifty metres.','Four: fifteen meters.'),
 ('Crushing cold, no light.','Crushing cold, light.'),
 ('Depths from 2-10 metres.','Depths from 2-10 meters.'),
])
def test_wrong_numbers_scales_words_and_missing_spoken_range_remain_errors(expected,heard):
 assert wer(expected,heard)>0

def test_an_explicitly_spoken_range_still_passes():
 assert wer('Depths from 2-10 metres.','Depths from two to ten meters.')==0


@pytest.mark.parametrize('recovered', [True, False])
def test_missing_answer_word_retries_only_verdict_even_below_wer_limit(tmp_path, monkeypatch, recovered):
 from studio import ladder_voice, shorts_ladder
 import numpy as np

 text = ('Real answer number 2 because photosynthesis can work under ice up to twenty metres thick. '
         'Life may have found a foothold there.')
 missed = text.replace('number ', '')
 assert wer(text, missed) < ladder_voice.MAX_WER
 lines = [{'kind': 'hook', 'text': 'Pick your depth.'}, {'kind': 'verdict', 'text': text}]
 monkeypatch.setattr(ladder_voice.ladder, 'segments', lambda p: lines)
 monkeypatch.setattr(ladder_voice, 'wpm', lambda segments: ladder_voice.TARGET_WPM)
 requested = []

 def take(rows, speed):
  requested.append(([row['id'] for row in rows], speed))
  return {row['id']: {'text': row['text'], 'audio': np.zeros(240, dtype=np.float32),
                     'duration': 0.01, 'words': []} for row in rows}

 heard = iter(['Pick your depth.', missed, 'Pick your depth.', text if recovered else missed])
 monkeypatch.setattr(ladder_voice, '_take', take)
 monkeypatch.setattr(ladder_voice, '_transcribe', lambda audio, sr: next(heard))
 report = ladder_voice.narrate({'answer_n': 2}, tmp_path)
 assert requested == [(['hook', 'verdict'], 1.0), (['verdict'], 0.96)]
 assert shorts_ladder.verdict_heard(report, 2) is recovered
