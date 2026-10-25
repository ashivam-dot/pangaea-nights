"""Actual Snowball ASR variants must not invent spoken errors or hide wrong content."""
import pytest
from studio.ladder_voice import wer


@pytest.mark.parametrize('samples,padded', [(16000, True), (48000, False)])
def test_asr_short_clip_margin_preserves_all_audio_and_unprompted_decoding(monkeypatch, samples, padded):
 from types import SimpleNamespace
 import sys
 import numpy as np
 from studio import ladder_voice, voice
 audio = np.linspace(-.5, .5, samples, dtype=np.float32)
 before = audio.copy()
 monkeypatch.setitem(sys.modules, 'librosa', SimpleNamespace(resample=lambda wav, **kwargs: wav.copy()))
 def transcribe(wav, **kwargs):
  assert kwargs == {'language': 'en', 'beam_size': 1, 'vad_filter': False,
                    'condition_on_previous_text': False}
  if padded:
   assert len(wav) == samples + 4800
   np.testing.assert_array_equal(wav[2400:-2400], before)
   assert not np.any(wav[:2400]) and not np.any(wav[-2400:])
  else:
   np.testing.assert_array_equal(wav, before)
  return [SimpleNamespace(text='Comment your number.')], None
 monkeypatch.setattr(voice, 'whisper', lambda: SimpleNamespace(transcribe=transcribe))
 assert ladder_voice._transcribe(audio, 16000) == 'Comment your number.'
 np.testing.assert_array_equal(audio, before)


def test_narration_preflight_refuses_real_asr_loop_and_late_timeline():
 from studio import ladder, ladder_voice, shorts_ladder
 p = ladder.find(ladder.load_banks(strict=True), 'how_deep-02')
 durations = shorts_ladder.estimated_durations(p, 165)
 report = {'wer': 0, 'wpm': 165, 'durations': durations,
           'segments': {ladder_voice.line_id(s): {'text': s['text'], 'heard': s['text'], 'wer': 0}
                        for s in ladder.segments(p)}}
 assert ladder_voice.preflight_problems(p, report) == []
 report['segments']['cta']['heard'] += ' eight' * 26
 report['segments']['cta']['wer'] = 8.667
 assert any('line narration WER' in e for e in ladder_voice.preflight_problems(p, report))
 report['segments']['cta']['wer'] = 0
 report['durations']['verdict'] = 60
 assert any('total' in e for e in ladder_voice.preflight_problems(p, report))


def test_narration_preflight_requires_own_complete_script_and_answer():
 from studio import ladder, ladder_voice, shorts_ladder
 p = ladder.find(ladder.load_banks(strict=True), 'how_deep-02')
 report = {'wer': 0, 'wpm': 165, 'durations': shorts_ladder.estimated_durations(p, 165),
           'segments': {ladder_voice.line_id(s): {'text': s['text'], 'heard': s['text'], 'wer': 0}
                        for s in ladder.segments(p)}}
 report['segments']['verdict']['heard'] = 'Real answer is six.'
 assert any('answer number' in e for e in ladder_voice.preflight_problems(p, report))
 report['segments']['o1']['text'] = 'a different script'
 assert any('exact planned' in e for e in ladder_voice.preflight_problems(p, report))

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


def test_answer_clause_pause_keeps_voice_script_and_actual_caption_times(monkeypatch):
 from studio import ladder_voice, voice
 import numpy as np

 calls = []
 def sentence(text, narrator, speed):
  calls.append((text, narrator, speed))
  return voice.Take(np.ones(2400, dtype=np.float32), [{'text': text, 'start': 0.01, 'end': 0.09}])
 monkeypatch.setattr(voice, '_sentence', sentence)
 text = 'Real answer: number 2, because life may survive there.'
 report = ladder_voice._take([{'id': 'verdict', 'text': text}, {'id': 'o2', 'text': 'Two: right under the ice.'}], 1.096)
 assert calls == [('Real answer.', ladder_voice.VOICE, 1.096),
                  ('number 2, because life may survive there.', ladder_voice.VOICE, 1.096),
                  ('Two: right under the ice.', ladder_voice.VOICE, 1.096)]
 verdict = report['verdict']
 assert verdict['text'] == text and verdict['duration'] == 0.34
 assert np.all(verdict['audio'][2400:5760] == 0)
 assert verdict['words'][1]['start'] == pytest.approx(0.25)
 assert verdict['words'][1]['end'] == pytest.approx(0.33)
 assert report['o2']['duration'] == 0.1


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
