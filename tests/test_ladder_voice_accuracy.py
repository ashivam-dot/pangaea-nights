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
