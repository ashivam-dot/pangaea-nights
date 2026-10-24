"""Composition exposure cannot bypass source/final gates or alter the darkness of deep water."""
import numpy as np
from studio.shorts_ladder import exposure_lut


def test_bounded_fixed_grade_lifts_midtone_foreground_and_preserves_black_water():
 frame=np.full((180,100,3),120,np.uint8);frame[:80]=0
 lut,report=exposure_lut([frame,frame.copy()])
 assert .8<=report['gamma']<1
 assert lut[0]==0 and lut[255]==255 and np.all(np.diff(lut.astype(int))>=0)
 assert report['after_min']>report['before_min']
 assert np.array_equal(lut[frame],lut[frame.copy()])


def test_already_readable_frame_is_unchanged():
 frame=np.full((180,100,3),120,np.uint8)
 lut,report=exposure_lut([frame])
 assert report['gamma']==1 and np.array_equal(lut,np.arange(256,dtype=np.uint8))


def test_flat_dark_source_is_not_rescued_past_the_bound():
 lut,report=exposure_lut([np.full((180,100,3),5,np.uint8)])
 assert report['gamma']==.8 and report['after_min']<60
 assert lut[0]==0
