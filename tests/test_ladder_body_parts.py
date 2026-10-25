"""Replay actual CLIP embeddings for the modern-shoes escape and directly reviewed controls."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from studio import qa

FIXTURE=Path(__file__).parent/'fixtures/ladder-modern-feet.npz'

@pytest.fixture
def actual_embeddings(monkeypatch):
    saved=np.load(FIXTURE,allow_pickle=False)
    vocabulary=saved['vocabulary'].tolist()
    vectors={text:torch.tensor(saved['labels'][i]) for i,text in enumerate(vocabulary)}
    class Encoder:
        def encode_text(self,labels):
            return torch.stack([vectors[text] for text in labels])
    monkeypatch.setattr(qa,'_figure_emb',None)
    monkeypatch.setattr(qa,'_load',lambda:(Encoder(),None,lambda labels:labels))
    monkeypatch.setattr(qa,'embed',lambda image:saved['images'][image.info['row']])
    return json.loads(FIXTURE.with_suffix('.json').read_text())

def image_for(index):
    image=Image.new('RGB',(1,1));image.info['row']=index;return image

def test_actual_modern_shoes_are_rejected(actual_embeddings):
    row=actual_embeddings['rows'][0]
    assert row['direct_decision']=='hold' and row['before']<qa.FIGURE_LIMIT
    assert qa.figure(image_for(0))>qa.FIGURE_LIMIT

def test_reviewed_clean_controls_remain_eligible(actual_embeddings):
    for i,row in enumerate(actual_embeddings['rows']):
        if row['direct_decision']=='pass' and row['before']<=qa.FIGURE_LIMIT:
            assert qa.figure(image_for(i))<=qa.FIGURE_LIMIT,row['file']
