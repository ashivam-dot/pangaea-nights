"""Replay actual CLIP embeddings for the modern-shoes escape and directly reviewed controls."""
import json
from pathlib import Path
import numpy as np
import pytest
from studio import qa

FIXTURE=Path(__file__).parent/'fixtures/ladder-modern-feet.npz'

@pytest.fixture
def actual_embeddings():
    saved=np.load(FIXTURE,allow_pickle=False)
    vocabulary=saved['vocabulary'].tolist()
    vectors={text:saved['labels'][i] for i,text in enumerate(vocabulary)}
    labels=np.stack([vectors[text] for text in qa.FIGURE_NO+qa.FIGURE_YES])
    labels/=np.linalg.norm(labels,axis=-1,keepdims=True)
    scores=[qa.figure_probability(image,labels) for image in saved['images']]
    return json.loads(FIXTURE.with_suffix('.json').read_text()),scores

def test_actual_modern_shoes_are_rejected(actual_embeddings):
    metadata,scores=actual_embeddings
    row=metadata['rows'][0]
    assert row['direct_decision']=='hold' and row['before']<qa.FIGURE_LIMIT
    assert scores[0]>qa.FIGURE_LIMIT
    assert scores[0]==pytest.approx(row['after'],abs=5e-6)

def test_reviewed_clean_controls_remain_eligible(actual_embeddings):
    metadata,scores=actual_embeddings
    for i,row in enumerate(metadata['rows']):
        assert scores[i]==pytest.approx(row['after'],abs=5e-6)
        if row['direct_decision']=='pass' and row['before']<=qa.FIGURE_LIMIT:
            assert scores[i]<=qa.FIGURE_LIMIT,row['file']
