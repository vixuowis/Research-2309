from f00652_create_speaker_embedding import *
import torch

def test_create_speaker_embedding():
    waveform = torch.randn(10, 1, 16000)
    speaker_embeddings = create_speaker_embedding(waveform)
    assert speaker_embeddings.shape == (10, 512)


def test_entry():
    test_create_speaker_embedding()

test_entry()
