from f00703_from_pretrained import *
import torch


def test_from_pretrained():
    model = from_pretrained('distilbert-base-uncased')
    assert isinstance(model, DistilBertModel)
    print('Test passed.')


test_from_pretrained()
