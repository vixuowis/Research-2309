from f00535_rescale_logits import *
import torch
import torch.nn.functional as F


def test_rescale_logits():
    logits = torch.randn(1, 2, 64, 64)
    image = Image.open('image.jpg')
    pred_seg = rescale_logits(logits, image)
    assert pred_seg.shape == (64, 64)


test_rescale_logits()
