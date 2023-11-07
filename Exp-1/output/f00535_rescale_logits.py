from typing import *
import torch.nn.functional as F

def rescale_logits(logits, image):
    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1],
        mode='bilinear',
        align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    return pred_seg
