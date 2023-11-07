from typing import *
from torch import nn

def apply_softmax(logits):
	return nn.functional.softmax(logits, dim=-1)

pt_predictions = apply_softmax(pt_outputs.logits)
print(pt_predictions)
