from typing import *
import torch

def get_predicted_label(logits, id2label):
    predicted_class_ids = torch.argmax(logits).item()
    predicted_label = id2label[predicted_class_ids]
    return predicted_label
