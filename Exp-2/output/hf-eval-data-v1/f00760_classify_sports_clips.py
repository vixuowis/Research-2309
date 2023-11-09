from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

def classify_sports_clips(video):
    """
    Classify sports clips by identifying the type of sports being played in the video.

    Args:
        video (list): A list of numpy arrays representing frames of the video. Each numpy array is of shape (3, 224, 224).

    Returns:
        str: The predicted class of the sports being played in the video.
    """
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k400')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k400')
    inputs = processor(video, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]