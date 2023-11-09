from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

def classify_video(video):
    '''
    Classify the exercise in a video using a pre-trained model.
    
    Args:
        video (list): A list of numpy arrays representing frames of a video. Each frame is a 3D array (channels, height, width).
    
    Returns:
        str: The predicted class of the exercise.
    
    Raises:
        ValueError: If the input video is not a list or if it's empty.
    '''
    if not isinstance(video, list) or not video:
        raise ValueError('Input video must be a non-empty list.')
    
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k600')
    
    inputs = processor(images=video, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
    return model.config.id2label[predicted_class_idx]