from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

def classify_video_highlights(video_path):
    """
    This function classifies video highlights using the TimesformerForVideoClassification model from Hugging Face Transformers.
    The model is pre-trained on the Kinetics-600 dataset.
    
    Args:
        video_path (str): The path to the video file.
    
    Returns:
        str: The predicted class of action or activity occurring in the video.
    """
    video_frames = load_video_frames(video_path)
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-hr-finetuned-k600')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-hr-finetuned-k600')
    inputs = processor(images=video_frames, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]