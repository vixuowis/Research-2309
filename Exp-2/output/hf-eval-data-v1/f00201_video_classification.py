from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

def video_classification(video):
    '''
    This function classifies the content of a video using a pre-trained model from Hugging Face Transformers.
    The model is fine-tuned on the Kinetics-400 dataset.
    
    Args:
    video (list): A list of video frames. Each frame is a 3D numpy array (channels, height, width).
    
    Returns:
    str: The predicted class of the video content.
    '''
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-short-finetuned-kinetics')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-short-finetuned-kinetics')
    inputs = processor(video, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]