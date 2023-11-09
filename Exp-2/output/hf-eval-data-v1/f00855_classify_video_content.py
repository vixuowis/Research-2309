from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

def classify_video_content(video):
    '''
    Classify a video's content into multiple categories like 'sports', 'comedy', and 'news'.
    
    Args:
        video (np.array): A numpy array representing the video frames.
    
    Returns:
        str: The category of the video content.
    
    '''
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-ssv2')
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-ssv2')
    inputs = processor(video, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]